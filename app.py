from __future__ import annotations
import os
import json
import base64
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import anthropic
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import pillow_heif
from playwright.sync_api import sync_playwright

# Register HEIF/HEIC format with Pillow
pillow_heif.register_heif_opener()

load_dotenv()

app = Flask(__name__)

# Known players and their characters
PLAYERS = ["Emma", "Emily", "Matty", "Jake", "Skinny"]
PLAYER_CHARACTERS = {
    "Emma": "Toad (white mushroom cap with red spots, small character)",
    "Emily": "Blue Yoshi (light blue/cyan colored dinosaur)",
    "Matty": "Green Yoshi (bright green colored dinosaur - the classic Yoshi color)",
    "Jake": "Donkey Kong (large brown gorilla with red tie)",
    "Skinny": "Black Yoshi (dark gray/black colored dinosaur - very dark, almost charcoal colored)"
}

# Google Sheets setup
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


def get_sheets_client():
    """Initialize Google Sheets client from credentials."""
    creds_json = os.getenv("GOOGLE_CREDENTIALS")
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set")

    # Try to decode as base64 first (more reliable for env vars with newlines)
    try:
        decoded = base64.b64decode(creds_json).decode('utf-8')
        creds_dict = json.loads(decoded)
    except Exception:
        # Fall back to direct JSON parsing
        creds_dict = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(credentials)


def get_spreadsheet():
    """Get the Beerio Tracker spreadsheet."""
    client = get_sheets_client()
    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    if not spreadsheet_id:
        raise ValueError("SPREADSHEET_ID environment variable not set")
    return client.open_by_key(spreadsheet_id)


def convert_image_to_jpeg(image_base64: str) -> tuple[str, str]:
    """Convert any image format (including HEIC) to JPEG base64.

    Returns: (base64_data, media_type)
    """
    # Decode the base64 image
    image_bytes = base64.b64decode(image_base64)

    # Open with Pillow (which now supports HEIC via pillow-heif)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (handles RGBA, etc.)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    # Resize if too large (max 2048px on longest side for efficiency)
    max_size = 2048
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Save as JPEG to bytes
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=85)
    output.seek(0)

    # Encode back to base64
    jpeg_base64 = base64.b64encode(output.read()).decode('utf-8')

    return jpeg_base64, "image/jpeg"


def parse_race_image(image_base64: str, participating_players: list[str] = None) -> dict:
    """Use Claude Vision to extract player positions from MK8D results screen."""
    client = anthropic.Anthropic()

    # Image is always JPEG after conversion
    media_type = "image/jpeg"

    # Use all players if none specified
    if participating_players is None:
        participating_players = PLAYERS

    # Build character descriptions for participating players only
    char_list = "\n".join([
        f"- {player}: {PLAYER_CHARACTERS[player]}"
        for player in participating_players
        if player in PLAYER_CHARACTERS
    ])

    # Build the example JSON with only participating players
    example_players = {player: (i + 1) for i, player in enumerate(participating_players)}
    example_json = json.dumps({"players_found": example_players, "confidence": "high"}, indent=2)

    prompt = f"""Find the finishing positions (1-12) of these players in the Mario Kart 8 Deluxe results screen.

PLAYERS TO FIND:
{char_list}

CRITICAL: Look at the small circular character ICON next to each racer name.

For BLACK YOSHI (Skinny): Look for ANY very dark/black circular icon. The icon will appear much darker than other Yoshis - almost black or very dark gray. Even if it's hard to see details, a dark round icon = Black Yoshi = Skinny.

For GREEN YOSHI (Matty): Bright green circular icon.
For BLUE YOSHI (Emily): Light blue/cyan circular icon.
For TOAD (Emma): Red and white mushroom icon.
For DONKEY KONG (Jake): Brown gorilla face icon.

IMPORTANT: If you see a dark/black icon that could be a Yoshi, it IS Black Yoshi (Skinny).

Position 1 is at top of the screen, position 12 at bottom.

Return JSON: {example_json}

Use null ONLY if you truly cannot find the character."""

    system_prompt = """You are a Mario Kart 8 Deluxe race results analyzer. You MUST respond with ONLY valid JSON - no explanations, no markdown code blocks, just raw JSON. Identify characters by their visual appearance (color and shape), not by player names which vary by console."""

    message = client.messages.create(
        model="claude-opus-4-20250514",  # Using Opus for better vision accuracy
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    # Parse the response
    response_text = message.content[0].text.strip()
    print(f"Claude raw response: {response_text}")  # Debug logging

    # Try to extract JSON from the response
    try:
        # Handle case where response might have markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        print(f"Parsed result: {result}")  # Debug logging
        return result
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")  # Debug logging
        return {"error": f"Failed to parse AI response: {str(e)}", "raw": response_text}


# MK8D points by finishing position (1st-12th)
MK8D_POINTS = {
    1: 15, 2: 12, 3: 10, 4: 9, 5: 8, 6: 7,
    7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}


def position_to_points(players_found: dict) -> dict:
    """Convert absolute positions (1-12) to MK8D points."""
    points = {}
    for player, position in players_found.items():
        if position is not None and position in MK8D_POINTS:
            points[player] = MK8D_POINTS[position]
        else:
            points[player] = None
    return points


@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html", players=PLAYERS)


@app.route("/api/parse", methods=["POST"])
def parse_image():
    """Parse a race results image and return player positions."""
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Remove data URL prefix if present
    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Get participating players (default to all)
    participating_players = data.get("participating_players", PLAYERS)

    try:
        # Convert image to JPEG (handles HEIC and other formats)
        image_data, _ = convert_image_to_jpeg(image_data)

        result = parse_race_image(image_data, participating_players)

        if "error" in result:
            return jsonify(result), 500

        # Convert positions to points
        absolute_positions = result.get("players_found", {})
        points = position_to_points(absolute_positions)

        return jsonify({
            "absolute_positions": absolute_positions,
            "points": points,
            "confidence": result.get("confidence", "unknown"),
            "participating_players": participating_players
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/submit", methods=["POST"])
def submit_results():
    """Submit race results to Google Sheets."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    race_type = data.get("race_type")
    points = data.get("points", {})

    if not race_type:
        return jsonify({"error": "Race type is required"}), 400

    try:
        spreadsheet = get_spreadsheet()

        # List all worksheets for debugging
        worksheets = [ws.title for ws in spreadsheet.worksheets()]
        print(f"Available worksheets: {worksheets}")

        # Find the correct worksheet (try variations of the name)
        target_sheet_name = None
        for name in ["Overall Inputs", "Overall inputs", "overall inputs", "Inputs"]:
            if name in worksheets:
                target_sheet_name = name
                break

        if not target_sheet_name:
            return jsonify({
                "error": f"Could not find 'Overall Inputs' worksheet. Available: {worksheets}"
            }), 400

        sheet = spreadsheet.worksheet(target_sheet_name)
        print(f"Using worksheet: {sheet.title}")

        # Get all values to find the next Race ID and row to write to
        all_values = sheet.get_all_values()

        # Find existing race data (non-empty Race ID in column A)
        next_race_id = 1
        next_row = 2  # Start after header (row 1)

        for i, row in enumerate(all_values[1:], start=2):  # Start at row 2 (skip header)
            if row and row[0] and row[0].strip():
                try:
                    race_id = int(row[0])
                    next_race_id = max(next_race_id, race_id + 1)
                    next_row = i + 1  # Next row after this one
                except ValueError:
                    pass

        print(f"Next Race ID: {next_race_id}, Writing to row: {next_row}")

        # Format date as "Mon D YYYY" (e.g., "Jun 5 2025")
        today = datetime.now().strftime("%b %-d %Y")

        # Build the row: Race ID, Race Type, Date, Emma, Emily, Matty, Jake, Skinny (points)
        row_data = [
            next_race_id,
            race_type,
            today,
            points.get("Emma", ""),
            points.get("Emily", ""),
            points.get("Matty", ""),
            points.get("Jake", ""),
            points.get("Skinny", "")
        ]

        # Update specific row instead of appending
        cell_range = f"A{next_row}:H{next_row}"
        sheet.update(values=[row_data], range_name=cell_range, value_input_option="USER_ENTERED")

        return jsonify({
            "success": True,
            "race_id": next_race_id,
            "message": f"Race #{next_race_id} added successfully!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/screenshot", methods=["GET"])
def get_summary_screenshot():
    """Take a screenshot of the Summary tab (clean, cropped view)."""
    try:
        spreadsheet_id = os.getenv("SPREADSHEET_ID")
        summary_gid = os.getenv("SUMMARY_GID", "1357725660")

        # Use the embedded/preview view for a cleaner look
        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/preview#gid={summary_gid}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Wide viewport to capture tables and graphs
            page = browser.new_page(viewport={"width": 1400, "height": 700})

            # Navigate to the sheet
            page.goto(sheet_url, wait_until="networkidle", timeout=60000)

            # Wait longer for charts to render
            page.wait_for_timeout(6000)

            # Take screenshot - capture full content
            screenshot_bytes = page.screenshot(
                clip={"x": 0, "y": 0, "width": 1400, "height": 580}
            )
            browser.close()

        # Return as base64 for easy frontend use
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{screenshot_base64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
