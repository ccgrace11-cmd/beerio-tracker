from __future__ import annotations
import os
import json
import base64
import io
from datetime import datetime
import random
import glob
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import anthropic
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image, ImageOps
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

    # Apply EXIF orientation - critical for iPhone photos!
    # Without this, images from phones may be rotated/flipped incorrectly
    image = ImageOps.exif_transpose(image)

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

    # Build the example JSON with only participating players (showing example points)
    example_points = {player: 45 - (i * 5) for i, player in enumerate(participating_players)}
    example_json = json.dumps({"points": example_points, "confidence": "high"}, indent=2)

    prompt = f"""Extract the TOTAL GRAND PRIX POINTS for each player from this Mario Kart 8 Deluxe results screen.

PLAYERS TO FIND:
{char_list}

INSTRUCTIONS:
1. Find each character's row in the results list
2. Read the POINT TOTAL displayed on the RIGHT SIDE of their row (typically 0-60)
3. Be VERY careful reading digits - common confusions are 5/6, 6/8, 3/5

CHARACTER IDENTIFICATION (in order of difficulty):

BLACK YOSHI (Skinny) - HARDEST TO SPOT:
- Very dark gray/charcoal colored dinosaur - almost black
- Can blend into dark backgrounds - look carefully!
- Same shape as other Yoshis but MUCH darker
- If you see a Yoshi that looks dark/shadowy, it's Black Yoshi

GREEN YOSHI (Matty): Bright lime green dinosaur - classic Yoshi color
BLUE YOSHI (Emily): Light blue/cyan/sky blue dinosaur
TOAD (Emma): Small character with white mushroom cap and red spots
DONKEY KONG (Jake): Large brown gorilla with red necktie

READING NUMBERS CAREFULLY:
- Look at each digit individually
- 5 has a flat top, 6 has a curved top
- 8 has two loops, 6 has one loop at bottom
- Double-check your reading before responding

Return JSON with the POINTS (the number shown on screen):
{example_json}

Use null ONLY if the character is truly not in this race."""

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

        # Get points directly from Claude's response
        points = result.get("points")

        # Handle case where Claude returned data in unexpected format
        if not points:
            # Check if Claude returned players_found instead (old format)
            if "players_found" in result:
                points = result.get("players_found")
            else:
                return jsonify({"error": f"Could not extract points. Raw response: {result}"}), 500

        return jsonify({
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


# Training interface routes
SAMPLE_IMAGES_DIR = os.path.expanduser("~/Downloads/Sample MK Results")
TRAINING_DATA_FILE = os.path.join(os.path.dirname(__file__), "training_data", "feedback.json")


def load_training_data():
    """Load existing training feedback data."""
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'r') as f:
            return json.load(f)
    return {"feedback": [], "stats": {"total": 0, "correct": 0}}


def save_training_data(data):
    """Save training feedback data."""
    os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
    with open(TRAINING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


@app.route("/train")
def train():
    """Serve the training interface."""
    return render_template("train.html", players=PLAYERS)


@app.route("/api/train/random")
def get_random_sample():
    """Get a random sample image for training."""
    # Get all image files
    patterns = ["*.HEIC", "*.heic", "*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for pattern in patterns:
        all_images.extend(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, pattern)))

    if not all_images:
        return jsonify({"error": "No sample images found"}), 404

    # Load training data to see which images have been reviewed
    training_data = load_training_data()
    reviewed = {f["filename"] for f in training_data["feedback"]}

    # Prefer unreveiwed images
    unreviewed = [img for img in all_images if os.path.basename(img) not in reviewed]

    if unreviewed:
        selected = random.choice(unreviewed)
    else:
        selected = random.choice(all_images)

    filename = os.path.basename(selected)

    return jsonify({
        "filename": filename,
        "total_images": len(all_images),
        "reviewed_count": len(reviewed),
        "remaining": len(unreviewed)
    })


@app.route("/api/train/image/<filename>")
def serve_sample_image(filename):
    """Serve a sample image file."""
    filepath = os.path.join(SAMPLE_IMAGES_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "Image not found"}), 404

    # Convert HEIC to JPEG for browser compatibility
    if filename.lower().endswith('.heic'):
        image = Image.open(filepath)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85)
        output.seek(0)
        return send_file(output, mimetype='image/jpeg')

    return send_file(filepath)


@app.route("/api/train/parse", methods=["POST"])
def parse_training_image():
    """Parse a training image by filename."""
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    filepath = os.path.join(SAMPLE_IMAGES_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "Image not found"}), 404

    try:
        # Read and convert image
        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data, _ = convert_image_to_jpeg(image_base64)

        # Parse with Claude
        result = parse_race_image(image_data, PLAYERS)

        if "error" in result:
            return jsonify(result), 500

        points = result.get("points") or result.get("players_found", {})

        return jsonify({
            "points": points,
            "confidence": result.get("confidence", "unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train/feedback", methods=["POST"])
def save_feedback():
    """Save training feedback."""
    data = request.get_json()

    filename = data.get("filename")
    ai_guess = data.get("ai_guess", {})
    correct_answer = data.get("correct_answer", {})

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Load existing data
    training_data = load_training_data()

    # Calculate accuracy for this sample
    correct_count = 0
    total_count = 0
    errors = []

    for player in PLAYERS:
        ai_val = ai_guess.get(player)
        correct_val = correct_answer.get(player)

        if correct_val is not None:
            total_count += 1
            if ai_val == correct_val:
                correct_count += 1
            else:
                errors.append({
                    "player": player,
                    "ai_guess": ai_val,
                    "correct": correct_val
                })

    # Add feedback entry
    feedback_entry = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "ai_guess": ai_guess,
        "correct_answer": correct_answer,
        "errors": errors,
        "accuracy": correct_count / total_count if total_count > 0 else 0
    }

    training_data["feedback"].append(feedback_entry)
    training_data["stats"]["total"] += 1
    if len(errors) == 0:
        training_data["stats"]["correct"] += 1

    save_training_data(training_data)

    return jsonify({
        "success": True,
        "accuracy": feedback_entry["accuracy"],
        "errors": errors,
        "overall_stats": training_data["stats"]
    })


@app.route("/api/train/stats")
def get_training_stats():
    """Get training statistics."""
    training_data = load_training_data()

    # Analyze common errors
    error_counts = {}
    for feedback in training_data["feedback"]:
        for error in feedback.get("errors", []):
            player = error["player"]
            if player not in error_counts:
                error_counts[player] = {"count": 0, "examples": []}
            error_counts[player]["count"] += 1
            if len(error_counts[player]["examples"]) < 5:
                error_counts[player]["examples"].append({
                    "ai": error["ai_guess"],
                    "correct": error["correct"]
                })

    return jsonify({
        "stats": training_data["stats"],
        "error_analysis": error_counts,
        "feedback_count": len(training_data["feedback"])
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
