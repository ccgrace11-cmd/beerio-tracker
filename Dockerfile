# Use official Playwright image with browsers pre-installed
FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render uses PORT env var)
EXPOSE 10000

# Run with gunicorn
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
