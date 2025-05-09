import os

# Default password if not set in environment
DEFAULT_PASSWORD = "123456"

# Get password from environment variable or use default
API_KEY = os.environ.get("API_KEY", DEFAULT_PASSWORD)

# Directory for service account credential files
CREDENTIALS_DIR = os.environ.get("CREDENTIALS_DIR", "/app/credentials")

# JSON string for service account credentials (can be one or multiple comma-separated)
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get("GOOGLE_CREDENTIALS_JSON")

# API Key for Vertex Express Mode
VERTEX_EXPRESS_API_KEY_VAL = os.environ.get("VERTEX_EXPRESS_API_KEY")

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = float(os.environ.get("FAKE_STREAMING_INTERVAL", "1.0"))

# Validation logic moved to app/auth.py