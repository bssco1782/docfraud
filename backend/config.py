"""
backend/config.py
Central configuration for the Document Fraud Detection API.

Every tunable value is read from environment variables (optionally loaded from a
.env file in the working directory) with a safe default. Other backend modules
import their settings from here instead of hardcoding them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import FrozenSet, List

# Load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Service metadata reported by /health and the OpenAPI schema
APP_VERSION: str = "0.1.0"

# Authentication. The default is for local development only; always set API_KEY in deployment.
DEFAULT_API_KEY: str = "default-api-key"
API_KEY: str = os.getenv("API_KEY", DEFAULT_API_KEY)

# Model file. An explicit MODEL_PATH that does not exist falls back to the bundled model.
DEFAULT_MODEL_PATH = BASE_DIR / "data" / "models" / "fraud_model.pkl"
_env_model_path = os.getenv("MODEL_PATH")
if _env_model_path and Path(_env_model_path).exists():
    MODEL_PATH: str = _env_model_path
elif DEFAULT_MODEL_PATH.exists():
    MODEL_PATH = str(DEFAULT_MODEL_PATH)
else:
    MODEL_PATH = _env_model_path or str(DEFAULT_MODEL_PATH)

# Upload limits
MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB default in bytes
ALLOWED_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png", ".pdf"})
# Extra bytes allowed on top of MAX_FILE_SIZE for multipart framing before a request is rejected
MULTIPART_OVERHEAD_BYTES: int = 64 * 1024

# Image limits, checked from the file header before the image is decoded.
# MAX_IMAGE_PIXELS blocks decompression bombs (a small file that expands to gigabytes);
# 40 MP covers an A4 page scanned at 600 DPI (~35 MP).
MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", str(40_000_000)))
# Shortest side (px) after the analyzer's downscale to 2000 px; smaller images crash feature extraction
MIN_IMAGE_SIDE: int = 16
# The analyzer renders PDF page 1 at this resolution (core/document_analyzer.py)
PDF_RENDER_DPI: int = 200

# Maximum number of documents analysed at the same time in one process; further requests wait.
# Analyses run in threads and contend for the GIL and OpenCV/numpy threads: measured on 4 CPUs
# (logs/performance_review.txt), 2 slots gave the best throughput and 4 slots was ~35% slower.
# Scale with more uvicorn worker processes rather than more slots.
MAX_CONCURRENT_ANALYSES: int = int(os.getenv("MAX_CONCURRENT_ANALYSES", str(min(2, os.cpu_count() or 1))))

# Risk score at or above which /verify reports "evidence of tampering"
TAMPER_THRESHOLD: float = float(os.getenv("TAMPER_THRESHOLD", "0.5"))

# Comma-separated list of allowed CORS origins ("*" allows any origin)
CORS_ORIGINS: List[str] = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

# Logging level for the backend and core loggers (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
