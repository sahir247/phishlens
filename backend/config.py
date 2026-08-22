"""
PhishLens Configuration
-----------------------
All tuneable values are read from environment variables (or a .env file).
Import this module instead of hard-coding values anywhere else.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ── Server ──────────────────────────────────────────────────────────────────
HOST: str = _get("PHISHLENS_HOST", "127.0.0.1")
PORT: int = int(_get("PHISHLENS_PORT", "8000"))

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH: str = _get(
    "PHISHLENS_DB_PATH",
    os.path.join(os.path.dirname(__file__), "phishlens.db"),
)
DB_URL: str = f"sqlite:///{DB_PATH}"

# ── ML Model ─────────────────────────────────────────────────────────────────
MODEL_PATH: str = _get(
    "PHISHLENS_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "phishlens_clf.pkl"),
)

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(_get("PHISHLENS_CACHE_TTL", "300"))   # seconds
CACHE_MAX_SIZE: int = int(_get("PHISHLENS_CACHE_SIZE", "512"))

# ── Rate Limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_CHECK: str = _get("PHISHLENS_RATE_CHECK", "60 per minute")
RATE_LIMIT_SIMULATE: str = _get("PHISHLENS_RATE_SIM", "30 per minute")
RATE_LIMIT_BATCH: str = _get("PHISHLENS_RATE_BATCH", "10 per minute")
BATCH_MAX_URLS: int = int(_get("PHISHLENS_BATCH_MAX", "20"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("PHISHLENS_LOG_LEVEL", "INFO")
LOG_FILE: str = _get(
    "PHISHLENS_LOG_FILE",
    os.path.join(os.path.dirname(__file__), "phishlens.log"),
)
LOG_MAX_BYTES: int = int(_get("PHISHLENS_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
LOG_BACKUP_COUNT: int = int(_get("PHISHLENS_LOG_BACKUP", "3"))

# ── Trust Engine data cache dir ───────────────────────────────────────────────
DATA_DIR: str = _get(
    "PHISHLENS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)
os.makedirs(DATA_DIR, exist_ok=True)
