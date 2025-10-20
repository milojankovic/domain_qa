"""Configuration helpers and constants for the knowledge base stack."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean environment variable with a fallback default."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))

DOMAINDATA_DIR = ROOT / "domaindata"
DATA_DIRS = [DOMAINDATA_DIR]
METADATA_JSONL = DOMAINDATA_DIR / "metadata.jsonl"

STORAGE_DIR = ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
ASSET_DIR = STORAGE_DIR / "assets"
ASSET_DB_PATH = ASSET_DIR / "assets.sqlite"

GEMINI_GENERATE_MODEL = os.getenv("GEMINI_GENERATE_MODEL", "gemini-2.0-flash")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
SENTENCE_TRANSFORMER_DEVICE = os.getenv("SENTENCE_TRANSFORMER_DEVICE")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng")
USE_OCR = env_bool("USE_OCR", True)
HI_RES_STRATEGY = os.getenv("HI_RES_STRATEGY", "hi_res")
MAX_CHARS_NARRATIVE = int(os.getenv("MAX_CHARS_NARRATIVE", "1600"))
MAX_CHARS_SLIDEY = int(os.getenv("MAX_CHARS_SLIDEY", "900"))
COMBINE_UNDER_N_CHARS = int(os.getenv("COMBINE_UNDER_N_CHARS", "500"))

TOP_K = int(os.getenv("TOP_K", "6"))
MAX_ASSET_ATTACH = int(os.getenv("MAX_ASSET_ATTACH", "4"))

TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))
DEBUG = env_bool("DEBUG", True)
