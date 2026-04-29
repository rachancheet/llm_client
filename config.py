"""
Configuration module for llm_client.py
Reads settings from .env file in the project root.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _csv_list(key: str, default: str = "") -> list[str]:
    """Parse a comma-separated env var into a list of stripped strings."""
    raw = os.getenv(key, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _csv_int_list(key: str, default: str = "") -> list[int]:
    """Parse a comma-separated env var into a list of ints."""
    return [int(v) for v in _csv_list(key, default)]


# --- Model list ---
LLM_MODELS: list[str] = _csv_list("LLM_MODELS", "gemma-3-27b-it")

# --- API keys ---
GOOGLE_API_KEYS: list[str] = _csv_list("GOOGLE_API_KEYS", "")

# --- Rate limits (per-model, comma-separated to match LLM_MODELS order) ---
LLM_REQUESTS_PER_MINUTE: list[int] = _csv_int_list("LLM_REQUESTS_PER_MINUTE", "15")
LLM_TOKENS_PER_MINUTE: list[int] = _csv_int_list("LLM_TOKENS_PER_MINUTE", "99999999")
LLM_REQUESTS_PER_DAY: list[int] = _csv_int_list("LLM_REQUESTS_PER_DAY", "1500")
LLM_MAX_CONSECUTIVE_FAILURES: list[int] = _csv_int_list("LLM_MAX_CONSECUTIVE_FAILURES", "5")
LLM_RETRY_DELAY_SECONDS: int = int(os.getenv("LLM_RETRY_DELAY_SECONDS", "80"))

# Pad lists to match LLM_MODELS length (repeat last value)
for _lst_name in ("LLM_REQUESTS_PER_MINUTE", "LLM_TOKENS_PER_MINUTE",
                   "LLM_REQUESTS_PER_DAY", "LLM_MAX_CONSECUTIVE_FAILURES"):
    _lst = globals()[_lst_name]
    while len(_lst) < len(LLM_MODELS):
        _lst.append(_lst[-1] if _lst else 15)
