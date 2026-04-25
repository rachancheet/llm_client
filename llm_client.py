import logging
import time
import os
import re
import json
import random
from datetime import datetime
from threading import Lock
from typing import Type, Optional

import tiktoken
from google import genai
from google.genai import types
from pydantic import BaseModel
from google.genai.errors import ClientError

# Silence the httpx logger used by the google genai client
logging.getLogger("httpx").setLevel(logging.WARNING)

from config import (
    LLM_MODELS,
    GOOGLE_API_KEYS,
    LLM_REQUESTS_PER_MINUTE,
    LLM_TOKENS_PER_MINUTE,
    LLM_REQUESTS_PER_DAY,
    LLM_MAX_CONSECUTIVE_FAILURES,
)

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")

config = types.GenerateContentConfig()


class LLMRateLimitError(Exception):
    pass


class LLMFatalError(Exception):
    pass


import json_repair


def _extract_json_block(text: str) -> dict:
    """Extract JSON even if model adds markdown or text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in output")

    raw_json = match.group()

    # Try strict parsing first
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        pass

    # Detect code expressions that json_repair would silently mangle
    # These patterns indicate the LLM used JS/Python code instead of plain JSON strings
    code_patterns = [
        (r'"[^"]*"\s*\+\s*"', "string concatenation with +"),
        (r'\.repeat\s*\(', ".repeat() function call"),
        (r'\.join\s*\(', ".join() function call"),
        (r'\*\s*\d+', "multiplication operator"),
        (r'"[^"]*"\s*\*\s*', "string repetition with *"),
    ]
    for pattern, description in code_patterns:
        if re.search(pattern, raw_json):
            raise ValueError(
                f"Invalid JSON: detected {description} in value. "
                f"JSON values must be plain string literals, not code expressions. "
                f"Write out the full string content directly. "
                f"For example, WRONG: \"hello\\n\" + \"hello\\n\".repeat(5) — "
                f"CORRECT: \"hello\\nhello\\nhello\\nhello\\nhello\\nhello\""
            )

    # Fall back to json_repair for minor issues (trailing commas, etc.)
    return json_repair.loads(raw_json)


class RateLimitedLLMClient:
    def __init__(self):
        self._lock = Lock()
        self._api_keys = GOOGLE_API_KEYS.copy() if GOOGLE_API_KEYS else []
        self._clients = {}  # cache genai clients per key
        self._state_file = ".llm_requests"
        self._current_pool_index = 0
        
        self._pool = []
        for key in self._api_keys:
            for i, model in enumerate(LLM_MODELS):
                self._pool.append({
                    "key": key,
                    "model": model,
                    "rpm_limit": LLM_REQUESTS_PER_MINUTE[i],
                    "tpm_limit": LLM_TOKENS_PER_MINUTE[i],
                    "rpd_limit": LLM_REQUESTS_PER_DAY[i],
                    "failures_limit": LLM_MAX_CONSECUTIVE_FAILURES[i]
                })

        if not self._pool and LLM_MODELS:
            for i, model in enumerate(LLM_MODELS):
                self._pool.append({
                    "key": None,
                    "model": model,
                    "rpm_limit": LLM_REQUESTS_PER_MINUTE[i],
                    "tpm_limit": LLM_TOKENS_PER_MINUTE[i],
                    "rpd_limit": LLM_REQUESTS_PER_DAY[i],
                    "failures_limit": LLM_MAX_CONSECUTIVE_FAILURES[i]
                })
        
        # Load state from file or initialize empty structures
        state = self._load_state()
        self._minute_requests = state.get("minute_requests", {})
        self._minute_tokens = state.get("minute_tokens", {})
        self._daily_requests = state.get("daily_requests", {})
        self._consecutive_failures = 0
        
        # Register the save method to run on exit
        if hasattr(self, "_save_state"):
            import atexit
            atexit.register(self._save_state)

    def _load_state(self) -> dict:
        if not os.path.exists(self._state_file):
            return {}
        try:
            with open(self._state_file, "r") as f:
                data = json.load(f)
                
            # Convert string timestamps back to datetime objects or format internally
            # For simplistic compatibility with existing structure: (count, timestamp/date)
            for k in data.get("minute_requests", {}):
                val = data["minute_requests"][k]
                data["minute_requests"][k] = (val[0], datetime.fromisoformat(val[1]))
                
            for k in data.get("minute_tokens", {}):
                val = data["minute_tokens"][k]
                data["minute_tokens"][k] = (val[0], datetime.fromisoformat(val[1]))
                
            for k in data.get("daily_requests", {}):
                val = data["daily_requests"][k]
                data["daily_requests"][k] = (val[0], datetime.fromisoformat(val[1]).date())
                
            return data
        except Exception as e:
            logger.warning(f"Could not load LLM state: {e}")
            return {}

    def _save_state(self):
        try:
            state = {
                "minute_requests": {k: (v[0], v[1].isoformat() if v[1] else None) for k, v in self._minute_requests.items()},
                "minute_tokens": {k: (v[0], v[1].isoformat() if v[1] else None) for k, v in self._minute_tokens.items()},
                "daily_requests": {k: (v[0], datetime.combine(v[1], datetime.min.time()).isoformat() if v[1] else None) for k, v in self._daily_requests.items()}
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save LLM state: {e}")

    # --------------------------------------------------
    # TOKEN ESTIMATE
    # --------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        return len(_tokenizer.encode(text))

    # --------------------------------------------------
    # KEY MANAGEMENT
    # --------------------------------------------------
    def _get_random_key(self):
        return random.choice(self._api_keys) if self._api_keys else None

    def _get_client(self, key):
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key, http_options={'timeout': 90000.0})
        return self._clients[key]

    # --------------------------------------------------
    # RATE LIMIT CHECKS
    # --------------------------------------------------
    def _wait_for_rate_limit(self, state_id, estimated_tokens, rpm_limit, tpm_limit):
        if not state_id:
            return

        now = datetime.now()
        current_minute = now.replace(second=0, microsecond=0)

        rpm = self._minute_requests.get(state_id, (0, None))
        tpm = self._minute_tokens.get(state_id, (0, None))

        rpm_hit = rpm[1] == current_minute and rpm[0] >= rpm_limit
        tpm_hit = tpm[1] == current_minute and (tpm[0] + estimated_tokens) >= tpm_limit

        if rpm_hit or tpm_hit:
            wait = 60 - now.second + 1
            logger.warning(f"LLM ⏳ Rate limit hit on {state_id}. Waiting {wait}s")
            time.sleep(wait)

    def _check_daily_limit(self, state_id, rpd_limit):
        if not state_id:
            return True
        today = datetime.now().date()
        count, day = self._daily_requests.get(state_id, (0, today))
        return not (day == today and count >= rpd_limit)

    def _update_counters(self, state_id, tokens):
        now = datetime.now()
        minute = now.replace(second=0, microsecond=0)
        day = now.date()

        req_count, req_minute = self._minute_requests.get(state_id, (0, minute))
        self._minute_requests[state_id] = (req_count + 1 if req_minute == minute else 1, minute)

        tok_count, tok_minute = self._minute_tokens.get(state_id, (0, minute))
        self._minute_tokens[state_id] = (tok_count + tokens if tok_minute == minute else tokens, minute)

        day_count, day_start = self._daily_requests.get(state_id, (0, day))
        self._daily_requests[state_id] = (day_count + 1 if day_start == day else 1, day)

    # --------------------------------------------------
    # RAW COMPLETION
    # --------------------------------------------------
    def completion(self, prompt: str, system: str = None) -> str:
        with self._lock:
            estimated_tokens = self._estimate_tokens(prompt)
            consecutive_round_failures: int = 0
            num_combinations = max(len(self._pool), 1)
            
            start_i = self._current_pool_index
            if self._pool:
                self._current_pool_index = (self._current_pool_index + 1) % len(self._pool)

            while True:
                for offset in range(num_combinations):
                    i = (start_i + offset) % num_combinations
                    combo = self._pool[i] if self._pool else None
                    if not combo:
                        raise LLMFatalError("No LLM pool configurations found.")
                        
                    key = combo["key"]
                    model = combo["model"]
                    state_id = f"{key}::{model}" if key else model

                    if not self._check_daily_limit(state_id, combo["rpd_limit"]):
                        continue

                    self._wait_for_rate_limit(state_id, estimated_tokens, combo["rpm_limit"], combo["tpm_limit"])

                    try:
                        client = self._get_client(key)

                        if system:
                            full_prompt = f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n{prompt}"
                        else:
                            full_prompt = prompt

                        logger.info(f"LLM Prompt [Model: {model}]: {full_prompt}")
                        resp = client.models.generate_content(
                            model=model,
                            contents=full_prompt,
                            config=config
                        )

                        if resp.text is None:
                            finish_reason = getattr(resp.candidates[0], 'finish_reason', 'unknown') if resp.candidates else 'no candidates'
                            token_count = getattr(resp.usage_metadata, 'total_token_count', '?') if resp.usage_metadata else '?'
                            logger.warning(
                                f"LLM returned empty response. model={model}, "
                                f"finish_reason={finish_reason}, prompt_tokens={token_count}. "
                                f"This is common with smaller models on long prompts. Retrying..."
                            )
                            continue

                        text = resp.text.strip()
                        used_tokens = estimated_tokens + self._estimate_tokens(text)
                        self._update_counters(state_id, used_tokens)
                        return text

                    except Exception as e:
                        logger.error(f"LLM error with key/model index {i} (model: {model}): {e}")

                consecutive_round_failures += 1
                max_allowed = max(LLM_MAX_CONSECUTIVE_FAILURES) if LLM_MAX_CONSECUTIVE_FAILURES else 5
                if consecutive_round_failures >= max_allowed:
                    raise LLMFatalError(f"All keys/models failed {consecutive_round_failures} rounds in a row. Giving up.")
                logger.warning(f"All permutations failed (round {consecutive_round_failures}/{max_allowed}). Waiting 60s before retrying...")
                time.sleep(60)

    # --------------------------------------------------
    # STRUCTURED COMPLETION
    # --------------------------------------------------
    def completion_structured(
        self,
        prompt: str,
        schema_model: Type[BaseModel],
        system: Optional[str] = None,
        max_retries: int = 1,
    ) -> BaseModel:

        schema_json = schema_model.model_json_schema()

        strict_system = f"""
You are a strict JSON API.
Return ONLY valid JSON.
No markdown. No explanation.
JSON values must be simple valid JSON — no code expressions, concatenation operators, or function calls as values.
For example, WRONG: "content": "hello\\n" + "hello\\n".repeat(36)
CORRECT: "content": "hello\\nhello\\nhello\\n..."
String values CAN contain code, or any text — just wrap them as a single JSON string literal.

Schema:
{json.dumps(schema_json, indent=2)}
"""

        if system:
            strict_system = system + "\n\n" + strict_system

        original_prompt = prompt
        last_error = None

        for attempt in range(max_retries):
            try:
                raw = self.completion(prompt, strict_system)
                logger.info(f"Raw LLM response:\n{raw}")
                parsed = _extract_json_block(raw)
                return schema_model.model_validate(parsed)

            except (LLMRateLimitError, LLMFatalError) as e:
                # Direct API/quota failures should be propagated immediately, not treated as parse errors
                raise e
            except Exception as e:
                last_error = e
                logger.warning(f"Structured parse failed attempt {attempt+1}: {e}")

                prompt = f"""
Previous output invalid.

Error:
{str(e)}

Fix and return ONLY valid JSON.

Original task:
{original_prompt}
"""

        raise LLMFatalError(f"Structured output failed: {last_error}")


# --------------------------------------------------
# GLOBAL CLIENT
# --------------------------------------------------
_client = RateLimitedLLMClient()


def llm_completion(prompt: str, system: str = None) -> str:
    return _client.completion(prompt, system)


def llm_completion_structured(prompt: str, schema_model: Type[BaseModel], system: str = None, max_retries: int = 1):
    return _client.completion_structured(prompt, schema_model, system, max_retries=max_retries)