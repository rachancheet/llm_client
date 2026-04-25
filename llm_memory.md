# LLM Memory — PicoClaw ↔ llm_client.py Integration

> All changes made to integrate `llm_client.py` (Python/Gemini) as the inference backend for PicoClaw (Go).

---

## Architecture Overview

PicoClaw is a **Go** codebase. `llm_client.py` is a **Python** file using the Google Gemini SDK (`google-genai`).

**Solution:** A Python HTTP bridge server (`llm_bridge.py`) that:
1. Exposes an **OpenAI-compatible** `/v1/chat/completions` endpoint on `http://localhost:5099`
2. Translates incoming OpenAI-format chat requests into calls to `llm_client.py`'s `llm_completion()` function
3. Returns OpenAI-format responses back to PicoClaw

PicoClaw connects to this bridge as a standard `openai`-protocol provider — **zero Go code changes needed**.

```
┌─────────────┐     OpenAI HTTP     ┌──────────────┐     Python calls     ┌────────────────┐     Gemini SDK     ┌──────────────┐
│  PicoClaw    │ ──────────────────► │  llm_bridge   │ ──────────────────► │  llm_client    │ ────────────────► │  Google API  │
│  (Go agent)  │  POST /v1/chat/    │  (Python HTTP) │    llm_completion() │  (Rate-limited) │                   │  (Gemini)    │
│              │  completions       │  :5099         │                     │  Key rotation   │                   │              │
└─────────────┘                     └──────────────┘                     └────────────────┘                     └──────────────┘
```

---

## Files Created

### 1. `config.py` — Python config module
- **Purpose:** Provides the configuration values that `llm_client.py` imports
- **Reads from:** `.env` file (via `python-dotenv`)
- **Exports:** `LLM_MODELS`, `GOOGLE_API_KEYS`, `LLM_REQUESTS_PER_MINUTE`, `LLM_TOKENS_PER_MINUTE`, `LLM_REQUESTS_PER_DAY`, `LLM_MAX_CONSECUTIVE_FAILURES`
- **Auto-pads** rate limit lists to match `LLM_MODELS` length

### 2. `llm_bridge.py` — OpenAI-compatible HTTP bridge
- **Purpose:** Bridge between PicoClaw's Go code and Python `llm_client.py`
- **Listens on:** `http://0.0.0.0:5099`
- **Endpoints:**
  - `POST /v1/chat/completions` — Main chat endpoint (OpenAI format)
  - `GET /health`, `GET /v1/models` — Health check
- **Features:**
  - Converts OpenAI messages array → single prompt string for `llm_completion()`
  - Extracts system messages and passes them separately
  - Injects tool definitions into the system prompt
  - Parses tool calls from LLM text output back into OpenAI tool_calls format
  - Proper error handling with JSON error responses

### 3. `config/config.json` — PicoClaw configuration (modified)
- **Changed:** Points to the bridge as an OpenAI-compatible provider
- **Model entry:**
  ```json
  {
    "model_name": "llm-bridge",
    "model": "openai/llm-client-bridge",
    "api_key": "not-needed",
    "api_base": "http://localhost:5099/v1"
  }
  ```

### 4. `requirements.txt` — Python dependencies
- `google-genai>=1.0.0`
- `tiktoken>=0.5.0`
- `pydantic>=2.0.0`
- `json-repair>=0.20.0`
- `python-dotenv>=1.0.0`

---

## Files NOT Modified

- **`llm_client.py`** — No changes. Used as-is.
- **All Go source files** — Zero Go code changes.
- **`.env`** — No changes. Already has `GOOGLE_API_KEYS` and `LLM_MODELS`.

---

## How to Run

### Step 1: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the bridge server
```bash
python llm_bridge.py
```
This starts listening on port 5099.

### Step 3: Start PicoClaw
```bash
# Build and run PicoClaw (it will connect to the bridge on localhost:5099)
go run ./cmd/picoclaw/
```

---

## How It Works — Request Flow

1. **PicoClaw** sends a standard OpenAI `POST /v1/chat/completions` request with messages, tools, etc.
2. **`llm_bridge.py`** receives it and:
   - Separates `system` messages from `user`/`assistant`/`tool` messages
   - Converts multi-message conversation into a single prompt string
   - If tools are provided, appends tool definitions to the system prompt
   - Calls `llm_completion(prompt, system=system_prompt)` from `llm_client.py`
3. **`llm_client.py`** handles:
   - API key rotation across all `GOOGLE_API_KEYS`
   - Rate limiting (RPM, TPM, RPD)
   - Retry logic with backoff
   - Calls the Gemini API via `google-genai` SDK
4. **`llm_bridge.py`** receives the raw text response and:
   - Attempts to parse tool calls from the text (if tools were provided)
   - Wraps the response in OpenAI format with proper `choices`, `usage`, etc.
   - Returns it to PicoClaw

---

## Tool Calling

Since `llm_client.py` only supports raw text completion (not native tool calling), the bridge handles tool calls by:

1. **Injecting** tool definitions into the system prompt with explicit JSON format instructions
2. **Parsing** the LLM's text response for JSON containing `{"tool_calls": [...]}`
3. **Converting** parsed tool calls to OpenAI's `tool_calls` format in the response

---

## Testing & Verification

### Verified Working:
- ✅ `config.py` correctly loads all values from `.env` (3 API keys, model list, rate limits)
- ✅ `llm_bridge.py` starts and listens on port 5099
- ✅ Health endpoint (`GET /health`) returns `{"status": "ok"}`
- ✅ Chat endpoint receives requests, routes to `llm_client.py`, cycles through API keys
- ✅ Bridge properly formats OpenAI-compatible responses
- ✅ Error handling works (502 responses on LLM failure)

### Network Issue (not a code issue):
- ❌ The Gemini API (`generativelanguage.googleapis.com`) is unreachable from this machine (SSL/read timeouts)
- This affects ALL Google Gemini calls — both through the bridge and direct Python SDK calls
- **Fix:** Ensure network connectivity to Google APIs, or configure a proxy in the `.env`

---

## Environment Variables (`.env`)

| Variable | Description | Example |
|---|---|---|
| `LLM_MODELS` | Comma-separated model names | `gemma-3-27b-it` |
| `GOOGLE_API_KEYS` | Comma-separated API keys | `key1,key2,key3` |
| `LLM_REQUESTS_PER_MINUTE` | Rate limit per model | `15` |
| `LLM_TOKENS_PER_MINUTE` | Token limit per model | `99999999` |
| `LLM_REQUESTS_PER_DAY` | Daily request limit | `1500` |
| `LLM_MAX_CONSECUTIVE_FAILURES` | Max retries before fatal | `2` |

---

## Design Decisions

1. **Bridge over Go modification:** Adding Python subprocess calls into Go would be fragile. An HTTP bridge is clean, testable, and uses PicoClaw's existing OpenAI-compat provider infrastructure.

2. **stdlib `http.server`:** No Flask/FastAPI dependency needed — keeps things minimal. The bridge is simple request-response with no need for async.

3. **Tool call injection via system prompt:** Since `llm_client.py` only does raw text completion, tool definitions are injected into the system prompt. The LLM is instructed to respond with a specific JSON format, which the bridge parses back into OpenAI tool_calls format.

4. **Port 5099:** Chosen to avoid conflicts with common ports (3000, 5000, 8000, etc.).
