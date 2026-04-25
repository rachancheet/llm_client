# LLM Memory — OpenClaw & PicoClaw ↔ llm_client.py Integration

> Documentation on integrating `llm_client.py` (Python/Gemini) as the inference backend for OpenClaw and PicoClaw.

---

## Architecture Overview

`llm_client.py` is a Python-based rate-limited client using the Google Gemini SDK (`google-genai`).

**Solution:** A Python HTTP bridge server (`llm_bridge.py`) that:
1. Exposes an **OpenAI-compatible** `/v1/chat/completions` endpoint on `http://localhost:5099`
2. Translates incoming OpenAI-format chat requests into calls to `llm_client.py`'s `llm_completion()` function
3. Returns OpenAI-format responses back to the caller (OpenClaw/PicoClaw)

This allows OpenClaw to connect to the bridge as a standard `openai`-protocol provider without any native code changes.

```text
┌─────────────┐     OpenAI HTTP     ┌──────────────┐     Python calls     ┌────────────────┐     Gemini SDK     ┌──────────────┐
│  OpenClaw    │ ──────────────────► │  llm_bridge   │ ──────────────────► │  llm_client    │ ────────────────► │  Google API  │
│  (Framework) │  POST /v1/chat/    │  (Python HTTP) │    llm_completion() │  (Rate-limited) │                   │  (Gemini)    │
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
- **Purpose:** Bridge between OpenClaw and Python `llm_client.py`
- **Listens on:** `http://0.0.0.0:5099`
- **Endpoints:**
  - `POST /v1/chat/completions` — Main chat endpoint (OpenAI format)
  - `GET /health`, `GET /v1/models` — Health check
- **Features:**
  - Converts OpenAI messages array → single prompt string for `llm_completion()`
  - Extracts system messages and passes them separately
  - Injects tool definitions into the system prompt to guide the LLM's response
  - Actively parses tool calls from the LLM's raw text output and converts them back into the standard OpenAI `tool_calls` format expected by OpenClaw.

### 3. `requirements.txt` — Python dependencies
- `google-genai>=1.0.0`
- `tiktoken>=0.5.0`
- `pydantic>=2.0.0`
- `json-repair>=0.20.0`
- `python-dotenv>=1.0.0`

---

## Tool Calling & OpenClaw Integration Verdict

A critical discovery was made regarding how OpenClaw handles tool calling compared to earlier assumptions:

1. **How OpenClaw sends tool definitions:** OpenClaw **DOES** send a `tools` array in its POST requests to `/v1/chat/completions`. It relies on standard OpenAI HTTP interactions (handled natively by its `openai-transport-stream.ts`).
2. **How OpenClaw expects responses:** OpenClaw **DOES NOT** expect the bridge to return raw JSON text inside `choices[0].message.content`. Instead, it explicitly expects standard OpenAI `tool_calls` formatting (i.e., `choices[0].message.tool_calls`) when a tool is invoked.
3. **The Bridge's Role:** Because `llm_client.py` natively returns raw text, `llm_bridge.py` must aggressively intercept LLM responses. If the LLM generates a JSON tool call, the bridge parses it out and correctly wraps it in `tool_calls`. Previously, a comment in the bridge claimed OpenClaw handled raw JSON directly—this was **factually incorrect**. The bridge has been updated to *always* attempt to parse tool calls from the response, ensuring flawless compatibility with OpenClaw's OpenAI transport expectations.

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

### Step 3: Configure OpenClaw
Set up OpenClaw to point to the bridge as an OpenAI-compatible provider:
- **API Base URL:** `http://localhost:5099/v1`
- **Model:** `llm-client-bridge`
- **API Key:** Not needed (or a dummy key)

---

## How It Works — Request Flow

1. **OpenClaw** sends a standard OpenAI `POST /v1/chat/completions` request with messages, and natively includes a `tools` array.
2. **`llm_bridge.py`** receives it and:
   - Separates `system` messages from `user`/`assistant`/`tool` messages
   - Converts multi-message conversation into a single prompt string
   - Appends tool definitions to the system prompt to guide the underlying model to output correct JSON
   - Calls `llm_completion(prompt, system=system_prompt)` from `llm_client.py`
3. **`llm_client.py`** handles:
   - API key rotation across all `GOOGLE_API_KEYS`
   - Rate limiting (RPM, TPM, RPD)
   - Calls the Gemini API via `google-genai` SDK
4. **`llm_bridge.py`** receives the raw text response and:
   - Actively parses the text for tool calls (`{"tool_calls": [...]}`)
   - Wraps the response in standard OpenAI format: populating `choices[0].message.tool_calls` instead of `content` if a tool is called.
   - Returns the standardized OpenAI response to OpenClaw.
