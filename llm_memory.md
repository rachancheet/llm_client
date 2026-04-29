# LLM Memory — OpenClaw & PicoClaw ↔ llm_client.py Integration

> Documentation on integrating `llm_client.py` (Python/Gemini) as the inference backend for OpenClaw and PicoClaw.

---

## Architecture Overview

`llm_client.py` is a Python-based rate-limited client using the Google Gemini SDK (`google-genai`).

**Solution:** A Python HTTP bridge server (`llm_bridge.py`) that:
1. Exposes an **OpenAI-compatible** `/v1/chat/completions` endpoint on `http://localhost:5099`
2. Translates incoming OpenAI-format chat requests (including tools and streaming) into calls to `llm_client.py`'s `llm_completion_raw()` function.
3. Returns OpenAI-format responses (supporting text, tool calls, and SSE streaming) back to the caller (OpenClaw/PicoClaw).

This allows OpenClaw to connect to the bridge as a standard `openai`-protocol provider without any native code changes.

```text
┌─────────────┐     OpenAI HTTP     ┌──────────────┐     Python calls     ┌────────────────┐     Gemini SDK     ┌──────────────┐
│  OpenClaw    │ ──────────────────► │  llm_bridge   │ ──────────────────► │  llm_client    │ ────────────────► │  Google API  │
│  (Framework) │  POST /v1/chat/    │  (Flask HTTP)  │    llm_completion_  │  (Rate-limited) │                   │  (Gemini)    │
│              │  completions       │  :5099         │    raw()            │  Key rotation   │                   │              │
└─────────────┘                     └──────────────┘                     └────────────────┘                     └──────────────┘
```

---

## Core Components

### 1. `config.py` — Configuration Management
- **Purpose:** Centralized configuration for the LLM stack.
- **Key Features:**
  - Reads from `.env` using `python-dotenv`.
  - Configures `GOOGLE_API_KEYS` for rotation.
  - Sets per-model rate limits (`RPM`, `TPM`, `RPD`).
  - Configures `LLM_RETRY_DELAY_SECONDS` for exponential backoff/retry.
  - Sets `LLM_MAX_CONSECUTIVE_FAILURES` to define when to give up.

### 2. `llm_client.py` — The Inference Engine
- **Purpose:** Robust, rate-limited interface to the Gemini API.
- **Key Capabilities:**
  - **Native Tool Calling:** Uses `google.genai` native types for tool definitions and function calls.
  - **Structured Output:** Supports Pydantic-based `response_schema` enforcement.
  - **Rate Limiting & Persistence:** Tracks usage across keys/models and persists state to `.llm_requests` (saved on exit via `atexit`).
  - **Reliability:** Implements a round-robin retry loop across all available key/model combinations with exponential backoff.
  - **Timeouts:** Configured with a 180s timeout (`http_options={'timeout': 180000}`) to handle high-latency "Time to First Token" scenarios.

### 3. `llm_bridge.py` — OpenAI-Compatible Bridge
- **Purpose:** Translates between OpenAI standards and Gemini's native capabilities.
- **Endpoints:** `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.
- **Key Features:**
  - **Message Translation:** Correctlly maps `system`, `user`, `assistant` (with tool calls), and `tool` (responses) roles.
  - **Native Tool Integration:** Translates OpenAI `tools` array into Gemini `FunctionDeclaration` objects.
  - **Streaming Support:** Supports `stream: true` for both standard text and `tool_calls` using Server-Sent Events (SSE).
  - **Schema Sanitization:** Automatically strips unsupported JSON Schema keys (e.g., `additionalProperties`, `patternProperties`) that cause Gemini API errors.
  - **Pastel Logging:** Enhanced terminal logging with color-coded prompts and responses for easier debugging.

---

## Tool Calling & Interaction Logic

The integration uses **Native Tool Calling** rather than raw text parsing:

1. **Definitions:** OpenClaw sends standard OpenAI tool definitions. The bridge sanitizes these (removing unsupported keys) and passes them as native Gemini `Tool` objects.
2. **Execution:** `automatic_function_calling` is **disabled** in the client. This ensures the model *proposes* a call, which the bridge then returns to OpenClaw. OpenClaw executes the tool and sends the result back in a subsequent request with `role: "tool"`.
3. **Response Formatting:** The bridge detects `function_call` parts in the Gemini response and formats them into the OpenAI `tool_calls` structure.

---

## State & Persistence
Usage counters (Requests Per Minute, Tokens Per Minute, etc.) are stored in a local JSON file: `.llm_requests`.
This ensures that rate limits are respected even if the bridge is restarted. The `RateLimitedLLMClient` automatically loads this on init and saves it on shutdown.

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Bridge
```bash
python llm_bridge.py
```
Default port is `5099`.

### Step 3: Configure OpenClaw
- **Provider:** OpenAI (or Custom)
- **Base URL:** `http://localhost:5099/v1`
- **Model:** `llm-client-bridge`
- **API Key:** `sk-dummy` (Required by some clients but ignored by the bridge)
