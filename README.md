# LLM Bridge

An OpenAI-compatible HTTP bridge for free AI inference, using the Google Gemini SDK behind the scenes. This bridge allows any application or agent framework (like OpenClaw, PicoClaw, AutoGPT, etc.) that speaks the standard OpenAI API protocol to route its inference through a custom Python client.

## Features

- **OpenAI Compatible API:** Exposes a `/v1/chat/completions` endpoint that acts exactly like OpenAI.
- **Free Inference:** Uses `llm_client.py` and the Google Gemini SDK for free inference (depending on your Gemini API tier).
- **Tool Calling Support:** Injects tool schemas into system prompts and cleanly parses out the JSON tool responses back into the standard OpenAI `tool_calls` structure.
- **Key Rotation & Rate Limiting:** Handles multiple API keys, RPM/TPM limits, and retries automatically through `llm_client.py`.
- **Zero-Code Integration:** No need to modify your agent frameworks or Go/Node.js apps. Just point the API Base URL to this bridge.

## Installation

1. Clone this repository (or copy the files).
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory (or use your existing one) with your API keys and configuration:

```env
LLM_MODELS=gemma-3-27b-it
GOOGLE_API_KEYS=your_gemini_api_key_1,your_gemini_api_key_2
LLM_REQUESTS_PER_MINUTE=15
LLM_TOKENS_PER_MINUTE=99999999
LLM_REQUESTS_PER_DAY=1500
LLM_MAX_CONSECUTIVE_FAILURES=2
```

## Running the Bridge

Start the bridge server by running:

```bash
python llm_bridge.py
```

The server will start listening on `http://0.0.0.0:5099`.

## Connecting OpenClaw (or other agents)

Update your framework's configuration to point to the bridge instead of the official OpenAI API.

### Environment Variables
If your framework uses environment variables:
```env
OPENAI_API_BASE=http://localhost:5099/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL_NAME=llm-bridge
```

### JSON Config
If your framework uses a configuration file (like OpenClaw/PicoClaw):
```json
{
  "model_name": "llm-bridge",
  "model": "openai/llm-client-bridge",
  "api_key": "not-needed",
  "api_base": "http://localhost:5099/v1"
}
```

## Health Checks
You can verify the bridge is running by visiting:
- `http://localhost:5099/health`
- `http://localhost:5099/v1/models`
