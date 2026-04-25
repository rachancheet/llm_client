# GemmaClaw Bridge

An ultra-lightweight, OpenAI-compatible HTTP bridge that allows any agent framework (like OpenClaw or PicoClaw) to use **free Google Gemini inference** with full tool-calling and streaming support.

<img width="1758" height="186" alt="image" src="https://github.com/user-attachments/assets/81527a56-a300-4b54-b785-d2c9343ac9f1" />


## Why is this needed?
The Google Gemini/Gemma models available in AI Studio offer incredibly generous free tiers and rate limits, making them perfect for powering local, always-on AI agents. However, many of the smaller or open-weights models (like Gemma) do not officially support native OpenAI-compatible tool calling, making them incompatible with major agent frameworks. 

This bridge sits between your agent and the Google API. It takes standard OpenAI tool definitions, injects them into the model's system prompt, forces the LLM to output strictly-typed JSON using Google's Structured Output functionality, and translates that JSON back into a flawless OpenAI `tool_calls` response. To the agent, it looks exactly like you are talking to GPT-4!



## Features
- **OpenAI Compatible:** Acts exactly like OpenAI's `/v1/chat/completions`.
- **Free Inference:** Routes all traffic through the official Google Gemini SDK (free tier).
- **Native Tool Calling:** Intercepts agent tool schemas and guarantees perfectly structured JSON `tool_calls` back to the agent.
- **Robust Streaming:** Uses Flask and Server-Sent Events (SSE) to prevent TUI hangs.
- **Rate Limit & Key Rotation:** Automatically manages API limits and rotates through multiple API keys.

## 1. Setup

### Get your Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account and click **Create API Key**. It's completely free!



### Install
Clone this repository and install the requirements:
```bash
git clone https://github.com/rachancheet/GemmaClaw-bridge
cd GemmaClaw-Bridge
pip install -r requirements.txt
```

### Configure
Create a `.env` file in the root directory:
```env
LLM_MODELS=gemma-4-31b-it,gemma-4-26b-a4b-it
GOOGLE_API_KEYS=your_gemini_api_key_1,your_gemini_api_key_2
LLM_REQUESTS_PER_MINUTE=15
LLM_TOKENS_PER_MINUTE=1000000
LLM_REQUESTS_PER_DAY=1500
LLM_MAX_CONSECUTIVE_FAILURES=2
```

## 2. Running the Bridge

You need to keep the bridge running in the background while your agent operates.

**Option A: Using `tmux` (Recommended)**
```bash
tmux new -s gemini_bridge
python llm_bridge.py
# Press Ctrl+B, then D to detach and leave it running in the background.
```

**Option B: Using `screen`**
```bash
screen -S gemini_bridge
python llm_bridge.py
# Press Ctrl+A, then D to detach.
```

**Option C: systemd Service (For permanent servers)**
Create `/etc/systemd/system/geminibridge.service`:
```ini
[Unit]
Description=GemmaClaw Bridge
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/repo
ExecStart=/usr/bin/python3 llm_bridge.py
Restart=always

[Install]
WantedBy=multi-user.target
```
Then run: `sudo systemctl enable --now geminibridge`

## 3. Connecting OpenClaw

Once the bridge is running (it listens on `http://0.0.0.0:5099` by default), simply update your OpenClaw or PicoClaw `config.json` to point to the bridge:

```json
{
  "model_list": [
    {
      "model_name": "gemini-bridge",
      "model": "openai/gemini-bridge",
      "api_key": "not-needed",
      "api_base": "http://127.0.0.1:5099/v1"
    }
  ]
}
```
That's it! OpenClaw will now seamlessly route its inference through Gemini completely free of charge.
