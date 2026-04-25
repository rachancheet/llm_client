"""
OpenAI-compatible HTTP bridge for llm_client.py using Flask

Exposes POST /v1/chat/completions so PicoClaw's Go code can use
the Python-based RateLimitedLLMClient (with Gemini key rotation,
rate limiting, and retry logic) as a standard OpenAI-compat provider.

Run:  python llm_bridge.py
Listens on http://0.0.0.0:5099/v1/chat/completions
"""

import json
import logging
import time
import uuid
import re
from flask import Flask, request, jsonify, Response, stream_with_context

from llm_client import llm_completion, llm_completion_structured
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

app = Flask(__name__)

# Reduce Flask default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class BridgeToolCall(BaseModel):
    name: str = Field(description="The name of the tool to call")
    arguments: Dict[str, Any] = Field(description="The JSON arguments to pass to the tool")

class BridgeChatResponse(BaseModel):
    content: Optional[str] = Field(None, description="The text response to the user. Use this if NO tool is being called.")
    tool_calls: Optional[List[BridgeToolCall]] = Field(None, description="The list of tools to call. Leave null if no tools are needed.")

class PastelFormatter(logging.Formatter):
    PASTEL_BLUE = "\033[38;5;117m"
    PASTEL_GREEN = "\033[38;5;114m"
    PASTEL_YELLOW = "\033[38;5;229m"
    PASTEL_RED = "\033[38;5;210m"
    PASTEL_PURPLE = "\033[38;5;183m"
    RESET = "\033[0m"

    def format(self, record):
        color = self.PASTEL_GREEN
        if record.levelno == logging.WARNING:
            color = self.PASTEL_YELLOW
        elif record.levelno >= logging.ERROR:
            color = self.PASTEL_RED
        elif record.levelno == logging.DEBUG:
            color = self.PASTEL_BLUE
            
        msg = str(record.msg)
        if "LLM Response" in msg:
            color = self.PASTEL_PURPLE
        elif "LLM Prompt" in msg:
            color = self.PASTEL_BLUE
        elif "Bridge request" in msg:
            color = self.PASTEL_BLUE
        elif "[MOCK]" in msg:
            color = self.PASTEL_YELLOW

        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(PastelFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(handler)
else:
    for handler in root_logger.handlers:
        handler.setFormatter(PastelFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logger = logging.getLogger("llm_bridge")

BRIDGE_PORT = 5099

def _try_parse_tool_calls(text: str) -> list[dict] | None:
    """
    Attempt to extract tool_calls from the LLM's raw text response.
    Returns OpenAI-format tool_calls list, or None if not a tool call.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1] == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "tool_calls" in data:
            tool_calls = data["tool_calls"]
            if isinstance(tool_calls, list):
                out = []
                for call in tool_calls:
                    if "name" in call and "arguments" in call:
                        out.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"])
                            }
                        })
                if out:
                    return out
    except json.JSONDecodeError:
        pass
    return None

@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "models": ["llm-client-bridge"]})

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    # To use the real LLM, uncomment the line below and comment out the mock:
    # return _handle_chat_completions_mock()
    return _handle_chat_completions()

def _handle_chat_completions_mock():
    body = request.json or {}
    
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    is_stream = body.get("stream", False)
    
    messages = body.get("messages", [])
    last_role = messages[-1].get("role") if messages else ""
    
    logger.info(f"[MOCK] Received request: stream={is_stream}, messages_count={len(messages)}, last_role={last_role}")
    logger.info(f"[MOCK] Full messages received:\n{json.dumps(messages, indent=2)}")
    
    if last_role == "tool":
        tool_calls_out = []
        final_content = "I successfully received the tool execution result! Everything works."
        logger.info("[MOCK] Action: Returning success text because last_role was 'tool'")
    else:
        tool_calls_out = [{
            "id": "call_mock123",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": "{\"path\": \"dilly_dally_mock.txt\", \"content\": \"hello from mock!\"}"
            }
        }]
        final_content = ""
        logger.info("[MOCK] Action: Returning 'write' tool call because last_role was NOT 'tool'")
    
    logger.info(f"[MOCK] Preparing response... Stream mode: {is_stream}")
    
    if is_stream:
        def generate():
            if tool_calls_out:
                chunk1 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge-mock",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": tool_calls_out[0]["id"], "type": "function", "function": {"name": "write", "arguments": "{\"path\": \"dilly_dally_mock.txt\", \"content\": \"hello from mock!\"}"}}]},
                        "finish_reason": None
                    }]
                }
                chunk2 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge-mock",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls"
                    }]
                }
            else:
                chunk1 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge-mock",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": final_content},
                        "finish_reason": None
                    }]
                }
                chunk2 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge-mock",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_content if not tool_calls_out else None,
            },
            "finish_reason": "tool_calls" if tool_calls_out else "stop",
        }
        if tool_calls_out:
            choice["message"]["tool_calls"] = tool_calls_out

        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "llm-client-bridge-mock",
            "choices": [choice],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
        return jsonify(response)


def _handle_chat_completions():
    body = request.json or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    system_prompt = None
    user_prompt = ""
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt += f"\nUser: {content}"
        elif role == "assistant":
            if "tool_calls" in msg:
                user_prompt += f"\nAssistant Tool Call: {json.dumps(msg['tool_calls'])}"
            else:
                user_prompt += f"\nAssistant: {content}"
        elif role == "tool":
            user_prompt += f"\nTool Result [{msg.get('name', 'unknown')}]: {content}"

    user_prompt = user_prompt.strip()

    tools = body.get("tools", [])
    if tools:
        tool_descriptions = []
        for t in tools:
            if t.get("type") == "function":
                f = t.get("function", {})
                desc = f.get("description", "")
                name = f.get("name", "unknown")
                params = json.dumps(f.get("parameters", {}))
                tool_descriptions.append(f"- {name}: {desc}\n  Schema: {params}")

        tools_section = (
            "\n\n## Available Tools\n"
            "You have access to the following tools. If you need to use them, return them in the 'tool_calls' field.\n\n"
            + "\n\n".join(tool_descriptions)
        )
        if system_prompt:
            system_prompt += tools_section
        else:
            system_prompt = tools_section

    logger.info(
        "Bridge request: %d messages, system=%d chars, prompt=%d chars, tools=%d",
        len(messages),
        len(system_prompt or ""),
        len(user_prompt),
        len(tools),
    )

    t0 = time.time()
    try:
        tool_calls_out = []
        final_content = ""
        raw_response = ""

        if tools:
            logger.info("Using structured completion for tool calls...")
            parsed_resp = llm_completion_structured(user_prompt, BridgeChatResponse, system=system_prompt, max_retries=2)
            raw_response = parsed_resp.model_dump_json()
            
            if parsed_resp.tool_calls:
                for call in parsed_resp.tool_calls:
                    tool_calls_out.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments)
                        }
                    })
            else:
                final_content = parsed_resp.content or ""
        else:
            raw_response = llm_completion(user_prompt, system=system_prompt)
            final_content = raw_response
            
            parsed_tool_calls = _try_parse_tool_calls(raw_response)
            if parsed_tool_calls:
                tool_calls_out = parsed_tool_calls
                final_content = ""

    except Exception as e:
        logger.error("llm_completion failed: %s", e)
        return jsonify({"error": f"LLM backend error: {e}"}), 502
        
    elapsed = time.time() - t0
    logger.info("Bridge response in %.2fs: %d chars", elapsed, len(raw_response))
    logger.info(f"LLM Response:\n{raw_response}\n")

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    is_stream = body.get("stream", False)
    
    if is_stream:
        def generate():
            if tool_calls_out:
                chunk_tool_calls = []
                for i, tc in enumerate(tool_calls_out):
                    chunk_tool_calls.append({
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
                chunk1 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "tool_calls": chunk_tool_calls},
                        "finish_reason": None
                    }]
                }
                chunk2 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls"
                    }]
                }
            else:
                chunk1 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": final_content},
                        "finish_reason": None
                    }]
                }
                chunk2 = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "llm-client-bridge",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"
            
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_content if not tool_calls_out else None,
            },
            "finish_reason": "tool_calls" if tool_calls_out else "stop",
        }
        if tool_calls_out:
            choice["message"]["tool_calls"] = tool_calls_out

        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "llm-client-bridge",
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(raw_response.split()),
                "total_tokens": len(user_prompt.split()) + len(raw_response.split()),
            },
        }
        return jsonify(response)

if __name__ == "__main__":
    logger.info(f"Starting Flask-based LLM bridge on port {BRIDGE_PORT}...")
    app.run(host="0.0.0.0", port=BRIDGE_PORT, threaded=True)
