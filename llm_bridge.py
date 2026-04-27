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

from llm_client import llm_completion_raw
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from google.genai import types

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

@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "models": ["llm-client-bridge"]})

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    return _handle_chat_completions()

def _handle_chat_completions():
    body = request.json or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get("role")
        raw_content = msg.get("content", "")
        
        if isinstance(raw_content, list):
            text_content = "".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and item.get("type") == "text"])
            user_parts = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    user_parts.append(types.Part.from_text(text=item.get("text", "")))
        else:
            text_content = raw_content
            user_parts = [types.Part.from_text(text=raw_content)]

        if role == "system":
            if system_instruction:
                system_instruction += "\n" + text_content
            else:
                system_instruction = text_content
        elif role == "user":
            contents.append(types.Content(role="user", parts=user_parts))
        elif role == "assistant":
            parts = []
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except:
                        args = {}
                    parts.append(types.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=args
                    ))
            if text_content:
                parts.append(types.Part.from_text(text=text_content))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            name = msg.get("name", "unknown")
            try:
                resp = json.loads(text_content)
            except:
                resp = {"result": text_content}
                
            contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(
                    name=name,
                    response=resp
                )
            ]))

    tools_in = body.get("tools", [])
    genai_tools = None
    if tools_in:
        function_declarations = []
        for t in tools_in:
            if t.get("type") == "function":
                f = t.get("function", {})
                function_declarations.append(types.FunctionDeclaration(
                    name=f.get("name", ""),
                    description=f.get("description", ""),
                    parameters=f.get("parameters", {})
                ))
        if function_declarations:
            genai_tools = [types.Tool(function_declarations=function_declarations)]

    logger.info(
        "Bridge request: %d messages, system=%d chars, tools=%d",
        len(messages),
        len(system_instruction or ""),
        len(tools_in),
    )

    response_format = body.get("response_format")
    response_mime_type = None
    response_schema = None
    
    if response_format:
        if response_format.get("type") == "json_object":
            response_mime_type = "application/json"

    t0 = time.time()
    try:
        if not contents and system_instruction:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Hi.")]))

        resp = llm_completion_raw(
            contents=contents,
            system_instruction=system_instruction,
            tools=genai_tools,
            response_mime_type=response_mime_type,
            response_schema=response_schema
        )
        
        tool_calls_out = []
        final_content = ""
        
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    tool_calls_out.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args)
                        }
                    })
                elif part.text:
                    final_content += part.text
                    
        raw_response = final_content + (" [TOOLS]" if tool_calls_out else "")
        prompt_tokens = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 10
        completion_tokens = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 10

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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return jsonify(response)

if __name__ == "__main__":
    logger.info(f"Starting Flask-based LLM bridge on port {BRIDGE_PORT}...")
    app.run(host="0.0.0.0", port=BRIDGE_PORT, threaded=True)
