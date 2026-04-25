"""
OpenAI-compatible HTTP bridge for llm_client.py

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
from http.server import HTTPServer, BaseHTTPRequestHandler

from llm_client import llm_completion, llm_completion_structured
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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


class BridgeHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI chat/completions endpoint backed by llm_client."""

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    # ---------- routing ----------
    def do_POST(self):
        if self.path.rstrip("/") in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions_mock()
            # self._handle_chat_completions()
        else:
            self._json_error(404, f"Not found: {self.path}")

    def _handle_chat_completions_mock(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._json_error(400, f"Bad request body: {e}")
            return
            
        import uuid
        import time
        import json
        
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
        self._send_json(200, response)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models", "/health", "/"):
            self._send_json(200, {"status": "ok", "models": ["llm-client-bridge"]})
        else:
            self._json_error(404, f"Not found: {self.path}")

    # ---------- core ----------
    def _handle_chat_completions(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._json_error(400, f"Bad request body: {e}")
            return

        messages = body.get("messages", [])
        if not messages:
            self._json_error(400, "messages is required")
            return

        # Build a single prompt string from the messages array.
        # Separate system message from the rest for llm_client's `system` arg.
        system_parts = []
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # content can be a list of content blocks (vision format)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)

            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"[Assistant]\n{content}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                conversation_parts.append(
                    f"[Tool Result for {tool_call_id}]\n{content}"
                )
            else:
                conversation_parts.append(f"[User]\n{content}")

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(conversation_parts)

        # Extract tool definitions if present and append them to system prompt
        tools = body.get("tools", [])
        if tools:
            tool_descriptions = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    name = func.get("name", "")
                    desc = func.get("description", "")
                    params = json.dumps(func.get("parameters", {}), indent=2)
                    tool_descriptions.append(
                        f"### {name}\n{desc}\nParameters:\n```json\n{params}\n```"
                    )

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
                
                # Fallback for parsing tool calls without explicit tools array
                parsed_tool_calls = self._try_parse_tool_calls(raw_response)
                if parsed_tool_calls:
                    tool_calls_out = parsed_tool_calls
                    final_content = ""

        except Exception as e:
            logger.error("llm_completion failed: %s", e)
            self._json_error(502, f"LLM backend error: {e}")
            return
            
        elapsed = time.time() - t0
        logger.info("Bridge response in %.2fs: %d chars", elapsed, len(raw_response))
        logger.info(f"LLM Response:\n{raw_response}\n")

        # Build OpenAI-format response
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

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
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
        self._send_json(200, response)

    # ---------- tool call parsing ----------
    @staticmethod
    def _try_parse_tool_calls(text: str) -> list[dict] | None:
        """
        Attempt to extract tool_calls from the LLM's raw text response.
        Returns OpenAI-format tool_calls list, or None if not a tool call.
        """
        import re

        # Try to find JSON with tool_calls key
        # First try: the whole response is JSON
        stripped = text.strip()
        if stripped.startswith("```"):
            # Strip markdown code fences
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
            stripped = stripped.strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to extract JSON block
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None

        raw_calls = parsed.get("tool_calls")
        if not raw_calls or not isinstance(raw_calls, list):
            return None

        result = []
        for i, call in enumerate(raw_calls):
            name = call.get("name", "")
            args = call.get("arguments", {})
            if not name:
                continue
            result.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })

        return result if result else None

    # ---------- helpers ----------
    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str):
        self._send_json(status, {"error": {"message": message, "type": "bridge_error"}})


def main():
    server = HTTPServer(("0.0.0.0", BRIDGE_PORT), BridgeHandler)
    logger.info("LLM Bridge listening on http://0.0.0.0:%d", BRIDGE_PORT)
    logger.info("PicoClaw endpoint: http://localhost:%d/v1/chat/completions", BRIDGE_PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge...")
        server.server_close()


if __name__ == "__main__":
    main()
