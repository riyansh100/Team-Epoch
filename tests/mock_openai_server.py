"""A small OpenAI-compatible server that imitates OpenCode Zen, for testing the Mac scripts offline.

    python tests/mock_openai_server.py --port 18766 [--fail-429 N] [--fail-500 N] [--empty N]

Endpoints
    GET  /v1/models             Zen-style listing (cost.text.*.price, capabilities.temperature, recommended_api)
    POST /v1/chat/completions   deterministic answers recognised from AutoDDG's prompts
    GET  /stats                 {"chat_requests": n, "by_kind": {...}}   (used by the tests)
    POST /reset                 reset counters; body may set {"fail_429": n, "fail_500": n, "empty": n}

Behaviour
    * no Authorization header        -> 429 FreeUsageLimitError (what Zen does)
    * key other than "test-key"      -> 401
    * unknown model                  -> 404
    * model "mock-notemp-free" + temperature in the body -> 400 "Unsupported parameter: temperature"
    * first N chat requests: 429 with Retry-After: 0 (--fail-429), then N x 500 (--fail-500), then N empty messages
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    {"id": "mock-free", "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
     "capabilities": {"temperature": True}, "recommended_api": "chat-completions"},
    {"id": "mock-notemp-free", "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
     "capabilities": {"temperature": False}, "recommended_api": "chat-completions"},
    {"id": "mock-messages-free", "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
     "capabilities": {"temperature": True}, "recommended_api": "messages"},
    {"id": "mock-paid", "cost": {"text": {"input": {"price": 1.5}, "output": {"price": 6}}},
     "capabilities": {"temperature": True}, "recommended_api": "chat-completions"},
]
MODEL_IDS = {m["id"] for m in MODELS}

STATE = {"chat_requests": 0, "by_kind": {}, "fail_429": 0, "fail_500": 0, "empty": 0}
LOCK = threading.Lock()


def semantic_json(cols: list[str]) -> dict:
    return {c: {"Domain-Specific Types": "Government", "Entity Type": "Attribute",
                "Function/Usage Context": "Measurement", "Spatial": {"isSpatial": "state" in c.lower(),
                                                                     "resolution": "State"},
                "Temporal": {"isTemporal": "year" in c.lower() or "date" in c.lower(), "resolution": "Year"}}
            for c in cols}


def answer(messages: list[dict]) -> tuple[str, str]:
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    user = " ".join(m["content"] for m in messages if m["role"] != "system")
    if "Analyze multiple columns" in user:
        cols = re.findall(r"^Column: (.*)$", user, flags=re.M)
        return "semantic_group", "```json\n" + json.dumps(semantic_json(cols)) + "\n```"
    if "classify the column into multiple semantic types" in user:
        col = re.search(r"Column name: (.*)", user).group(1)
        return "semantic_single", json.dumps(semantic_json([col])[col])
    if "generating concise dataset topics" in system:
        title = re.search(r"Title: (.*)", user).group(1)
        return "topic", " ".join(title.split()[:2]) + " Data"
    if "expand the description" in user:
        topic = re.search(r"about the topic (.*?), with", user).group(1)
        return "sfd", f"Dataset Overview: expanded for {topic}. Keywords: {topic.lower()}, statistics, records."
    if "provide a dataset description" in user:
        header = re.search(r"First, consider the dataset sample:\s*\n\s*\n?(.*)", user).group(1)
        return "ufd", f"This dataset has columns {header.strip()}. It supports analysis of public records."
    return "other", "pong"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": MODELS})
        if self.path == "/stats":
            with LOCK:
                return self._send(200, {k: STATE[k] for k in ("chat_requests", "by_kind")})
        return self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/reset":
            with LOCK:
                STATE.update(chat_requests=0, by_kind={}, fail_429=body.get("fail_429", 0),
                             fail_500=body.get("fail_500", 0), empty=body.get("empty", 0))
            return self._send(200, {"ok": True})
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "not found"}})
        auth = self.headers.get("Authorization")
        if not auth:
            return self._send(429, {"type": "error", "error": {"type": "FreeUsageLimitError",
                                                              "message": "Free usage limit reached."}})
        if auth != "Bearer test-key":
            return self._send(401, {"error": {"type": "AuthError", "message": "Invalid API key"}})
        if body.get("model") not in MODEL_IDS:
            return self._send(404, {"error": {"type": "ModelError", "message": f"Model {body.get('model')} not found"}})
        if body["model"] == "mock-notemp-free" and "temperature" in body:
            return self._send(400, {"error": {"type": "invalid_request_error",
                                              "message": "Unsupported parameter: temperature"}})
        with LOCK:
            STATE["chat_requests"] += 1
            if STATE["fail_429"] > 0:
                STATE["fail_429"] -= 1
                return self._send(429, {"error": {"type": "rate_limit", "message": "Too many requests"}},
                                  {"Retry-After": "0"})
            if STATE["fail_500"] > 0:
                STATE["fail_500"] -= 1
                return self._send(500, {"error": {"type": "server_error", "message": "upstream failed"}})
            empty = STATE["empty"] > 0
            if empty:
                STATE["empty"] -= 1
        kind, text = answer(body["messages"])
        if kind in ("topic", "ufd", "sfd"):  # make every answer depend on its prompt, like a real model would
            text += f" ref{hashlib.sha1(json.dumps(body['messages']).encode()).hexdigest()[:6]}"
        if empty:
            text = ""
        with LOCK:
            STATE["by_kind"][kind] = STATE["by_kind"].get(kind, 0) + 1
        prompt_tokens = sum(len(m["content"]) for m in body["messages"]) // 4
        return self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text) // 4,
                      "total_tokens": prompt_tokens + len(text) // 4}})


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18766)
    ap.add_argument("--fail-429", type=int, default=0)
    ap.add_argument("--fail-500", type=int, default=0)
    ap.add_argument("--empty", type=int, default=0)
    a = ap.parse_args()
    STATE.update(fail_429=a.fail_429, fail_500=a.fail_500, empty=a.empty)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock Zen on http://127.0.0.1:{a.port}/v1", flush=True)
    srv.serve_forever()
