#!/usr/bin/env python3
"""Mock OpenAI-compatible server for verifying the benchmark harness without a GPU.

Implements: GET /health, GET /version, GET /v1/models, POST /v1/chat/completions (SSE).
Streams a fixed number of tokens with a fixed per-token delay, so the harness
output can be checked against known values.

Usage: python tests/mock_server.py --port 8765 --tokens 20 --delay-ms 10
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep test output quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/version":
            self._json(200, {"version": "mock-0.0.0"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": CFG["model"], "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        n_tokens = min(CFG["tokens"], req.get("max_tokens", CFG["tokens"]))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload):
            data = ("data: " + json.dumps(payload) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        # Small pre-token delay to mimic prefill, then one token per delay.
        time.sleep(CFG["prefill_ms"] / 1000.0)
        for i in range(n_tokens):
            time.sleep(CFG["delay_ms"] / 1000.0)
            chunk({"id": "mock", "object": "chat.completion.chunk", "model": CFG["model"],
                   "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]})
        if include_usage:
            chunk({"id": "mock", "object": "chat.completion.chunk", "model": CFG["model"],
                   "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": n_tokens,
                                            "total_tokens": 10 + n_tokens}})
        end = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="mock/model")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--delay-ms", type=float, default=10)
    ap.add_argument("--prefill-ms", type=float, default=50)
    args = ap.parse_args()
    CFG.update(model=args.model, tokens=args.tokens, delay_ms=args.delay_ms,
               prefill_ms=args.prefill_ms)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock server on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
