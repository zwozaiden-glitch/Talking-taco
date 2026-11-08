"""A tiny OpenAI-compatible server for local testing.

Run it, point AI_BASE_URL at it, and the full bot pipeline works with real
HTTP traffic -- no credentials, no cost. Handy for verifying your .env before
switching to your real endpoint.

    python scripts/fake_api_server.py            # listens on 127.0.0.1:8000
    AI_BASE_URL=http://127.0.0.1:8000/v1 AI_MODEL=fake python -m talkingtaco
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[fake-api] {self.address_string()} {fmt % args}")

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "unknown path")
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        model = body.get("model", "fake")
        reply = (
            f"FakeAI ({model}) heard: {last_user[:80]!r}. "
            f"You sent {len(messages)} message(s) of context."
        )

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for word in reply.split(" "):
                payload = {
                    "choices": [{"delta": {"content": word + " "}, "index": 0}]
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        payload = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[fake-api] OpenAI-compatible server on http://{host}:{port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fake-api] bye")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
