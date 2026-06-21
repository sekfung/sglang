#!/usr/bin/env python3
"""Minimal Streamable-HTTP MCP server (stdlib only, no `mcp` SDK).

The gateway's regular gRPC responses MCP loop only drives *dynamic* MCP
servers referenced per-request by `server_url` (http/https). Static stdio
servers from --mcp-config-path are not used by that path. So we expose the
same two tools over the MCP Streamable-HTTP transport instead.

Endpoint: POST /mcp  (JSON-RPC 2.0; responses sent back as a one-shot SSE
`event: message` frame, which the rmcp Streamable client accepts). A session
id is issued on `initialize` via the `Mcp-Session-Id` header.

Tools:
  - add(a, b)        -> a + b
  - get_secret_code  -> SGLANG-MCP-OK-7421  (un-guessable; proves real exec)
"""
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"
SECRET_CODE = "SGLANG-MCP-OK-7421"

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "get_secret_code",
        "description": "Return the project's secret access code. The model "
        "cannot know this value without calling the tool.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def log(msg):
    print(f"[dummy-mcp-http] {msg}", file=sys.stderr, flush=True)


def call_tool(name, args):
    if name == "add":
        return [{"type": "text", "text": str(int(args["a"]) + int(args["b"]))}]
    if name == "get_secret_code":
        return [{"type": "text", "text": SECRET_CODE}]
    raise KeyError(name)


def handle_rpc(msg):
    """Return a JSON-RPC response dict, or None for notifications."""
    method = msg.get("method")
    req_id = msg.get("id")

    if method == "initialize":
        client_ver = (msg.get("params") or {}).get("protocolVersion")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_ver or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "dummy-http", "version": "0.1.0"},
            },
        }
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            content = call_tool(name, args)
            log(f"tools/call {name}({args}) -> {content}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": content, "isError": False},
            }
        except Exception as e:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                },
            }
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_sse(self, payload, session_id=None):
        body = f"event: message\ndata: {json.dumps(payload)}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, code=202, session_id=None):
        self.send_response(code)
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._send_empty(400)
            return

        session_id = self.headers.get("Mcp-Session-Id")
        is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
        if is_init and not session_id:
            session_id = uuid.uuid4().hex

        resp = handle_rpc(msg) if isinstance(msg, dict) else None
        if resp is None:
            self._send_empty(202, session_id)
        else:
            self._send_sse(resp, session_id)

    def do_GET(self):
        # rmcp may open a GET SSE stream for server->client messages; we have
        # none to push, so keep it minimal.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):
        self._send_empty(200)

    def log_message(self, *a):
        pass  # silence default logging


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    log(f"listening on 0.0.0.0:{port} (POST /mcp)")
    server.serve_forever()


if __name__ == "__main__":
    main()
