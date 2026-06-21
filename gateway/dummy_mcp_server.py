#!/usr/bin/env python3
"""Minimal self-contained STDIO MCP server (stdlib only, no `mcp` SDK).

Purpose: validate the sgl-model-gateway server-side tool-execution loop
end-to-end without pulling any dependency. The gateway spawns this script as
a subprocess (STDIO transport) and discovers/executes its tools.

Transport: MCP stdio == newline-delimited JSON-RPC 2.0 (one JSON object per
line on stdin/stdout). Logs go to stderr so they never corrupt the channel.

Tools exposed:
  - add(a, b)        -> a + b           (deterministic; easy to verify)
  - get_secret_code  -> a fixed token   (proves the answer came via the tool,
                                          not the model's prior knowledge)
"""
import json
import sys

PROTOCOL_VERSION = "2025-06-18"  # echo client's version when provided
SERVER_INFO = {"name": "dummy-mcp", "version": "0.1.0"}

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "first addend"},
                "b": {"type": "integer", "description": "second addend"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "get_secret_code",
        "description": (
            "Return the project's secret access code. The model cannot know "
            "this value without calling the tool."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]

SECRET_CODE = "SGLANG-MCP-OK-7421"


def log(msg: str) -> None:
    print(f"[dummy-mcp] {msg}", file=sys.stderr, flush=True)


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def make_result(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def call_tool(name: str, args: dict):
    """Return the MCP tool-result `content` list for a tool call."""
    if name == "add":
        total = int(args["a"]) + int(args["b"])
        text = str(total)
    elif name == "get_secret_code":
        text = SECRET_CODE
    else:
        raise KeyError(name)
    return [{"type": "text", "text": text}]


def handle(msg: dict):
    """Dispatch one JSON-RPC message. Returns a response dict or None (notify)."""
    method = msg.get("method")
    req_id = msg.get("id")

    if method == "initialize":
        client_ver = (msg.get("params") or {}).get("protocolVersion")
        return make_result(
            req_id,
            {
                "protocolVersion": client_ver or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        )

    if method in ("notifications/initialized", "initialized"):
        return None  # notification, no response

    if method == "ping":
        return make_result(req_id, {})

    if method == "tools/list":
        return make_result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            content = call_tool(name, args)
            log(f"tools/call {name}({args}) -> {content}")
            return make_result(req_id, {"content": content, "isError": False})
        except Exception as e:  # noqa: BLE001 - surface as MCP tool error
            log(f"tools/call {name} failed: {e}")
            return make_result(
                req_id,
                {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                },
            )

    if req_id is not None:
        return make_error(req_id, -32601, f"method not found: {method}")
    return None


def main() -> None:
    log("started (stdio)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"bad json: {e}")
            continue
        try:
            resp = handle(msg)
        except Exception as e:  # noqa: BLE001
            log(f"handler crashed: {e}")
            resp = make_error(msg.get("id"), -32603, str(e))
        if resp is not None:
            send(resp)
    log("stdin closed, exiting")


if __name__ == "__main__":
    main()
