#!/usr/bin/env bash
# Gateway + MCP HTTP server entrypoint.
# The MCP HTTP server exposes tools to the gateway's regular responses pipeline.
# It dies with the container; docker restart brings both back up.
set -euo pipefail

MCP_PORT="${MCP_PORT:-9000}"

# Start HTTP MCP server in background
echo "[entrypoint] starting MCP HTTP server on port ${MCP_PORT}"
python3 /opt/models/sglang/gateway/dummy_mcp_http.py "${MCP_PORT}" &
MCP_PID=$!
trap 'kill ${MCP_PID} 2>/dev/null; exit 0' EXIT TERM INT

# Wait for MCP server to be ready
for i in $(seq 1 20); do
  if curl -sf -m 1 -X POST "http://127.0.0.1:${MCP_PORT}/mcp" \
          -H 'Accept: application/json, text/event-stream' \
          -d '{"jsonrpc":"2.0","method":"ping"}' >/dev/null 2>&1; then
    echo "[entrypoint] MCP HTTP server ready"
    break
  fi
  sleep 1
done

# Launch the gateway (foreground) — $@ = compose command args after entrypoint
exec "$@"
