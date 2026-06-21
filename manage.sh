#!/usr/bin/env bash
# =============================================================================
# SGLang + Gateway — 一键管理
# =============================================================================
#   ./manage.sh up 256k     # 启动 256K worker + gateway
#   ./manage.sh up 128k     # 启动 128K worker + gateway
#   ./manage.sh up           # 只启动 gateway (不加 worker)
#   ./manage.sh down 256k   # 停止 256K + gateway
#   ./manage.sh restart 256k # 重启 256K + gateway
#   ./manage.sh restart      # 只重启 gateway
#   ./manage.sh logs 256k   # 查看 256K 日志
#   ./manage.sh status      # 全部状态
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ACTION="${1:-up}"
PROFILE="${2:-}"  # optional

# --help / -h anywhere triggers usage
for arg in "$@"; do
  [[ "$arg" =~ ^(--help|-h)$ ]] && ACTION="help"
done

usage() {
  cat <<EOF
用法: $0 <action> [profile]

action:
  up        启动服务
  down      停止服务
  restart   重启服务
  logs      查看日志
  status    查看状态 (同 ps)
  ps        列出容器

profile (可选, 不填只操作 gateway):
  128k      DeepSeek-V4 128K worker
  256k      DeepSeek-V4 256K worker

示例:
  $0 up 256k            启动 256K worker + gateway
  $0 up                 只启动 gateway (无 worker)
  $0 restart 256k       重启 256K worker + gateway
  $0 restart            只重启 gateway (不重启 worker)
  $0 down 256k          停止 256K worker + gateway
  $0 logs 256k          查看 256K 服务日志 (实时跟踪)
  $0 status             查看所有运行中的服务

服务架构:
  :8080  → gateway (OpenAI 兼容入口 + MCP 工具编排)
            └─ grpc://:21000 → worker (DeepSeek-V4 推理)
EOF
  exit 0
}

[[ "$ACTION" == "help" ]] && usage
[[ "$ACTION" =~ ^(up|down|restart|logs|status|ps)$ ]] || usage

# Map status → ps (compose doesn't have a "status" subcommand)
COMPOSE_ACTION="$ACTION"
[[ "$COMPOSE_ACTION" == "status" ]] && COMPOSE_ACTION="ps"

if [ -n "$PROFILE" ]; then
  docker compose --profile "$PROFILE" -f "$COMPOSE_FILE" "$COMPOSE_ACTION"
else
  docker compose -f "$COMPOSE_FILE" "$COMPOSE_ACTION"
fi
