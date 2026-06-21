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

usage() {
  echo "用法: $0 <action> [profile]"
  echo "  action:  up | down | restart | logs | status | ps"
  echo "  profile: 128k | 256k (不填只操作 gateway)"
  echo ""
  echo "示例:"
  echo "  $0 up 256k         启动 256K worker + gateway"
  echo "  $0 up              只启动 gateway"
  echo "  $0 restart 256k    重启 256K + gateway"
  echo "  $0 restart         只重启 gateway"
  echo "  $0 down 256k       停止 256K + gateway"
  exit 1
}

[[ "$ACTION" =~ ^(up|down|restart|logs|status|ps)$ ]] || usage

if [ -n "$PROFILE" ]; then
  docker compose --profile "$PROFILE" -f "$COMPOSE_FILE" "$ACTION"
else
  docker compose -f "$COMPOSE_FILE" "$ACTION"
fi
