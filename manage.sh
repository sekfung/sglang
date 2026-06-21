#!/usr/bin/env bash
# =============================================================================
# SGLang + Gateway — 一键管理
# =============================================================================
#   ./manage.sh 128k up     # 启动 128K worker + gateway + postgres
#   ./manage.sh 256k down   # 停止 256K
#   ./manage.sh all status  # 全部状态
#   ./manage.sh nvfp4 logs  # NVFP4 日志
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_ARGS=(-f "$SCRIPT_DIR/docker-compose.yml")
PROFILE="${1:-}"
ACTION="${2:-up}"

usage() {
  echo "用法: $0 <profile> <action>"
  echo "  profile: 128k | 256k | nvfp4 | all (仅 gateway+pg, 不加 worker)"
  echo "  action:  up | down | restart | logs | status | ps"
  exit 1
}

[ -z "$PROFILE" ] && usage
[[ "$ACTION" =~ ^(up|down|restart|logs|status|ps)$ ]] || usage

if [ "$PROFILE" = "all" ]; then
  docker compose "${COMPOSE_ARGS[@]}" "$ACTION"
else
  docker compose --profile "$PROFILE" "${COMPOSE_ARGS[@]}" "$ACTION"
fi
