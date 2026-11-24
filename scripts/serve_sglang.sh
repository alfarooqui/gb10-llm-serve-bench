#!/usr/bin/env bash
# Start or stop an SGLang server for one profile.
# Usage: scripts/serve_sglang.sh start|stop <profile.env>
set -euo pipefail

ACTION="${1:?usage: $0 start|stop <profile.env>}"
PROFILE="${2:?usage: $0 start|stop <profile.env>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source "$ROOT/configs/sglang-image.env"
source "$PROFILE"

NAME="sglang-${PROFILE_NAME}"
LOG="$ROOT/logs/${PROFILE_NAME}.log"
mkdir -p "$ROOT/logs"

case "$ACTION" in
  stop)
    if docker rm -f "$NAME" >/dev/null 2>&1; then
      echo "stopped $NAME"
    else
      echo "$NAME is not running"
    fi
    exit 0
    ;;
  start) ;;
  *) echo "unknown action: $ACTION (use start or stop)" >&2; exit 1 ;;
esac

# Refuse to start if any inference server (vLLM or SGLang) is already running.
RUNNING="$(docker ps --format '{{.Names}}' | grep -E '^(vllm-|sglang-)' || true)"
if [[ -n "$RUNNING" ]]; then
  echo "An inference server is already running:" >&2
  echo "$RUNNING" >&2
  exit 1
fi

docker run -d --name "$NAME" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:30000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$SGLANG_IMAGE" \
  sglang serve --model-path "$MODEL" --host 0.0.0.0 --port 30000 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$MAX_MODEL_LEN" \
  --max-running-requests "$MAX_RUNNING" >/dev/null

# Follow the container log into logs/ (runs in the background)
nohup docker logs -f "$NAME" >"$LOG" 2>&1 &

echo "waiting for http://localhost:${PORT}/v1/models (model load can take several minutes)"
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
    echo "ready after about $((i * 5)) s; log: $LOG"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]]; then
    echo "container exited; see $LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "timed out waiting for the server; see $LOG" >&2
exit 1
