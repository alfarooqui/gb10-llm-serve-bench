#!/usr/bin/env bash
# Start or stop a vLLM server from a profile file.
#   ./scripts/02_serve.sh configs/profiles/<profile>.env
#   ./scripts/02_serve.sh --stop
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/vllm-image.env
mkdir -p logs

if [[ "${1:-}" == "--stop" ]]; then
  for c in $(docker ps -a --format '{{.Names}}' | grep '^vllm-' || true); do
    echo "Stopping $c"
    docker rm -f "$c" >/dev/null
  done
  exit 0
fi

PROFILE="${1:?usage: 02_serve.sh <profile.env> | --stop}"
source "$PROFILE"

NAME="vllm-${PROFILE_NAME}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Refuse to start a second server: it would contaminate the memory numbers.
if docker ps --format '{{.Names}}' | grep -q '^vllm-'; then
  echo "A vLLM server is already running. Stop it first: ./scripts/02_serve.sh --stop" >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "Starting $NAME with model $MODEL"
docker run -d --name "$NAME" \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:8000" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    ${EXTRA_ARGS:-} >/dev/null

# Wait for health, up to 15 minutes (first load downloads weights and compiles kernels).
echo "Waiting for /health (first load can take several minutes) ..."
DEADLINE=$((SECONDS + 900))
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  if [[ $SECONDS -gt $DEADLINE ]]; then
    echo "Timed out. Last log lines:" >&2
    docker logs --tail 40 "$NAME" >&2 || true
    exit 1
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]]; then
    echo "Container exited. Last log lines:" >&2
    docker logs --tail 40 "$NAME" >&2 || true
    exit 1
  fi
  sleep 5
done

docker logs -f "$NAME" > "logs/${PROFILE_NAME}.log" 2>&1 &
echo "Ready: http://localhost:${PORT}/v1  (log: logs/${PROFILE_NAME}.log)"
