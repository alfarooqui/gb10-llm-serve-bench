#!/usr/bin/env bash
# Pull the NGC vLLM container and record the resolved vLLM version.
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/vllm-image.env
mkdir -p results

echo "Pulling $VLLM_IMAGE ..."
docker pull "$VLLM_IMAGE"

echo "Recording vLLM version ..."
VERSION=$(docker run --rm --gpus all --ipc=host "$VLLM_IMAGE" \
  python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -n1 || echo "unknown")
ARCH=$(docker image inspect "$VLLM_IMAGE" --format '{{.Architecture}}')

{
  echo "image: $VLLM_IMAGE"
  echo "vllm_version: $VERSION"
  echo "image_arch: $ARCH"
  echo "recorded_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee results/container.txt

if [[ "$ARCH" != "$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')" ]]; then
  echo "WARNING: image architecture ($ARCH) does not match host ($(uname -m))." >&2
  exit 1
fi
