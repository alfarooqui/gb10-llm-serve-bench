#!/usr/bin/env bash
# Preflight: verify the host can run the serve + bench pipeline.
# Usage: ./scripts/00_preflight.sh [--all]
#   --all  also checks the Python client and the local container image.
set -u

PASS=0; FAIL=0; WARN=0
ok()   { echo "OK    $*"; PASS=$((PASS+1)); }
fail() { echo "FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "WARN  $*"; WARN=$((WARN+1)); }

ALL=0; [[ "${1:-}" == "--all" ]] && ALL=1
IMAGE_ENV="$(dirname "$0")/../configs/vllm-image.env"
[[ -f "$IMAGE_ENV" ]] && source "$IMAGE_ENV"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.09-py3}"

# 1. Architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" || "$ARCH" == "x86_64" ]]; then
  ok "architecture: $ARCH"
else
  fail "architecture: $ARCH (expected aarch64 or x86_64)"
fi

# 2. GPU present and driver
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
  ok "GPU: $GPU_NAME | driver $DRIVER | compute capability ${CC:-unknown}"
else
  fail "nvidia-smi not found"
fi

# 3. Memory (unified on GB10, dedicated VRAM elsewhere)
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
if [[ "$MEM_GB" -ge 64 ]]; then
  ok "system memory: ${MEM_GB} GB"
else
  warn "system memory: ${MEM_GB} GB (small for 7B+ models)"
fi

# 4. Disk for model weights
FREE_GB=$(df -BG --output=avail "$HOME" | tail -n1 | tr -dc '0-9')
if [[ "$FREE_GB" -ge 100 ]]; then
  ok "free disk in \$HOME: ${FREE_GB} GB"
else
  warn "free disk in \$HOME: ${FREE_GB} GB (plan 100 GB+ for weights)"
fi

# 5. Docker and GPU passthrough
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  ok "docker reachable"
  if docker info 2>/dev/null | grep -qi "nvidia"; then
    ok "docker: NVIDIA runtime registered"
  else
    warn "docker: NVIDIA runtime not listed in 'docker info' (--gpus may still work)"
  fi
else
  fail "docker not reachable (is your user in the docker group?)"
fi

# 6. Hugging Face token (needed only for gated models)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN set"
else
  warn "HF_TOKEN not set (fine for open models, required for gated ones)"
fi

if [[ $ALL -eq 1 ]]; then
  # 7. Python for the benchmark client
  if command -v python3 >/dev/null 2>&1 && python3 -c "import http.client, urllib.request" 2>/dev/null; then
    ok "python3 available (benchmark uses the standard library only)"
  else
    fail "python3 missing"
  fi

  # 8. Image present locally
  if docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    ok "image present: $VLLM_IMAGE"
  else
    fail "image not pulled: $VLLM_IMAGE (run scripts/01_pull_container.sh)"
  fi
fi

echo
echo "Preflight: $PASS OK, $WARN WARN, $FAIL FAIL"
[[ $FAIL -eq 0 ]]
