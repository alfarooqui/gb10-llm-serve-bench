#!/usr/bin/env bash
# Run the full sweep for one profile against the already-running server.
#   ./scripts/run_bench.sh configs/profiles/<profile>.env
set -euo pipefail

cd "$(dirname "$0")/.."
PROFILE="${1:?usage: run_bench.sh <profile.env>}"
source "$PROFILE"

# Fail fast if the server is not up or serves a different model.
SERVED=$(curl -sf "http://localhost:${PORT:-8000}/v1/models" | python3 -c \
  'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])') || {
  echo "No server on port ${PORT:-8000}. Start it with scripts/02_serve.sh." >&2
  exit 1
}
if [[ "$SERVED" != "$MODEL" ]]; then
  echo "Server serves '$SERVED' but profile expects '$MODEL'." >&2
  exit 1
fi

python3 scripts/03_bench.py \
  --base-url "http://localhost:${PORT:-8000}" \
  --model "$MODEL" \
  --profile "$PROFILE_NAME" \
  --levels "${BENCH_LEVELS:-1,2,4,8,16}" \
  --max-tokens "${BENCH_MAX_TOKENS:-256}" \
  --out results
