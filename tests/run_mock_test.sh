#!/usr/bin/env bash
# Self-check for the benchmark harness. Needs no GPU and no Docker.
# Starts the mock server, runs the harness, and asserts the known values.
#   ./tests/run_mock_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8765
TOKENS=20
DELAY_MS=10
PREFILL_MS=50
TMP=$(mktemp -d)
PASS=0; FAIL=0
check() { if eval "$2"; then echo "PASS  $1"; PASS=$((PASS+1)); else echo "FAIL  $1"; FAIL=$((FAIL+1)); fi; }

python3 tests/mock_server.py --port $PORT --tokens $TOKENS --delay-ms $DELAY_MS --prefill-ms $PREFILL_MS >"$TMP/mock.log" 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true; rm -rf "$TMP"' EXIT
for _ in $(seq 1 50); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 0.1; done

# Run 1: valid sweep
python3 scripts/03_bench.py --base-url http://127.0.0.1:$PORT --model mock/model \
  --profile test-a --levels 1,2 --max-tokens 256 --out "$TMP/out" >"$TMP/run1.log" 2>&1
RUN_A=$(ls -d "$TMP"/out/test-a/*)
check "valid sweep exits 0" "[[ -f '$RUN_A/summary.json' ]]"
check "summary.json has 2 levels" "python3 -c \"import json,sys; d=json.load(open('$RUN_A/summary.json')); sys.exit(0 if len(d['levels'])==2 else 1)\""
check "zero failures" "python3 -c \"import json,sys; d=json.load(open('$RUN_A/summary.json')); sys.exit(0 if all(l['failed']==0 for l in d['levels']) else 1)\""
check "tokens per request == $TOKENS" "python3 -c \"import json,sys; r=[json.loads(l) for l in open('$RUN_A/raw.jsonl')]; sys.exit(0 if all(x['out_tokens']==$TOKENS for x in r) else 1)\""
check "TTFT ~ prefill + 1 token (expect 0.06 s, accept 0.05-0.09)" "python3 -c \"import json,sys; d=json.load(open('$RUN_A/summary.json')); v=d['levels'][0]['ttft_p50_s']; sys.exit(0 if 0.05<=v<=0.09 else 1)\""
check "ITL ~ 10 ms (accept 8-14)" "python3 -c \"import json,sys; d=json.load(open('$RUN_A/summary.json')); v=d['levels'][0]['itl_p50_ms']; sys.exit(0 if 8<=v<=14 else 1)\""
check "E2E ~ 0.25 s (accept 0.23-0.30)" "python3 -c \"import json,sys; d=json.load(open('$RUN_A/summary.json')); v=d['levels'][0]['e2e_p50_s']; sys.exit(0 if 0.23<=v<=0.30 else 1)\""
check "summary.md written" "[[ -s '$RUN_A/summary.md' ]]"
check "env.json written" "[[ -s '$RUN_A/env.json' ]]"

# Run 2: deliberately different max_tokens, compare must warn
python3 scripts/03_bench.py --base-url http://127.0.0.1:$PORT --model mock/model \
  --profile test-b --levels 1,2 --max-tokens 128 --out "$TMP/out" >"$TMP/run2.log" 2>&1
RUN_B=$(ls -d "$TMP"/out/test-b/*)
python3 scripts/04_compare.py "$RUN_A" "$RUN_B" --out "$TMP/cmp.md" >"$TMP/cmp.log" 2>&1
check "compare warns on mismatched max_tokens" "grep -q 'settings differ' '$TMP/cmp.log'"

# Run 3: matched settings, compare must not warn
python3 scripts/03_bench.py --base-url http://127.0.0.1:$PORT --model mock/model \
  --profile test-c --levels 1,2 --max-tokens 256 --out "$TMP/out" >"$TMP/run3.log" 2>&1
RUN_C=$(ls -d "$TMP"/out/test-c/*)
python3 scripts/04_compare.py "$RUN_A" "$RUN_C" --out "$TMP/cmp2.md" >"$TMP/cmp2.log" 2>&1
check "compare silent on matched settings" "! grep -q 'settings differ' '$TMP/cmp2.log'"

# Run 4: broken endpoint, harness must exit 2 and count failures
set +e
python3 scripts/03_bench.py --base-url http://127.0.0.1:8799 --model mock/model \
  --profile test-fail --levels 1 --out "$TMP/out" >"$TMP/run4.log" 2>&1
RC=$?
set -e
check "broken endpoint exits 2" "[[ $RC -eq 2 ]]"

echo
echo "Mock test: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
