# 03 — Benchmark

Goal: a reproducible concurrency sweep that produces the same metrics on every GPU, so results can be compared directly.

## What the harness does

`scripts/03_bench.py` sends streaming chat requests to the running server at a series of concurrency levels, then writes:

- `raw.jsonl`: one line per request (TTFT, E2E, token count, ITL, errors)
- `summary.json`: one object per concurrency level, plus host and server metadata
- `summary.md`: a readable table
- `env.json`: host, GPU, driver, and vLLM version at the time of the run

Method:
- **Warmup**: two requests at concurrency 1, discarded. Removes first-run compilation effects.
- **Sweep**: concurrency levels from `--levels`, default `1,2,4,8,16`.
- **Requests per level**: `max(8, 4 × concurrency)` by default, so every level has enough samples.
- **Fixed prompts**: the same prompt set in the same order every run.
- **Deterministic output**: `temperature=0`, fixed `max_tokens`.
- **Exact token counts**: taken from the server's `usage` block when present, otherwise counted from streamed chunks.

## Run it

With the server up (see [02-serve.md](02-serve.md)):

```bash
./scripts/run_bench.sh configs/profiles/gb10-128g.env
```

That reads `MODEL` and `PROFILE_NAME` from the profile, runs the sweep, and writes to `results/<PROFILE_NAME>/<timestamp>/`.

Direct form, if you want to tune it:

```bash
python scripts/03_bench.py \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --profile gb10-128g \
  --levels 1,2,4,8,16,32 \
  --max-tokens 256
```

The script exits with status 2 if any request failed. Treat exit 2 as a broken run, even though the outputs are still written.

## Reading the results

Look at three things, in this order:

1. **Failed must be 0.** A level with failures is not a valid data point. Lower the concurrency or `MAX_NUM_SEQS` headroom and rerun.
2. **Aggregate output tok/s vs concurrency.** This is the scaling curve. It should rise and then flatten. Where it flattens is the saturation point for this hardware and model.
3. **TTFT p95 vs concurrency.** This is the user-facing cost of queueing. It climbs sharply once the server is saturated.

Per-stream tok/s at concurrency 1 is your single-user speed. Compare it with aggregate tok/s at the highest level to see how much batching buys you.

## Rules that keep runs comparable

- Same model, same prompts, same `max_tokens`, same `temperature`, across every profile you compare.
- One server at a time. Stop the previous profile before starting the next.
- Record the run. Do not delete failed runs; note why they failed in `summary.md`.
- Run each profile at least twice. If the two runs differ by more than about 10 percent on aggregate tok/s, the box is not in a steady state. Check for background load or thermal throttling before trusting either number.

## Optional: GPU telemetry during a run

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv -l 1 > results/<PROFILE_NAME>/<timestamp>/gpu.csv &
```

Stop it after the sweep. This gives you the power and memory curve, which matters for the GB10's shared memory pool.
