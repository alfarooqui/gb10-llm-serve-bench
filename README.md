# gb10-vllm-serve-bench

Serve and benchmark harness for LLM inference on **NVIDIA GB10 (DGX Spark / Dell Pro Max with GB10)**. The steps, scripts, and metrics are built to run the same way on any GPU, so results can be compared across platforms.

> Scope: this validates that serving works and measures what one GB10 can do. It is an engineering baseline, not a datacenter performance claim. Port the pipeline to target hardware and benchmark there.

## Pipeline

| Phase | What you do | Doc | Script |
|---|---|---|---|
| 0 | Preflight: architecture, GPU, container runtime, disk | [docs/01-install.md](docs/01-install.md) | `scripts/00_preflight.sh` |
| 1 | Install: pull the NGC vLLM container, Hugging Face access | [docs/01-install.md](docs/01-install.md) | `scripts/01_pull_container.sh` |
| 2 | Serve: start the vLLM OpenAI-compatible endpoint from a profile | [docs/02-serve.md](docs/02-serve.md) | `scripts/02_serve.sh` |
| 3 | Benchmark: concurrency sweep, TTFT / ITL / throughput | [docs/03-benchmark.md](docs/03-benchmark.md) | `scripts/03_bench.py`, `scripts/run_bench.sh` |
| 4 | Compare: merge runs across profiles or GPUs into one table | [docs/04-replicate-gpu-flavors.md](docs/04-replicate-gpu-flavors.md) | `scripts/04_compare.py` |

Quick run, once phases 0 and 1 are done:

```bash
./scripts/00_preflight.sh --all
./scripts/02_serve.sh configs/profiles/gb10-128g.env
./scripts/run_bench.sh configs/profiles/gb10-128g.env
./scripts/02_serve.sh --stop
```

Each step was first run by hand and then through its script. See the docs for the manual commands.

## Layout

```
configs/profiles/     one env file per hardware and model combination
configs/vllm-image.env   NGC container tag
docs/                 step-by-step guides: install, serve, benchmark, replicate
scripts/              preflight, container pull, serve, benchmark, compare
tests/                self-test for the benchmark client (no GPU needed)
results/              one folder per run (not committed)
logs/                 server logs (not committed)
```

## Metrics

| Metric | Meaning |
|---|---|
| TTFT p50 / p95 | Time to first token, in seconds. What a user waits for before the reply starts. |
| ITL p50 / p95 | Inter-token latency, in milliseconds. How smoothly the reply streams. |
| E2E p50 / p95 | Full request time, in seconds. |
| Per-user tok/s | Output tokens per second for one request, median. |
| Aggregate output tok/s | Total output tokens divided by wall-clock time at that concurrency. This is the scaling curve. |
| Req/s | Completed requests per second at that concurrency. |
| Failed | Requests that errored or returned no tokens. Must be 0 for a valid level. |

## Baseline results: GB10, single node

- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`, BF16
- **Engine:** vLLM 0.29.0, NGC container `nvcr.io/nvidia/vllm:26.09-py3` (arm64)
- **Settings:** `--gpu-memory-utilization 0.70`, `--max-model-len 8192`, `--max-num-seqs 128`
- **Benchmark:** 256 output tokens, temperature 0, 8 fixed prompts, 2 warmup requests, one sweep
- **Host:** driver 580.178.04, 121 GiB unified memory

| Conc | Agg out tok/s | Per-user tok/s (p50) | TTFT p50 s | TTFT p95 s | ITL p50 ms | E2E p50 s | Failed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.93 | 15.93 | 0.076 | 0.132 | 62.6 | 16.07 | 0 |
| 2 | 31.58 | 16.52 | 0.184 | 0.187 | 60.0 | 15.48 | 0 |
| 4 | 61.22 | 16.45 | 0.180 | 0.182 | 60.3 | 15.55 | 0 |
| 8 | 121.52 | 16.35 | 0.184 | 0.189 | 60.7 | 15.61 | 0 |
| 16 | 240.85 | 16.21 | 0.190 | 0.217 | 61.2 | 15.78 | 0 |
| 32 | 429.24 | 14.36 | 0.210 | 0.229 | 69.1 | 17.81 | 0 |
| 64 | 836.86 | 14.04 | 0.233 | 0.269 | 70.5 | 18.20 | 0 |
| 128 | 1407.38 | 11.71 | 0.293 | 0.373 | 84.3 | 21.80 | 0 |

**What the curve shows**

- Aggregate throughput scales almost linearly up to 16 users, then keeps growing more slowly.
- Per-user speed stays near 16 tok/s up to 16 users, drops at 32, and falls to about 11.7 tok/s at 128.
- The knee is between 64 and 128 users. Going from 64 to 128 users gives 1.68× the tokens.
- The KV cache holds 561,776 tokens. That is not the limit for this workload, which uses about 300 tokens per request. At full 8,192-token context the server fits about 68 concurrent requests.

**Limits of this baseline**

- One 7B model, with 256-token outputs and short prompts. Longer prompts reduce how many requests fit.
- One node. These numbers are not a datacenter comparison and should not be extrapolated to other GPUs.
- The run-to-run check passed: two sweeps agree within 2.3% on aggregate tok/s at every level.

## Status

- [x] Benchmark harness verified against a mock server (`tests/run_mock_test.sh`, 12 of 12 checks)
- [x] Preflight passes on GB10
- [x] Container pulled and vLLM version recorded
- [x] Baseline profile served by hand, then through `02_serve.sh`
- [x] Concurrency sweep completed, run twice, results recorded above
- [x] Stop and guard behavior verified in `02_serve.sh`
- [ ] Second profile (different model or quantization)
- [ ] Cross-GPU comparison table

## License

MIT
