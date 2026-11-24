# GB10 LLM serve benchmark

Benchmarks OpenAI-compatible inference servers on one NVIDIA GB10 (Dell Pro Max GB10 / DGX Spark: aarch64, sm_121, 121 GiB unified memory). It serves a model from an NGC container, runs a streaming concurrency sweep, and compares the results across engines.

- Engines: vLLM 26.09 (`nvcr.io/nvidia/vllm:26.09-py3`) and SGLang 26.08 (`nvcr.io/nvidia/sglang:26.08-py3`)
- Model: `mistralai/Mistral-7B-Instruct-v0.3`, BF16
- Workload: 256 output tokens per request, 2 warmup requests, concurrency 1 to 128, failed requests counted

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 00 | `scripts/00_preflight.sh` | Checks arch, GPU, driver, memory, disk, Docker, NVIDIA runtime |
| 01 | `scripts/01_pull_container.sh` | Pulls the vLLM image and records its version |
| 02 | `scripts/02_serve.sh` | Starts or stops the vLLM server for a profile |
| SGLang | `scripts/serve_sglang.sh` | Starts or stops the SGLang server for a profile |
| run | `scripts/run_bench.sh` | Checks the served model, then runs the benchmark |
| 03 | `scripts/03_bench.py` | Streaming concurrency sweep (standard library only) |
| 04 | `scripts/04_compare.py` | Merges run summaries and warns when inputs differ |

## Results

Aggregate output tokens per second (higher is better):

| Concurrency | vLLM 26.09 | SGLang run 1 | SGLang run 2 |
|---:|---:|---:|---:|
| 1 | 15.9 | 15.8 | 15.8 |
| 2 | 31.6 | 31.0 | 30.9 |
| 4 | 61.2 | 59.5 | 59.4 |
| 8 | 121.5 | 117.3 | 117.3 |
| 16 | 240.8 | 230.7 | 230.9 |
| 32 | 429.2 | 400.6 | 401.3 |
| 64 | 836.9 | 779.3 | 782.4 |
| 128 | 1407.4 | 1333.7 | 1309.6 |

Time to first token, p95 in seconds (lower is better):

| Concurrency | vLLM 26.09 | SGLang run 1 | SGLang run 2 |
|---:|---:|---:|---:|
| 1 | 0.132 | 0.129 | 0.128 |
| 2 | 0.187 | 0.189 | 0.189 |
| 4 | 0.182 | 0.189 | 0.188 |
| 8 | 0.189 | 0.188 | 0.188 |
| 16 | 0.216 | 0.190 | 0.192 |
| 32 | 0.229 | 0.207 | 1.581 |
| 64 | 0.269 | 0.218 | 0.218 |
| 128 | 0.372 | 0.251 | 0.283 |

No run had a failed request. The full comparison is in `docs/comparison-gb10.md`.

## What the numbers show

- **Throughput:** SGLang delivers 93–99% of vLLM's output tokens. The gap is about 1% at one user and grows to 5–7% from 32 users up.
- **Per-user speed:** at low load both engines decode at about 16 tok/s, which is the memory-bandwidth limit for a 7B BF16 model on this hardware. Under load, SGLang's median per-user rate is lower (10.8–11.1 tok/s vs 11.7 at 128 users).
- **First token at one user:** SGLang's median is about 0.13 s, against 0.08 s for vLLM. Both SGLang runs agree, so the gap is reproducible. The cause is untested. Candidates include the prefill attention backend, CUDA graph settings, and the 1 GiB multimodal transport reservation.
- **Tail latency under load:** from 16 users up, SGLang's p95 first-token time is lower (0.218 s vs 0.269 s at 64 users, 0.25–0.28 s vs 0.37 s at 128).
- **Outlier:** SGLang run 2 at 32 users has a normal median first token (0.205 s) but a p95 of 1.58 s, so a few requests waited about 1.5 s. Run 1 and the other levels don't show this. It is unexplained.
- **Repeatability:** the two SGLang runs agree within 2% on aggregate throughput at every level. Earlier vLLM runs agreed within 2.3%.

## Limits

- One GB10, one model (7B BF16), one prompt set, 256 output tokens.
- vLLM has several runs; SGLang has two full sweeps.
- No shared-prefix workload, so RadixAttention (SGLang's prefix cache) is not measured in a way that shows its benefit.
- Engine flags are not identical: both use a 0.70 memory fraction, but SGLang also reserves up to 1 GiB for multimodal transport.
- Output quality is not checked.
- These results compare two engines on this hardware with these settings. They are not a general ranking of the engines.

## Reproduce

```bash
# Host checks and image
./scripts/00_preflight.sh
./scripts/01_pull_container.sh
docker pull nvcr.io/nvidia/sglang:26.08-py3

# vLLM (see the header of scripts/02_serve.sh for arguments)
./scripts/02_serve.sh   # start the vLLM profile, then:
python3 scripts/03_bench.py --base-url http://localhost:8000 \
  --model mistralai/Mistral-7B-Instruct-v0.3 --profile gb10-128g \
  --levels 1,2,4,8,16,32,64,128 --max-tokens 256 --out results

# SGLang
scripts/serve_sglang.sh start configs/profiles/gb10-128g-sglang.env
python3 scripts/03_bench.py --base-url http://localhost:30000 \
  --model mistralai/Mistral-7B-Instruct-v0.3 --profile gb10-128g-sglang \
  --levels 1,2,4,8,16,32,64,128 --max-tokens 256 --out results
scripts/serve_sglang.sh stop configs/profiles/gb10-128g-sglang.env

# Compare (paths are the run folders under results/)
python3 scripts/04_compare.py results/manual-mistral-final/20261123-215338 \
  results/sglang-manual/20261124-013050 results/sglang-manual/20261124-014440 \
  --out docs/comparison-gb10.md
```

## Status

- [x] Preflight, image pull, and vLLM serve verified on GB10
- [x] Benchmark client verified against a mock server and on hardware
- [x] vLLM full sweep (1–128) with zero failures
- [x] SGLang image, flags, and full sweep (1–128) with zero failures, two runs
- [x] SGLang start/stop script verified (start, readiness, guard, stop)
- [x] Cross-engine comparison
- [ ] Third SGLang sweep to check the concurrency 32 p95 outlier
- [ ] Shared-prefix workload
- [ ] Record engine and server flags in `env.json`
