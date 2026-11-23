# 04 — Replicate on other GPU flavors

The pattern is the same everywhere. Only the profile file and the container tag change. This section tells you what to change and what must stay fixed so the numbers can be compared.

## The fixed parts (never change between platforms)

- The benchmark script and its metrics definitions
- The prompt set and `max_tokens`
- The concurrency levels
- The model you compare (same model id, same quantization where possible)
- The method: warmup, sweep, two runs per profile

## The per-platform parts (change these)

| Item | GB10 (this repo) | RTX PRO 6000 | H100 / B200 (x86) |
|---|---|---|---|
| Architecture | aarch64 | x86_64 | x86_64 |
| Container | `nvcr.io/nvidia/vllm` arm64 | `nvcr.io/nvidia/vllm` amd64 | `nvcr.io/nvidia/vllm` amd64 |
| GPU memory | 128 GB unified, shared with OS | dedicated VRAM (96 GB) | dedicated HBM |
| `GPU_MEM_UTIL` starting point | 0.70 | 0.90 | 0.90 |
| Multi-GPU | 2 nodes over ConnectX-7 | NVLink / PCIe TP | NVLink TP |
| Profile file | `configs/profiles/gb10-128g.env` | `configs/profiles/rtx-pro-6000-96g.env` | `configs/profiles/h100-80g.env` |

## Steps to add a new platform

1. **Preflight.** Run `scripts/00_preflight.sh` on the new host. Fix every FAIL.
2. **Copy a profile.** `cp configs/profiles/gb10-128g.env configs/profiles/<platform>.env`, then change `PROFILE_NAME`, `GPU_MEM_UTIL`, and `EXTRA_ARGS`. Leave `MODEL`, `MAX_MODEL_LEN`, and `MAX_NUM_SEQS` alone for the first run so the comparison is fair.
3. **Pull the matching image.** Confirm the arch matches (`uname -m`) before pulling.
4. **Serve and bench** with the same commands from [02-serve.md](02-serve.md) and [03-benchmark.md](03-benchmark.md).
5. **Compare.** Run `scripts/04_compare.py` over the result folders from all platforms. It writes one table keyed by concurrency.

## Multi-node (two GB10s over ConnectX-7)

Add these to `EXTRA_ARGS` and environment for the multi-node profile:

- `NCCL_NET_GDR_LEVEL=0`. On GB10 the unified-memory SoC hard-locks on model load without it. The launcher does not set it, so pass it explicitly.
- Ray head and worker, with the worker started after the head is reachable. Start vLLM only after both nodes have joined. Otherwise vLLM can fall back to single-GPU mode without warning.
- Tensor parallel size 2 (`--tensor-parallel-size 2`).

Keep a separate profile, for example `gb10-2node-tp2.env`, so it is never mixed with single-node results.

## What not to conclude

A GB10 number is not a datacenter number. Use GB10 results to check that the pipeline works, that the flags behave as documented, and to find the failure modes early. Do not extrapolate GB10 throughput to RTX PRO 6000 or H100. Run the same harness there and report those numbers separately.

## Adding a new model

Copy a profile, change `MODEL` and `PROFILE_NAME`, keep the hardware the same, and run the sweep again. Compare models on the same platform first, and platforms on the same model second. Do not mix both in one comparison.
