# 02 — Serve

Goal: a running OpenAI-compatible vLLM endpoint on `localhost:8000`, driven by a profile file so the same command works on any GPU.

## 1. Pick a profile

A profile is a small env file that describes one hardware + model combination:

```bash
cat configs/profiles/gb10-128g.env
```

Key settings:

| Variable | Meaning |
|---|---|
| `PROFILE_NAME` | Folder name used under `results/`. Keep it unique per hardware + model. |
| `MODEL` | Hugging Face model id. |
| `GPU_MEM_UTIL` | Fraction of GPU memory vLLM may use. On GB10 this is a share of the 128 GB unified pool, so leave headroom for the OS. |
| `MAX_MODEL_LEN` | Max context length. Keep it modest for the first run. |
| `MAX_NUM_SEQS` | Max concurrent sequences the scheduler admits. |
| `EXTRA_ARGS` | Any extra `vllm serve` flags. |

## 2. Start the server

```bash
./scripts/02_serve.sh configs/profiles/gb10-128g.env
```

The script:
1. Starts the container named `vllm-<PROFILE_NAME>` with `--gpus all`, `--ipc=host`, the Hugging Face cache mounted, and port 8000 published.
2. Waits for `/health` to return 200. First load can take several minutes because weights are downloaded and loaded.
3. Writes the container log to `logs/<PROFILE_NAME>.log`.

## 3. Smoke test

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL>","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}' \
  | python3 -m json.tool
```

If you get a completion back, the serving path works end to end.

## 4. Watch memory while it runs

In a second terminal:

```bash
watch -n 2 nvidia-smi
free -h
```

Note the steady-state memory with the model loaded and idle. This goes into your results as the baseline footprint.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits immediately | Bad flag or missing model | `docker logs vllm-<PROFILE_NAME>` |
| OOM at load | `GPU_MEM_UTIL` too high or `MAX_MODEL_LEN` too big | Lower `MAX_MODEL_LEN` first, then `GPU_MEM_UTIL` |
| `401` or gated-repo error | Token missing or license not accepted | Set `HF_TOKEN`, accept the model license on Hugging Face |
| Hangs at "loading weights" | Slow first download | Wait; check `du -sh ~/.cache/huggingface` grows |
| `/health` never 200 | Startup still compiling kernels | Check the log; CUDA graph capture is normal and slow on first start |
| Works on your laptop, not here | Image is x86 | Confirm `uname -m` is aarch64 and the pulled image is the arm64 variant |

## 6. Stop

```bash
./scripts/02_serve.sh --stop
```

Stop the server before running a different profile, so the memory numbers are not contaminated.
