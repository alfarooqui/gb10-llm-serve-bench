# 01: Install

Goal: a host that passes preflight, the NGC vLLM container pulled, and the host ready to serve.

## 1. Check the host

On the GB10, over SSH or the NVIDIA Sync terminal:

```bash
uname -m                     # expect: aarch64
nvidia-smi                   # expect: GB10 listed, driver 580.x
free -h                      # expect: about 121 GiB total (unified memory, shared with the CPU)
df -h ~                      # expect: plenty of room for model weights (plan 100 GB or more)
```

Run the preflight script, which checks all of this and the container runtime:

```bash
chmod +x scripts/*.sh
./scripts/00_preflight.sh
```

Every line should say `OK`. Fix any `FAIL` before going on. `WARN` is acceptable for `HF_TOKEN` when you use only open models.

## 2. Container runtime

DGX OS ships Docker with the NVIDIA container runtime. Confirm the GPU is visible from inside a container:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Expect the same GB10 table as on the host. If `--gpus all` fails, the NVIDIA container runtime is not registered with Docker. On DGX OS, check the runtime setup with `nvidia-ctk` and restart Docker.

## 3. Pull the vLLM container

Do not `pip install vllm` on the GB10. Prebuilt aarch64 wheels did not include sm_121 kernels as of late 2026, so use NVIDIA's NGC container, which includes the Blackwell patches.

```bash
./scripts/01_pull_container.sh
```

The script pulls the tag in `configs/vllm-image.env`, checks that the image is arm64, and writes the vLLM version to `results/container.txt`. Check the tag in the NGC catalog (`catalog.ngc.nvidia.com`) before the first pull, because release tags change. The default is `nvcr.io/nvidia/vllm:26.09-py3`.

If the pull fails with `unauthorized`, log in to NGC with `docker login nvcr.io` (username `$oauthtoken`, password your NGC API key).

## 4. Hugging Face access

Models are downloaded from Hugging Face. Gated models need a token that has accepted the model license.

```bash
export HF_TOKEN=hf_xxx            # read token; keep it out of git
```

The baseline profile uses `mistralai/Mistral-7B-Instruct-v0.3`, which is open and needs no license acceptance.

## 5. Benchmark client (no install)

The benchmark client runs on the host and uses only the Python standard library, so there is nothing to install. Confirm the interpreter:

```bash
python3 --version     # 3.10 or newer
```

## 6. Verify

```bash
./scripts/00_preflight.sh --all
```

`--all` also checks that the container image is present locally. When every check is OK, go to [02-serve.md](02-serve.md).

## Driver and container compatibility

The baseline ran with host driver 580.178.04, which `nvidia-smi` reports as CUDA 13.0. The NGC container reports `CUDA Forward Compatibility mode ENABLED`: it uses CUDA 13.4 user-space libraries on the 580.178.04 kernel driver.

Record the host driver and this message with any results you publish. A host with a different driver may behave differently.
