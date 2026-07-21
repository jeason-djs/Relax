# Featurize Runbook with the Official Relax Image

This note records the working path for Featurize instances where `docker run --gpus all` fails with:

```text
could not select device driver "" with capabilities: [[gpu]]
```

On the tested Featurize instance, Docker listed an `nvidia` runtime but the host was missing `nvidia-container-runtime`. The workaround is to mount the NVIDIA device files and driver libraries manually.

## 0. Resource Sizing

The beginner task uses Qwen3-0.6B, but it is not a plain single-model inference job. Relax runs an online RL stack with the Megatron actor, rollout service, SGLang engine, Ray Serve, transfer queues, datasource actors, reward calculation, logprob calculation, and metrics services.

Observed on a small Featurize instance:

| Resource | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| Container-visible memory | 27.40 GB |
| Failure point | rollout engine initialization, before completing training steps |
| Peak memory at failure | 26.86 GB / 27.40 GB |
| Largest process | `ray::MegatronTrainRayActor`, about 11.31 GB RSS |

This means 12 GB GPU plus about 28 GB host memory is not enough for a reliable run, even with optimizer CPU offload and reduced rollout settings. Use the following as a practical sizing guide:

| Tier | GPU | CPU | Host memory | Notes |
| --- | --- | --- | --- | --- |
| Minimum to try | 24 GB single GPU, such as RTX 4090, L20, or A10 | 8+ cores | 48 GB+ | Use the conservative environment overrides below. |
| Recommended | 48 GB+ single GPU, such as L40S, A40, or A6000 | 16+ cores | 64 GB+ | Better for finishing 10 steps without repeated memory tuning. |
| Most stable | H20 80 GB or A100 80 GB | 16-32 cores | 80 GB+ | Best choice if the goal is to finish quickly and collect clean curves. |

The rough host-memory budget is:

```text
Megatron actor process             10-14 GB
SGLang scheduler and detokenizer    2-4 GB
Ray Serve / rollout replicas        3-6 GB
Transfer queue and datasource       1-2 GB
Ray head, dashboard, object store   4-8 GB
Python import and framework overhead 2-4 GB
Peak temporary objects and margin   8-16 GB
```

So a stable lower bound is around 48 GB host memory, with 64 GB preferred.

Before renting the instance, prefer:

```text
GPU memory >= 24 GB
Host memory >= 48 GB
CPU cores >= 8
```

For a no-drama run, prefer:

```text
GPU memory >= 48 GB
Host memory >= 64 GB
CPU cores >= 16
```

## 1. Start the Official Image

First check whether the standard NVIDIA Docker path works:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this command succeeds, start the official Relax image with the simpler command:

```bash
cd /home/featurize

docker pull ghcr.io/redai-infra/relaxrl:latest

docker run -it --name relax-beginner-run --rm \
  --gpus all \
  --privileged \
  --ipc=host \
  -v /home/featurize/Relax:/root/Relax \
  -v /home/featurize/model:/root/model \
  -v /home/featurize/data:/root/data \
  ghcr.io/redai-infra/relaxrl:latest \
  /bin/bash
```

Do not add `--network=host` unless there is a specific reason. In one failed run, host networking made the container connect to stale Ray state on the host and triggered a Ray session mismatch.

If `docker run --gpus all` fails with `could not select device driver "" with capabilities: [[gpu]]`, use the manual device and library mounts instead:

```bash
cd /home/featurize

docker pull ghcr.io/redai-infra/relaxrl:latest

docker run -it --name relax-beginner-run --rm \
  --privileged \
  --ipc=host \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-modeset \
  -v /dev/nvidia-caps:/dev/nvidia-caps \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/nvidia/lib64/libcuda.so.1:ro \
  -v /lib/x86_64-linux-gnu/libcuda.so:/usr/local/nvidia/lib64/libcuda.so:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ml.so:/usr/local/nvidia/lib64/libnvidia-ml.so:ro \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
  -v /home/featurize/Relax:/root/Relax \
  -v /home/featurize/model:/root/model \
  -v /home/featurize/data:/root/data \
  ghcr.io/redai-infra/relaxrl:latest \
  /bin/bash
```

Inside the container:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
git config --global --add safe.directory /root/Relax
```

Expected:

```text
True 1
```

## 2. Convert Data

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

gsm = Path("/root/data/gsm8k/main/train-00000-of-00001.parquet")
df = pd.read_parquet(gsm)
out = Path("/root/data/gsm8k/train.jsonl")
with out.open("w") as f:
    for _, row in df.iterrows():
        answer = str(row["answer"]).split("####")[-1].strip()
        f.write(json.dumps({"question": row["question"], "answer": answer}, ensure_ascii=False) + "\n")
print(out, sum(1 for _ in out.open()))

aime = Path("/root/data/aime-2024/data/train-00000-of-00001.parquet")
df = pd.read_parquet(aime)
out = Path("/root/data/aime-2024/aime-2024.jsonl")
with out.open("w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({"prompt": row["problem"], "label": str(row["answer"])}, ensure_ascii=False) + "\n")
print(out, sum(1 for _ in out.open()))
PY
```

Expected counts on the tested download:

```text
/root/data/gsm8k/train.jsonl 7473
/root/data/aime-2024/aime-2024.jsonl 90
```

## 3. Run a 10-Step Smoke Test

On resource-constrained machines, use conservative settings:

```bash
cd /root/Relax

export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export NUM_CPUS=16
export MODEL_DIR=/root/model
export DATA_DIR=/root/data

export NUM_ROLLOUT=5
export ROLLOUT_BATCH_SIZE=1
export N_SAMPLES=4
export GLOBAL_BATCH_SIZE=2
export ROLLOUT_MAX_RESPONSE_LEN=512
export EVAL_MAX_RESPONSE_LEN=512
export MAX_TOKENS_PER_GPU=2048
export LOG_PROBS_MAX_TOKENS_PER_GPU=2048
export SGLANG_MEM_FRACTION_STATIC=0.25
export OPTIMIZER_CPU_OFFLOAD=1
export USE_CLEARML=0
export USE_METRICS_SERVICE=0
export SGLANG_EXTRA_ARGS="--sglang-disable-cuda-graph --sglang-max-running-requests 4 --sglang-disable-radix-cache"

bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

The step count is:

```text
NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES / GLOBAL_BATCH_SIZE
= 5 * 1 * 4 / 2
= 10
```

If Ray Serve reports a replica stuck with no CPU resources, increase `NUM_CPUS`. This is a Ray scheduling declaration; it does not create physical CPU cores, but it can unblock local smoke tests where service actors reserve more logical CPU slots than the small instance advertises by default.

On a 12 GB RTX 3060 with only about 27 GB container-visible host memory, this configuration still failed due to Ray memory pressure during rollout engine initialization:

```text
RuntimeError: [engine-init-barrier:default] engine rank=0 init failed:
worker(s) were killed due to the node running low on memory.
Memory on the node was 26.86GB / 27.40GB (98.0%).
```

Treat that failure as a resource limitation, not a model or dataset installation issue.

## 4. Troubleshooting Notes

### `ray: command not found`

This means the base host environment is missing Relax runtime dependencies. Use the official image instead of trying to patch the host conda environment package by package.

### `ModuleNotFoundError: No module named 'transfer_queue'`

The host Python environment is incomplete. The official image includes Relax's native and Python runtime dependencies.

### `docker run --gpus all` fails

If the error is:

```text
could not select device driver "" with capabilities: [[gpu]]
```

and `docker info` shows an `nvidia` runtime but `--runtime=nvidia` cannot find `nvidia-container-runtime`, use the manual device/library mount command in section 1.

### Ray session mismatch

Avoid `--network=host` for this single-node container run. It can make the container see stale host Ray/GCS state.

### Ray worker OOM at SGLang initialization

First try the conservative environment variables in section 3. If the host memory is below 48 GB, switch to a larger instance instead of continuing to reduce task parameters, because the run may fail before training starts.

## 5. Cleanup

```bash
docker rm -f relax-beginner-run 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 -f ray 2>/dev/null || true
```
