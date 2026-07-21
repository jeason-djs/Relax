# Featurize Runbook with the Official Relax Image

This note records the working path for Featurize instances where `docker run --gpus all` fails with:

```text
could not select device driver "" with capabilities: [[gpu]]
```

On the tested Featurize instance, Docker listed an `nvidia` runtime but the host was missing `nvidia-container-runtime`. The workaround is to mount the NVIDIA device files and driver libraries manually.

## 1. Start the Official Image

```bash
cd /home/featurize

docker pull ghcr.io/redai-infra/relaxrl:latest

docker run -it --name relax-beginner-run --rm \
  --privileged \
  --ipc=host \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
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

For a small 12 GB GPU such as RTX 3060, use conservative settings:

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

bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

The step count is:

```text
NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES / GLOBAL_BATCH_SIZE
= 5 * 1 * 4 / 2
= 10
```

If Ray Serve reports a replica stuck with no CPU resources, increase `NUM_CPUS`. This is a Ray scheduling declaration; it does not create physical CPU cores, but it can unblock local smoke tests where service actors reserve more logical CPU slots than the small instance advertises by default.

## 4. Cleanup

```bash
docker rm -f relax-beginner-run 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 -f ray 2>/dev/null || true
```

