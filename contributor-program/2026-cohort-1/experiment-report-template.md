# Relax Beginner Task Experiment Report

## 1. Basic Information

| Item | Value |
| --- | --- |
| School / Major / Name | TODO |
| Completion date | TODO |
| Code branch / commit id | `feat/beginner-task` / TODO |
| GPU model and count | TODO |
| Python / CUDA / image version | TODO |

## 2. Task Details

| Item | Value |
| --- | --- |
| Model | Qwen3-0.6B |
| Dataset | GSM8K |
| Algorithm | GRPO |
| Actual training steps | TODO |
| Total time | TODO |
| Log / output directory | TODO |
| Main parameter changes | TODO |

AutoDL smoke-run reference, if reusing the verified 32 GB setup:

```text
GPU: 1 x NVIDIA GeForce RTX 4080 SUPER, 32760 MiB
CPU / memory: 12 cores / 62 GB
Ray job: raysubmit_dFf7yDNygAFYScB3
Log: /root/autodl-tmp/relax-beginner/autodl-run-20260722-030847-gb4-bshd-staticmbs-softmaxpatch-gpu1.out
Result: succeeded; internal train step reached 11; "All training steps finished"
```

Full startup command:

```bash
export MODEL_DIR=/root/model
export DATA_DIR=/root/data
bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

Data conversion commands used:

```bash
python3 - <<'PY'
import json
import pandas as pd

df = pd.read_parquet("/root/data/gsm8k/main/train-00000-of-00001.parquet")
with open("/root/data/gsm8k/train.jsonl", "w") as f:
    for _, row in df.iterrows():
        answer = row["answer"].split("####")[-1].strip()
        f.write(json.dumps({"question": row["question"], "answer": answer}) + "\n")

df = pd.read_parquet("/root/data/aime-2024/data/train-00000-of-00001.parquet")
with open("/root/data/aime-2024/aime-2024.jsonl", "w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({"prompt": row["problem"], "label": str(row["answer"])}) + "\n")
PY
```

For a 10-step smoke run:

```bash
export MODEL_DIR=/root/model
export DATA_DIR=/root/data
NUM_ROLLOUT=5 bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

For a resource-constrained 32 GB single-GPU smoke run:

```bash
export MODEL_DIR=/root/model
export DATA_DIR=/root/data
NUM_ROLLOUT=3 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=4 \
GLOBAL_BATCH_SIZE=4 \
REWARD_NUM_WORKERS=4 \
ROLLOUT_MAX_RESPONSE_LEN=512 \
EVAL_MAX_RESPONSE_LEN=512 \
MAX_TOKENS_PER_GPU=2048 \
LOG_PROBS_MAX_TOKENS_PER_GPU=2048 \
SGLANG_MEM_FRACTION_STATIC=0.35 \
OPTIMIZER_CPU_OFFLOAD=1 \
USE_CLEARML=0 \
USE_METRICS_SERVICE=0 \
SGLANG_EXTRA_ARGS="--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph --sglang-max-running-requests 4 --sglang-disable-radix-cache" \
bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

When using a non-official host conda environment instead of the official image, record any local Megatron/SGLang fallback patches separately in the troubleshooting section. Do not present those host-only patches as required Relax code changes.

This profile should report 12 planned training steps:

```text
3 * 4 * 4 / 4 = 12
```

## 3. Experiment Curves

### 3.1 Reward Curve

Insert `reward_curve.png`.

Metric: `rollout/raw_reward`

Data source: TODO

### 3.2 Loss Curve

Insert `loss_curve.png`.

Metric: `train/pg_loss`

Data source: TODO

### 3.3 Optional Curves

Insert KL, grad norm, response length, or AIME pass rate curves if available.

## 4. Issues and Solutions

| Symptom / Error | Cause Analysis | Solution | Confirmation |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |

If no blocking error occurred, record one checked risk such as GPU memory pressure, incorrect model/data path, Ray startup failure, or missing reward metrics.

## 5. Summary

TODO: In 3 to 5 sentences, state whether the task was completed, the final step count, your understanding of the training flow, and what you would tune first for a longer run.
