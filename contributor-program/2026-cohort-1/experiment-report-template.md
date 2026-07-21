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
