# Beginner Task: Qwen3-0.6B Single-GPU GRPO on GSM8K

Full task description: https://github.com/redai-infra/Relax/issues/76

## Goal

Run a complete Relax reinforcement-learning post-training flow on one GPU and finish at least 10 training steps.

| Item | Configuration |
| --- | --- |
| Model | Qwen3-0.6B |
| Training dataset | GSM8K |
| Algorithm | GRPO |
| Hardware | 1 GPU |
| Expected work | About 0.5 day; no code change required |

## Activity Notes

The beginner task is required before joining the formal task phase. The final submission should be a ZIP named with school, major, and name, and submitted to the WPS form linked in the issue.

Required materials:

1. Complete training log.
2. PDF experiment report with reward/loss curves and troubleshooting notes.

## Prepare Model and Data

```bash
export HF_ENDPOINT=https://hf-mirror.com  # optional in China
export MODEL_DIR=/root/model
export DATA_DIR=/root/data

可能需要设置这个环境变量
export HF_HUB_DISABLE_XET=1

检查是否生效
echo $HF_HUB_DISABLE_XET

hf download Qwen/Qwen3-0.6B --local-dir $MODEL_DIR/Qwen3-0.6B

hf download openai/gsm8k main \
  --repo-type dataset \
  --local-dir $DATA_DIR/gsm8k

hf download AI-MO/aimo-validation-aime \
  --repo-type dataset \
  --local-dir $DATA_DIR/aime-2024
```

GSM8K is downloaded as parquet and must be converted to JSONL:

```python
import json
import pandas as pd

df = pd.read_parquet("/root/data/gsm8k/main/train-00000-of-00001.parquet")
with open("/root/data/gsm8k/train.jsonl", "w") as f:
    for _, row in df.iterrows():
        answer = row["answer"].split("####")[-1].strip()
        f.write(json.dumps({"question": row["question"], "answer": answer}) + "\n")
```

AIME is also downloaded as parquet and must be converted to the field names expected by the eval arguments:

```python
import json
import pandas as pd

df = pd.read_parquet("/root/data/aime-2024/data/train-00000-of-00001.parquet")
with open("/root/data/aime-2024/aime-2024.jsonl", "w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({"prompt": row["problem"], "label": str(row["answer"])}) + "\n")
```

The training script expects:

```text
$MODEL_DIR/Qwen3-0.6B
$DATA_DIR/gsm8k/train.jsonl
$DATA_DIR/aime-2024/aime-2024.jsonl
```

## Run

```bash
cd /root/Relax
export MODEL_DIR=/root/model
export DATA_DIR=/root/data
bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

For Featurize instances where `docker run --gpus all` cannot start the official image, see `featurize-official-image.md`.

Before renting a GPU instance, also check the resource sizing notes in `featurize-official-image.md`. The full GRPO stack is heavier than loading the 0.6B model alone because it starts Megatron actor training, SGLang rollout, Ray Serve, queues, and metrics components at the same time. A 12 GB GPU with about 28 GB host memory was not enough in testing; prefer at least 24 GB GPU memory and 48 GB host memory, with 64 GB host memory recommended.

For a smoke run that still satisfies the 10-step requirement, override rollout count:

```bash
NUM_ROLLOUT=5 bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

With the default batch settings, training steps are:

```text
train_iters = NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES / GLOBAL_BATCH_SIZE
```

Default values produce `100 * 4 * 8 / 16 = 200` steps. `NUM_ROLLOUT=5` produces 10 steps.

## Main Flow

```text
Startup script
  -> argument parsing
  -> Controller orchestration
  -> Rollout generation with SGLang
  -> math reward calculation
  -> GRPO advantage estimation and policy update
  -> metrics logging
  -> AIME-2024 evaluation every 10 steps
```

## Metrics to Capture

Use the training log and ClearML/TensorBoard curves to capture:

| Curve | Suggested metric |
| --- | --- |
| Reward | `rollout/raw_reward` |
| Loss | `train/pg_loss` |
| Optional | `train/grad_norm`, KL, response length, `eval/aime-2024-pass@1` |
