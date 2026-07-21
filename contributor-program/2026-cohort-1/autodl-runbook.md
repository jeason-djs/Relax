# AutoDL Runbook and Lessons

This note records the AutoDL/SeetaCloud attempt for the beginner task. The main lesson is simple: use the official Relax image whenever possible. A non-official host conda environment can be made to start, but the dependency surface is wide and brittle.

## Tested Machine

| Item | Value |
| --- | --- |
| Provider | AutoDL / SeetaCloud |
| GPU | 1 x NVIDIA GeForce RTX 4080 SUPER |
| GPU memory | 32760 MiB |
| CPU quota | 12 cores |
| Host memory quota | 62 GB |
| Work dir | `/root/autodl-tmp/relax-beginner` |

The 32 GB GPU was enough for a conservative smoke run after CPU offload and shorter rollout lengths. The successful run was:

| Item | Value |
| --- | --- |
| Ray job | `raysubmit_dFf7yDNygAFYScB3` |
| Log file | `/root/autodl-tmp/relax-beginner/autodl-run-20260722-030847-gb4-bshd-staticmbs-softmaxpatch-gpu1.out` |
| Planned train steps | `3 * 4 * 4 / 4 = 12` |
| Proof | Ray reported the job succeeded; logs reached `step 11` and `All training steps finished`. |
| Training window | Started at `2026-07-22 03:08:47`; final training log at `2026-07-22 03:17:36`. |
| Peak observed GPU memory in logs | About 8.9 GB after update weight sync in the patched host run; earlier rollout-only checks reached about 15.7 GB by `nvidia-smi`. |

This does not mean the full default task is that small, because longer responses, larger rollout batches, eval, CUDA graphs, fused kernels, and less offload can all raise the peak.

## Recommended Future Setup

Prefer an instance that can run:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Then use:

```bash
docker pull ghcr.io/redai-infra/relaxrl:latest
```

For the beginner task, choose at least:

| Tier | GPU memory | CPU | Host memory | Use case |
| --- | --- | --- | --- | --- |
| Minimum smoke | 32 GB | 12 cores | 62 GB | Works only with conservative knobs and low reward workers. |
| Recommended | 48 GB+ | 16+ cores | 64-96 GB | Cleaner run with less dependency and memory tuning. |
| Comfortable | 80 GB | 24+ cores | 96 GB+ | Best for default settings and faster iteration. |

## Conservative Smoke Command

Use this when the machine has one 32 GB GPU and about 12 CPU cores:

```bash
cd /root/autodl-tmp/relax-beginner/Relax

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/relax-beginner/Relax:/root/autodl-tmp/relax-beginner/Megatron-LM:${PYTHONPATH:-}
export RELAX=/root/autodl-tmp/relax-beginner/Relax
export MEGATRON=/root/autodl-tmp/relax-beginner/Megatron-LM
export MODEL_DIR=/root/autodl-tmp/relax-beginner/model
export DATA_DIR=/root/autodl-tmp/relax-beginner/data
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export NUM_CPUS=12

export NUM_ROLLOUT=3
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES=4
export GLOBAL_BATCH_SIZE=4
export REWARD_NUM_WORKERS=4

export ROLLOUT_MAX_RESPONSE_LEN=512
export EVAL_MAX_RESPONSE_LEN=512
export MAX_TOKENS_PER_GPU=2048
export LOG_PROBS_MAX_TOKENS_PER_GPU=2048
export SGLANG_MEM_FRACTION_STATIC=0.35
export OPTIMIZER_CPU_OFFLOAD=1
export USE_CLEARML=0
export USE_METRICS_SERVICE=0
export SGLANG_EXTRA_ARGS="--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph --sglang-max-running-requests 4 --sglang-disable-radix-cache"

bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

If the official image is not available and the host conda environment has no Transformer Engine/Apex-compatible fused kernels, add the local Megatron fallback knobs used in the successful host smoke run:

```bash
export ATTENTION_BACKEND=local
export USE_DYNAMIC_BATCH_SIZE=0
export MICRO_BATCH_SIZE=1
export MEGATRON_EXTRA_ARGS="--qkv-format bshd --no-masked-softmax-fusion --spec local --no-rope-fusion --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-bias-swiglu-fusion"
```

In this non-official host path, Megatron still tried to probe `scaled_masked_softmax_cuda` even with `--no-masked-softmax-fusion`. The smoke run only finished after patching `/root/Megatron-LM/megatron/core/fusions/fused_softmax.py` to return `False` from `is_kernel_available()` when `scaled_masked_softmax_cuda` is missing. Treat that as a host-environment workaround, not an upstream Relax code change.

This produces:

```text
train_iters = NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES / GLOBAL_BATCH_SIZE
            = 3 * 4 * 4 / 4
            = 12 steps
```

`ROLLOUT_BATCH_SIZE=1` is invalid with the default `num_rollout_minis=2`:

```text
rollout_batch_size must be divisible by num_rollout_minis
```

Relax derives `num_rollout_minis` as:

```text
num_rollout_minis = ROLLOUT_BATCH_SIZE * N_SAMPLES / GLOBAL_BATCH_SIZE
```

It also requires `GLOBAL_BATCH_SIZE` to match one rollout mini, so with `N_SAMPLES=4`, use `GLOBAL_BATCH_SIZE=4`. `GLOBAL_BATCH_SIZE=2` is invalid for this smoke profile; it produced `num_rollout_minis=8` when `ROLLOUT_BATCH_SIZE=4`, and then failed the same divisibility check.

## Data Layout

Model:

```text
/root/autodl-tmp/relax-beginner/model/Qwen3-0.6B
```

Data:

```text
/root/autodl-tmp/relax-beginner/data/gsm8k/train.jsonl
/root/autodl-tmp/relax-beginner/data/aime-2024/aime-2024.jsonl
```

Expected counts:

```text
gsm8k/train.jsonl: 7473 lines
aime-2024/aime-2024.jsonl: 90 lines
```

## What Went Wrong on the Non-Official Host

The host image did not have Docker, so the official image could not be used. Installing dependencies into the host conda environment exposed a chain of compatibility issues:

| Symptom | Cause | Resolution |
| --- | --- | --- |
| `ray: command not found` | Base env did not include Relax runtime dependencies. | Install requirements or use official image. |
| `ModuleNotFoundError: transfer_queue` | Native queue package missing. | Install `transferqueue` from the repository commit used by Relax. |
| Megatron import failed on `transformer_engine` | Host image lacked TE. | Use local Megatron specs/fusions only as a smoke fallback. |
| `apply_rope_fusion is not available` | TE-only fused rope path. | Disable rope fusion in fallback mode. |
| Apex gradient accumulation fusion error | Apex extension missing. | Disable gradient accumulation and bias fusions in fallback mode. |
| SGLang `sgl_kernel` import failed with `libnvrtc.so.13` / `sm89` | Installed wheel did not match the CUDA/GPU stack. | Official image or matching `sglang-kernel` wheel is the real fix. Temporary torch fallbacks were only for smoke testing. |
| Piecewise CUDA graph compile failed | TorchDynamo could not compile the fallback path. | Add `--sglang-disable-piecewise-cuda-graph`. |
| Reward workers stuck in `PENDING_CREATION` | 16 reward workers over-reserved CPU on a 12-core Ray cluster. | Set `REWARD_NUM_WORKERS=4`. |
| Actor training failed on `rollout_batch_size must be divisible by num_rollout_minis` | The initial memory-saving batch sizes used `GLOBAL_BATCH_SIZE < N_SAMPLES`, which makes the derived rollout mini plan impossible. | Use `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES=4`, `GLOBAL_BATCH_SIZE=4`, and reduce `NUM_ROLLOUT` instead. |
| `Packed sequence is not supported by DotProductAttention` | Local Megatron attention fallback does not support packed `thd` sequences. | Use `--qkv-format bshd`, then disable dynamic batch size and set `--micro-batch-size 1`. |
| `ModuleNotFoundError: scaled_masked_softmax_cuda` | Host Megatron lacked the fused softmax CUDA extension but still probed it before falling back. | Official image is preferred. For smoke only, patch the Megatron softmax availability probe to return `False` when the extension is missing. |

## Practical Rules

1. Start from the official image. It avoids most of the Python, CUDA, SGLang, Megatron, Apex, and Transformer Engine mismatch work.
2. Match Ray resources to the rented machine. On a 12-core machine, export `NUM_CPUS=12` and reduce `REWARD_NUM_WORKERS`.
3. Keep `GLOBAL_BATCH_SIZE >= N_SAMPLES` for the fixed-n-samples rollout mini plan. In this smoke profile, use `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES=4`, and `GLOBAL_BATCH_SIZE=4`.
4. On 32 GB GPU smoke runs, disable CUDA graph paths and shorten generation length.
5. Treat host memory as a first-class resource. Even Qwen3-0.6B post-training launches actor training, rollout inference, Ray Serve, transfer queue, reward workers, and data actors together.
6. Do not mistake a host dependency workaround for the normal contribution path. Submit the script knobs and runbook, but keep local Megatron/SGLang patching out of the Relax code changes unless the task explicitly asks for it.
