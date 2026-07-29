#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 4xGPU hybrid-async precheck / benchmark script.
#
# 本脚本由 run-qwen3-4B-8xgpu-hybrid-async.sh 派生，用于 Task 22 的
# 4×RTX PRO 6000 (96GB) 单机环境。相对 8 卡脚本的差异（仅必要项）：
#   1. 拓扑：Actor 2 + Rollout 2（原为 Actor 4 + Rollout 4）。
#   2. NUM_ROLLOUT 默认 20，用于短预检；正式跑 200 step 请显式传 NUM_ROLLOUT=200。
#   3. 关闭 checkpoint 落盘（--save / --save-interval），交付物不需要权重，避免撑爆磁盘。
#   4. 预检阶段关闭 eval（aime-2024 数据尚未就绪）；正式跑需恢复 EVAL_ARGS 以保留 pass-rate 护栏。
#   5. 不启用 CPU optimizer offload、不启用 activation recompute（96GB 显存充足，保持 Actor 计时干净）。
# 其余 batch / 序列长度 / 采样 / 算法参数与 8 卡脚本一致。
#
# Usage:
#   NUM_ROLLOUT=20 \
#   MODEL_DIR=/root/autodl-fs/exps \
#   DATA_DIR=/root/autodl-fs/exps \
#   bash scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

# 4 卡拓扑：确保 Ray head 只申请 4 张 GPU。
export NUM_GPUS="${NUM_GPUS:=4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

TASK22_TIMING_PREFLIGHT="${SCRIPT_DIR}/../../task22/prepare_sglang_timing_transport.sh"
bash "$TASK22_TIMING_PREFLIGHT"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=20}"



CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   # 预检不保存 checkpoint（交付物不需要权重，磁盘受限）。正式跑如需续训再放开：
   # --load ${EXP_DIR}/Qwen3-4B_mcore_4xgpu/
   # --save ${EXP_DIR}/Qwen3-4B_mcore_4xgpu/
   # --save-interval 100
   )

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 64
   --use-fault-tolerance
   --balance-data

)

EVAL_ARGS=(
   --skip-eval-before-train
   # 预检阶段关闭 eval：aime-2024 数据尚未上传。正式跑请恢复以下参数以保留 pass-rate 护栏：
   # --log-passrate
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 8
   # --eval-max-response-len 16384
   # --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # 96GB 显存充足，预检与正式 baseline 均不启用 recompute：
   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1

   # --micro-batch-size 1 # avoid OOM
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   # 96GB 显存充足，不启用 CPU optimizer offload：
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-show-time-cost
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --timeline-dump-dir /tmp/timeline
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-GRPO-gpu4-hybrid-async-${now}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

   # --num-iters-per-train-update 4 \
   #  --use-health-check \
mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2]}'\
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-GRPO-gpu4-hybrid-async-${now}.log
