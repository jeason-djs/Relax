#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Task 22 four-GPU matched probe:
#   2 actor GPUs + 2 single-GPU rollout engines.
#
# Admission examples:
#   PARTITION_ADMISSION_MODE=off bash <this-script>
#   PARTITION_ADMISSION_MODE=shadow PARTITION_ADMISSION_MIN=4 \
#     PARTITION_ADMISSION_MAX=8 PARTITION_ADMISSION_SLACK=2 bash <this-script>

set -euo pipefail
set -x

now="$(date '+%Y-%m-%d-%H:%M:%S')"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_REQUEST="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
if [[ "$PYTHON_REQUEST" != /* || ! -x "$PYTHON_REQUEST" ]]; then
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 2
fi
PYTHON_BIN="$PYTHON_REQUEST"
export TASK22_PYTHON="$PYTHON_REQUEST"

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NUM_GPUS="${NUM_GPUS:-4}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/task22/p1-admission}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-15}"
PARTITION_ADMISSION_MODE="${PARTITION_ADMISSION_MODE:-off}"
REQUEST_PLACEMENT_MODE="${REQUEST_PLACEMENT_MODE:-off}"
REQUEST_PLACEMENT_POLICY="${REQUEST_PLACEMENT_POLICY:-least_predicted_work}"
REQUEST_OBSERVABILITY_DIR="${REQUEST_OBSERVABILITY_DIR:-}"
TIMELINE_DUMP_DIR="${TIMELINE_DUMP_DIR:-/tmp/timeline}"
DRIVER_LOG_PATH="${DRIVER_LOG_PATH:-log/qwen3-4b-task22-p1-${PARTITION_ADMISSION_MODE}-${now}.log}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
MAX_STALENESS="${MAX_STALENESS:-2}"

if [[ "$PARTITION_ADMISSION_MODE" != "off" ]]; then
    : "${PARTITION_ADMISSION_MIN:?Set PARTITION_ADMISSION_MIN for shadow/on}"
    : "${PARTITION_ADMISSION_MAX:?Set PARTITION_ADMISSION_MAX for shadow/on}"
    : "${PARTITION_ADMISSION_SLACK:?Set PARTITION_ADMISSION_SLACK for shadow/on}"
fi
if [[ "$REQUEST_PLACEMENT_MODE" != "off" && "$REQUEST_PLACEMENT_MODE" != "shadow" && "$REQUEST_PLACEMENT_MODE" != "on" ]]; then
    echo "REQUEST_PLACEMENT_MODE must be off, shadow, or on" >&2
    exit 2
fi
if [[ "$REQUEST_PLACEMENT_POLICY" != "round_robin" \
    && "$REQUEST_PLACEMENT_POLICY" != "least_active_requests" \
    && "$REQUEST_PLACEMENT_POLICY" != "least_predicted_work" ]]; then
    echo "Unsupported REQUEST_PLACEMENT_POLICY=$REQUEST_PLACEMENT_POLICY" >&2
    exit 2
fi
if [[ "$REQUEST_PLACEMENT_MODE" != "off" && "${USE_SLIME_ROUTER:-0}" != "1" ]]; then
    echo "Request placement shadow/on requires USE_SLIME_ROUTER=1" >&2
    exit 2
fi
if [[ "$REQUEST_PLACEMENT_MODE" != "off" && -z "$REQUEST_OBSERVABILITY_DIR" ]]; then
    echo "Request placement shadow/on requires REQUEST_OBSERVABILITY_DIR" >&2
    exit 2
fi

export RELAX_REQUEST_PLACEMENT_MODE="$REQUEST_PLACEMENT_MODE"
export RELAX_REQUEST_PLACEMENT_POLICY="$REQUEST_PLACEMENT_POLICY"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
    --ref-load "${MODEL_DIR}/Qwen3-4B/"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size 8
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --rollout-seed "$ROLLOUT_SEED"
    --global-batch-size 64
    --use-fault-tolerance
    --balance-data
    --partition-critical-admission-mode "$PARTITION_ADMISSION_MODE"
)

if [[ "$PARTITION_ADMISSION_MODE" != "off" ]]; then
    ROLLOUT_ARGS+=(
        --partition-critical-admission-min-inflight-groups "$PARTITION_ADMISSION_MIN"
        --partition-critical-admission-max-inflight-groups "$PARTITION_ADMISSION_MAX"
        --partition-critical-admission-slack-groups "$PARTITION_ADMISSION_SLACK"
    )
fi

if [[ -n "$REQUEST_OBSERVABILITY_DIR" ]]; then
    mkdir -p "$REQUEST_OBSERVABILITY_DIR"
    bash "$REPO/scripts/task22/prepare_rollout_observability.sh"
    export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"
    export SGLANG_LOG_SCHEDULER_STATUS_INTERVAL="${SGLANG_LOG_SCHEDULER_STATUS_INTERVAL:-1.0}"
    export RELAX_RID_ONLY_REQUEST_LOGGING=1
    export RAY_DEDUP_LOGS=0
    RUNTIME_ENV_JSON="$(
        RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON" "$PYTHON_BIN" -c '
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "SGLANG_LOG_SCHEDULER_STATUS_TARGET",
    "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL",
    "RELAX_RID_ONLY_REQUEST_LOGGING",
    "RELAX_REQUEST_PLACEMENT_MODE",
    "RELAX_REQUEST_PLACEMENT_POLICY",
    "RAY_DEDUP_LOGS",
    "TASK22_PYTHON",
    "TASK22_INPUT_MANIFEST",
    "TASK22_INPUT_ROOTS_JSON",
):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime_env))
'
    )"
    export RUNTIME_ENV_JSON
    ROLLOUT_ARGS+=(--rollout-request-observability-dir "$REQUEST_OBSERVABILITY_DIR")
fi

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
    --advantage-estimator grpo
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
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-show-time-cost
)

if [[ "${USE_SLIME_ROUTER:-0}" == "1" ]]; then
    SGLANG_ARGS+=(--use-slime-router)
fi

if [[ -n "$REQUEST_OBSERVABILITY_DIR" ]]; then
    SGLANG_ARGS+=(
        --sglang-log-requests
        --sglang-log-requests-level 0
        --sglang-log-requests-format json
        --sglang-log-requests-target stdout
    )
fi

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --timeline-dump-dir "$TIMELINE_DUMP_DIR"
    --tb-project-name "$PROJECT_NAME"
    --tb-experiment-name "qwen3-4b-task22-p1-${PARTITION_ADMISSION_MODE}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed "$TRAIN_SEED"
)

mkdir -p "$(dirname -- "$DRIVER_LOG_PATH")" "$TIMELINE_DUMP_DIR"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- env RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON" \
    TASK22_RUNTIME_ATTESTATION_DIR="${TASK22_RUNTIME_ATTESTATION_DIR:-}" \
    TASK22_INPUT_MANIFEST="$TASK22_INPUT_MANIFEST" \
    TASK22_INPUT_ROOTS_JSON="$TASK22_INPUT_ROOTS_JSON" \
    "$PYTHON_BIN" -m relax.entrypoints.train \
    --resource '{"actor": [1, 2], "rollout": [1, 2]}' \
    --max-staleness "$MAX_STALENESS" \
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
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "$DRIVER_LOG_PATH"
