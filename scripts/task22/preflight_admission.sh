#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:-python3}"
MODE="local"

if [[ "${1:-}" == "--formal" ]]; then
    MODE="formal"
elif [[ -n "${1:-}" && "${1:-}" != "--local" ]]; then
    echo "usage: $0 [--local|--formal]" >&2
    exit 2
fi

cd "$REPO"

echo "TASK22_PREFLIGHT mode=$MODE repo=$REPO"

"$PYTHON_BIN" -m py_compile \
    scripts/task22/analyze_rollout_observability.py \
    scripts/task22/compare_admission_pair.py \
    scripts/task22/simulate_request_placement.py \
    scripts/task22/validate_admission_run.py \
    relax/engine/router/placement.py \
    relax/engine/router/router.py \
    relax/engine/rollout/admission.py \
    relax/engine/rollout/request_observability.py

bash -n \
    scripts/task22/preflight_admission.sh \
    scripts/task22/run_admission_matched_ab.sh \
    scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh

"$PYTHON_BIN" -m pytest -q \
    tests/engine/router/test_placement.py \
    tests/engine/router/test_router_placement.py \
    tests/engine/rollout/test_admission.py \
    tests/engine/rollout/test_request_observability.py \
    tests/engine/rollout/test_request_placement_simulator.py \
    tests/engine/rollout/test_rollout_observability_analyzer.py \
    tests/engine/rollout/test_admission_matched_runner.py \
    tests/engine/rollout/test_admission_run_validator.py \
    tests/engine/rollout/test_admission_pair_comparator.py

for source_file in \
    relax/backends/megatron/actor.py \
    relax/backends/megatron/weight_update/update_weight_from_tensor.py \
    relax/components/rollout.py \
    relax/engine/rollout/sglang_rollout.py \
    relax/utils/utils.py; do
    current_count="$(grep -E -c 'cuda\.synchronize|dist\.barrier|ray\.get' "$source_file" || true)"
    baseline_count="$(git show "HEAD:$source_file" | grep -E -c 'cuda\.synchronize|dist\.barrier|ray\.get' || true)"
    if (( current_count > baseline_count )); then
        echo "Task 22 changes increase synchronization primitives in $source_file: $baseline_count -> $current_count" >&2
        exit 4
    fi
done

echo "TASK22_PREFLIGHT static=PASS"

if [[ "$MODE" == "local" ]]; then
    if "$PYTHON_BIN" -c 'import sglang' >/dev/null 2>&1; then
        bash scripts/task22/prepare_rollout_observability.sh
        echo "TASK22_PREFLIGHT sglang_smoke=PASS"
    else
        echo "TASK22_PREFLIGHT sglang_smoke=SKIP reason=sglang_not_installed"
    fi
    echo "TASK22_PREFLIGHT formal_environment=SKIP"
    echo "TASK22_PREFLIGHT verdict=PASS_LOCAL"
    exit 0
fi

for command in git ray nvidia-smi timeout sha256sum; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required formal-run command is missing: $command" >&2
        exit 4
    fi
done

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" != "4" ]]; then
    echo "Formal Task 22 runner requires exactly 4 visible GPUs, found $gpu_count" >&2
    exit 4
fi

"$PYTHON_BIN" -c 'import ray, sglang'
bash scripts/task22/prepare_rollout_observability.sh

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR for formal preflight}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR for formal preflight}"
[[ -r "$MODEL_DIR/Qwen3-4B/config.json" ]] || {
    echo "Qwen3-4B checkpoint is missing under MODEL_DIR=$MODEL_DIR" >&2
    exit 4
}
[[ -r "$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl" ]] || {
    echo "Dapo prompt data is missing under DATA_DIR=$DATA_DIR" >&2
    exit 4
}

echo "TASK22_PREFLIGHT sglang_smoke=PASS"
echo "TASK22_PREFLIGHT gpu_count=$gpu_count"
echo "TASK22_PREFLIGHT verdict=PASS_FORMAL"
