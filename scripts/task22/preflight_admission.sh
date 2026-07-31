#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_REQUEST="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
if [[ "$PYTHON_REQUEST" != /* || ! -x "$PYTHON_REQUEST" ]]; then
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 4
fi
PYTHON_BIN="$PYTHON_REQUEST"
export TASK22_PYTHON="$PYTHON_REQUEST"
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
    scripts/task22/input_guard.py \
    scripts/task22/enforce_process_deadline.py \
    scripts/task22/monitor_admission_run.py \
    scripts/task22/sample_gpu_state.py \
    scripts/task22/simulate_request_placement.py \
    scripts/task22/validate_admission_run.py \
    relax/backends/sglang/sglang_engine.py \
    relax/distributed/ray/actor_group.py \
    relax/distributed/ray/placement_group.py \
    relax/distributed/ray/rollout.py \
    relax/distributed/ray/train_actor.py \
    relax/entrypoints/train.py \
    relax/engine/router/placement.py \
    relax/engine/router/router.py \
    relax/engine/rollout/admission.py \
    relax/engine/rollout/request_observability.py \
    relax/utils/task22_runtime_attestation.py

bash -n \
    scripts/task22/preflight_admission.sh \
    scripts/task22/run_admission_matched_ab.sh \
    scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh

"$PYTHON_BIN" -m pytest -q \
    tests/engine/router/test_placement.py \
    tests/engine/router/test_router_placement.py \
    tests/engine/rollout/test_admission.py \
    tests/engine/rollout/test_sglang_rollout_cleanup.py \
    tests/engine/rollout/test_request_observability.py \
    tests/engine/rollout/test_request_placement_simulator.py \
    tests/engine/rollout/test_rollout_observability_analyzer.py \
    tests/engine/rollout/test_admission_matched_runner.py \
    tests/engine/rollout/test_admission_online_monitor.py \
    tests/engine/rollout/test_admission_preflight.py \
    tests/engine/rollout/test_admission_run_validator.py \
    tests/engine/rollout/test_admission_pair_comparator.py \
    tests/engine/rollout/test_task22_input_guard.py \
    tests/components/test_rollout_terminal_state.py \
    tests/utils/test_timeline_trace.py

"$PYTHON_BIN" - <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path("scripts/task22/approved_sync_baseline.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1 or not manifest.get("approved_revision"):
    raise SystemExit("Invalid versioned Task 22 synchronization baseline manifest")
pattern = re.compile(manifest["pattern"])
for source_file, approved_count in manifest["sources"].items():
    current_count = len(pattern.findall(pathlib.Path(source_file).read_text(encoding="utf-8")))
    if current_count > approved_count:
        print(
            "Task 22 changes increase synchronization primitives in "
            f"{source_file}: approved baseline {approved_count} -> {current_count}",
            file=sys.stderr,
        )
        raise SystemExit(4)
PY

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

for command in cmp git ray nvidia-smi timeout sha256sum; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required formal-run command is missing: $command" >&2
        exit 4
    fi
done

if [[ "${NUM_GPUS:-}" != "4" ]]; then
    echo "Formal Task 22 Ray declaration requires NUM_GPUS=4; got ${NUM_GPUS:-unset}" >&2
    exit 4
fi
if ! normalized_visible="$(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$PYTHON_BIN" -c '
import os

raw = os.environ["CUDA_VISIBLE_DEVICES"]
if not raw.strip():
    print("")
    raise SystemExit(0)
devices = [value.strip() for value in raw.split(",")]
if any(not value for value in devices) or len(devices) != 4 or len(set(devices)) != 4:
    raise SystemExit(4)
print(",".join(devices))
'
)"; then
    echo "Formal Task 22 requires CUDA_VISIBLE_DEVICES to be empty or 4 unique devices" >&2
    exit 4
fi
if [[ "$normalized_visible" != "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Formal Task 22 requires normalized CUDA_VISIBLE_DEVICES=$normalized_visible" >&2
    exit 4
fi

gpu_count="$(env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" != "4" ]]; then
    echo "Formal Task 22 runner requires exactly 4 physical GPUs, found $gpu_count" >&2
    exit 4
fi

"$PYTHON_BIN" -c 'import ray, sglang'
"$PYTHON_BIN" -m pytest -q tests/utils/test_metrics_service.py
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
echo "TASK22_PREFLIGHT ray_declared_gpus=$NUM_GPUS"
echo "TASK22_PREFLIGHT gpu_count=$gpu_count"
echo "TASK22_PREFLIGHT verdict=PASS_FORMAL"
