#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
MODE="${1:---local}"
[[ "$MODE" == "--local" || "$MODE" == "--formal" ]] || { echo "usage: $0 [--local|--formal]" >&2; exit 2; }
cd "$REPO"

BASE_SHA="0bc99af8dd39de8fd99c588a98b3f3a463bc818c"
EXPECTED_PATHS=(
    relax/backends/megatron/actor.py
    relax/engine/rollout/permit_observability.py
    relax/engine/rollout/request_permit.py
    relax/engine/rollout/sglang_rollout.py
    scripts/task22/analyze_phase_elastic_calibration.py
    scripts/task22/megatron_frozen_weight_dgrad.patch
    scripts/task22/preflight_clean_main_calibration.sh
    scripts/task22/prepare_megatron_calibration.sh
    scripts/task22/prepare_sglang_calibration.sh
    scripts/task22/run_clean_main_calibration.sh
    scripts/task22/sglang_idle_heartbeat.patch
    scripts/task22/sglang_idle_heartbeat_legacy_upgrade.patch
    scripts/task22/sglang_rollout_observability.patch
    scripts/task22/smoke_sglang_calibration.py
    tests/engine/rollout/test_permit_observability.py
    tests/engine/rollout/test_request_permit.py
    tests/engine/rollout/test_request_permit_snapshot.py
    tests/tools/test_task22_phase_elastic_calibration_analyzer.py
)

[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "Calibration preflight requires a clean worktree" >&2
    exit 4
}
git merge-base --is-ancestor "$BASE_SHA" HEAD || {
    echo "Calibration HEAD does not contain upstream/main calibration base $BASE_SHA" >&2
    exit 4
}
[[ "$(git rev-list --count "$BASE_SHA"..HEAD)" == "1" ]] || {
    echo "Calibration branch must contain exactly one commit above the clean upstream base" >&2
    exit 4
}
[[ -z "$(git rev-list --merges "$BASE_SHA"..HEAD)" ]] || {
    echo "Calibration commit must not be a merge" >&2
    exit 4
}
diff -u \
    <(printf '%s\n' "${EXPECTED_PATHS[@]}" | LC_ALL=C sort) \
    <(git diff --name-only "$BASE_SHA" HEAD | LC_ALL=C sort) || {
    echo "Calibration diff contains a missing or unexpected path" >&2
    exit 4
}
git diff --check "$BASE_SHA" HEAD
[[ ! -e relax/engine/rollout/admission.py ]] || {
    echo "Clean-main calibration must not contain the prior Task22 admission implementation" >&2
    exit 4
}

"$PYTHON_BIN" -m py_compile \
    relax/backends/megatron/actor.py \
    relax/engine/rollout/permit_observability.py \
    relax/engine/rollout/request_permit.py \
    relax/engine/rollout/sglang_rollout.py \
    scripts/task22/analyze_phase_elastic_calibration.py \
    scripts/task22/smoke_sglang_calibration.py
bash -n scripts/task22/run_clean_main_calibration.sh
bash -n scripts/task22/prepare_megatron_calibration.sh
"$PYTHON_BIN" -m pytest -q \
    tests/engine/rollout/test_permit_observability.py \
    tests/engine/rollout/test_request_permit.py \
    tests/engine/rollout/test_request_permit_snapshot.py \
    tests/backends/megatron/test_frozen_weight_dgrad.py \
    tests/tools/test_task22_phase_elastic_calibration_analyzer.py

if [[ "$MODE" == "--local" ]]; then
    echo "TASK22_CLEAN_MAIN_PREFLIGHT verdict=PASS_LOCAL"
    exit 0
fi

[[ "${NUM_GPUS:-}" == "4" ]] || { echo "Formal preflight requires NUM_GPUS=4" >&2; exit 4; }
[[ -n "${TASK22_EXPECTED_GIT_SHA:-}" ]] || { echo "Formal preflight requires TASK22_EXPECTED_GIT_SHA" >&2; exit 4; }
[[ "${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-}" == "stdout" ]] || {
    echo "Formal preflight requires scheduler status on stdout" >&2
    exit 4
}
[[ "${SGLANG_LOG_SCHEDULER_STATUS_INTERVAL:-}" == "1.0" ]] || {
    echo "Formal preflight requires a 1.0-second scheduler heartbeat" >&2
    exit 4
}
[[ "${FLASHINFER_CUDA_ARCH_LIST:-}" == "12.0a" ]] || {
    echo "Formal preflight requires FLASHINFER_CUDA_ARCH_LIST=12.0a for SM120" >&2
    exit 4
}
[[ "$(git rev-parse HEAD)" == "$TASK22_EXPECTED_GIT_SHA" ]] || {
    echo "Formal preflight HEAD does not match TASK22_EXPECTED_GIT_SHA" >&2
    exit 4
}
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$gpu_count" == "4" ]] || { echo "Formal preflight requires exactly 4 GPUs, found $gpu_count" >&2; exit 4; }
"$PYTHON_BIN" - <<'PY'
import csv
import io
import subprocess

expected_name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = [row for row in csv.reader(io.StringIO(raw)) if row]
if len(rows) != 4:
    raise SystemExit(f"expected 4 GPU inventory rows, found {len(rows)}")
for row in rows:
    index, uuid, name, memory_mib, driver = (field.strip() for field in row)
    if name != expected_name:
        raise SystemExit(f"GPU {index} model mismatch: {name!r} != {expected_name!r}")
    if int(memory_mib) < 95_000:
        raise SystemExit(f"GPU {index} memory too small for 96GB contract: {memory_mib} MiB")
    if not uuid or not driver:
        raise SystemExit(f"GPU {index} inventory is incomplete")
print("TASK22_CLEAN_MAIN_PREFLIGHT gpu_contract=4x_RTX_PRO_6000_96GB_PASS")
PY
"$PYTHON_BIN" - <<'PY'
import torch
from flashinfer.jit import core
from flashinfer.norm import rmsnorm

targets = core.current_compilation_context.TARGET_CUDA_ARCHS
if (12, "0a") not in targets:
    raise SystemExit(f"FlashInfer did not resolve the required SM120a target: {targets!r}")
core.check_cuda_arch()
x = torch.randn((4, 2560), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((2560,), device="cuda", dtype=torch.bfloat16)
actual = rmsnorm(x, weight, 1e-6)
torch.cuda.synchronize()
expected = (
    x.float()
    * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    * weight.float()
).to(torch.bfloat16)
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
print("TASK22_CLEAN_MAIN_PREFLIGHT flashinfer_arch=SM120a_RMSNORM_PASS")
PY
"$PYTHON_BIN" -c 'import ray, sglang, torch'
TASK22_PYTHON="$PYTHON_BIN" scripts/task22/prepare_sglang_calibration.sh
TASK22_PYTHON="$PYTHON_BIN" scripts/task22/prepare_megatron_calibration.sh
"$PYTHON_BIN" - "$REPO" <<'PY'
from pathlib import Path
import inspect
import relax

repo = Path(__import__("sys").argv[1]).resolve()
origin = Path(inspect.getfile(relax)).resolve()
if repo not in origin.parents:
    raise SystemExit(f"formal preflight imported Relax outside the checked worktree: {origin}")
print(f"TASK22_CLEAN_MAIN_PREFLIGHT relax_origin={origin}")
PY
"$PYTHON_BIN" - <<'PY'
import inspect

from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight

source = inspect.getsource(LinearWithFrozenWeight.backward)
needle = "grad_output.reshape(-1, grad_output.size(-1))"
if needle not in source:
    raise SystemExit("installed Megatron-LM lacks the frozen-weight DGRAD fold from PR #162")
print("TASK22_CLEAN_MAIN_PREFLIGHT megatron_dgrad_fold=PASS")
PY
"$PYTHON_BIN" -m pytest -q tests/backends/megatron/test_frozen_weight_dgrad.py

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
[[ -r "$MODEL_DIR/Qwen3-4B/config.json" ]] || { echo "Qwen3-4B checkpoint missing" >&2; exit 4; }
[[ -r "$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl" ]] || { echo "Dapo dataset missing" >&2; exit 4; }
echo "TASK22_CLEAN_MAIN_PREFLIGHT gpu_count=$gpu_count"
echo "TASK22_CLEAN_MAIN_PREFLIGHT verdict=PASS_FORMAL"
