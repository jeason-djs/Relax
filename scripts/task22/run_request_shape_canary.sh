#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-/root/RelaxRepo}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-fs/task22/request_shape_canary}"
BASE_WRAPPER="${BASE_WRAPPER:-$REPO/scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async.sh}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-fs/exps}"
DATA_DIR="${DATA_DIR:-/root/autodl-fs/exps}"
EXP_DIR="${EXP_DIR:-/root/autodl-fs/exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:-6}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/canary_${NUM_ROLLOUT}step_$STAMP}"
RESUME_DIR="$RUN_DIR/task22_resume"
STALENESS_DIR="$RUN_DIR/task22_staleness"
REQUEST_DIR="$RUN_DIR/task22_requests"
PREFLIGHT="$REPO/scripts/task22/prepare_task22_observability.sh"
ANALYZER="$REPO/scripts/task22/analyze_request_engine_shape.py"

if [ "$(git -C "$REPO" status --porcelain | wc -l)" -ne 0 ]; then
    echo "Relax repository must be clean for a commit-based canary" >&2
    exit 4
fi

mkdir -p "$RUN_DIR/logs" "$RESUME_DIR" "$STALENESS_DIR" "$REQUEST_DIR"

cleanup() {
    if [ -n "${NVML_PID:-}" ]; then
        kill "$NVML_PID" >/dev/null 2>&1 || true
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

source /root/activate_relax_image.sh >/dev/null 2>&1
bash "$PREFLIGHT" > "$RUN_DIR/logs/observability_preflight.log" 2>&1
grep -q "TASK22_OBSERVABILITY_PREFLIGHT=PASS" "$RUN_DIR/logs/observability_preflight.log"

ray stop --force >/dev/null 2>&1 || true
sleep 3

{
    echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
    echo "num_rollout=$NUM_ROLLOUT"
    echo "purpose=request-engine-shape-health-only"
    echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$RUN_DIR/INFO.txt"
echo RUNNING > "$RUN_DIR/STATUS"

(
    while true; do
        date "+%F %T.%N"
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw \
            --format=csv,noheader
        sleep 1
    done
) > "$RUN_DIR/logs/nvidia_smi_1s.csv" 2>&1 &
NVML_PID=$!

set +e
RUN_DIR="$RUN_DIR" \
NUM_ROLLOUT="$NUM_ROLLOUT" \
TASK22_RESUME_DIR="$RESUME_DIR" \
TASK22_STALENESS_DIR="$STALENESS_DIR" \
TASK22_REQUEST_DIR="$REQUEST_DIR" \
SGLANG_LOG_SCHEDULER_STATUS_TARGET=stdout \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=1.0 \
RAY_DEDUP_LOGS=0 \
MODEL_DIR="$MODEL_DIR" \
DATA_DIR="$DATA_DIR" \
EXP_DIR="$EXP_DIR" \
timeout --signal=TERM --kill-after=180 "${RUN_TIMEOUT_S:-3600}" \
    bash "$BASE_WRAPPER" > "$RUN_DIR/driver.log" 2>&1
RUN_RC=$?
set -e

echo "$RUN_RC" > "$RUN_DIR/EXIT_CODE"
date '+%Y-%m-%dT%H:%M:%S%z' > "$RUN_DIR/FINISHED_AT"
if [ "$RUN_RC" -ne 0 ]; then
    echo "FAILED($RUN_RC)" > "$RUN_DIR/STATUS"
    exit "$RUN_RC"
fi

set +e
python3 "$ANALYZER" \
    --driver-log "$RUN_DIR/driver.log" \
    --request-dir "$REQUEST_DIR" \
    --expected-engines 2 \
    --require-resume \
    --output-json "$RUN_DIR/request_engine_shape_health.json" \
    > "$RUN_DIR/logs/request_engine_shape_health.log" 2>&1
ANALYZER_RC=$?
set -e

echo "$ANALYZER_RC" > "$RUN_DIR/ANALYZER_EXIT_CODE"
if [ "$ANALYZER_RC" -eq 0 ]; then
    echo SUCCEEDED > "$RUN_DIR/STATUS"
else
    echo "FAILED_ANALYZER($ANALYZER_RC)" > "$RUN_DIR/STATUS"
fi

(
    cd "$RUN_DIR"
    find . -type f ! -name ARTIFACT_SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum > ARTIFACT_SHA256SUMS
)

echo "RUN_DIR=$RUN_DIR"
exit "$ANALYZER_RC"
