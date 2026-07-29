#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-/root/RelaxRepo}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-fs/task22/overlap_baseline}"
ANALYZER="${ANALYZER:-$RUN_ROOT/analyze_overlap_baseline_v2.py}"
BASE_WRAPPER="${BASE_WRAPPER:-$REPO/scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async.sh}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-fs/exps}"
DATA_DIR="${DATA_DIR:-/root/autodl-fs/exps}"
EXP_DIR="${EXP_DIR:-/root/autodl-fs/exps}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/probe15_committed_$STAMP}"
RESUME_DIR="$RUN_DIR/task22_resume"
STALENESS_DIR="$RUN_DIR/task22_staleness"
PREFLIGHT="$REPO/scripts/task22/prepare_sglang_timing_transport.sh"

if [ "$(git -C "$REPO" status --porcelain | wc -l)" -ne 0 ]; then
    echo "Relax repository must be clean for a commit-based probe" >&2
    exit 4
fi
if [ ! -x "$BASE_WRAPPER" ] || [ ! -r "$ANALYZER" ]; then
    echo "Task22 base wrapper or analyzer is missing" >&2
    exit 4
fi

mkdir -p "$RUN_DIR/logs" "$RESUME_DIR" "$STALENESS_DIR"
if find "$RESUME_DIR" "$STALENESS_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Task22 output directories must be empty" >&2
    exit 4
fi

cleanup() {
    if [ -n "${NVML_PID:-}" ]; then
        kill "$NVML_PID" >/dev/null 2>&1 || true
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

source /root/activate_relax_image.sh >/dev/null 2>&1
bash "$PREFLIGHT" > "$RUN_DIR/logs/sglang_timing_preflight.log" 2>&1
grep -q "TASK22_SGLANG_TIMING_PREFLIGHT=PASS" "$RUN_DIR/logs/sglang_timing_preflight.log"

ray stop --force >/dev/null 2>&1 || true
sleep 3

{
    echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
    echo "num_rollout=15"
    echo "headline_logical_steps=5..14"
    echo "resume_dir=$RESUME_DIR"
    echo "staleness_dir=$STALENESS_DIR"
    echo "ray_dedup_logs=0"
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
NUM_ROLLOUT=15 \
TASK22_RESUME_DIR="$RESUME_DIR" \
TASK22_STALENESS_DIR="$STALENESS_DIR" \
RAY_DEDUP_LOGS=0 \
MODEL_DIR="$MODEL_DIR" \
DATA_DIR="$DATA_DIR" \
EXP_DIR="$EXP_DIR" \
timeout --signal=TERM --kill-after=180 "${RUN_TIMEOUT_S:-5400}" \
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
    --resume-dir "$RESUME_DIR" \
    --headline-lo 5 \
    --headline-hi 14 \
    --base-step-wall 145.9 \
    > "$RUN_DIR/analysis_steps_5_14.md" 2>&1
ANALYZER_RC=$?
set -e

echo "$ANALYZER_RC" > "$RUN_DIR/ANALYZER_EXIT_CODE"
ANALYZER_VERDICT="$(
    awk '/^- verdict:/{print $3; exit}' "$RUN_DIR/analysis_steps_5_14.md"
)"
if [ -z "$ANALYZER_VERDICT" ]; then
    ANALYZER_VERDICT=UNPARSEABLE
    ANALYZER_RC=4
fi
echo "$ANALYZER_VERDICT" > "$RUN_DIR/ANALYZER_VERDICT"
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
echo "ANALYZER_VERDICT=$(cat "$RUN_DIR/ANALYZER_VERDICT")"
exit "$ANALYZER_RC"
