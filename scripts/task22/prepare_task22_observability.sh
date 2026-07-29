#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
TIMING_PREFLIGHT="$REPO/scripts/task22/prepare_sglang_timing_transport.sh"
PATCH="$REPO/scripts/task22/sglang_request_shape_observability.patch"
UNIT_TEST="$REPO/scripts/task22/test_sglang_request_shape_observability.py"

bash "$TIMING_PREFLIGHT"

SGLANG_IMPORT_ROOT="$(
    python3 -c \
        'from pathlib import Path; from sglang.srt.utils import scheduler_status_logger; print(Path(scheduler_status_logger.__file__).resolve().parents[3])'
)"
SCHEDULER_STATUS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/utils/scheduler_status_logger.py"
SCHEDULER_METRICS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/observability/scheduler_metrics_mixin.py"

if grep -q -- '"running_seq_lens"' "$SCHEDULER_STATUS_FILE" \
    && grep -q -- "TASK22_REQUEST_SHAPE_PREFILL_STATUS" "$SCHEDULER_METRICS_FILE"; then
    echo "Task22 request/shape observability patch already applied"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH"
    echo "Applied Task22 request/shape observability patch"
else
    echo "Task22 request/shape observability patch state is broken" >&2
    exit 4
fi

grep -q -- '"running_seq_lens"' "$SCHEDULER_STATUS_FILE"
grep -q -- "TASK22_REQUEST_SHAPE_PREFILL_STATUS" "$SCHEDULER_METRICS_FILE"
python3 "$UNIT_TEST"
echo "TASK22_OBSERVABILITY_PREFLIGHT=PASS"
