#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PATCH="${RELAX_SGLANG_OBSERVABILITY_PATCH:-$REPO/scripts/task22/sglang_rollout_observability.patch}"
SMOKE_TEST="${RELAX_SGLANG_OBSERVABILITY_SMOKE:-$REPO/scripts/task22/smoke_rollout_observability.py}"

if [ ! -r "$PATCH" ] || [ ! -r "$SMOKE_TEST" ]; then
    echo "SGLang observability patch or smoke test is missing" >&2
    exit 4
fi

SGLANG_IMPORT_ROOT="$(
    python3 -c \
        'from pathlib import Path; from sglang.srt.utils import scheduler_status_logger; print(Path(scheduler_status_logger.__file__).resolve().parents[3])'
)"
TIMING_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/observability/req_time_stats.py"
METRICS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/observability/scheduler_metrics_mixin.py"
STATUS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/utils/scheduler_status_logger.py"

if grep -Eq -- "_(relax|task22)_forward_timing_payload" "$TIMING_FILE" \
    && grep -Eq -- "(RELAX_REQUEST_SHAPE_OBSERVABILITY|TASK22_REQUEST_SHAPE_PREFILL_STATUS)" "$METRICS_FILE" \
    && grep -q -- '"running_seq_lens"' "$STATUS_FILE"; then
    echo "Compatible Relax rollout observability patch already applied"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH"
    echo "Applied Relax rollout observability patch"
else
    echo "Relax rollout observability patch state is incompatible" >&2
    exit 4
fi

python3 "$SMOKE_TEST"
echo "RELAX_ROLLOUT_OBSERVABILITY_PREFLIGHT=PASS"
