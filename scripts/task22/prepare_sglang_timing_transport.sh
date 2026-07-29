#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PATCH="${TASK22_SGLANG_TIMING_PATCH:-$REPO/scripts/task22/sglang_timing_transport.patch}"
UNIT_TEST="${TASK22_SGLANG_TIMING_TEST:-$REPO/scripts/task22/test_sglang_timing_transport.py}"

if [ ! -r "$PATCH" ] || [ ! -r "$UNIT_TEST" ]; then
    echo "Task22 SGLang timing patch or unit test is missing" >&2
    exit 4
fi

SGLANG_FILE="$(
    python3 -c \
        'from sglang.srt.observability import req_time_stats; print(req_time_stats.__file__)'
)"
SGLANG_IMPORT_ROOT="$(
    python3 -c \
        'from pathlib import Path; from sglang.srt.observability import req_time_stats; print(Path(req_time_stats.__file__).resolve().parents[3])'
)"

if [ ! -r "$SGLANG_FILE" ]; then
    echo "Imported SGLang timing source not found: $SGLANG_FILE" >&2
    exit 4
fi

if grep -q -- "_task22_forward_timing_payload" "$SGLANG_FILE"; then
    echo "Task22 SGLang timing transport patch already applied: $SGLANG_FILE"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$PATCH"
    echo "Applied Task22 SGLang timing transport patch: $SGLANG_FILE"
else
    echo "Task22 SGLang timing patch state is broken: $SGLANG_FILE" >&2
    exit 4
fi

grep -q -- "_task22_forward_timing_payload" "$SGLANG_FILE"
python3 "$UNIT_TEST"
echo "TASK22_SGLANG_TIMING_PREFLIGHT=PASS"
