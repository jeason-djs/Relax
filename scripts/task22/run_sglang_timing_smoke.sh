#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-/root/RelaxRepo}"
SGLANG_REPO="${SGLANG_REPO:-/sgl-workspace/sglang}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/exps/Qwen3-4B}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-fs/task22/sglang_timing_smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$STAMP}"
PATCH="$REPO/scripts/task22/sglang_timing_transport.patch"
UNIT_TEST="$REPO/scripts/task22/test_sglang_timing_transport.py"
GPU_SMOKE="$REPO/scripts/task22/smoke_sglang_timing_transport.py"

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/smoke.log") 2>&1

if [ ! -r "$MODEL_PATH/config.json" ]; then
    echo "Model config not found: $MODEL_PATH/config.json" >&2
    exit 4
fi

if [ -r "$SGLANG_REPO/python/sglang/srt/observability/req_time_stats.py" ]; then
    if git -C "$SGLANG_REPO" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
        echo "SGLang timing transport patch already applied to source tree"
    elif git -C "$SGLANG_REPO" apply --check "$PATCH" >/dev/null 2>&1; then
        git -C "$SGLANG_REPO" apply "$PATCH"
        echo "Applied SGLang timing transport patch to source tree"
    else
        echo "SGLang source-tree timing patch state is broken" >&2
        exit 4
    fi
    git -C "$SGLANG_REPO" apply --check --reverse "$PATCH"
else
    SGLANG_FILE="$(python3 -c 'from sglang.srt.observability import req_time_stats; print(req_time_stats.__file__)')"
    SGLANG_SITE_PACKAGES="$(python3 -c 'from pathlib import Path; from sglang.srt.observability import req_time_stats; print(Path(req_time_stats.__file__).resolve().parents[3])')"
    if [ ! -r "$SGLANG_FILE" ]; then
        echo "Installed SGLang timing source not found: $SGLANG_FILE" >&2
        exit 4
    fi
    if patch --dry-run --batch --reverse -d "$SGLANG_SITE_PACKAGES" -p2 < "$PATCH" >/dev/null 2>&1; then
        echo "SGLang timing transport patch already applied to installed package"
    elif patch --dry-run --batch -d "$SGLANG_SITE_PACKAGES" -p2 < "$PATCH" >/dev/null 2>&1; then
        patch --batch -d "$SGLANG_SITE_PACKAGES" -p2 < "$PATCH"
        echo "Applied SGLang timing transport patch to installed package"
    else
        echo "Installed SGLang timing patch state is broken" >&2
        exit 4
    fi
    patch --dry-run --batch --reverse -d "$SGLANG_SITE_PACKAGES" -p2 < "$PATCH" >/dev/null
fi

python3 "$UNIT_TEST"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
python3 "$GPU_SMOKE" --model-path "$MODEL_PATH" --output "$RUN_DIR/result.json"
sha256sum "$RUN_DIR/result.json" "$RUN_DIR/smoke.log" > "$RUN_DIR/SHA256SUMS"
echo "SUCCEEDED" > "$RUN_DIR/STATUS"
echo "RUN_DIR=$RUN_DIR"
