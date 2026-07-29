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

if [ ! -d "$SGLANG_REPO" ]; then
    echo "SGLang source directory not found: $SGLANG_REPO" >&2
    exit 4
fi
if [ ! -r "$MODEL_PATH/config.json" ]; then
    echo "Model config not found: $MODEL_PATH/config.json" >&2
    exit 4
fi

if git -C "$SGLANG_REPO" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
    echo "SGLang timing transport patch already applied"
elif git -C "$SGLANG_REPO" apply --check "$PATCH" >/dev/null 2>&1; then
    git -C "$SGLANG_REPO" apply "$PATCH"
    echo "Applied SGLang timing transport patch"
else
    echo "SGLang timing transport patch state is broken" >&2
    exit 4
fi

git -C "$SGLANG_REPO" apply --check --reverse "$PATCH"
python3 "$UNIT_TEST"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
python3 "$GPU_SMOKE" --model-path "$MODEL_PATH" --output "$RUN_DIR/result.json"
sha256sum "$RUN_DIR/result.json" "$RUN_DIR/smoke.log" > "$RUN_DIR/SHA256SUMS"
echo "SUCCEEDED" > "$RUN_DIR/STATUS"
echo "RUN_DIR=$RUN_DIR"
