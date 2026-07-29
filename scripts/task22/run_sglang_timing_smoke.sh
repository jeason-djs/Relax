#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-/root/RelaxRepo}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/exps/Qwen3-4B}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-fs/task22/sglang_timing_smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$STAMP}"
PREFLIGHT="$REPO/scripts/task22/prepare_task22_observability.sh"
GPU_SMOKE="$REPO/scripts/task22/smoke_sglang_timing_transport.py"

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/smoke.log") 2>&1

if [ ! -r "$MODEL_PATH/config.json" ]; then
    echo "Model config not found: $MODEL_PATH/config.json" >&2
    exit 4
fi

bash "$PREFLIGHT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
python3 "$GPU_SMOKE" --model-path "$MODEL_PATH" --output "$RUN_DIR/result.json"
(cd "$RUN_DIR" && sha256sum result.json > SHA256SUMS)
echo "SUCCEEDED" > "$RUN_DIR/STATUS"
echo "RUN_DIR=$RUN_DIR"
