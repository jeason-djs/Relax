#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Bring an older saved runtime image up to the latest-main image-patch contract.
# This is public baseline preparation, not a Task 22 candidate optimization.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
PATCH_FILE="$REPO/scripts/task22/megatron_frozen_weight_dgrad.patch"

MEGATRON_ROOT="$(
    "$PYTHON_BIN" -c \
        'from pathlib import Path; from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight; print(Path(__import__("inspect").getfile(LinearWithFrozenWeight)).resolve().parents[3])'
)"
LAYERS_FILE="$MEGATRON_ROOT/megatron/core/tensor_parallel/layers.py"
FOLD_NEEDLE='grad_output.reshape(-1, grad_output.size(-1))'

if grep -qF -- "$FOLD_NEEDLE" "$LAYERS_FILE"; then
    printf '%s\n' "Megatron frozen-weight DGRAD fold already present"
elif patch --dry-run --batch --forward -d "$MEGATRON_ROOT" -p1 < "$PATCH_FILE" >/dev/null 2>&1; then
    patch --batch --forward -d "$MEGATRON_ROOT" -p1 < "$PATCH_FILE"
    printf '%s\n' "Applied latest-main Megatron frozen-weight DGRAD image hunk"
else
    printf '%s\n' "Megatron frozen-weight DGRAD patch state is incompatible" >&2
    exit 4
fi

"$PYTHON_BIN" -m py_compile "$LAYERS_FILE"
