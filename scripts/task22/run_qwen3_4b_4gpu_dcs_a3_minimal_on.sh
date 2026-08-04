#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Task 22 four-GPU minimal A3 entrypoint:
#   Actor TP2 + two independent one-GPU Rollout engines.
# Enables request-level A3 and in-place DCS publication while leaving
# Admission, priority scheduling, and work-aware placement disabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NUM_GPUS=4
export NUM_ROLLOUT="${NUM_ROLLOUT:-11}"
export TASK22_EXPERIMENT_ARM=dcs_a3_minimal_on

exec "$SCRIPT_DIR/run_clean_main_calibration.sh"
