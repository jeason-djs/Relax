#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Default environment configuration for local single-node development.
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job.sh) has been used.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local.sh
#
# Environment variables:
#   NUM_GPUS               - Number of GPUs to use (optional, auto-detect from CUDA_VISIBLE_DEVICES)
#   CUDA_VISIBLE_DEVICES   - Comma-separated GPU IDs (e.g., "0,1,2,3" → 4 GPUs)
#   MASTER_ADDR            - Head node IP address (default: 127.0.0.1)
#   MEGATRON               - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX                  - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

_relax_ray() {
    if [ -n "${TASK22_PYTHON:-}" ] && [ -x "${TASK22_PYTHON}" ]; then
        "${TASK22_PYTHON}" -m ray.scripts.scripts "$@"
    else
        command ray "$@"
    fi
}

_relax_ray_status() {
    if [ -n "${TASK22_PYTHON:-}" ] && [ -x "${TASK22_PYTHON}" ]; then
        timeout 5 "${TASK22_PYTHON}" -m ray.scripts.scripts status
    else
        timeout 5 ray status
    fi
}

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && _relax_ray_status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
_relax_ray stop --force 2>/dev/null || true

# Never use process-name-wide pkill here: local.sh is sourced from the training
# wrapper while qualification monitors and other users' jobs may be Python
# processes on the same host. A caller may explicitly identify stale PIDs owned
# by this job (space-separated) for bounded cleanup.
for _relax_pid in ${RELAX_LOCAL_CLEANUP_PIDS:-}; do
    case "$_relax_pid" in
        ''|*[!0-9]*) continue ;;
    esac
    if [ "$_relax_pid" -gt 1 ] && [ "$_relax_pid" -ne "$$" ]; then
        kill -TERM "$_relax_pid" 2>/dev/null || true
    fi
done
unset _relax_pid

set -x

# ── environment setup ───────────────────────────────────────────────────────
unset MASTER_ADDR 2>/dev/null || true
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── NVLink detection ────────────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export HAS_NVLINK=1
else
    export HAS_NVLINK=0
fi
if [ -n "$NCCL_NVLS_ENABLE" ] && [ "$NCCL_NVLS_ENABLE" -eq 0 ]; then
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── GPU count detection ───────────────────────────────────────────────────────
# Priority: NUM_GPUS env > CUDA_VISIBLE_DEVICES > default 8
if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Count GPUs from CUDA_VISIBLE_DEVICES (comma-separated)
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
    else
        NUM_GPUS=8
    fi
fi

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "Starting Ray head node: MASTER_ADDR=$MASTER_ADDR, NUM_GPUS=$NUM_GPUS"

_relax_ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# ── set entrypoint mode ────────────────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="local"

# Runtime env for single-node (empty, env inherited from Ray cluster).
# A qualification runner may pre-freeze this value into its contract.
# Cap OMP/MKL/OpenBLAS threads (default 24) to avoid CPU oversubscription when colocating multiple Ray actors per node.
if [ -z "${RUNTIME_ENV_JSON:-}" ]; then
export RUNTIME_ENV_JSON="{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-24}\",
   \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-24}\",
   \"OPENBLAS_NUM_THREADS\": \"${OPENBLAS_NUM_THREADS:-24}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}\",
   \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-1}\",
   \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\",
   \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}\",
   \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM:-1}\",
   \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-1}\",
   \"INDEXER_ROPE_NEOX_STYLE\": \"${INDEXER_ROPE_NEOX_STYLE:-0}\"
}
}"
fi

echo "=== Local environment ready ==="
