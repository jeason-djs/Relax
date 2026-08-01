#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WRAPPER="$REPO/scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
VALIDATOR="$REPO/scripts/task22/validate_admission_run.py"
MONITOR="$REPO/scripts/task22/monitor_admission_run.py"
HEALTH_MONITOR="$REPO/scripts/task22/monitor_admission_health.py"
COMPARATOR="$REPO/scripts/task22/compare_admission_pair.py"
PREFLIGHT="$REPO/scripts/task22/preflight_admission.sh"
INPUT_GUARD="$REPO/scripts/task22/input_guard.py"
GPU_SAMPLER="$REPO/scripts/task22/sample_gpu_state.py"
PROCESS_DEADLINE="$REPO/scripts/task22/enforce_process_deadline.py"
PYTHON_REQUEST="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
if [[ "$PYTHON_REQUEST" != /* || ! -x "$PYTHON_REQUEST" ]]; then
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 4
fi
PYTHON_BIN="$PYTHON_REQUEST"
MODE=""
STOP_AFTER_SHADOW=0
RESUME_ON_DIR=""
ON_FIRST=0
ON_ONLY=0

usage() {
    cat <<EOF
usage: $0 [--check|--run] [--stop-after-shadow | --resume-on PAIR_DIR | --on-first | --on-only]

  --check              run formal preflight without starting training (default)
  --run                run the authorized experiment
  --stop-after-shadow  stop after strict Shadow validation; never start ON
  --resume-on PAIR_DIR resume a validated qualification pair at the ON leg
  --on-first           run matched ON first, then Shadow
  --on-only            run one authorized ON qualification leg
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --check|--run)
            if [[ -n "$MODE" ]]; then
                echo "Specify exactly one of --check or --run" >&2
                usage >&2
                exit 2
            fi
            MODE="$1"
            ;;
        --stop-after-shadow)
            if [[ "$STOP_AFTER_SHADOW" == "1" ]]; then
                echo "--stop-after-shadow may be specified only once" >&2
                usage >&2
                exit 2
            fi
            STOP_AFTER_SHADOW=1
            ;;
        --resume-on)
            if [[ -n "$RESUME_ON_DIR" || $# -lt 2 ]]; then
                echo "--resume-on requires exactly one pair directory" >&2
                usage >&2
                exit 2
            fi
            RESUME_ON_DIR="$2"
            shift
            ;;
        --on-first)
            if [[ "$ON_FIRST" == "1" ]]; then
                echo "--on-first may be specified only once" >&2
                usage >&2
                exit 2
            fi
            ON_FIRST=1
            ;;
        --on-only)
            if [[ "$ON_ONLY" == "1" ]]; then
                echo "--on-only may be specified only once" >&2
                usage >&2
                exit 2
            fi
            ON_ONLY=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done
MODE="${MODE:---check}"
if [[ -n "$RESUME_ON_DIR" && ( "$MODE" != "--run" || "$STOP_AFTER_SHADOW" == "1" ) ]]; then
    echo "--resume-on requires --run and cannot be combined with --stop-after-shadow" >&2
    usage >&2
    exit 2
fi
if [[ "$ON_FIRST" == "1" && ( "$MODE" != "--run" || "$STOP_AFTER_SHADOW" == "1" || -n "$RESUME_ON_DIR" ) ]]; then
    echo "--on-first requires --run and cannot be combined with staged qualification options" >&2
    usage >&2
    exit 2
fi
if [[ "$ON_ONLY" == "1" && ( "$MODE" != "--run" || "$STOP_AFTER_SHADOW" == "1" || -n "$RESUME_ON_DIR" || "$ON_FIRST" == "1" ) ]]; then
    echo "--on-only requires --run and cannot be combined with other run-order options" >&2
    usage >&2
    exit 2
fi
if [[ "$ON_ONLY" == "1" ]]; then
    RUN_SCOPE="on_qualification"
elif [[ "$STOP_AFTER_SHADOW" == "1" || -n "$RESUME_ON_DIR" ]]; then
    RUN_SCOPE="shadow_qualification"
else
    RUN_SCOPE="matched_pair"
fi

cd "$REPO"

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Formal matched A/B requires a clean git worktree" >&2
    exit 4
fi

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ROOT="${RUN_ROOT:-$(dirname -- "$REPO")/task22_probe_artifacts}"
RUN_ROOT="$(cd -- "$(dirname -- "$RUN_ROOT")" && pwd -P)/$(basename -- "$RUN_ROOT")"
NUM_ROLLOUT="${NUM_ROLLOUT:-15}"
EXPECTED_SAMPLES_PER_PARTITION="${EXPECTED_SAMPLES_PER_PARTITION:-64}"
EXPECTED_ENGINES="${EXPECTED_ENGINES:-2}"
MAX_STALENESS="${MAX_STALENESS:-2}"
HEADLINE_LO="${HEADLINE_LO:-5}"
HEADLINE_HI="${HEADLINE_HI:-14}"
TASK22_ADMISSION_PROFILE="${TASK22_ADMISSION_PROFILE:-legacy_v1}"
if [[ "$TASK22_ADMISSION_PROFILE" == "work_conserving_v2" ]]; then
    PARTITION_ADMISSION_POLICY="${PARTITION_ADMISSION_POLICY:-work_conserving}"
    PARTITION_ADMISSION_MIN="${PARTITION_ADMISSION_MIN:-12}"
    PARTITION_ADMISSION_MAX="${PARTITION_ADMISSION_MAX:-16}"
    PARTITION_ADMISSION_SLACK="${PARTITION_ADMISSION_SLACK:-4}"
    REQUEST_PLACEMENT_MODE="${REQUEST_PLACEMENT_MODE:-on}"
    REQUEST_PLACEMENT_POLICY="${REQUEST_PLACEMENT_POLICY:-least_predicted_work}"
    USE_SLIME_ROUTER="${USE_SLIME_ROUTER:-1}"
elif [[ "$TASK22_ADMISSION_PROFILE" == "legacy_v1" ]]; then
    PARTITION_ADMISSION_POLICY="${PARTITION_ADMISSION_POLICY:-legacy_debt_window}"
    PARTITION_ADMISSION_MIN="${PARTITION_ADMISSION_MIN:-4}"
    PARTITION_ADMISSION_MAX="${PARTITION_ADMISSION_MAX:-8}"
    PARTITION_ADMISSION_SLACK="${PARTITION_ADMISSION_SLACK:-2}"
    REQUEST_PLACEMENT_MODE="${REQUEST_PLACEMENT_MODE:-off}"
    REQUEST_PLACEMENT_POLICY="${REQUEST_PLACEMENT_POLICY:-least_predicted_work}"
    USE_SLIME_ROUTER="${USE_SLIME_ROUTER:-0}"
else
    echo "TASK22_ADMISSION_PROFILE must be legacy_v1 or work_conserving_v2" >&2
    exit 4
fi
TRAIN_SEED="${TRAIN_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-5400}"
TASK22_EVIDENCE_PROFILE="${TASK22_EVIDENCE_PROFILE:-qualification_v1}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
if [[ "$TASK22_EVIDENCE_PROFILE" != "qualification_v1" && "$TASK22_EVIDENCE_PROFILE" != "clean_ab_v1" ]]; then
    echo "TASK22_EVIDENCE_PROFILE must be qualification_v1 or clean_ab_v1" >&2
    exit 4
fi
if [[ "$ON_FIRST" == "1" && "$TASK22_EVIDENCE_PROFILE" != "clean_ab_v1" ]]; then
    echo "--on-first requires TASK22_EVIDENCE_PROFILE=clean_ab_v1" >&2
    exit 4
fi
if [[ "$ON_ONLY" == "1" && "$TASK22_EVIDENCE_PROFILE" != "clean_ab_v1" ]]; then
    echo "--on-only requires TASK22_EVIDENCE_PROFILE=clean_ab_v1" >&2
    exit 4
fi
TASK22_MONITOR_POLL_INTERVAL_WAS_SET="${TASK22_MONITOR_POLL_INTERVAL+x}"
TASK22_MONITOR_EVIDENCE_GRACE_WAS_SET="${TASK22_MONITOR_EVIDENCE_GRACE+x}"
TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S_WAS_SET="${TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S+x}"
TASK22_GPU_MAX_SNAPSHOT_AGE_S_WAS_SET="${TASK22_GPU_MAX_SNAPSHOT_AGE_S+x}"
TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S_WAS_SET="${TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S+x}"
TASK22_GPU_SAMPLE_INTERVAL_S_WAS_SET="${TASK22_GPU_SAMPLE_INTERVAL_S+x}"
TASK22_HARD_FAILURE_GRACE_S_WAS_SET="${TASK22_HARD_FAILURE_GRACE_S+x}"
TASK22_MONITOR_TIMEOUT_S_WAS_SET="${TASK22_MONITOR_TIMEOUT_S+x}"
TASK22_MONITOR_TERM_GRACE_S_WAS_SET="${TASK22_MONITOR_TERM_GRACE_S+x}"
TASK22_TRAINING_TERM_TIMEOUT_S_WAS_SET="${TASK22_TRAINING_TERM_TIMEOUT_S+x}"
NUM_GPUS_WAS_SET="${NUM_GPUS+x}"
CUDA_VISIBLE_DEVICES_WAS_SET="${CUDA_VISIBLE_DEVICES+x}"
if [[ "$TASK22_EVIDENCE_PROFILE" == "clean_ab_v1" ]]; then
    TASK22_MONITOR_POLL_INTERVAL="${TASK22_MONITOR_POLL_INTERVAL:-5}"
    TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S="${TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S:-1200}"
    TASK22_GPU_MAX_SNAPSHOT_AGE_S="${TASK22_GPU_MAX_SNAPSHOT_AGE_S:-30}"
    TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S="${TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S:-30}"
    TASK22_GPU_SAMPLE_INTERVAL_S="${TASK22_GPU_SAMPLE_INTERVAL_S:-5}"
    TASK22_HARD_FAILURE_GRACE_S="${TASK22_HARD_FAILURE_GRACE_S:-300}"
    TASK22_PAIR_COOLDOWN_S="${TASK22_PAIR_COOLDOWN_S:-60}"
else
    TASK22_MONITOR_POLL_INTERVAL="${TASK22_MONITOR_POLL_INTERVAL:-1}"
    TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S="${TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S:-600}"
    TASK22_GPU_MAX_SNAPSHOT_AGE_S="${TASK22_GPU_MAX_SNAPSHOT_AGE_S:-5}"
    TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S="${TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S:-2}"
    TASK22_GPU_SAMPLE_INTERVAL_S="${TASK22_GPU_SAMPLE_INTERVAL_S:-1}"
    TASK22_HARD_FAILURE_GRACE_S="${TASK22_HARD_FAILURE_GRACE_S:-0}"
    TASK22_PAIR_COOLDOWN_S="${TASK22_PAIR_COOLDOWN_S:-0}"
fi
TASK22_MONITOR_EVIDENCE_GRACE="${TASK22_MONITOR_EVIDENCE_GRACE:-5}"
TASK22_MONITOR_TIMEOUT_S="${TASK22_MONITOR_TIMEOUT_S:-$((RUN_TIMEOUT_S + 300))}"
TASK22_MONITOR_TERM_GRACE_S="${TASK22_MONITOR_TERM_GRACE_S:-1}"
TASK22_TRAINING_TERM_TIMEOUT_S="${TASK22_TRAINING_TERM_TIMEOUT_S:-10}"
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
CUDA_VISIBLE_DEVICES="$(
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -c '
import os

raw = os.environ["CUDA_VISIBLE_DEVICES"]
if not raw.strip():
    print("")
    raise SystemExit(0)
devices = [value.strip() for value in raw.split(",")]
if any(not value for value in devices) or len(devices) != 4 or len(set(devices)) != 4:
    raise SystemExit(
        "CUDA_VISIBLE_DEVICES must be empty or identify exactly 4 unique devices"
    )
print(",".join(devices))
'
)" || exit 4
TASK22_CUDA_VISIBLE_DEVICES_CONTRACT="$CUDA_VISIBLE_DEVICES"
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES
else
    unset CUDA_VISIBLE_DEVICES
fi
if [[ -z "$RESUME_ON_DIR" ]]; then
    MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
    DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
    EXP_DIR="${EXP_DIR:-$MODEL_DIR}"
    MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd -P)"
    DATA_DIR="$(cd -- "$DATA_DIR" && pwd -P)"
    EXP_DIR="$(cd -- "$EXP_DIR" && pwd -P)"
else
    MODEL_DIR=""
    DATA_DIR=""
    EXP_DIR=""
fi
SOURCE_MODEL_INPUT_ROOT="$MODEL_DIR/Qwen3-4B"
SOURCE_DATA_INPUT_FILE="$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl"
SNAPSHOT_ROOT="$EXP_DIR/task22_input_snapshots"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
qualification_contract=(
    "NUM_ROLLOUT:$NUM_ROLLOUT:15"
    "EXPECTED_SAMPLES_PER_PARTITION:$EXPECTED_SAMPLES_PER_PARTITION:64"
    "EXPECTED_ENGINES:$EXPECTED_ENGINES:2"
    "MAX_STALENESS:$MAX_STALENESS:2"
    "HEADLINE_LO:$HEADLINE_LO:5"
    "HEADLINE_HI:$HEADLINE_HI:14"
    "RUN_TIMEOUT_S:$RUN_TIMEOUT_S:5400"
    "NUM_GPUS:$NUM_GPUS:4"
)
if [[ "$TASK22_ADMISSION_PROFILE" == "work_conserving_v2" ]]; then
    qualification_contract+=(
        "PARTITION_ADMISSION_POLICY:$PARTITION_ADMISSION_POLICY:work_conserving"
        "PARTITION_ADMISSION_MIN:$PARTITION_ADMISSION_MIN:12"
        "PARTITION_ADMISSION_MAX:$PARTITION_ADMISSION_MAX:16"
        "PARTITION_ADMISSION_SLACK:$PARTITION_ADMISSION_SLACK:4"
        "REQUEST_PLACEMENT_MODE:$REQUEST_PLACEMENT_MODE:on"
        "REQUEST_PLACEMENT_POLICY:$REQUEST_PLACEMENT_POLICY:least_predicted_work"
        "USE_SLIME_ROUTER:$USE_SLIME_ROUTER:1"
    )
else
    qualification_contract+=(
        "PARTITION_ADMISSION_POLICY:$PARTITION_ADMISSION_POLICY:legacy_debt_window"
        "PARTITION_ADMISSION_MIN:$PARTITION_ADMISSION_MIN:4"
        "PARTITION_ADMISSION_MAX:$PARTITION_ADMISSION_MAX:8"
        "PARTITION_ADMISSION_SLACK:$PARTITION_ADMISSION_SLACK:2"
        "REQUEST_PLACEMENT_MODE:$REQUEST_PLACEMENT_MODE:off"
        "USE_SLIME_ROUTER:$USE_SLIME_ROUTER:0"
    )
fi
for contract_entry in "${qualification_contract[@]}"; do
    IFS=: read -r contract_name contract_actual contract_expected <<< "$contract_entry"
    if [[ "$contract_actual" != "$contract_expected" ]]; then
        echo "Task 22 qualification requires $contract_name=$contract_expected; got $contract_actual" >&2
        exit 4
    fi
done
if [[ "$TASK22_EVIDENCE_PROFILE" == "qualification_v1" && "$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" != "600" ]]; then
    echo "Task 22 qualification requires TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S=600; got $TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" >&2
    exit 4
fi
STAMP="${TASK22_RUN_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
if [[ -n "$RESUME_ON_DIR" ]]; then
    if [[ ! -d "$RESUME_ON_DIR" ]]; then
        echo "Resume pair directory does not exist: $RESUME_ON_DIR" >&2
        exit 4
    fi
    PAIR_DIR="$(cd -- "$RESUME_ON_DIR" && pwd -P)"
else
    PAIR_DIR="$RUN_ROOT/admission_matched_${GIT_COMMIT:0:12}_$STAMP"
fi

export RELAX_REQUEST_PLACEMENT_MODE="$REQUEST_PLACEMENT_MODE"
export RELAX_REQUEST_PLACEMENT_POLICY="$REQUEST_PLACEMENT_POLICY"
export RELAX_RID_ONLY_REQUEST_LOGGING=1
export TASK22_EVIDENCE_PROFILE
export FLASHINFER_CUDA_ARCH_LIST
export RAY_DEDUP_LOGS=0
if [[ "$TASK22_EVIDENCE_PROFILE" == "qualification_v1" ]]; then
    export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"
    export SGLANG_LOG_SCHEDULER_STATUS_INTERVAL="${SGLANG_LOG_SCHEDULER_STATUS_INTERVAL:-1.0}"
else
    export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"
    export SGLANG_LOG_SCHEDULER_STATUS_INTERVAL="${SGLANG_LOG_SCHEDULER_STATUS_INTERVAL:-5.0}"
fi
export TASK22_PYTHON="$PYTHON_REQUEST"
if [[ -n "$RESUME_ON_DIR" ]]; then
    WORKING_DIR=""
    RUNTIME_ENV_JSON=""
else
    WORKING_DIR="${WORKING_DIR:-$REPO}"
    WORKING_DIR="$(cd -- "$WORKING_DIR" && pwd -P)"
    if [[ -z "${RUNTIME_ENV_JSON+x}" ]]; then
        RUNTIME_ENV_JSON='{}'
    fi
    export RUNTIME_ENV_JSON
    RUNTIME_ENV_JSON="$(
        WORKING_DIR="$WORKING_DIR" RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON" "$PYTHON_BIN" -c '
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env["working_dir"] = os.environ["WORKING_DIR"]
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "RELAX_REQUEST_PLACEMENT_MODE",
    "RELAX_REQUEST_PLACEMENT_POLICY",
    "RELAX_RID_ONLY_REQUEST_LOGGING",
    "TASK22_EVIDENCE_PROFILE",
    "FLASHINFER_CUDA_ARCH_LIST",
    "RAY_DEDUP_LOGS",
    "TASK22_PYTHON",
):
    env_vars[name] = os.environ[name]
for name in (
    "SGLANG_LOG_SCHEDULER_STATUS_TARGET",
    "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL",
):
    if name in os.environ:
        env_vars[name] = os.environ[name]
print(json.dumps(runtime_env, sort_keys=True, separators=(",", ":")))
'
    )"
fi

export MODEL_DIR DATA_DIR EXP_DIR WORKING_DIR RUNTIME_ENV_JSON TASK22_RUNTIME_ATTESTATION_DIR

run_formal_preflight() {
    env \
        -u TASK22_AUTHORIZE_GPU_RUN \
        -u TASK22_AUTHORIZE_ON_RUN \
        -u TASK22_EVIDENCE_PROFILE \
        -u TASK22_PAIR_COOLDOWN_S \
        -u TASK22_MONITOR_POLL_INTERVAL \
        -u TASK22_MONITOR_EVIDENCE_GRACE \
        -u TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S \
        -u TASK22_GPU_MAX_SNAPSHOT_AGE_S \
        -u TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S \
        -u TASK22_GPU_SAMPLE_INTERVAL_S \
        -u TASK22_HARD_FAILURE_GRACE_S \
        -u TASK22_MONITOR_TIMEOUT_S \
        -u TASK22_MONITOR_TERM_GRACE_S \
        -u TASK22_TRAINING_TERM_TIMEOUT_S \
        -u TASK22_RUN_STAMP \
        bash "$PREFLIGHT" --formal
}

if [[ "$MODE" == "--check" ]]; then
    run_formal_preflight
    echo "TASK22_MATCHED_AB verdict=READY"
    echo "TASK22_MATCHED_AB commit=$GIT_COMMIT"
    echo "TASK22_MATCHED_AB pair_dir=$PAIR_DIR"
    exit 0
fi

if [[ "${TASK22_AUTHORIZE_GPU_RUN:-0}" != "1" ]]; then
    echo "Set TASK22_AUTHORIZE_GPU_RUN=1 to authorize the formal GPU run" >&2
    exit 4
fi
if [[ -n "$RESUME_ON_DIR" && "${TASK22_AUTHORIZE_ON_RUN:-0}" != "1" ]]; then
    echo "Set TASK22_AUTHORIZE_ON_RUN=1 after human review to resume the ON leg" >&2
    exit 4
fi
if [[ "$ON_FIRST" == "1" && "${TASK22_AUTHORIZE_ON_RUN:-0}" != "1" ]]; then
    echo "Set TASK22_AUTHORIZE_ON_RUN=1 to authorize the ON-first matched pair" >&2
    exit 4
fi
if [[ "$ON_ONLY" == "1" && "${TASK22_AUTHORIZE_ON_RUN:-0}" != "1" ]]; then
    echo "Set TASK22_AUTHORIZE_ON_RUN=1 to authorize the ON-only qualification" >&2
    exit 4
fi
if [[ -z "$RESUME_ON_DIR" && -e "$PAIR_DIR" ]]; then
    echo "Pair artifact path already exists: $PAIR_DIR" >&2
    exit 4
fi

verify_snapshot() {
    "$PYTHON_BIN" "$INPUT_GUARD" snapshot-verify --snapshot "$INPUT_SNAPSHOT" \
        >/dev/null
}

if [[ -n "$RESUME_ON_DIR" ]]; then
    if [[ "$(cat "$PAIR_DIR/GIT_COMMIT" 2>/dev/null || true)" != "$GIT_COMMIT" ]]; then
        echo "Resume pair commit does not match current HEAD" >&2
        exit 4
    fi
    if [[ "$(cat "$PAIR_DIR/SHADOW_VALID" 2>/dev/null || true)" != "PASS" ]]; then
        echo "Resume pair lacks SHADOW_VALID=PASS" >&2
        exit 4
    fi
    if [[ "$(cat "$PAIR_DIR/PAIR_STATUS" 2>/dev/null || true)" != "AWAITING_ON_AUTHORIZATION" ]]; then
        echo "Resume pair is not awaiting ON authorization" >&2
        exit 4
    fi
    if [[ -e "$PAIR_DIR/on" || ! -f "$PAIR_DIR/ARTIFACT_SHA256SUMS" ]]; then
        echo "Resume pair has an ON artifact or lacks frozen checksums" >&2
        exit 4
    fi
    if ! (cd "$PAIR_DIR" && sha256sum --check --strict ARTIFACT_SHA256SUMS); then
        echo "Resume pair artifact checksum verification failed" >&2
        exit 4
    fi
    restored_contract_values=""
    if ! restored_contract_values="$(
        TASK22_MONITOR_POLL_INTERVAL_WAS_SET="$TASK22_MONITOR_POLL_INTERVAL_WAS_SET" \
        TASK22_MONITOR_EVIDENCE_GRACE_WAS_SET="$TASK22_MONITOR_EVIDENCE_GRACE_WAS_SET" \
        TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S_WAS_SET="$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S_WAS_SET" \
        TASK22_GPU_MAX_SNAPSHOT_AGE_S_WAS_SET="$TASK22_GPU_MAX_SNAPSHOT_AGE_S_WAS_SET" \
        TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S_WAS_SET="$TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S_WAS_SET" \
        TASK22_GPU_SAMPLE_INTERVAL_S_WAS_SET="$TASK22_GPU_SAMPLE_INTERVAL_S_WAS_SET" \
        TASK22_HARD_FAILURE_GRACE_S_WAS_SET="$TASK22_HARD_FAILURE_GRACE_S_WAS_SET" \
        TASK22_MONITOR_TIMEOUT_S_WAS_SET="$TASK22_MONITOR_TIMEOUT_S_WAS_SET" \
        TASK22_MONITOR_TERM_GRACE_S_WAS_SET="$TASK22_MONITOR_TERM_GRACE_S_WAS_SET" \
        TASK22_TRAINING_TERM_TIMEOUT_S_WAS_SET="$TASK22_TRAINING_TERM_TIMEOUT_S_WAS_SET" \
        NUM_GPUS_WAS_SET="$NUM_GPUS_WAS_SET" \
        CUDA_VISIBLE_DEVICES_WAS_SET="$CUDA_VISIBLE_DEVICES_WAS_SET" \
        TASK22_MONITOR_POLL_INTERVAL="$TASK22_MONITOR_POLL_INTERVAL" \
        TASK22_MONITOR_EVIDENCE_GRACE="$TASK22_MONITOR_EVIDENCE_GRACE" \
        TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S="$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" \
        TASK22_GPU_MAX_SNAPSHOT_AGE_S="$TASK22_GPU_MAX_SNAPSHOT_AGE_S" \
        TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S="$TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S" \
        TASK22_GPU_SAMPLE_INTERVAL_S="$TASK22_GPU_SAMPLE_INTERVAL_S" \
        TASK22_HARD_FAILURE_GRACE_S="$TASK22_HARD_FAILURE_GRACE_S" \
        TASK22_MONITOR_TIMEOUT_S="$TASK22_MONITOR_TIMEOUT_S" \
        TASK22_MONITOR_TERM_GRACE_S="$TASK22_MONITOR_TERM_GRACE_S" \
        TASK22_TRAINING_TERM_TIMEOUT_S="$TASK22_TRAINING_TERM_TIMEOUT_S" \
        NUM_GPUS="$NUM_GPUS" CUDA_VISIBLE_DEVICES="$TASK22_CUDA_VISIBLE_DEVICES_CONTRACT" \
        "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" <<'PY'
import json
import math
import os
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    contract = json.load(source)
if contract.get("schema_version") != 6:
    raise SystemExit("Shadow contract is not schema v6")

numeric = {
    "TASK22_MONITOR_POLL_INTERVAL": "monitor_poll_interval_s",
    "TASK22_MONITOR_EVIDENCE_GRACE": "monitor_evidence_grace_s",
    "TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S": "monitor_no_progress_timeout_s",
    "TASK22_GPU_MAX_SNAPSHOT_AGE_S": "gpu_max_snapshot_age_s",
    "TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S": "gpu_max_snapshot_interval_s",
    "TASK22_GPU_SAMPLE_INTERVAL_S": "gpu_sample_interval_s",
    "TASK22_HARD_FAILURE_GRACE_S": "hard_failure_grace_s",
    "TASK22_MONITOR_TIMEOUT_S": "monitor_timeout_s",
    "TASK22_MONITOR_TERM_GRACE_S": "monitor_term_grace_s",
    "TASK22_TRAINING_TERM_TIMEOUT_S": "training_term_timeout_s",
}
for env_name, key in numeric.items():
    frozen = contract.get(key)
    if (
        not isinstance(frozen, (int, float))
        or isinstance(frozen, bool)
        or not math.isfinite(float(frozen))
    ):
        raise SystemExit(f"Shadow schema v6 lacks valid {key}")
    if os.environ.get(f"{env_name}_WAS_SET"):
        try:
            current = float(os.environ[env_name])
        except ValueError as exc:
            raise SystemExit(f"Invalid resume environment {env_name}") from exc
        if current != float(frozen):
            raise SystemExit(f"Resume environment drift for {env_name}")
    print(f"{env_name}={frozen}")

frozen_num_gpus = contract.get("num_gpus")
if frozen_num_gpus != 4:
    raise SystemExit("Shadow schema v6 lacks num_gpus=4")
if os.environ.get("NUM_GPUS_WAS_SET") and os.environ["NUM_GPUS"] != "4":
    raise SystemExit("Resume environment drift for NUM_GPUS")
print("NUM_GPUS=4")

frozen_visible = contract.get("cuda_visible_devices")
if not isinstance(frozen_visible, str):
    raise SystemExit("Shadow schema v6 lacks normalized cuda_visible_devices")
if (
    os.environ.get("CUDA_VISIBLE_DEVICES_WAS_SET")
    and os.environ["CUDA_VISIBLE_DEVICES"] != frozen_visible
):
    raise SystemExit("Resume environment drift for CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES={frozen_visible}")
PY
    )"; then
        echo "Resume pair supervision or GPU contract restoration failed" >&2
        exit 4
    fi
    while IFS='=' read -r name value; do
        printf -v "$name" '%s' "$value"
    done <<< "$restored_contract_values"
    export TASK22_MONITOR_POLL_INTERVAL TASK22_MONITOR_EVIDENCE_GRACE
    export TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S TASK22_GPU_MAX_SNAPSHOT_AGE_S
    export TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S TASK22_GPU_SAMPLE_INTERVAL_S
    export TASK22_HARD_FAILURE_GRACE_S TASK22_MONITOR_TIMEOUT_S
    export TASK22_MONITOR_TERM_GRACE_S TASK22_TRAINING_TERM_TIMEOUT_S
    export NUM_GPUS
    TASK22_CUDA_VISIBLE_DEVICES_CONTRACT="$CUDA_VISIBLE_DEVICES"
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES
    else
        unset CUDA_VISIBLE_DEVICES
    fi
    INPUT_SNAPSHOT="$(
        "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    contract = json.load(source)
model_dir = contract.get("model_dir")
if not isinstance(model_dir, str) or model_dir != contract.get("data_dir"):
    raise SystemExit("Shadow contract does not identify one shared input snapshot")
print(os.path.realpath(model_dir))
PY
    )" || {
        echo "Resume pair lacks a valid shared input snapshot contract" >&2
        exit 4
    }
    EXP_DIR="$(
        "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    contract = json.load(source)
exp_dir = contract.get("exp_dir")
if not isinstance(exp_dir, str) or not os.path.isabs(exp_dir):
    raise SystemExit("Shadow contract lacks an absolute experiment directory")
print(os.path.realpath(exp_dir))
PY
    )" || {
        echo "Resume pair lacks a valid experiment directory contract" >&2
        exit 4
    }
    WORKING_DIR="$(
        "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    contract = json.load(source)
working_dir = contract.get("working_dir")
if not isinstance(working_dir, str) or not os.path.isabs(working_dir):
    raise SystemExit("Shadow contract lacks an absolute working directory")
print(os.path.realpath(working_dir))
PY
    )" || {
        echo "Resume pair lacks a valid working directory contract" >&2
        exit 4
    }
    RUNTIME_ENV_JSON="$(
        "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    contract = json.load(source)
raw = contract.get("runtime_env_json")
if not isinstance(raw, str):
    raise SystemExit("Shadow contract lacks normalized runtime_env_json")
runtime_env = json.loads(raw)
canonical = json.dumps(runtime_env, sort_keys=True, separators=(",", ":"))
if raw != canonical:
    raise SystemExit("Shadow runtime_env_json is not normalized")
actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
if actual_hash != contract.get("runtime_env_json_sha256"):
    raise SystemExit("Shadow runtime_env_json hash mismatch")
if runtime_env.get("working_dir") != contract.get("working_dir"):
    raise SystemExit("Shadow runtime_env_json working_dir mismatch")
print(canonical)
PY
    )" || {
        echo "Resume pair lacks a valid normalized runtime environment contract" >&2
        exit 4
    }
    export WORKING_DIR RUNTIME_ENV_JSON EXP_DIR
    if ! verify_snapshot; then
        echo "Resume pair snapshot verification failed" >&2
        exit 4
    fi
else
    if ! INPUT_SNAPSHOT="$(
        "$PYTHON_BIN" "$INPUT_GUARD" snapshot-create \
            --snapshot-root "$SNAPSHOT_ROOT" \
            --model-root "$SOURCE_MODEL_INPUT_ROOT" \
            --data-file "$SOURCE_DATA_INPUT_FILE"
    )"; then
        echo "Input snapshot creation failed" >&2
        exit 4
    fi
    if ! verify_snapshot; then
        echo "New input snapshot verification failed" >&2
        exit 4
    fi
fi

MODEL_DIR="$INPUT_SNAPSHOT"
DATA_DIR="$INPUT_SNAPSHOT"
MODEL_INPUT_ROOT="$INPUT_SNAPSHOT/Qwen3-4B"
DATA_INPUT_FILE="$INPUT_SNAPSHOT/dapo-math-17k/dapo-math-17k.jsonl"
export MODEL_DIR DATA_DIR
run_formal_preflight

if [[ -z "$RESUME_ON_DIR" ]]; then
    mkdir -p "$PAIR_DIR"
    printf '%s\n' "$GIT_COMMIT" > "$PAIR_DIR/GIT_COMMIT"
    git status --porcelain=v1 --untracked-files=all > "$PAIR_DIR/GIT_STATUS"
    git diff --binary > "$PAIR_DIR/WORKTREE.patch"
fi

sampler_pid=""
training_pid=""
monitor_pid=""
monitor_watchdog_pid=""
training_start_identity=""
sampler_start_identity=""
process_start_identity() {
    "$PYTHON_BIN" - "$1" <<'PY'
import pathlib
import subprocess
import sys

pid = sys.argv[1]
try:
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
except OSError:
    identity = subprocess.run(
        ["ps", "-o", "lstart=", "-p", pid],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
else:
    suffix = stat.rsplit(")", 1)[1].split()
    identity = suffix[19] if len(suffix) > 19 else ""
if not identity:
    raise SystemExit(f"cannot identify process start for pid {pid}")
print(identity)
PY
}
stop_training_safely() {
    local pid="$1"
    local expected_identity="$2"
    local term_timeout="${3:-10}"
    "$PYTHON_BIN" - "$pid" "$expected_identity" "$term_timeout" <<'PY'
import os
import pathlib
import signal
import subprocess
import sys
import time

pid = int(sys.argv[1])
expected = sys.argv[2]
term_timeout = float(sys.argv[3])


def identity():
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        try:
            return subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except OSError:
            return ""
    suffix = stat.rsplit(")", 1)[1].split()
    return suffix[19] if len(suffix) > 19 else ""


def group_exists():
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


actual = identity()
if actual:
    if actual != expected:
        raise SystemExit(f"refusing to stop reused pid {pid}")
    if os.getpgid(pid) != pid:
        raise SystemExit(f"refusing to stop unsafe process group for pid {pid}")
elif not group_exists():
    raise SystemExit(0)

try:
    os.killpg(pid, signal.SIGTERM)
except ProcessLookupError:
    raise SystemExit(0)
deadline = time.monotonic() + max(term_timeout, 0.0)
while group_exists() and time.monotonic() < deadline:
    time.sleep(0.05)
if not group_exists():
    raise SystemExit(0)

# Identity/group membership is rechecked immediately before KILL. If the
# leader exited, the extant group itself prevents this PGID from being reused.
actual = identity()
if actual and (actual != expected or os.getpgid(pid) != pid):
    raise SystemExit(f"refusing KILL after identity/group change for pid {pid}")
try:
    os.killpg(pid, signal.SIGKILL)
except ProcessLookupError:
    pass
PY
}
stop_pid_safely() {
    local pid="$1"
    local expected_identity="$2"
    "$PYTHON_BIN" - "$pid" "$expected_identity" <<'PY'
import os
import pathlib
import signal
import subprocess
import sys

pid = int(sys.argv[1])
expected = sys.argv[2]
try:
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
except OSError:
    try:
        actual = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        actual = ""
else:
    suffix = stat.rsplit(")", 1)[1].split()
    actual = suffix[19] if len(suffix) > 19 else ""
if not actual:
    # This PID is an unreaped direct child created by this runner, so it cannot
    # be reused before the immediately following wait().
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise SystemExit(0)
if actual != expected:
    raise SystemExit(f"refusing to stop reused pid {pid}")
os.kill(pid, signal.SIGTERM)
PY
}
cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        monitor_pid=""
    fi
    if [[ -n "$monitor_watchdog_pid" ]]; then
        kill "$monitor_watchdog_pid" >/dev/null 2>&1 || true
        wait "$monitor_watchdog_pid" >/dev/null 2>&1 || true
        monitor_watchdog_pid=""
    fi
    if [[ -n "$training_pid" ]]; then
        stop_training_safely "$training_pid" "$training_start_identity" \
            "${TASK22_TRAINING_TERM_TIMEOUT_S:-10}" >/dev/null 2>&1 || true
        wait "$training_pid" >/dev/null 2>&1 || true
        training_pid=""
        training_start_identity=""
    fi
    if [[ -n "$sampler_pid" ]]; then
        stop_pid_safely "$sampler_pid" "$sampler_start_identity" >/dev/null 2>&1 || true
        wait "$sampler_pid" >/dev/null 2>&1 || true
        sampler_pid=""
        sampler_start_identity=""
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

write_contract() {
    local mode="$1"
    local output_path="$2"
    TASK22_CUDA_VISIBLE_DEVICES_CONTRACT="$TASK22_CUDA_VISIBLE_DEVICES_CONTRACT" \
        "$PYTHON_BIN" - "$output_path" "$mode" "$REPO" <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from scripts.task22.input_guard import _exclusive_json, load_json_nofollow, manifest_sha256
from relax.utils.task22_runtime_attestation import (
    task22_env_sha256,
    working_dir_content_hashes,
    working_dir_content_sha256,
)

output_path, mode, repo = sys.argv[1:]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


sglang_modules = (
    "sglang.srt.observability.req_time_stats",
    "sglang.srt.observability.scheduler_metrics_mixin",
    "sglang.srt.utils.request_logger",
    "sglang.srt.utils.scheduler_status_logger",
)
sglang_source_sha256 = {}
for module_name in sglang_modules:
    module_path = os.path.realpath(importlib.import_module(module_name).__file__)
    sglang_source_sha256[module_path] = sha256_file(module_path)


source_sha256 = working_dir_content_hashes(repo)
launch_path = os.environ["TASK22_PYTHON"]
executable_realpath = os.path.realpath(sys.executable)
launcher_target = os.path.realpath(launch_path)
pip_freeze = subprocess.run(
    [launch_path, "-m", "pip", "freeze", "--all"],
    check=True,
    capture_output=True,
    text=True,
).stdout
gpu_fingerprint = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()
runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env_canonical = json.dumps(
    runtime_env,
    sort_keys=True,
    separators=(",", ":"),
).encode()
input_manifest = load_json_nofollow(Path(os.environ["TASK22_INPUT_MANIFEST"]))
contract = {
    "schema_version": 6,
    "git_commit": os.environ["GIT_COMMIT"],
    "admission_mode": mode,
    "admission_profile": os.environ["TASK22_ADMISSION_PROFILE"],
    "admission_policy": os.environ["PARTITION_ADMISSION_POLICY"],
    "admission_debt_basis": "logical_previous_partition",
    "transfer_flush_policy": "previous_boundary_current_preferred",
    "evidence_profile": os.environ["TASK22_EVIDENCE_PROFILE"],
    "scheduler_status_interval_s": float(os.environ["SGLANG_LOG_SCHEDULER_STATUS_INTERVAL"]),
    "flashinfer_cuda_arch_list": os.environ["FLASHINFER_CUDA_ARCH_LIST"],
    "runner_scope": os.environ["RUN_SCOPE"],
    "request_placement_mode": os.environ["REQUEST_PLACEMENT_MODE"],
    "request_placement_policy": os.environ["REQUEST_PLACEMENT_POLICY"],
    "use_slime_router": os.environ["USE_SLIME_ROUTER"] == "1",
    "num_rollout": int(os.environ["NUM_ROLLOUT"]),
    "expected_samples_per_partition": int(os.environ["EXPECTED_SAMPLES_PER_PARTITION"]),
    "expected_engines": int(os.environ["EXPECTED_ENGINES"]),
    "max_staleness": int(os.environ["MAX_STALENESS"]),
    "headline_lo": int(os.environ["HEADLINE_LO"]),
    "headline_hi": int(os.environ["HEADLINE_HI"]),
    "admission_min": int(os.environ["PARTITION_ADMISSION_MIN"]),
    "admission_max": int(os.environ["PARTITION_ADMISSION_MAX"]),
    "admission_slack": int(os.environ["PARTITION_ADMISSION_SLACK"]),
    "train_seed": int(os.environ["TRAIN_SEED"]),
    "rollout_seed": int(os.environ["ROLLOUT_SEED"]),
    "run_timeout_s": int(os.environ["RUN_TIMEOUT_S"]),
    "model_dir": os.path.realpath(os.environ["MODEL_DIR"]),
    "data_dir": os.path.realpath(os.environ["DATA_DIR"]),
    "exp_dir": os.path.realpath(os.environ["EXP_DIR"]),
    "working_dir": os.path.realpath(os.environ["WORKING_DIR"]),
    "working_dir_content_sha256": working_dir_content_sha256(os.environ["WORKING_DIR"]),
    "runtime_env_json_sha256": hashlib.sha256(runtime_env_canonical).hexdigest(),
    "runtime_env_json": runtime_env_canonical.decode(),
    "task22_env_sha256": task22_env_sha256(),
    "input_manifest_sha256": manifest_sha256(input_manifest),
    "monitor_poll_interval_s": float(os.environ["TASK22_MONITOR_POLL_INTERVAL"]),
    "monitor_evidence_grace_s": float(os.environ["TASK22_MONITOR_EVIDENCE_GRACE"]),
    "monitor_no_progress_timeout_s": float(
        os.environ["TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S"]
    ),
    "gpu_max_snapshot_age_s": float(os.environ["TASK22_GPU_MAX_SNAPSHOT_AGE_S"]),
    "gpu_max_snapshot_interval_s": float(
        os.environ["TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S"]
    ),
    "gpu_sample_interval_s": float(os.environ["TASK22_GPU_SAMPLE_INTERVAL_S"]),
    "hard_failure_grace_s": float(os.environ["TASK22_HARD_FAILURE_GRACE_S"]),
    "pair_cooldown_s": float(os.environ["TASK22_PAIR_COOLDOWN_S"]),
    "monitor_timeout_s": int(os.environ["TASK22_MONITOR_TIMEOUT_S"]),
    "monitor_term_grace_s": float(os.environ["TASK22_MONITOR_TERM_GRACE_S"]),
    "training_term_timeout_s": float(os.environ["TASK22_TRAINING_TERM_TIMEOUT_S"]),
    "num_gpus": int(os.environ["NUM_GPUS"]),
    "cuda_visible_devices": os.environ["TASK22_CUDA_VISIBLE_DEVICES_CONTRACT"],
    "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
    "nccl_socket_ifname": os.environ["NCCL_SOCKET_IFNAME"],
    "training_python": {
        "launch_path": launch_path,
        "executable_realpath": executable_realpath,
        "launcher_target_sha256": sha256_file(launcher_target),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
        "version": platform.python_version(),
    },
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "dependency_versions": {
        name: package_version(name)
        for name in ("ray", "sglang", "torch", "transformers")
    },
    "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
    "gpu_fingerprint": gpu_fingerprint,
    "sglang_source_sha256": sglang_source_sha256,
    "source_sha256": source_sha256,
}
_exclusive_json(Path(output_path), contract)
PY
}

run_one() {
    local mode="$1"
    local run_dir="$PAIR_DIR/$mode"
    local prepared_contract="${2:-}"
    if ! verify_snapshot; then
        echo "Input snapshot is invalid before $mode; refusing run" >&2
        return 4
    fi
    mkdir -p "$run_dir/observability" "$run_dir/logs"
    if [[ "$TASK22_EVIDENCE_PROFILE" == "qualification_v1" ]]; then
        mkdir -p "$run_dir/timeline"
    fi
    mkdir "$run_dir/runtime_attestation"
    local runtime_attestation_dir="$run_dir/runtime_attestation"
    "$PYTHON_BIN" "$INPUT_GUARD" snapshot-verify \
        --snapshot "$INPUT_SNAPSHOT" \
        --output "$run_dir/input_manifest_before.json" >/dev/null
    printf '%s\n' RUNNING > "$run_dir/STATUS"
    date '+%Y-%m-%dT%H:%M:%S%z' > "$run_dir/STARTED_AT"
    if [[ -n "$prepared_contract" ]]; then
        "$PYTHON_BIN" - "$prepared_contract" "$run_dir/run_contract.json" <<'PY'
import sys
from pathlib import Path
from scripts.task22.input_guard import _exclusive_json, load_json_nofollow

source, destination = map(Path, sys.argv[1:])
_exclusive_json(destination, load_json_nofollow(source))
source.unlink()
PY
    else
        write_contract "$mode" "$run_dir/run_contract.json"
    fi

    local gpu_log_name="nvidia_smi_1s.csv"
    if [[ "$TASK22_EVIDENCE_PROFILE" == "clean_ab_v1" ]]; then
        gpu_log_name="nvidia_smi.csv"
    fi
    "$PYTHON_BIN" -u "$GPU_SAMPLER" --interval "$TASK22_GPU_SAMPLE_INTERVAL_S" \
        > "$run_dir/logs/$gpu_log_name" 2>&1 &
    sampler_pid=$!
    sampler_start_identity="$(process_start_identity "$sampler_pid")"
    local run_started_at
    run_started_at="$("$PYTHON_BIN" -c 'import time; print(time.time())')"

    set +e
    GIT_COMMIT="$GIT_COMMIT" \
    NUM_ROLLOUT="$NUM_ROLLOUT" \
    EXPECTED_SAMPLES_PER_PARTITION="$EXPECTED_SAMPLES_PER_PARTITION" \
    EXPECTED_ENGINES="$EXPECTED_ENGINES" \
    MAX_STALENESS="$MAX_STALENESS" \
    HEADLINE_LO="$HEADLINE_LO" \
    HEADLINE_HI="$HEADLINE_HI" \
    PARTITION_ADMISSION_MIN="$PARTITION_ADMISSION_MIN" \
    PARTITION_ADMISSION_MAX="$PARTITION_ADMISSION_MAX" \
    PARTITION_ADMISSION_SLACK="$PARTITION_ADMISSION_SLACK" \
    PARTITION_ADMISSION_POLICY="$PARTITION_ADMISSION_POLICY" \
    REQUEST_PLACEMENT_MODE="$REQUEST_PLACEMENT_MODE" \
    REQUEST_PLACEMENT_POLICY="$REQUEST_PLACEMENT_POLICY" \
    USE_SLIME_ROUTER="$USE_SLIME_ROUTER" \
    TRAIN_SEED="$TRAIN_SEED" \
    ROLLOUT_SEED="$ROLLOUT_SEED" \
    RUN_TIMEOUT_S="$RUN_TIMEOUT_S" \
    MODEL_DIR="$MODEL_DIR" \
    DATA_DIR="$DATA_DIR" \
    EXP_DIR="$EXP_DIR" \
    NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
    NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    PARTITION_ADMISSION_MODE="$mode" \
    REQUEST_OBSERVABILITY_DIR="$run_dir/observability" \
    TIMELINE_DUMP_DIR="$(
        if [[ "$TASK22_EVIDENCE_PROFILE" == "qualification_v1" ]]; then
            printf '%s\n' "$run_dir/timeline"
        fi
    )" \
    TASK22_EVIDENCE_PROFILE="$TASK22_EVIDENCE_PROFILE" \
    DRIVER_LOG_PATH="$run_dir/driver.log" \
    TASK22_RUNTIME_ATTESTATION_DIR="$runtime_attestation_dir" \
    "$PYTHON_BIN" -c \
        'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
        timeout --signal=TERM --kill-after=180 "$RUN_TIMEOUT_S" \
        env TASK22_INPUT_MANIFEST="$TASK22_INPUT_MANIFEST" \
        TASK22_INPUT_ROOTS_JSON="$TASK22_INPUT_ROOTS_JSON" \
        bash "$WRAPPER" &
    training_pid=$!
    training_start_identity="$(process_start_identity "$training_pid")"
    if [[ "$TASK22_EVIDENCE_PROFILE" == "clean_ab_v1" ]]; then
        timeout --signal=TERM --kill-after="$TASK22_MONITOR_TERM_GRACE_S" \
            "$TASK22_MONITOR_TIMEOUT_S" "$PYTHON_BIN" "$HEALTH_MONITOR" \
            --run-dir "$run_dir" \
            --pid "$training_pid" \
            --pid-start-identity "$training_start_identity" \
            --process-group-id "$training_pid" \
            --sampler-pid "$sampler_pid" \
            --sampler-start-identity "$sampler_start_identity" \
            --poll-interval "$TASK22_MONITOR_POLL_INTERVAL" \
            --no-progress-timeout "$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" \
            --hard-failure-grace "$TASK22_HARD_FAILURE_GRACE_S" \
            --term-timeout "$TASK22_TRAINING_TERM_TIMEOUT_S" \
            > "$run_dir/logs/online_monitor.log" 2>&1 &
    else
        timeout --signal=TERM --kill-after="$TASK22_MONITOR_TERM_GRACE_S" \
            "$TASK22_MONITOR_TIMEOUT_S" "$PYTHON_BIN" "$MONITOR" \
            --run-dir "$run_dir" \
            --pid "$training_pid" \
            --pid-start-identity "$training_start_identity" \
            --process-group-id "$training_pid" \
            --sampler-pid "$sampler_pid" \
            --sampler-start-identity "$sampler_start_identity" \
            --run-started-at "$run_started_at" \
            --expected-mode "$mode" \
            --expected-rollouts "$NUM_ROLLOUT" \
            --expected-samples-per-partition "$EXPECTED_SAMPLES_PER_PARTITION" \
            --expected-engines "$EXPECTED_ENGINES" \
            --max-staleness "$MAX_STALENESS" \
            --admission-min "$PARTITION_ADMISSION_MIN" \
            --admission-max "$PARTITION_ADMISSION_MAX" \
            --admission-slack "$PARTITION_ADMISSION_SLACK" \
            --headline-lo "$HEADLINE_LO" \
            --headline-hi "$HEADLINE_HI" \
            --poll-interval "$TASK22_MONITOR_POLL_INTERVAL" \
            --evidence-grace "$TASK22_MONITOR_EVIDENCE_GRACE" \
            --no-progress-timeout "$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" \
            --term-timeout "$TASK22_TRAINING_TERM_TIMEOUT_S" \
            --gpu-max-snapshot-age "$TASK22_GPU_MAX_SNAPSHOT_AGE_S" \
            --gpu-max-snapshot-interval "$TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S" \
            --monitor-timeout "$TASK22_MONITOR_TIMEOUT_S" \
            --monitor-term-grace "$TASK22_MONITOR_TERM_GRACE_S" \
            --num-gpus "$NUM_GPUS" \
            --cuda-visible-devices "$TASK22_CUDA_VISIBLE_DEVICES_CONTRACT" \
            > "$run_dir/logs/online_monitor.log" 2>&1 &
    fi
    monitor_pid=$!
    local monitor_timeout_marker="$run_dir/logs/.monitor_timeout"
    "$PYTHON_BIN" "$PROCESS_DEADLINE" \
        --pid "$monitor_pid" \
        --timeout "$TASK22_MONITOR_TIMEOUT_S" \
        --term-grace "$TASK22_MONITOR_TERM_GRACE_S" \
        --marker "$monitor_timeout_marker" \
        </dev/null >/dev/null 2>&1 &
    monitor_watchdog_pid=$!
    printf '%s\n' WAITING_MONITOR > "$run_dir/SUPERVISION_STATE"
    wait "$monitor_pid"
    local monitor_rc=$?
    printf 'MONITOR_EXITED(%s)\n' "$monitor_rc" > "$run_dir/SUPERVISION_STATE"
    kill "$monitor_watchdog_pid" >/dev/null 2>&1 || true
    wait "$monitor_watchdog_pid" >/dev/null 2>&1 || true
    monitor_watchdog_pid=""
    if [[ -f "$monitor_timeout_marker" ]]; then
        monitor_rc=124
    fi
    if [[ "$monitor_rc" -eq 124 || "$monitor_rc" -eq 137 ]]; then
        monitor_rc=124
        printf 'monitor hard timeout after %ss\n' "$TASK22_MONITOR_TIMEOUT_S" \
            >> "$run_dir/logs/online_monitor.log"
    fi
    monitor_pid=""
    local monitor_stopped_training=0
    if [[ -f "$run_dir/logs/.health_monitor_stopped_training" ]]; then
        monitor_stopped_training=1
    fi
    if [[ "$monitor_rc" -ne 0 && ( "$TASK22_EVIDENCE_PROFILE" != "clean_ab_v1" || "$monitor_stopped_training" == "1" ) ]]; then
        printf 'STOPPING_TRAINING(monitor=%s)\n' "$monitor_rc" > "$run_dir/SUPERVISION_STATE"
        stop_training_safely "$training_pid" "$training_start_identity" \
            "$TASK22_TRAINING_TERM_TIMEOUT_S" \
            >> "$run_dir/logs/online_monitor.log" 2>&1 || true
    elif [[ "$monitor_rc" -ne 0 ]]; then
        printf 'MONITOR_DEGRADED(monitor=%s)\n' "$monitor_rc" > "$run_dir/SUPERVISION_STATE"
        printf 'clean A/B health monitor degraded; training continues\n' \
            >> "$run_dir/logs/online_monitor.log"
    fi
    printf 'WAITING_TRAINING(monitor=%s)\n' "$monitor_rc" > "$run_dir/SUPERVISION_STATE"
    wait "$training_pid"
    local run_rc=$?
    printf 'TRAINING_EXITED(training=%s,monitor=%s)\n' "$run_rc" "$monitor_rc" \
        > "$run_dir/SUPERVISION_STATE"
    training_pid=""
    training_start_identity=""
    set -e

    printf 'STOPPING_SAMPLER(training=%s,monitor=%s)\n' "$run_rc" "$monitor_rc" \
        > "$run_dir/SUPERVISION_STATE"
    stop_pid_safely "$sampler_pid" "$sampler_start_identity" >/dev/null 2>&1 || true
    wait "$sampler_pid" >/dev/null 2>&1 || true
    sampler_pid=""
    sampler_start_identity=""
    ray stop --force >/dev/null 2>&1 || true
    printf 'SUPERVISION_COMPLETE(training=%s,monitor=%s)\n' "$run_rc" "$monitor_rc" \
        > "$run_dir/SUPERVISION_STATE"
    local input_guard_rc=0
    if ! "$PYTHON_BIN" "$INPUT_GUARD" snapshot-verify \
        --snapshot "$INPUT_SNAPSHOT" \
        --output "$run_dir/input_manifest_after.json" >/dev/null; then
        input_guard_rc=4
        echo "Input snapshot changed during $mode" >&2
    fi
    printf '%s\n' "$run_rc" > "$run_dir/EXIT_CODE"
    printf '%s\n' "$monitor_rc" > "$run_dir/ONLINE_MONITOR_EXIT_CODE"
    date '+%Y-%m-%dT%H:%M:%S%z' > "$run_dir/FINISHED_AT"
    if [[ "$run_rc" -eq 0 && "$input_guard_rc" -eq 0 && ( "$monitor_rc" -eq 0 || ( "$TASK22_EVIDENCE_PROFILE" == "clean_ab_v1" && "$monitor_stopped_training" == "0" ) ) ]]; then
        printf '%s\n' SUCCEEDED > "$run_dir/STATUS"
    elif [[ "$input_guard_rc" -eq 0 ]]; then
        printf 'FAILED(training=%s,monitor=%s)\n' "$run_rc" "$monitor_rc" > "$run_dir/STATUS"
    else
        printf 'FAILED(training=%s,monitor=%s,input_guard=%s)\n' \
            "$run_rc" "$monitor_rc" "$input_guard_rc" > "$run_dir/STATUS"
    fi

    set +e
    "$PYTHON_BIN" "$VALIDATOR" \
        --run-dir "$run_dir" \
        --expected-mode "$mode" \
        --expected-rollouts "$NUM_ROLLOUT" \
        --expected-samples-per-partition "$EXPECTED_SAMPLES_PER_PARTITION" \
        --expected-engines "$EXPECTED_ENGINES" \
        --max-staleness "$MAX_STALENESS" \
        --headline-lo "$HEADLINE_LO" \
        --headline-hi "$HEADLINE_HI" \
        --monitor-poll-interval "$TASK22_MONITOR_POLL_INTERVAL" \
        --monitor-evidence-grace "$TASK22_MONITOR_EVIDENCE_GRACE" \
        --monitor-no-progress-timeout "$TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S" \
        --gpu-max-snapshot-age "$TASK22_GPU_MAX_SNAPSHOT_AGE_S" \
        --gpu-max-snapshot-interval "$TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S" \
        --evidence-profile "$TASK22_EVIDENCE_PROFILE" \
        --monitor-timeout "$TASK22_MONITOR_TIMEOUT_S" \
        --monitor-term-grace "$TASK22_MONITOR_TERM_GRACE_S" \
        --training-term-timeout "$TASK22_TRAINING_TERM_TIMEOUT_S" \
        --num-gpus "$NUM_GPUS" \
        --cuda-visible-devices "$TASK22_CUDA_VISIBLE_DEVICES_CONTRACT" \
        --require-resume \
        --output-json "$run_dir/validation.json" \
        > "$run_dir/validation.stdout.json"
    local validator_rc=$?
    set -e
    printf '%s\n' "$validator_rc" > "$run_dir/VALIDATOR_EXIT_CODE"
    if [[ "$run_rc" -ne 0 || "$input_guard_rc" -ne 0 || "$validator_rc" -ne 0 ]]; then
        return 4
    fi
    if [[ "$monitor_rc" -ne 0 && ( "$TASK22_EVIDENCE_PROFILE" != "clean_ab_v1" || "$monitor_stopped_training" == "1" ) ]]; then
        return 4
    fi
}

write_artifact_checksums() {
    (
        cd "$PAIR_DIR"
        find . -type f ! -name ARTIFACT_SHA256SUMS -print0 \
            | sort -z \
            | xargs -0 sha256sum > ARTIFACT_SHA256SUMS
    )
}

prepare_matched_contract() {
    local source_mode="$1"
    local target_mode="$2"
    local candidate="$PAIR_DIR/.${target_mode}_contract_candidate.json"
    write_contract "$target_mode" "$candidate"
    if ! "$PYTHON_BIN" - "$PAIR_DIR/$source_mode/run_contract.json" "$candidate" \
            "$source_mode" "$target_mode" <<'PY'
import json
import sys

source_path, candidate_path, source_mode, target_mode = sys.argv[1:]
with open(source_path, encoding="utf-8") as source:
    frozen = json.load(source)
with open(candidate_path, encoding="utf-8") as source:
    candidate = json.load(source)
keys = set(frozen) | set(candidate)
diffs = {key for key in keys if frozen.get(key) != candidate.get(key)}
if diffs != {"admission_mode"}:
    print(
        f"{target_mode} contract differs from validated {source_mode} outside admission_mode: "
        + ", ".join(sorted(diffs)),
        file=sys.stderr,
    )
    raise SystemExit(4)
if frozen.get("admission_mode") != source_mode or candidate.get("admission_mode") != target_mode:
    print("Invalid admission mode order in matched pair", file=sys.stderr)
    raise SystemExit(4)
PY
    then
        rm -f "$candidate"
        return 4
    fi
    printf '%s\n' "$candidate"
}

export GIT_COMMIT NUM_ROLLOUT EXPECTED_SAMPLES_PER_PARTITION EXPECTED_ENGINES
export MAX_STALENESS HEADLINE_LO HEADLINE_HI
export PARTITION_ADMISSION_MIN PARTITION_ADMISSION_MAX PARTITION_ADMISSION_SLACK
export PARTITION_ADMISSION_POLICY TASK22_ADMISSION_PROFILE
export TRAIN_SEED ROLLOUT_SEED RUN_TIMEOUT_S MODEL_DIR DATA_DIR EXP_DIR
export NCCL_NVLS_ENABLE NCCL_SOCKET_IFNAME
export RUN_SCOPE
export TASK22_MONITOR_POLL_INTERVAL TASK22_MONITOR_EVIDENCE_GRACE
export TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S TASK22_GPU_MAX_SNAPSHOT_AGE_S
export TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S TASK22_GPU_SAMPLE_INTERVAL_S
export TASK22_HARD_FAILURE_GRACE_S TASK22_MONITOR_TIMEOUT_S
export TASK22_MONITOR_TERM_GRACE_S TASK22_TRAINING_TERM_TIMEOUT_S
export TASK22_EVIDENCE_PROFILE
export TASK22_PAIR_COOLDOWN_S
export NUM_GPUS TASK22_CUDA_VISIBLE_DEVICES_CONTRACT
export TASK22_INPUT_MANIFEST="$INPUT_SNAPSHOT/MANIFEST.json"
export TASK22_INPUT_ROOTS_JSON
TASK22_INPUT_ROOTS_JSON="$(
    MODEL_INPUT_ROOT="$MODEL_INPUT_ROOT" DATA_INPUT_FILE="$DATA_INPUT_FILE" \
    "$PYTHON_BIN" -c \
    'import json, os; print(json.dumps([os.environ["MODEL_INPUT_ROOT"], os.environ["DATA_INPUT_FILE"]]))'
)"
RUNTIME_ENV_JSON="$(
    RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON" \
    TASK22_INPUT_MANIFEST="$TASK22_INPUT_MANIFEST" \
    TASK22_INPUT_ROOTS_JSON="$TASK22_INPUT_ROOTS_JSON" \
    "$PYTHON_BIN" -c '
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
for name in ("TASK22_INPUT_MANIFEST", "TASK22_INPUT_ROOTS_JSON"):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime_env, sort_keys=True, separators=(",", ":")))
'
)"
export RUNTIME_ENV_JSON
export REQUEST_PLACEMENT_MODE REQUEST_PLACEMENT_POLICY USE_SLIME_ROUTER

if [[ "$ON_ONLY" == "1" ]]; then
    printf '%s\n' ON_RUNNING > "$PAIR_DIR/PAIR_STATUS"
    if ! run_one on; then
        printf '%s\n' ON_FAILED > "$PAIR_DIR/PAIR_STATUS"
        echo "ON-only qualification failed strict validation" >&2
        exit 5
    fi
    printf '%s\n' PASS > "$PAIR_DIR/ON_VALID"
    printf '%s\n' ON_VALIDATED > "$PAIR_DIR/PAIR_STATUS"
    write_artifact_checksums
    echo "TASK22_MATCHED_AB verdict=ON_PASS"
    echo "TASK22_MATCHED_AB scope=$RUN_SCOPE"
    echo "TASK22_MATCHED_AB pair_dir=$PAIR_DIR"
    exit 0
elif [[ "$ON_FIRST" == "1" ]]; then
    printf '%s\n' ON_RUNNING > "$PAIR_DIR/PAIR_STATUS"
    if ! run_one on; then
        printf '%s\n' ON_FAILED > "$PAIR_DIR/PAIR_STATUS"
        echo "ON failed strict validation; Shadow was not started" >&2
        exit 5
    fi
    printf '%s\n' ON_VALIDATED > "$PAIR_DIR/PAIR_STATUS"
    sleep "$TASK22_PAIR_COOLDOWN_S"
    shadow_contract="$(prepare_matched_contract on shadow)" || {
        echo "Shadow contract preflight failed" >&2
        exit 5
    }
    printf '%s\n' SHADOW_RUNNING > "$PAIR_DIR/PAIR_STATUS"
    if ! run_one shadow "$shadow_contract"; then
        printf '%s\n' SHADOW_FAILED > "$PAIR_DIR/PAIR_STATUS"
        echo "Shadow failed strict validation" >&2
        exit 5
    fi
    printf '%s\n' SHADOW_VALIDATED > "$PAIR_DIR/PAIR_STATUS"
    if ! "$PYTHON_BIN" "$COMPARATOR" \
            --shadow-dir "$PAIR_DIR/shadow" \
            --on-dir "$PAIR_DIR/on" \
            --output-json "$PAIR_DIR/pair_summary.json" \
            > "$PAIR_DIR/pair_summary.stdout.json"; then
        printf '%s\n' PAIR_COMPARISON_FAILED > "$PAIR_DIR/PAIR_STATUS"
        echo "Admission pair comparison failed" >&2
        exit 5
    fi
    printf '%s\n' PASS > "$PAIR_DIR/PAIR_VALID"
    printf '%s\n' PASS > "$PAIR_DIR/PAIR_STATUS"
    write_artifact_checksums
    echo "TASK22_MATCHED_AB verdict=PASS"
    echo "TASK22_MATCHED_AB scope=$RUN_SCOPE"
    echo "TASK22_MATCHED_AB order=on_then_shadow"
    echo "TASK22_MATCHED_AB pair_dir=$PAIR_DIR"
    exit 0
elif [[ -z "$RESUME_ON_DIR" ]]; then
    if ! run_one shadow; then
        echo "SHADOW failed strict validation; ON was not started" >&2
        exit 5
    fi

    printf '%s\n' PASS > "$PAIR_DIR/SHADOW_VALID"
    if [[ "$STOP_AFTER_SHADOW" == "1" ]]; then
        printf '%s\n' AWAITING_ON_AUTHORIZATION > "$PAIR_DIR/PAIR_STATUS"
        write_artifact_checksums
        echo "TASK22_MATCHED_AB verdict=SHADOW_PASS"
        echo "TASK22_MATCHED_AB scope=$RUN_SCOPE"
        echo "TASK22_MATCHED_AB pair_dir=$PAIR_DIR"
        exit 0
    fi
fi

if [[ "$(git rev-parse HEAD)" != "$GIT_COMMIT" || -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Repository changed after SHADOW; refusing ON" >&2
    exit 5
fi
if ! verify_snapshot; then
    echo "Input snapshot changed after SHADOW; refusing ON" >&2
    exit 5
fi

on_contract="$(prepare_matched_contract shadow on)" || {
    echo "ON contract preflight failed" >&2
    exit 5
}
printf '%s\n' ON_RUNNING > "$PAIR_DIR/PAIR_STATUS"
if ! run_one on "$on_contract"; then
    printf '%s\n' ON_FAILED > "$PAIR_DIR/PAIR_STATUS"
    echo "ON failed strict validation" >&2
    exit 5
fi

printf '%s\n' ON_VALIDATED > "$PAIR_DIR/PAIR_STATUS"
if ! "$PYTHON_BIN" "$COMPARATOR" \
        --shadow-dir "$PAIR_DIR/shadow" \
        --on-dir "$PAIR_DIR/on" \
        --output-json "$PAIR_DIR/pair_summary.json" \
        > "$PAIR_DIR/pair_summary.stdout.json"; then
    printf '%s\n' PAIR_COMPARISON_FAILED > "$PAIR_DIR/PAIR_STATUS"
    echo "Admission pair comparison failed" >&2
    exit 5
fi

printf '%s\n' PASS > "$PAIR_DIR/PAIR_VALID"
printf '%s\n' PASS > "$PAIR_DIR/PAIR_STATUS"
write_artifact_checksums

echo "TASK22_MATCHED_AB verdict=PASS"
echo "TASK22_MATCHED_AB scope=$RUN_SCOPE"
echo "TASK22_MATCHED_AB pair_dir=$PAIR_DIR"
