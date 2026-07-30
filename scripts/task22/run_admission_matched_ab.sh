#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WRAPPER="$REPO/scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
VALIDATOR="$REPO/scripts/task22/validate_admission_run.py"
COMPARATOR="$REPO/scripts/task22/compare_admission_pair.py"
PREFLIGHT="$REPO/scripts/task22/preflight_admission.sh"
PYTHON_BIN="${TASK22_PYTHON:-python3}"
MODE=""
STOP_AFTER_SHADOW=0
RESUME_ON_DIR=""

usage() {
    cat <<EOF
usage: $0 [--check|--run] [--stop-after-shadow | --resume-on PAIR_DIR]

  --check              run formal preflight without starting training (default)
  --run                run the authorized experiment
  --stop-after-shadow  stop after strict Shadow validation; never start ON
  --resume-on PAIR_DIR resume a validated qualification pair at the ON leg
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
if [[ "$STOP_AFTER_SHADOW" == "1" || -n "$RESUME_ON_DIR" ]]; then
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
NUM_ROLLOUT="${NUM_ROLLOUT:-15}"
EXPECTED_SAMPLES_PER_PARTITION="${EXPECTED_SAMPLES_PER_PARTITION:-64}"
EXPECTED_ENGINES="${EXPECTED_ENGINES:-2}"
MAX_STALENESS="${MAX_STALENESS:-2}"
HEADLINE_LO="${HEADLINE_LO:-5}"
HEADLINE_HI="${HEADLINE_HI:-14}"
PARTITION_ADMISSION_MIN="${PARTITION_ADMISSION_MIN:-4}"
PARTITION_ADMISSION_MAX="${PARTITION_ADMISSION_MAX:-8}"
PARTITION_ADMISSION_SLACK="${PARTITION_ADMISSION_SLACK:-2}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-5400}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
EXP_DIR="${EXP_DIR:-$MODEL_DIR}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
if [[ "$STOP_AFTER_SHADOW" == "1" ]]; then
    qualification_contract=(
        "NUM_ROLLOUT:$NUM_ROLLOUT:15"
        "EXPECTED_SAMPLES_PER_PARTITION:$EXPECTED_SAMPLES_PER_PARTITION:64"
        "EXPECTED_ENGINES:$EXPECTED_ENGINES:2"
        "MAX_STALENESS:$MAX_STALENESS:2"
        "HEADLINE_LO:$HEADLINE_LO:5"
        "HEADLINE_HI:$HEADLINE_HI:14"
        "PARTITION_ADMISSION_MIN:$PARTITION_ADMISSION_MIN:4"
        "PARTITION_ADMISSION_MAX:$PARTITION_ADMISSION_MAX:8"
        "PARTITION_ADMISSION_SLACK:$PARTITION_ADMISSION_SLACK:2"
        "RUN_TIMEOUT_S:$RUN_TIMEOUT_S:5400"
    )
    for contract_entry in "${qualification_contract[@]}"; do
        IFS=: read -r contract_name contract_actual contract_expected <<< "$contract_entry"
        if [[ "$contract_actual" != "$contract_expected" ]]; then
            echo "Shadow qualification requires $contract_name=$contract_expected; got $contract_actual" >&2
            exit 4
        fi
    done
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

export MODEL_DIR DATA_DIR EXP_DIR
export TASK22_PYTHON="$PYTHON_BIN"

bash "$PREFLIGHT" --formal

if [[ "$MODE" == "--check" ]]; then
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
if [[ -z "$RESUME_ON_DIR" && -e "$PAIR_DIR" ]]; then
    echo "Pair artifact path already exists: $PAIR_DIR" >&2
    exit 4
fi

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
else
    mkdir -p "$PAIR_DIR"
    printf '%s\n' "$GIT_COMMIT" > "$PAIR_DIR/GIT_COMMIT"
    git status --porcelain=v1 --untracked-files=all > "$PAIR_DIR/GIT_STATUS"
    git diff --binary > "$PAIR_DIR/WORKTREE.patch"
fi

sampler_pid=""
cleanup() {
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" >/dev/null 2>&1 || true
        wait "$sampler_pid" >/dev/null 2>&1 || true
        sampler_pid=""
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

write_contract() {
    local mode="$1"
    local output_path="$2"
    "$PYTHON_BIN" - "$output_path" "$mode" "$REPO" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

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


fingerprinted_files = (
    "relax/engine/router/placement.py",
    "relax/engine/router/router.py",
    "scripts/task22/analyze_rollout_observability.py",
    "scripts/task22/compare_admission_pair.py",
    "scripts/task22/preflight_admission.sh",
    "scripts/task22/prepare_rollout_observability.sh",
    "scripts/task22/run_admission_matched_ab.sh",
    "scripts/task22/simulate_request_placement.py",
    "scripts/task22/sglang_rid_only_request_logging.patch",
    "scripts/task22/sglang_rollout_observability.patch",
    "scripts/task22/validate_admission_run.py",
    "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh",
)
source_sha256 = {
    path: sha256_file(os.path.join(repo, path))
    for path in fingerprinted_files
}
pip_freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"],
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
contract = {
    "schema_version": 1,
    "git_commit": os.environ["GIT_COMMIT"],
    "admission_mode": mode,
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
    "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
    "nccl_socket_ifname": os.environ["NCCL_SOCKET_IFNAME"],
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "dependency_versions": {
        name: package_version(name)
        for name in ("ray", "sglang", "torch", "transformers")
    },
    "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
    "gpu_fingerprint": gpu_fingerprint,
    "source_sha256": source_sha256,
}
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(contract, output, indent=2, sort_keys=True)
    output.write("\n")
PY
}

run_one() {
    local mode="$1"
    local run_dir="$PAIR_DIR/$mode"
    local prepared_contract="${2:-}"
    mkdir -p "$run_dir/observability" "$run_dir/timeline" "$run_dir/logs"
    printf '%s\n' RUNNING > "$run_dir/STATUS"
    date '+%Y-%m-%dT%H:%M:%S%z' > "$run_dir/STARTED_AT"
    if [[ -n "$prepared_contract" ]]; then
        mv "$prepared_contract" "$run_dir/run_contract.json"
    else
        write_contract "$mode" "$run_dir/run_contract.json"
    fi

    (
        while true; do
            date '+%Y-%m-%dT%H:%M:%S.%N%z'
            nvidia-smi \
                --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw \
                --format=csv,noheader
            sleep 1
        done
    ) > "$run_dir/logs/nvidia_smi_1s.csv" 2>&1 &
    sampler_pid=$!

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
    TIMELINE_DUMP_DIR="$run_dir/timeline" \
    DRIVER_LOG_PATH="$run_dir/driver.log" \
    timeout --signal=TERM --kill-after=180 "$RUN_TIMEOUT_S" bash "$WRAPPER"
    local run_rc=$?
    set -e

    kill "$sampler_pid" >/dev/null 2>&1 || true
    wait "$sampler_pid" >/dev/null 2>&1 || true
    sampler_pid=""
    ray stop --force >/dev/null 2>&1 || true
    printf '%s\n' "$run_rc" > "$run_dir/EXIT_CODE"
    date '+%Y-%m-%dT%H:%M:%S%z' > "$run_dir/FINISHED_AT"
    if [[ "$run_rc" -eq 0 ]]; then
        printf '%s\n' SUCCEEDED > "$run_dir/STATUS"
    else
        printf 'FAILED(%s)\n' "$run_rc" > "$run_dir/STATUS"
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
        --output-json "$run_dir/validation.json" \
        > "$run_dir/validation.stdout.json"
    local validator_rc=$?
    set -e
    printf '%s\n' "$validator_rc" > "$run_dir/VALIDATOR_EXIT_CODE"
    if [[ "$run_rc" -ne 0 || "$validator_rc" -ne 0 ]]; then
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

prepare_on_contract() {
    local candidate="$PAIR_DIR/.on_contract_candidate.json"
    write_contract on "$candidate"
    if ! "$PYTHON_BIN" - "$PAIR_DIR/shadow/run_contract.json" "$candidate" <<'PY'
import json
import sys

shadow_path, candidate_path = sys.argv[1:]
with open(shadow_path, encoding="utf-8") as source:
    shadow = json.load(source)
with open(candidate_path, encoding="utf-8") as source:
    candidate = json.load(source)
keys = set(shadow) | set(candidate)
diffs = {key for key in keys if shadow.get(key) != candidate.get(key)}
if diffs != {"admission_mode"}:
    print(
        "ON contract differs from validated Shadow outside admission_mode: "
        + ", ".join(sorted(diffs)),
        file=sys.stderr,
    )
    raise SystemExit(4)
if shadow.get("admission_mode") != "shadow" or candidate.get("admission_mode") != "on":
    print("Invalid admission mode order in resumed pair", file=sys.stderr)
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
export TRAIN_SEED ROLLOUT_SEED RUN_TIMEOUT_S MODEL_DIR DATA_DIR EXP_DIR
export NCCL_NVLS_ENABLE NCCL_SOCKET_IFNAME
export RUN_SCOPE
export REQUEST_PLACEMENT_MODE=off
export REQUEST_PLACEMENT_POLICY=least_predicted_work
export USE_SLIME_ROUTER=0

if [[ -z "$RESUME_ON_DIR" ]]; then
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

on_contract="$(prepare_on_contract)" || {
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
