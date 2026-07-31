# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from relax.utils.task22_runtime_attestation import (
    attestation_matches_contract,
    merge_runtime_env,
    runtime_source_root,
    working_dir_content_hashes,
    working_dir_content_sha256,
    working_dir_runtime_files,
)
from scripts.task22.monitor_admission_run import _scan
from tests.engine.rollout.test_admission_run_validator import _build_valid_run, _validate


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_source_root_tracks_loaded_package_instead_of_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    assert runtime_source_root() == REPO_ROOT
RUNNER = REPO_ROOT / "scripts" / "task22" / "run_admission_matched_ab.sh"
INPUT_GUARD = REPO_ROOT / "scripts" / "task22" / "input_guard.py"
RUNTIME_ATTESTATION = REPO_ROOT / "relax" / "utils" / "task22_runtime_attestation.py"
GPU_SAMPLER = REPO_ROOT / "scripts" / "task22" / "sample_gpu_state.py"
PROCESS_DEADLINE = REPO_ROOT / "scripts" / "task22" / "enforce_process_deadline.py"
LOCAL_ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint" / "local.sh"
TASK22_TRAINING_ENTRYPOINT = (
    REPO_ROOT / "scripts" / "training" / "text" / "run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
)
PRODUCTION_FINGERPRINTED_FILES = (
    "relax/components/rollout.py",
    "relax/engine/rollout/admission.py",
    "relax/engine/rollout/request_observability.py",
    "relax/engine/rollout/sglang_rollout.py",
    "relax/utils/metrics/service.py",
    "relax/utils/metrics/timeline_trace.py",
)


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_real_local_wrapper_only_stops_explicit_job_pid_and_keeps_monitor(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write(fake_bin / "ray", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    monitor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    owned = subprocess.Popen(["sleep", "60"])
    try:
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "NUM_GPUS": "0",
            "RELAX_LOCAL_CLEANUP_PIDS": str(owned.pid),
        }
        result = subprocess.run(
            ["bash", "-c", f'source "{LOCAL_ENTRYPOINT}"; kill -0 {monitor.pid}'],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, result.stderr
        assert monitor.poll() is None
        assert owned.wait(timeout=5) == -15
    finally:
        for process in (monitor, owned):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_real_task22_entrypoint_passes_input_guard_env_in_runtime_env_and_command(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "ray-argv"
    _write(
        fake_bin / "ray",
        '#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$TASK22_TEST_RAY_ARGV"\n',
        executable=True,
    )
    manifest = str(tmp_path / "snapshot/MANIFEST.json")
    roots_json = json.dumps([str(tmp_path / "snapshot/Qwen3-4B"), str(tmp_path / "snapshot/data.jsonl")])
    runtime_env = {
        "env_vars": {
            "TASK22_INPUT_MANIFEST": manifest,
            "TASK22_INPUT_ROOTS_JSON": roots_json,
        }
    }
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RELAX_ENTRYPOINT_MODE": "test",
        "MODEL_CONFIG_DIR": str(REPO_ROOT / "scripts/models"),
        "TASK22_PYTHON": os.path.realpath(sys.executable),
        "TASK22_INPUT_MANIFEST": manifest,
        "TASK22_INPUT_ROOTS_JSON": roots_json,
        "TASK22_TEST_RAY_ARGV": str(capture),
        "RUNTIME_ENV_JSON": json.dumps(runtime_env),
        "MODEL_DIR": str(tmp_path / "snapshot"),
        "DATA_DIR": str(tmp_path / "snapshot"),
        "EXP_DIR": str(tmp_path),
        "PARTITION_ADMISSION_MODE": "off",
        "REQUEST_OBSERVABILITY_DIR": "",
        "TIMELINE_DUMP_DIR": str(tmp_path / "timeline"),
        "DRIVER_LOG_PATH": str(tmp_path / "driver.log"),
        "TRAIN_SEED": "1234",
        "ROLLOUT_SEED": "42",
        "MAX_STALENESS": "2",
    }

    result = subprocess.run(
        ["bash", str(TASK22_TRAINING_ENTRYPOINT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    argv = capture.read_bytes().decode().split("\0")[:-1]
    runtime_arg = next(value for value in argv if value.startswith("--runtime-env-json="))
    submitted_runtime_env = json.loads(runtime_arg.split("=", 1)[1])
    assert submitted_runtime_env["env_vars"]["TASK22_INPUT_MANIFEST"] == manifest
    assert submitted_runtime_env["env_vars"]["TASK22_INPUT_ROOTS_JSON"] == roots_json
    assert "TASK22_RAY_JOB_RUNTIME_ENV_APPLIED=1" in argv
    assert f"TASK22_INPUT_MANIFEST={manifest}" in argv
    assert f"TASK22_INPUT_ROOTS_JSON={roots_json}" in argv
    assert str(os.path.realpath(sys.executable)) in argv
    assert ["-m", "relax.entrypoints.train"] == argv[
        argv.index(str(os.path.realpath(sys.executable))) + 1 : argv.index(str(os.path.realpath(sys.executable))) + 3
    ]


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    fake_bin = tmp_path / "bin"
    repo.mkdir()
    fake_bin.mkdir()
    _write(repo / "relax/__init__.py", "")
    _write(repo / "relax/utils/__init__.py", "")

    for relative_path in (
        *PRODUCTION_FINGERPRINTED_FILES,
        "relax/utils/task22_runtime_attestation.py",
        "scripts/task22/enforce_process_deadline.py",
        "scripts/task22/monitor_admission_health.py",
        "scripts/task22/sample_gpu_state.py",
        "scripts/task22/run_admission_matched_ab.sh",
        "scripts/task22/input_guard.py",
    ):
        path = repo / relative_path
        if relative_path == "scripts/task22/run_admission_matched_ab.sh":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNNER, path)
        elif relative_path == "scripts/task22/input_guard.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(INPUT_GUARD, path)
        elif relative_path == "relax/utils/task22_runtime_attestation.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNTIME_ATTESTATION, path)
        elif relative_path == "scripts/task22/sample_gpu_state.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(GPU_SAMPLER, path)
        elif relative_path == "scripts/task22/monitor_admission_health.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative_path, path)
        elif relative_path == "scripts/task22/enforce_process_deadline.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROCESS_DEADLINE, path)
        else:
            _write(path, "# runner contract fixture\n")
    _write(repo / "pyproject.toml", "[tool.pytest.ini_options]\n")

    for relative_path in (
        "sglang/__init__.py",
        "sglang/srt/__init__.py",
        "sglang/srt/observability/__init__.py",
        "sglang/srt/observability/req_time_stats.py",
        "sglang/srt/observability/scheduler_metrics_mixin.py",
        "sglang/srt/utils/__init__.py",
        "sglang/srt/utils/request_logger.py",
        "sglang/srt/utils/scheduler_status_logger.py",
    ):
        _write(repo / relative_path, "# fake imported sglang source\n")

    _write(
        repo / "scripts/task22/monitor_admission_run.py",
        "#!/usr/bin/env python3\nraise SystemExit(0)\n",
        executable=True,
    )
    _write(
        repo / "scripts/task22/preflight_admission.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' TASK22_PREFLIGHT=PASS\n",
        executable=True,
    )
    _write(
        repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${TASK22_TEST_REPLACE_SOURCES_DURING_RUN:-0}" == "1" ]]; then
  printf '%s\n' '{"model_type":"source-replaced"}' > "$TASK22_TEST_SOURCE_MODEL"
  printf '%s\n' '{"prompt":"source-replaced"}' > "$TASK22_TEST_SOURCE_DATA"
fi
IFS= read -r snapshot_model < "$MODEL_DIR/Qwen3-4B/config.json"
IFS= read -r snapshot_data < "$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl"
printf 'TASK22_SNAPSHOT_MODEL=%s\n' "$snapshot_model" >> "$DRIVER_LOG_PATH"
printf 'TASK22_SNAPSHOT_DATA=%s\n' "$snapshot_data" >> "$DRIVER_LOG_PATH"
printf '%s\n' '{"outcome":"committed","attempt_id":"fixture"}' \
  > "$REQUEST_OBSERVABILITY_DIR/request_lifecycle_rollout_0.jsonl"
printf '%s\n' '{"record_type":"attempt","outcome":"committed","attempt_id":"fixture"}' \
  > "$REQUEST_OBSERVABILITY_DIR/admission_ledger_rollout_0.jsonl"
for phase in pause flush transfer continue; do
  printf 'TASK22_EVENT phase=%s sync_id=bootstrap t_begin=0 t_end=1 dur=1\n' "$phase" \
    >> "$DRIVER_LOG_PATH"
done
for sync_id in $(seq 1 15); do
  for phase in gate pause flush transfer continue; do
    printf 'TASK22_EVENT phase=%s sync_id=%s t_begin=0 t_end=1 dur=1\n' "$phase" "$sync_id" \
      >> "$DRIVER_LOG_PATH"
  done
done
for step in $(seq 5 14); do
  if [[ -n "$TIMELINE_DUMP_DIR" ]]; then
    printf '[{"name":"train","ph":"X","ts":1,"dur":1,"pid":1,"tid":1}]\n' \
      > "$TIMELINE_DUMP_DIR/timeline_step_${step}.json"
  fi
  printf 'perf %s: {"perf/step_time": 1.0}\n' "$step" >> "$DRIVER_LOG_PATH"
done
printf '%s\n' TASK22_WRAPPER=PASS
""",
        executable=True,
    )
    validator = """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output-json", required=True)
parser.add_argument("--require-resume", action="store_true")
args, _ = parser.parse_known_args()
if not args.require_resume:
    raise SystemExit("--require-resume is mandatory")
payload = {
    "verdict": "PASS",
    "failures": {},
    "quality_metrics": [],
    "require_resume": args.require_resume,
}
with open(args.output_json, "w", encoding="utf-8") as output:
    json.dump(payload, output)
print(json.dumps(payload))
"""
    _write(repo / "scripts/task22/validate_admission_run.py", validator, executable=True)
    comparator = """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output-json", required=True)
args, _ = parser.parse_known_args()
payload = {"verdict": "PASS"}
with open(args.output_json, "w", encoding="utf-8") as output:
    json.dump(payload, output)
print(json.dumps(payload))
"""
    _write(repo / "scripts/task22/compare_admission_pair.py", comparator, executable=True)
    _write(
        fake_bin / "ray",
        "#!/usr/bin/env bash\nexit 0\n",
        executable=True,
    )
    _write(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
printf '%s\n' \
  '0, GPU-0, Fake GPU, 1.0' \
  '1, GPU-1, Fake GPU, 1.0' \
  '2, GPU-2, Fake GPU, 1.0' \
  '3, GPU-3, Fake GPU, 1.0'
""",
        executable=True,
    )
    _write(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -euo pipefail
while [[ "${1:-}" == --* ]]; do
    shift
done
shift
exec "$@"
""",
        executable=True,
    )
    _write(
        fake_bin / "setsid",
        """#!/usr/bin/env python3
import os
import sys

os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
""",
        executable=True,
    )
    sha256sum = """#!/usr/bin/env python3
import hashlib
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


if "--check" in sys.argv:
    manifest = pathlib.Path(sys.argv[-1])
    valid = True
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = digest(manifest.parent / name)
        if actual != expected:
            valid = False
            print(f"{name}: FAILED")
        else:
            print(f"{name}: OK")
    raise SystemExit(0 if valid else 1)

for raw_path in sys.argv[1:]:
    print(f"{digest(raw_path)}  {raw_path}")
"""
    _write(fake_bin / "sha256sum", sha256sum, executable=True)

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _write(tmp_path / "model/Qwen3-4B/config.json", '{"model_type":"fixture"}\n')
    _write(tmp_path / "data/dapo-math-17k/dapo-math-17k.jsonl", '{"prompt":"fixture"}\n')
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "runner fixture",
        ],
        check=True,
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHONPATH": f"{repo}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TASK22_PYTHON": os.path.realpath(sys.executable),
        "TASK22_AUTHORIZE_GPU_RUN": "1",
        "MODEL_DIR": str(tmp_path / "model"),
        "DATA_DIR": str(tmp_path / "data"),
        "EXP_DIR": str(tmp_path / "model"),
        "RUN_ROOT": str(run_root),
        "TASK22_RUN_STAMP": "fixture",
        "RUN_TIMEOUT_S": "5400",
        "NUM_GPUS": "4",
        "CUDA_VISIBLE_DEVICES": "",
        "TASK22_MONITOR_POLL_INTERVAL": "0.01",
    }
    return repo, run_root, env


def test_runner_fixture_passes_real_online_monitor_and_real_final_validator(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    contract_path = run_dir / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "schema_version": 6,
            "working_dir": str(tmp_path.resolve()),
            "runtime_env_json_sha256": "a" * 64,
            "task22_env_sha256": "e" * 64,
            "monitor_poll_interval_s": 1.0,
            "monitor_evidence_grace_s": 5.0,
            "monitor_no_progress_timeout_s": 600.0,
            "gpu_max_snapshot_age_s": 5.0,
            "gpu_max_snapshot_interval_s": 2.0,
            "monitor_timeout_s": 5700,
            "monitor_term_grace_s": 1.0,
            "training_term_timeout_s": 10.0,
            "num_gpus": 4,
            "cuda_visible_devices": "",
            "training_python": {
                "launch_path": str(Path(sys.executable).absolute()),
                "executable_realpath": os.path.realpath(sys.executable),
                "launcher_target_sha256": "b" * 64,
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "pip_freeze_sha256": "d" * 64,
                "version": "fixture",
            },
            "sglang_source_sha256": {"/fixture/sglang.py": "c" * 64},
        }
    )
    contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
    attestation_dir = run_dir / "runtime_attestation"
    attestation_dir.mkdir(exist_ok=True)
    common_attestation = {
        "working_dir": contract["working_dir"],
        "working_dir_content_sha256": contract["working_dir_content_sha256"],
        "runtime_env_json_sha256": contract["runtime_env_json_sha256"],
        "task22_env_sha256": contract["task22_env_sha256"],
        "python": contract["training_python"],
        "sglang_source_sha256": contract["sglang_source_sha256"],
        "input_manifest": {"sha256": contract["input_manifest_sha256"]},
    }
    for role, rank in (
        ("driver", None),
        ("actor", 0),
        ("actor", 1),
        ("rollout_engine", 0),
    ):
        suffix = role if rank is None else f"{role}_{rank}"
        (attestation_dir / f"runtime_attestation_{suffix}.json").write_text(
            json.dumps({**common_attestation, "role": role, "rank": rank}) + "\n",
            encoding="utf-8",
        )

    _scan(
        run_dir,
        expected_mode="shadow",
        expected_rollouts=1,
        expected_samples_per_partition=1,
        expected_engines=1,
        max_staleness=2,
        admission_min=4,
        admission_max=8,
        admission_slack=2,
        headline_lo=0,
        headline_hi=0,
        final=True,
        reported_lifecycle=set(),
        event_log=run_dir / "online_monitor.jsonl",
    )
    result = _validate(run_dir)

    assert result["verdict"] == "PASS", result["failures"]


@pytest.mark.parametrize(
    "field",
    (
        "launch_path",
        "executable_realpath",
        "launcher_target_sha256",
        "prefix",
        "base_prefix",
        "pip_freeze_sha256",
    ),
)
def test_driver_and_worker_python_identity_fields_must_match_contract(field) -> None:
    python_identity = {
        "launch_path": "/fixture/venv/bin/python",
        "executable_realpath": "/fixture/base/bin/python3",
        "launcher_target_sha256": "a" * 64,
        "prefix": "/fixture/venv",
        "base_prefix": "/fixture/base",
        "pip_freeze_sha256": "b" * 64,
        "version": "3.11.0",
    }
    contract = {
        "working_dir": "/fixture/repo",
        "working_dir_content_sha256": "c" * 64,
        "runtime_env_json_sha256": "d" * 64,
        "task22_env_sha256": "f" * 64,
        "training_python": python_identity,
        "sglang_source_sha256": {},
        "input_manifest_sha256": "e" * 64,
    }
    attestation = {
        "working_dir": contract["working_dir"],
        "working_dir_content_sha256": contract["working_dir_content_sha256"],
        "runtime_env_json_sha256": contract["runtime_env_json_sha256"],
        "task22_env_sha256": contract["task22_env_sha256"],
        "python": dict(python_identity),
        "sglang_source_sha256": {},
        "input_manifest": {"sha256": contract["input_manifest_sha256"]},
    }

    assert attestation_matches_contract(attestation, contract)
    attestation["python"][field] += "-worker-drift"
    assert not attestation_matches_contract(attestation, contract)


def test_runtime_attestation_compares_cwd_content_across_ray_unpack_paths(tmp_path) -> None:
    submitted_repo = tmp_path / "source/repo"
    ray_working_dir = tmp_path / "tmp/ray/session/working_dir"
    runtime_files = (
        "relax/components/rollout.py",
        "relax/engine/rollout/new_runtime.py",
        "scripts/task22/runner.sh",
        "scripts/task22/config.json",
        "pyproject.toml",
    )
    for relative_path in runtime_files:
        _write(submitted_repo / relative_path, f"controlled content: {relative_path}\n")
        _write(ray_working_dir / relative_path, f"controlled content: {relative_path}\n")

    content_sha256 = working_dir_content_sha256(submitted_repo)
    assert working_dir_content_sha256(ray_working_dir) == content_sha256
    contract = {
        "working_dir": str(submitted_repo.resolve()),
        "working_dir_content_sha256": content_sha256,
        "runtime_env_json_sha256": "d" * 64,
        "task22_env_sha256": "f" * 64,
        "training_python": {},
        "sglang_source_sha256": {},
        "input_manifest_sha256": "e" * 64,
    }
    attestation = {
        "working_dir": str(ray_working_dir.resolve()),
        "working_dir_content_sha256": working_dir_content_sha256(ray_working_dir),
        "runtime_env_json_sha256": contract["runtime_env_json_sha256"],
        "task22_env_sha256": contract["task22_env_sha256"],
        "python": {},
        "sglang_source_sha256": {},
        "input_manifest": {"sha256": contract["input_manifest_sha256"]},
    }

    assert attestation["working_dir"] != contract["working_dir"]
    assert attestation_matches_contract(attestation, contract)

    changed_file = ray_working_dir / runtime_files[0]
    changed_file.write_text("different unpacked content\n", encoding="utf-8")
    attestation["working_dir_content_sha256"] = working_dir_content_sha256(ray_working_dir)
    assert not attestation_matches_contract(attestation, contract)


def test_runtime_hash_rejects_rollout_component_drift(tmp_path) -> None:
    source = tmp_path / "source"
    worker = tmp_path / "worker"
    relative_path = "relax/components/rollout.py"
    _write(source / relative_path, "ROLLOUT_RUNTIME = 1\n")
    _write(worker / relative_path, "ROLLOUT_RUNTIME = 1\n")

    assert working_dir_content_sha256(source) == working_dir_content_sha256(worker)
    _write(worker / relative_path, "ROLLOUT_RUNTIME = 2\n")
    assert working_dir_content_sha256(source) != working_dir_content_sha256(worker)


def test_runtime_hash_rejects_any_new_runtime_python_drift(tmp_path) -> None:
    source = tmp_path / "source"
    worker = tmp_path / "worker"
    _write(source / "relax/new_runtime_feature.py", "VALUE = 'source'\n")
    _write(worker / "relax/new_runtime_feature.py", "VALUE = 'worker'\n")

    assert working_dir_runtime_files(source) == ("relax/new_runtime_feature.py",)
    assert working_dir_content_hashes(source) != working_dir_content_hashes(worker)
    contract = {"working_dir_content_sha256": working_dir_content_sha256(source)}
    attestation = {"working_dir_content_sha256": working_dir_content_sha256(worker)}
    assert not attestation_matches_contract(attestation, contract)


def test_runtime_env_merge_preserves_working_dir_and_overrides_role_env() -> None:
    original = {
        "working_dir": "/fixture/source",
        "pip": ["fixture-wheel"],
        "env_vars": {"SHARED": "base", "ROLE": "old"},
    }

    merged = merge_runtime_env(original, {"ROLE": "actor", "TASK22_PYTHON": "/venv/python"})

    assert merged == {
        "working_dir": "/fixture/source",
        "pip": ["fixture-wheel"],
        "env_vars": {
            "SHARED": "base",
            "ROLE": "actor",
            "TASK22_PYTHON": "/venv/python",
        },
    }
    assert original["env_vars"]["ROLE"] == "old"


def test_runtime_file_enumeration_includes_config_and_excludes_tests_and_temporary_files(
    tmp_path,
) -> None:
    included = (
        "relax/runtime.py",
        "scripts/task22/run.sh",
        "scripts/task22/change.patch",
        "scripts/task22/settings.json",
        "configs/env.yaml",
        "pyproject.toml",
        "requirements.txt",
    )
    excluded = (
        "relax/__pycache__/runtime.py",
        "relax/tests/helper.py",
        "relax/test_runtime.py",
        "scripts/task22/tmp_output.json",
        "scripts/task22/run.sh.bak",
    )
    for relative_path in (*included, *excluded):
        _write(tmp_path / relative_path, relative_path)

    assert working_dir_runtime_files(tmp_path) == tuple(sorted(included))


def test_runner_stops_after_shadow_and_resumes_same_pair_for_on(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    original_runtime_env = {
        "env_vars": {"PRESERVED_FROM_SHADOW": "yes"},
        "pip": ["shadow-only-wheel"],
    }
    shadow_env = {
        **env,
        "WORKING_DIR": str(repo),
        "RUNTIME_ENV_JSON": json.dumps(original_runtime_env),
        "TASK22_MONITOR_POLL_INTERVAL": "0.02",
        "TASK22_MONITOR_EVIDENCE_GRACE": "6",
        "TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S": "600",
        "TASK22_GPU_MAX_SNAPSHOT_AGE_S": "7",
        "TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S": "3",
        "TASK22_MONITOR_TIMEOUT_S": "5800",
        "TASK22_MONITOR_TERM_GRACE_S": "2",
        "TASK22_TRAINING_TERM_TIMEOUT_S": "11",
        "CUDA_VISIBLE_DEVICES": "3, 1,2,0",
    }

    shadow = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=shadow_env,
    )
    pair_dirs = list(run_root.glob("admission_matched_*_fixture"))
    assert len(pair_dirs) == 1
    pair_dir = pair_dirs[0]
    assert "verdict=SHADOW_PASS" in shadow.stdout
    assert (pair_dir / "shadow" / "validation.json").is_file()
    assert not (pair_dir / "on").exists()
    assert (pair_dir / "SHADOW_VALID").read_text(encoding="utf-8").strip() == "PASS"
    assert (
        pair_dir / "PAIR_STATUS"
    ).read_text(encoding="utf-8").strip() == "AWAITING_ON_AUTHORIZATION"
    assert (pair_dir / "ARTIFACT_SHA256SUMS").is_file()

    resume_env = {**shadow_env, "TASK22_AUTHORIZE_ON_RUN": "1"}
    for name in (
        "MODEL_DIR",
        "DATA_DIR",
        "EXP_DIR",
        "WORKING_DIR",
        "RUNTIME_ENV_JSON",
        "NUM_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "TASK22_MONITOR_POLL_INTERVAL",
        "TASK22_MONITOR_EVIDENCE_GRACE",
        "TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S",
        "TASK22_GPU_MAX_SNAPSHOT_AGE_S",
        "TASK22_GPU_MAX_SNAPSHOT_INTERVAL_S",
        "TASK22_MONITOR_TIMEOUT_S",
        "TASK22_MONITOR_TERM_GRACE_S",
        "TASK22_TRAINING_TERM_TIMEOUT_S",
    ):
        resume_env.pop(name, None)
    resumed = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=True,
        capture_output=True,
        text=True,
        env=resume_env,
    )

    assert "verdict=PASS" in resumed.stdout
    assert (pair_dir / "on" / "validation.json").is_file()
    assert (pair_dir / "PAIR_VALID").read_text(encoding="utf-8").strip() == "PASS"
    assert (pair_dir / "PAIR_STATUS").read_text(encoding="utf-8").strip() == "PASS"
    shadow_contract = json.loads(
        (pair_dir / "shadow" / "run_contract.json").read_text(encoding="utf-8")
    )
    on_contract = json.loads(
        (pair_dir / "on" / "run_contract.json").read_text(encoding="utf-8")
    )
    contract_diffs = {
        key
        for key in set(shadow_contract) | set(on_contract)
        if shadow_contract.get(key) != on_contract.get(key)
    }
    assert contract_diffs == {"admission_mode"}
    assert shadow_contract["request_placement_mode"] == "off"
    assert on_contract["request_placement_mode"] == "off"
    assert shadow_contract["monitor_no_progress_timeout_s"] == 600
    assert on_contract["monitor_no_progress_timeout_s"] == 600
    assert shadow_contract["monitor_poll_interval_s"] == 0.02
    assert shadow_contract["monitor_evidence_grace_s"] == 6
    assert shadow_contract["gpu_max_snapshot_age_s"] == 7
    assert shadow_contract["gpu_max_snapshot_interval_s"] == 3
    assert shadow_contract["monitor_timeout_s"] == 5800
    assert shadow_contract["monitor_term_grace_s"] == 2
    assert shadow_contract["training_term_timeout_s"] == 11
    assert shadow_contract["num_gpus"] == 4
    assert shadow_contract["cuda_visible_devices"] == "3,1,2,0"
    assert shadow_contract["working_dir"] == str(repo.resolve())
    assert shadow_contract["runtime_env_json"] == on_contract["runtime_env_json"]
    normalized_runtime_env = json.loads(shadow_contract["runtime_env_json"])
    assert normalized_runtime_env["pip"] == ["shadow-only-wheel"]
    assert normalized_runtime_env["env_vars"]["PRESERVED_FROM_SHADOW"] == "yes"
    assert normalized_runtime_env["working_dir"] == str(repo.resolve())
    assert (
        hashlib.sha256(shadow_contract["runtime_env_json"].encode()).hexdigest()
        == shadow_contract["runtime_env_json_sha256"]
    )


def test_runner_accepts_absolute_python_symlink_and_rejects_relative_path(tmp_path) -> None:
    repo, _, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    python_link = tmp_path / "python-link"
    python_link.symlink_to(os.path.realpath(sys.executable))

    accepted = subprocess.run(
        ["bash", str(runner), "--check"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_PYTHON": str(python_link)},
    )
    rejected = subprocess.run(
        ["bash", str(runner), "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_PYTHON": "python-link"},
    )

    assert accepted.returncode == 0, accepted.stderr
    assert "verdict=READY" in accepted.stdout
    assert rejected.returncode == 4
    assert "must be an executable absolute path" in rejected.stderr


def test_runner_resume_requires_separate_on_authorization(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert denied.returncode == 4
    assert "TASK22_AUTHORIZE_ON_RUN=1" in denied.stderr
    assert not (pair_dir / "on").exists()


def test_shadow_qualification_rejects_contract_drift_before_creating_artifacts(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    drifted_env = {**env, "PARTITION_ADMISSION_MIN": "3"}

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env=drifted_env,
    )

    assert denied.returncode == 4
    assert "requires PARTITION_ADMISSION_MIN=4" in denied.stderr
    assert not run_root.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"NUM_GPUS": "3"}, "requires NUM_GPUS=4"),
        (
            {"CUDA_VISIBLE_DEVICES": "0,1,2"},
            "CUDA_VISIBLE_DEVICES must be empty or identify exactly 4 unique devices",
        ),
        (
            {"CUDA_VISIBLE_DEVICES": "0,1,1,2"},
            "CUDA_VISIBLE_DEVICES must be empty or identify exactly 4 unique devices",
        ),
    ),
)
def test_runner_rejects_invalid_four_gpu_contract_before_creating_artifacts(
    tmp_path, override, message
) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, **override},
    )

    assert denied.returncode == 4
    assert message in denied.stderr
    assert not run_root.exists()


def test_runner_resume_rejects_explicit_supervision_environment_drift(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env={**env, "TASK22_GPU_MAX_SNAPSHOT_AGE_S": "7"},
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_AUTHORIZE_ON_RUN": "1",
            "TASK22_GPU_MAX_SNAPSHOT_AGE_S": "8",
        },
    )

    assert denied.returncode == 4
    assert "Resume environment drift for TASK22_GPU_MAX_SNAPSHOT_AGE_S" in denied.stderr
    assert not (pair_dir / "on").exists()


def test_runner_resume_rejects_tampered_shadow_artifacts(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    validation_path = pair_dir / "shadow" / "validation.json"
    validation_path.write_text('{"verdict": "FAIL"}\n', encoding="utf-8")

    resume_env = {**env, "TASK22_AUTHORIZE_ON_RUN": "1"}
    denied = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=resume_env,
    )

    assert denied.returncode == 4
    assert "checksum verification failed" in denied.stderr
    assert not (pair_dir / "on").exists()


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        ("model/Qwen3-4B/config.json", '{"model_type":"drifted"}\n'),
        ("data/dapo-math-17k/dapo-math-17k.jsonl", '{"prompt":"drifted"}\n'),
    ],
)
def test_runner_resume_uses_same_snapshot_after_source_path_content_is_replaced(
    tmp_path, relative_path, replacement
) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    shadow_contract = json.loads(
        (pair_dir / "shadow/run_contract.json").read_text(encoding="utf-8")
    )
    snapshot = Path(shadow_contract["model_dir"])
    assert shadow_contract["data_dir"] == str(snapshot)
    assert snapshot.parent == (tmp_path / "model/task22_input_snapshots").resolve()
    assert not snapshot.is_relative_to(pair_dir)
    (tmp_path / relative_path).write_text(replacement, encoding="utf-8")
    source_root = (tmp_path / relative_path).parent
    source_root.rename(source_root.with_name(f"{source_root.name}-replaced"))

    resumed = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=True,
        capture_output=True,
        text=True,
        env={**env, "TASK22_AUTHORIZE_ON_RUN": "1"},
    )

    assert "verdict=PASS" in resumed.stdout
    on_contract = json.loads((pair_dir / "on/run_contract.json").read_text(encoding="utf-8"))
    assert on_contract["model_dir"] == str(snapshot)
    assert on_contract["data_dir"] == str(snapshot)


def test_runner_reads_snapshot_when_sources_are_replaced_during_shadow(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    model_source = tmp_path / "model/Qwen3-4B/config.json"
    data_source = tmp_path / "data/dapo-math-17k/dapo-math-17k.jsonl"

    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_TEST_REPLACE_SOURCES_DURING_RUN": "1",
            "TASK22_TEST_SOURCE_MODEL": str(model_source),
            "TASK22_TEST_SOURCE_DATA": str(data_source),
        },
    )

    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    driver_log = (pair_dir / "shadow/driver.log").read_text(encoding="utf-8")
    assert "TASK22_SNAPSHOT_MODEL={\"model_type\":\"fixture\"}" in driver_log
    assert "TASK22_SNAPSHOT_DATA={\"prompt\":\"fixture\"}" in driver_log
    assert "source-replaced" in model_source.read_text(encoding="utf-8")
    assert "source-replaced" in data_source.read_text(encoding="utf-8")


def test_runner_resume_rejects_corrupted_snapshot(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    contract = json.loads((pair_dir / "shadow/run_contract.json").read_text(encoding="utf-8"))
    snapshot_file = Path(contract["model_dir"]) / "Qwen3-4B/config.json"
    snapshot_file.chmod(0o644)
    snapshot_file.write_text('{"model_type":"corrupted"}\n', encoding="utf-8")

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_AUTHORIZE_ON_RUN": "1"},
    )

    assert denied.returncode == 4
    assert "snapshot verification failed" in denied.stderr
    assert not (pair_dir / "on").exists()


@pytest.mark.parametrize("relative_path", PRODUCTION_FINGERPRINTED_FILES)
def test_runner_resume_rejects_production_source_drift_without_head_change(
    tmp_path, relative_path
) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    frozen_contract = json.loads(
        (pair_dir / "shadow" / "run_contract.json").read_text(encoding="utf-8")
    )
    assert relative_path in frozen_contract["source_sha256"]
    frozen_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    source_path = repo / relative_path
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "# drift after Shadow freeze\n",
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == frozen_head
    )

    resume_env = {**env, "TASK22_AUTHORIZE_ON_RUN": "1"}
    denied = subprocess.run(
        ["bash", str(runner), "--run", "--resume-on", str(pair_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=resume_env,
    )

    assert denied.returncode == 4
    assert "requires a clean git worktree" in denied.stderr
    assert not (pair_dir / "on").exists()


def test_runner_default_run_still_executes_complete_pair(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    handoff_venv = tmp_path / "handoff-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(handoff_venv)],
        check=True,
    )
    launch_path = handoff_venv / "bin" / "python"

    completed = subprocess.run(
        ["bash", str(runner), "--run"],
        check=True,
        capture_output=True,
        text=True,
        env={**env, "TASK22_PYTHON": str(launch_path)},
    )

    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert "verdict=PASS" in completed.stdout
    assert "scope=matched_pair" in completed.stdout
    assert (pair_dir / "shadow" / "validation.json").is_file()
    assert (pair_dir / "on" / "validation.json").is_file()
    assert (pair_dir / "PAIR_VALID").read_text(encoding="utf-8").strip() == "PASS"
    shadow_contract = json.loads(
        (pair_dir / "shadow" / "run_contract.json").read_text(encoding="utf-8")
    )
    python_contract = shadow_contract["training_python"]
    assert python_contract["launch_path"] == str(launch_path)
    assert python_contract["executable_realpath"] == os.path.realpath(launch_path)
    assert len(python_contract["launcher_target_sha256"]) == 64
    assert python_contract["prefix"] == str(handoff_venv)
    assert python_contract["base_prefix"] != python_contract["prefix"]
    assert len(python_contract["pip_freeze_sha256"]) == 64
    assert len(shadow_contract["sglang_source_sha256"]) == 4
    assert all(
        str(repo / "sglang") in path
        for path in shadow_contract["sglang_source_sha256"]
    )
    validation = json.loads(
        (pair_dir / "shadow" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["require_resume"] is True


def test_runner_on_first_clean_pair_keeps_only_admission_mode_as_contract_diff(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    completed = subprocess.run(
        ["bash", str(runner), "--run", "--on-first"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_AUTHORIZE_ON_RUN": "1",
            "TASK22_EVIDENCE_PROFILE": "clean_ab_v1",
            "TASK22_PAIR_COOLDOWN_S": "0",
        },
    )

    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert "order=on_then_shadow" in completed.stdout
    assert (pair_dir / "on" / "validation.json").is_file()
    assert (pair_dir / "shadow" / "validation.json").is_file()
    assert (pair_dir / "PAIR_VALID").read_text().strip() == "PASS"
    on_contract = json.loads((pair_dir / "on/run_contract.json").read_text())
    shadow_contract = json.loads((pair_dir / "shadow/run_contract.json").read_text())
    assert on_contract["evidence_profile"] == "clean_ab_v1"
    assert on_contract["flashinfer_cuda_arch_list"] == "12.0a"
    assert on_contract["gpu_sample_interval_s"] == 5
    assert on_contract["hard_failure_grace_s"] == 300
    assert on_contract["runtime_env_json"] == shadow_contract["runtime_env_json"]
    assert (
        json.loads(on_contract["runtime_env_json"])["env_vars"][
            "FLASHINFER_CUDA_ARCH_LIST"
        ]
        == "12.0a"
    )
    assert {
        key
        for key in set(on_contract) | set(shadow_contract)
        if on_contract.get(key) != shadow_contract.get(key)
    } == {"admission_mode"}
    assert not any((pair_dir / "on").glob("timeline/timeline_step_*.json"))
    assert not any((pair_dir / "shadow").glob("timeline/timeline_step_*.json"))


def test_runner_on_only_runs_no_shadow_leg(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    completed = subprocess.run(
        ["bash", str(runner), "--run", "--on-only"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_AUTHORIZE_ON_RUN": "1",
            "TASK22_EVIDENCE_PROFILE": "clean_ab_v1",
        },
    )

    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert "verdict=ON_PASS" in completed.stdout
    assert "scope=on_qualification" in completed.stdout
    assert (pair_dir / "on" / "validation.json").is_file()
    assert not (pair_dir / "shadow").exists()
    assert (pair_dir / "ON_VALID").read_text(encoding="utf-8").strip() == "PASS"
    assert (pair_dir / "PAIR_STATUS").read_text(encoding="utf-8").strip() == "ON_VALIDATED"


def test_runner_on_first_requires_explicit_on_authorization(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    denied = subprocess.run(
        ["bash", str(repo / "scripts/task22/run_admission_matched_ab.sh"), "--run", "--on-first"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_EVIDENCE_PROFILE": "clean_ab_v1"},
    )

    assert denied.returncode == 4
    assert "TASK22_AUTHORIZE_ON_RUN=1" in denied.stderr
    assert not run_root.exists()


def test_runner_isolates_formal_preflight_from_run_authorization_and_profile(tmp_path) -> None:
    repo, _, env = _build_fake_repo(tmp_path)
    preflight = repo / "scripts/task22/preflight_admission.sh"
    _write(
        preflight,
        """#!/usr/bin/env bash
set -euo pipefail
for name in \
  TASK22_AUTHORIZE_GPU_RUN \
  TASK22_AUTHORIZE_ON_RUN \
  TASK22_EVIDENCE_PROFILE \
  TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S \
  TASK22_HARD_FAILURE_GRACE_S; do
  if [[ -n "${!name+x}" ]]; then
    printf 'leaked=%s\n' "$name" >&2
    exit 4
  fi
done
printf '%s\n' TASK22_PREFLIGHT=PASS
""",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(preflight)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "preflight environment fixture",
        ],
        check=True,
    )

    checked = subprocess.run(
        ["bash", str(repo / "scripts/task22/run_admission_matched_ab.sh"), "--check"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_AUTHORIZE_ON_RUN": "1",
            "TASK22_EVIDENCE_PROFILE": "clean_ab_v1",
            "TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S": "1200",
            "TASK22_HARD_FAILURE_GRACE_S": "300",
        },
    )

    assert checked.returncode == 0, checked.stderr
    assert "verdict=READY" in checked.stdout
    assert "leaked=" not in checked.stderr


def test_runner_default_pair_rejects_qualification_contract_drift(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"

    denied = subprocess.run(
        ["bash", str(runner), "--run"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "HEADLINE_LO": "4"},
    )

    assert denied.returncode == 4
    assert "requires HEADLINE_LO=5" in denied.stderr
    assert not run_root.exists()


def test_runner_rejects_monitor_contract_drift_before_creating_artifacts(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"

    denied = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S": "300"},
    )

    assert denied.returncode == 4
    assert "requires TASK22_MONITOR_NO_PROGRESS_TIMEOUT_S=600" in denied.stderr
    assert not run_root.exists()


def test_runner_online_monitor_fails_closed_and_stops_training(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    monitor = repo / "scripts/task22/monitor_admission_run.py"
    wrapper = repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
    _write(
        monitor,
        """#!/usr/bin/env python3
import argparse
import json
import os
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--pid", required=True, type=int)
parser.add_argument("--process-group-id", required=True, type=int)
args, _ = parser.parse_known_args()
event_log = os.path.join(args.run_dir, "online_monitor.jsonl")
with open(event_log, "a", encoding="utf-8") as output:
    output.write(json.dumps({"event": "monitor_failed", "reason": "fixture_failure"}) + "\\n")
for _ in range(100):
    try:
        if os.getpgid(args.pid) == args.process_group_id:
            break
    except ProcessLookupError:
        pass
    time.sleep(0.01)
else:
    raise SystemExit("training process group was not ready")
os.killpg(args.process_group_id, signal.SIGTERM)
with open(event_log, "a", encoding="utf-8") as output:
    output.write(json.dumps({"event": "training_stop_requested"}) + "\\n")
raise SystemExit(4)
""",
        executable=True,
    )
    _write(
        wrapper,
        """#!/usr/bin/env bash
set -euo pipefail
sleep 10
printf '%s\n' TRAINING_COMPLETED > "$REQUEST_OBSERVABILITY_DIR/training_completed"
""",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(monitor), str(wrapper)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "broken monitor fixture",
        ],
        check=True,
    )

    completed = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )

    assert completed.returncode == 5
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert (pair_dir / "shadow" / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() == "4"
    assert (pair_dir / "shadow" / "EXIT_CODE").read_text().strip() != "0"
    status = (pair_dir / "shadow" / "STATUS").read_text().strip()
    assert status.startswith("FAILED(training=")
    assert "monitor=4" in status
    assert not (pair_dir / "shadow" / "observability" / "training_completed").exists()
    monitor_rows = [
        json.loads(line)
        for line in (pair_dir / "shadow" / "online_monitor.jsonl").read_text().splitlines()
    ]
    assert any(row["event"] == "monitor_failed" for row in monitor_rows)
    assert any(row["event"] == "training_stop_requested" for row in monitor_rows)


def test_runner_supervises_monitor_and_stops_long_training_when_monitor_crashes(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    monitor = repo / "scripts/task22/monitor_admission_run.py"
    wrapper = repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
    _write(
        monitor,
        "#!/usr/bin/env python3\nraise RuntimeError('unexpected monitor crash')\n",
        executable=True,
    )
    _write(
        wrapper,
        """#!/usr/bin/env bash
set -euo pipefail
sleep 10
printf '%s\n' TRAINING_COMPLETED > "$REQUEST_OBSERVABILITY_DIR/training_completed"
""",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(monitor), str(wrapper)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "crashing monitor fixture",
        ],
        check=True,
    )

    failed = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )

    assert failed.returncode == 5
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert (pair_dir / "shadow" / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() != "0"
    assert (pair_dir / "shadow" / "EXIT_CODE").read_text().strip() != "0"
    assert not (pair_dir / "shadow" / "observability" / "training_completed").exists()


def test_clean_runner_does_not_stop_training_when_health_monitor_crashes(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    monitor = repo / "scripts/task22/monitor_admission_health.py"
    wrapper = repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
    _write(
        monitor,
        "#!/usr/bin/env python3\nraise RuntimeError('unexpected monitor crash')\n",
        executable=True,
    )
    _write(
        wrapper,
        """#!/usr/bin/env bash
set -euo pipefail
sleep 0.2
printf '%s\n' TRAINING_COMPLETED > "$REQUEST_OBSERVABILITY_DIR/training_completed"
""",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(monitor), str(wrapper)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "crashing clean monitor fixture",
        ],
        check=True,
    )

    completed = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_EVIDENCE_PROFILE": "clean_ab_v1",
            "TASK22_PAIR_COOLDOWN_S": "0",
        },
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    shadow = pair_dir / "shadow"
    assert (shadow / "observability" / "training_completed").is_file()
    assert (shadow / "EXIT_CODE").read_text().strip() == "0"
    assert (shadow / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() != "0"
    assert (shadow / "STATUS").read_text().strip() == "SUCCEEDED"
    assert "MONITOR_DEGRADED" not in (shadow / "SUPERVISION_STATE").read_text()
    assert "health monitor degraded" in (shadow / "logs/online_monitor.log").read_text()


def test_runner_status_reports_monitor_failure_and_supervised_training_stop(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    monitor = repo / "scripts/task22/monitor_admission_run.py"
    _write(
        monitor,
        "#!/usr/bin/env python3\nraise SystemExit(4)\n",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(monitor)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "monitor failure fixture",
        ],
        check=True,
    )

    failed = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert failed.returncode == 5
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    training_rc = (pair_dir / "shadow" / "EXIT_CODE").read_text().strip()
    assert training_rc.lstrip("-").isdigit()
    assert (pair_dir / "shadow" / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() == "4"
    assert (
        pair_dir / "shadow" / "STATUS"
    ).read_text().strip() == f"FAILED(training={training_rc},monitor=4)"


def test_runner_hard_times_out_hung_monitor_and_stops_training(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"
    monitor = repo / "scripts/task22/monitor_admission_run.py"
    wrapper = repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh"
    _write(
        monitor,
        "#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n",
        executable=True,
    )
    _write(
        wrapper,
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 60\n",
        executable=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", str(monitor), str(wrapper)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task22 Test",
            "-c",
            "user.email=task22@example.invalid",
            "commit",
            "-qm",
            "hung monitor fixture",
        ],
        check=True,
    )

    failed = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_MONITOR_TIMEOUT_S": "1",
            "TASK22_MONITOR_TERM_GRACE_S": "0",
            "TASK22_TRAINING_TERM_TIMEOUT_S": "0",
        },
        timeout=5,
    )

    assert failed.returncode == 5
    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert (pair_dir / "shadow" / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() == "124"
    assert (pair_dir / "shadow" / "EXIT_CODE").read_text().strip() != "0"
    assert "monitor hard timeout" in (
        pair_dir / "shadow" / "logs" / "online_monitor.log"
    ).read_text()
