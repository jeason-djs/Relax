# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "task22" / "run_admission_matched_ab.sh"
PRODUCTION_FINGERPRINTED_FILES = (
    "relax/engine/rollout/admission.py",
    "relax/engine/rollout/request_observability.py",
    "relax/engine/rollout/sglang_rollout.py",
    "relax/utils/metrics/service.py",
    "relax/utils/metrics/timeline_trace.py",
)
FINGERPRINTED_FILES = (
    *PRODUCTION_FINGERPRINTED_FILES,
    "relax/engine/router/placement.py",
    "relax/engine/router/router.py",
    "scripts/task22/analyze_rollout_observability.py",
    "scripts/task22/compare_admission_pair.py",
    "scripts/task22/monitor_admission_run.py",
    "scripts/task22/preflight_admission.sh",
    "scripts/task22/prepare_rollout_observability.sh",
    "scripts/task22/run_admission_matched_ab.sh",
    "scripts/task22/simulate_request_placement.py",
    "scripts/task22/sglang_rid_only_request_logging.patch",
    "scripts/task22/sglang_rollout_observability.patch",
    "scripts/task22/validate_admission_run.py",
    "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh",
)


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    fake_bin = tmp_path / "bin"
    repo.mkdir()
    fake_bin.mkdir()

    for relative_path in FINGERPRINTED_FILES:
        path = repo / relative_path
        if relative_path == "scripts/task22/run_admission_matched_ab.sh":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNNER, path)
        else:
            _write(path, "# runner contract fixture\n")

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
  printf '[{"name":"train","ph":"X","ts":1,"dur":1,"pid":1,"tid":1}]\n' \
    > "$TIMELINE_DUMP_DIR/timeline_step_${step}.json"
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
args, _ = parser.parse_known_args()
payload = {"verdict": "PASS", "failures": {}, "quality_metrics": []}
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
        "TASK22_PYTHON": sys.executable,
        "TASK22_AUTHORIZE_GPU_RUN": "1",
        "MODEL_DIR": str(tmp_path / "model"),
        "DATA_DIR": str(tmp_path / "data"),
        "RUN_ROOT": str(run_root),
        "TASK22_RUN_STAMP": "fixture",
        "RUN_TIMEOUT_S": "5400",
        "TASK22_MONITOR_POLL_INTERVAL": "0.01",
    }
    return repo, run_root, env


def test_runner_stops_after_shadow_and_resumes_same_pair_for_on(tmp_path) -> None:
    repo, run_root, env = _build_fake_repo(tmp_path)
    runner = repo / "scripts/task22/run_admission_matched_ab.sh"

    shadow = subprocess.run(
        ["bash", str(runner), "--run", "--stop-after-shadow"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
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

    resume_env = {**env, "TASK22_AUTHORIZE_ON_RUN": "1"}
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

    completed = subprocess.run(
        ["bash", str(runner), "--run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    pair_dir = next(run_root.glob("admission_matched_*_fixture"))
    assert "verdict=PASS" in completed.stdout
    assert "scope=matched_pair" in completed.stdout
    assert (pair_dir / "shadow" / "validation.json").is_file()
    assert (pair_dir / "on" / "validation.json").is_file()
    assert (pair_dir / "PAIR_VALID").read_text(encoding="utf-8").strip() == "PASS"


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


def test_runner_status_reports_monitor_failure_when_training_exits_zero(tmp_path) -> None:
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
    assert (pair_dir / "shadow" / "EXIT_CODE").read_text().strip() == "0"
    assert (pair_dir / "shadow" / "ONLINE_MONITOR_EXIT_CODE").read_text().strip() == "4"
    assert (
        pair_dir / "shadow" / "STATUS"
    ).read_text().strip() == "FAILED(training=0,monitor=4)"
