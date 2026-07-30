# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "task22" / "run_admission_matched_ab.sh"
FINGERPRINTED_FILES = (
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
        repo / "scripts/task22/preflight_admission.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' TASK22_PREFLIGHT=PASS\n",
        executable=True,
    )
    _write(
        repo / "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' TASK22_WRAPPER=PASS\n",
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
