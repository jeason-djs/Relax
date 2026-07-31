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
PREFLIGHT = REPO_ROOT / "scripts" / "task22" / "preflight_admission.sh"
SYNC_SOURCES = (
    "relax/backends/megatron/actor.py",
    "relax/backends/megatron/weight_update/update_weight_from_tensor.py",
    "relax/components/rollout.py",
    "relax/engine/rollout/sglang_rollout.py",
    "relax/utils/utils.py",
)
SHELL_SOURCES = (
    "scripts/task22/preflight_admission.sh",
    "scripts/task22/run_admission_matched_ab.sh",
    "scripts/training/text/run-qwen3-4B-4xgpu-hybrid-async-task22.sh",
)


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _commit(repo: Path, message: str) -> None:
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
            message,
        ],
        check=True,
    )


def _build_preflight_repo(tmp_path: Path, *, target_in_parent: bool) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    fake_python = tmp_path / "task22-python"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for path in SHELL_SOURCES:
        destination = repo / path
        if path.endswith("preflight_admission.sh"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PREFLIGHT, destination)
        else:
            _write(destination, "#!/usr/bin/env bash\nexit 0\n", executable=True)
    for path in SYNC_SOURCES:
        if target_in_parent or path != SYNC_SOURCES[0]:
            _write(repo / path, "# baseline\n")
    _write(repo / "scripts/task22/input_guard.py", "# fixture\n")
    _write(
        repo / "scripts/task22/approved_sync_baseline.json",
        json.dumps(
            {
                "schema_version": 1,
                "approved_revision": "fixture-v1",
                "pattern": r"cuda\.synchronize|dist\.barrier|ray\.get",
                "sources": {path: 0 for path in SYNC_SOURCES},
            }
        ),
    )
    _write(
        fake_python,
        """#!/usr/bin/env python3
import os
import subprocess
import sys

args = sys.argv[1:]
if args[:1] == ["-c"] and "os.path.realpath(sys.executable)" in args[1]:
    print(os.path.realpath(sys.argv[0]))
    raise SystemExit(0)
if args[:2] == ["-m", "py_compile"]:
    raise SystemExit(0)
if args[:2] == ["-m", "pytest"]:
    handoff = os.environ.get("HANDOFF_LAUNCH_PATH")
    if handoff and os.path.abspath(sys.argv[0]) != handoff:
        print("No module named pytest", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)
if args[:2] == ["-c", "import sglang"]:
    raise SystemExit(1)
raise SystemExit(subprocess.run([os.environ["REAL_PYTHON"], *args]).returncode)
""",
        executable=True,
    )
    _commit(repo, "parent baseline")
    _write(repo / SYNC_SOURCES[0], "ray.get(value)\n")
    _commit(repo, "current task change")
    env = {
        **os.environ,
        "TASK22_PYTHON": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return repo, env


def test_preflight_compares_synchronization_primitives_to_approved_manifest(tmp_path) -> None:
    repo, env = _build_preflight_repo(tmp_path, target_in_parent=True)

    result = subprocess.run(
        ["bash", "scripts/task22/preflight_admission.sh", "--local"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 4
    assert "increase synchronization primitives" in result.stderr
    assert "approved baseline 0 -> 1" in result.stderr


def test_preflight_uses_manifest_even_when_source_did_not_exist_in_parent(tmp_path) -> None:
    repo, env = _build_preflight_repo(tmp_path, target_in_parent=False)

    result = subprocess.run(
        ["bash", "scripts/task22/preflight_admission.sh", "--local"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 4
    assert "approved baseline 0 -> 1" in result.stderr
    assert "HEAD^" not in result.stderr


def test_preflight_preserves_absolute_symlink_launcher_for_pytest(tmp_path) -> None:
    repo, env = _build_preflight_repo(tmp_path, target_in_parent=True)
    manifest_path = repo / "scripts/task22/approved_sync_baseline.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][SYNC_SOURCES[0]] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    launcher = tmp_path / "handoff-venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(env["TASK22_PYTHON"]))

    result = subprocess.run(
        ["bash", "scripts/task22/preflight_admission.sh", "--local"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "TASK22_PYTHON": str(launcher),
            "HANDOFF_LAUNCH_PATH": str(launcher),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "verdict=PASS_LOCAL" in result.stdout


def test_preflight_rejects_relative_python_launcher(tmp_path) -> None:
    repo, env = _build_preflight_repo(tmp_path, target_in_parent=True)

    result = subprocess.run(
        ["bash", "scripts/task22/preflight_admission.sh", "--local"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**env, "TASK22_PYTHON": "../task22-python"},
    )

    assert result.returncode == 4
    assert "must be an executable absolute path" in result.stderr


def test_formal_preflight_runs_metrics_service_test_only_after_formal_dependency_check() -> None:
    source = PREFLIGHT.read_text(encoding="utf-8")
    metrics_test = '"$PYTHON_BIN" -m pytest -q tests/utils/test_metrics_service.py'
    formal_import = '"$PYTHON_BIN" -c \'import ray, sglang\''

    assert source.count(metrics_test) == 1
    assert source.index(metrics_test) > source.index(formal_import)
    assert source.index(metrics_test) > source.index('if [[ "$MODE" == "local" ]]')


@pytest.mark.parametrize(
    ("extra_env", "gpu_lines", "message"),
    (
        ({"NUM_GPUS": "3"}, 4, "Ray declaration requires NUM_GPUS=4"),
        (
            {"NUM_GPUS": "4", "CUDA_VISIBLE_DEVICES": "0,1,2"},
            4,
            "CUDA_VISIBLE_DEVICES to be empty or 4 unique devices",
        ),
        ({"NUM_GPUS": "4"}, 3, "requires exactly 4 physical GPUs"),
    ),
)
def test_formal_preflight_rejects_non_four_gpu_contract_before_launch(
    tmp_path, extra_env, gpu_lines, message
) -> None:
    repo, env = _build_preflight_repo(tmp_path, target_in_parent=True)
    manifest_path = repo / "scripts/task22/approved_sync_baseline.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][SYNC_SOURCES[0]] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fake_bin = tmp_path / "formal-bin"
    fake_bin.mkdir()
    _write(fake_bin / "ray", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(fake_bin / "timeout", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(fake_bin / "sha256sum", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n" + "".join(f"printf '%s\\n' {index}\n" for index in range(gpu_lines)),
        executable=True,
    )

    result = subprocess.run(
        ["bash", "scripts/task22/preflight_admission.sh", "--formal"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={
            **env,
            "PATH": f"{fake_bin}:{env['PATH']}",
            **extra_env,
        },
    )

    assert result.returncode == 4
    assert message in result.stderr
