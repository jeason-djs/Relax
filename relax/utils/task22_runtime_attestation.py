"""Task 22 runtime identity attestation for the Ray driver and worker."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.task22.input_guard import (
    _exclusive_json,
    build_manifest,
    build_snapshot_manifest,
    load_json_nofollow,
    manifest_sha256,
)


SGLANG_MODULES = (
    "sglang.srt.observability.req_time_stats",
    "sglang.srt.observability.scheduler_metrics_mixin",
    "sglang.srt.utils.request_logger",
    "sglang.srt.utils.scheduler_status_logger",
)
RUNTIME_FILE_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".patch", ".py", ".sh", ".toml", ".yaml", ".yml"}
)
RUNTIME_SOURCE_DIRS = ("relax", "scripts", "configs")
TOP_LEVEL_RUNTIME_NAMES = frozenset({"pyproject.toml", "setup.cfg", "setup.py"})
TOP_LEVEL_RUNTIME_PREFIXES = ("requirements",)
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "test",
        "tests",
        "tmp",
        "temp",
    }
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_env_sha256(value: str) -> str:
    parsed = json.loads(value)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _is_runtime_file(relative_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES or part.startswith(".") for part in relative_path.parts[:-1]):
        return False
    name = relative_path.name
    lowered = name.lower()
    if (
        lowered.startswith(("test_", "tmp_", "temp_"))
        or lowered.endswith(("~", ".bak", ".orig", ".rej", ".swp", ".tmp"))
    ):
        return False
    return relative_path.suffix.lower() in RUNTIME_FILE_SUFFIXES


def working_dir_runtime_files(working_dir: str | os.PathLike[str]) -> tuple[str, ...]:
    """Enumerate runtime-bearing files independently of the working directory path."""
    root = Path(working_dir).resolve()
    paths: set[Path] = set()
    for source_dir_name in RUNTIME_SOURCE_DIRS:
        source_dir = root / source_dir_name
        if source_dir.is_dir():
            paths.update(
                path
                for path in source_dir.rglob("*")
                if path.is_file() and _is_runtime_file(path.relative_to(root))
            )
    paths.update(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in TOP_LEVEL_RUNTIME_NAMES
            or (
                path.name.startswith(TOP_LEVEL_RUNTIME_PREFIXES)
                and path.suffix.lower() in {".in", ".txt"}
            )
        )
    )
    return tuple(sorted(path.relative_to(root).as_posix() for path in paths))


def working_dir_content_hashes(working_dir: str | os.PathLike[str]) -> dict[str, str]:
    root = Path(working_dir).resolve()
    return {name: sha256_file(root / name) for name in working_dir_runtime_files(root)}


def working_dir_content_sha256(working_dir: str | os.PathLike[str]) -> str:
    content = working_dir_content_hashes(working_dir)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _input_manifest_attestation(
    expected_path: str | None = None,
    roots_json: str | None = None,
) -> dict[str, Any]:
    expected = load_json_nofollow(
        Path(expected_path if expected_path is not None else os.environ["TASK22_INPUT_MANIFEST"])
    )
    roots = [
        Path(value)
        for value in json.loads(
            roots_json if roots_json is not None else os.environ["TASK22_INPUT_ROOTS_JSON"]
        )
    ]
    if expected.get("schema_version") == 3:
        if len(roots) != 2:
            raise RuntimeError("snapshot attestation requires model root and data file")
        actual = build_snapshot_manifest(roots[0], roots[1])
    else:
        actual = build_manifest(roots)
    if actual != expected:
        raise RuntimeError("model/data changed before training input-open attestation")
    return {"sha256": manifest_sha256(actual), "manifest": actual}


def collect_attestation(
    role: str,
    runtime_env_json: str | None = None,
    input_manifest_path: str | None = None,
    input_roots_json: str | None = None,
) -> dict[str, Any]:
    requested_python = os.environ["TASK22_PYTHON"]
    if not os.path.isabs(requested_python) or not os.access(requested_python, os.X_OK):
        raise RuntimeError("TASK22_PYTHON must be an executable absolute path")
    executable_realpath = os.path.realpath(sys.executable)
    launcher_target = os.path.realpath(requested_python)
    pip_freeze = subprocess.run(
        [requested_python, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    working_dir = os.path.realpath(os.getcwd())
    sglang_source_sha256 = {}
    for module_name in SGLANG_MODULES:
        module_path = os.path.realpath(importlib.import_module(module_name).__file__)
        sglang_source_sha256[module_path] = sha256_file(module_path)
    return {
        "schema_version": 3,
        "role": role,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "working_dir": working_dir,
        "working_dir_content_sha256": working_dir_content_sha256(working_dir),
        "runtime_env_json_sha256": runtime_env_sha256(
            runtime_env_json if runtime_env_json is not None else os.environ["RUNTIME_ENV_JSON"]
        ),
        "python": {
            "launch_path": requested_python,
            "executable_realpath": executable_realpath,
            "launcher_target_sha256": sha256_file(launcher_target),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
            "version": platform.python_version(),
        },
        "sglang_source_sha256": sglang_source_sha256,
        "input_manifest": _input_manifest_attestation(input_manifest_path, input_roots_json),
    }


def write_attestation(
    directory: str | os.PathLike[str],
    role: str,
    runtime_env_json: str | None = None,
    input_manifest_path: str | None = None,
    input_roots_json: str | None = None,
) -> str:
    output_dir = Path(directory)
    suffix = "driver" if role == "driver" else f"worker_{socket.gethostname()}_{os.getpid()}"
    output_path = output_dir / f"runtime_attestation_{suffix}.json"
    _exclusive_json(
        output_path,
        collect_attestation(
            role,
            runtime_env_json,
            input_manifest_path,
            input_roots_json,
        ),
    )
    return str(output_path)


def attestation_matches_contract(attestation: dict[str, Any], contract: dict[str, Any]) -> bool:
    # Ray unpacks the submitted working directory under a node-local absolute
    # path. Keep both paths in the evidence for diagnosis, but prove source
    # identity from the controlled-file content digest instead.
    return (
        attestation.get("working_dir_content_sha256")
        == contract.get("working_dir_content_sha256")
        and attestation.get("runtime_env_json_sha256") == contract.get("runtime_env_json_sha256")
        and attestation.get("python") == contract.get("training_python")
        and attestation.get("sglang_source_sha256") == contract.get("sglang_source_sha256")
        and attestation.get("input_manifest", {}).get("sha256")
        == contract.get("input_manifest_sha256")
    )
