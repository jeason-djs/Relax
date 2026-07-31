#!/usr/bin/env python3
"""Race-resistant input manifests and exclusive output creation for Task 22."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _exclusive_json(path: Path, payload: Any) -> None:
    absolute = Path(os.path.abspath(path))
    parent_fd = _open_directory(absolute.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW
    try:
        fd = os.open(absolute.name, flags, 0o600, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"output is not a regular file: {absolute}")
            data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _open_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, os.O_RDONLY | DIRECTORY | NOFOLLOW)
    for component in absolute.parts[1:]:
        try:
            child = os.open(
                component,
                os.O_RDONLY | DIRECTORY | NOFOLLOW,
                dir_fd=descriptor,
            )
        except OSError as exc:
            os.close(descriptor)
            raise ValueError(f"unsafe directory path component: {absolute}") from exc
        os.close(descriptor)
        descriptor = child
    return descriptor


def _stable_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _walk(root: Path) -> list[dict[str, Any]]:
    root = Path(os.path.abspath(root))
    root_fd = _open_directory(root)
    records: list[dict[str, Any]] = []

    def visit(directory_fd: int, relative: str) -> None:
        before = os.fstat(directory_fd)
        names = sorted(os.listdir(directory_fd))
        for name in names:
            child_relative = f"{relative}/{name}" if relative else name
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                raise ValueError(f"input symlink is forbidden: {root / child_relative}")
            if stat.S_ISDIR(entry.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | DIRECTORY | NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if _stable_identity(opened) != _stable_identity(entry):
                        raise ValueError(f"directory changed while opening: {root / child_relative}")
                    records.append(
                        {
                            "path": child_relative,
                            "type": "directory",
                            "mode": stat.S_IMODE(opened.st_mode),
                            "dev": opened.st_dev,
                            "ino": opened.st_ino,
                        }
                    )
                    visit(child_fd, child_relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError(f"non-regular input is forbidden: {root / child_relative}")
            fd = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=directory_fd)
            try:
                opened_before = os.fstat(fd)
                if _stable_identity(opened_before) != _stable_identity(entry):
                    raise ValueError(f"file changed while opening: {root / child_relative}")
                digest = hashlib.sha256()
                while chunk := os.read(fd, 1024 * 1024):
                    digest.update(chunk)
                opened_after = os.fstat(fd)
                if _stable_identity(opened_before) != _stable_identity(opened_after):
                    raise ValueError(f"file changed while hashing: {root / child_relative}")
                records.append(
                    {
                        "path": child_relative,
                        "type": "file",
                        "mode": stat.S_IMODE(opened_after.st_mode),
                        "dev": opened_after.st_dev,
                        "ino": opened_after.st_ino,
                        "size": opened_after.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            finally:
                os.close(fd)
        after = os.fstat(directory_fd)
        if _stable_identity(before) != _stable_identity(after):
            raise ValueError(f"directory changed while scanning: {root / relative}")

    try:
        visit(root_fd, "")
        if not any(record["type"] == "file" for record in records):
            raise ValueError(f"input manifest root has no files: {root}")
        root_info = os.fstat(root_fd)
        return [
            {
                "root": str(root),
                "root_mode": stat.S_IMODE(root_info.st_mode),
                "root_dev": root_info.st_dev,
                "root_ino": root_info.st_ino,
                "entries": records,
            }
        ]
    finally:
        os.close(root_fd)


def build_manifest(roots: list[Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": 2, "roots": []}
    for root in roots:
        payload["roots"].extend(_walk(root))
    return payload


def manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _portable_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    scanned = _walk(root)[0]["entries"]
    entries: list[dict[str, Any]] = [{"path": prefix, "type": "directory"}]
    for record in scanned:
        portable = {
            "path": f"{prefix}/{record['path']}",
            "type": record["type"],
        }
        if record["type"] == "file":
            portable.update(size=record["size"], sha256=record["sha256"])
        entries.append(portable)
    return entries


def build_snapshot_manifest(model_root: Path, data_file: Path) -> dict[str, Any]:
    data_file = Path(os.path.abspath(data_file))
    parent_fd = _open_directory(data_file.parent)
    try:
        entry = os.stat(data_file.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError(f"snapshot data input is not a regular file: {data_file}")
        fd = os.open(data_file.name, os.O_RDONLY | NOFOLLOW, dir_fd=parent_fd)
        try:
            before = os.fstat(fd)
            if _stable_identity(before) != _stable_identity(entry):
                raise ValueError(f"file changed while opening: {data_file}")
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(fd)
            if _stable_identity(before) != _stable_identity(after):
                raise ValueError(f"file changed while hashing: {data_file}")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    entries = _portable_entries(model_root, "Qwen3-4B")
    entries.extend(
        [
            {"path": "dapo-math-17k", "type": "directory"},
            {
                "path": "dapo-math-17k/dapo-math-17k.jsonl",
                "type": "file",
                "size": after.st_size,
                "sha256": digest.hexdigest(),
            },
        ]
    )
    return {"schema_version": 3, "entries": entries}


def _copy_with_reflink(source: Path, destination: Path) -> None:
    result = subprocess.run(
        ["cp", "--reflink=auto", "--", str(source), str(destination)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _copy_model_tree(source: Path, destination: Path) -> None:
    scanned = _walk(source)[0]
    records = scanned["entries"]
    destination.mkdir(mode=scanned["root_mode"])
    destination.chmod(scanned["root_mode"])
    for record in records:
        target = destination / record["path"]
        if record["type"] == "directory":
            target.mkdir(mode=record["mode"])
            target.chmod(record["mode"])
        else:
            _copy_with_reflink(source / record["path"], target)
            target.chmod(record["mode"])


def _make_read_only(root: Path, *, include_root: bool = True) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"snapshot symlink is forbidden: {path}")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"snapshot symlink is forbidden: {path}")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        if include_root or current_path != root:
            current_path.chmod(stat.S_IMODE(current_path.stat().st_mode) & ~0o222)


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    snapshot = Path(os.path.abspath(snapshot))
    expected = load_json_nofollow(snapshot / "MANIFEST.json")
    actual = build_snapshot_manifest(
        snapshot / "Qwen3-4B",
        snapshot / "dapo-math-17k/dapo-math-17k.jsonl",
    )
    digest = manifest_sha256(actual)
    if actual != expected or snapshot.name != digest:
        raise ValueError("snapshot manifest verification failed")
    marker = snapshot / "COMPLETE"
    if marker.is_symlink():
        raise ValueError("snapshot completion marker verification failed")
    marker_value = marker.read_text(encoding="utf-8").strip()
    if marker_value != digest:
        raise ValueError("snapshot completion marker verification failed")
    for current, directories, files in os.walk(snapshot, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"snapshot symlink is forbidden: {path}")
            if stat.S_IMODE(info.st_mode) & 0o222:
                raise ValueError(f"snapshot path is writable: {path}")
    if snapshot.stat().st_mode & 0o222:
        raise ValueError(f"snapshot path is writable: {snapshot}")
    return actual


def create_snapshot(snapshot_root: Path, model_root: Path, data_file: Path) -> Path:
    snapshot_root = Path(os.path.abspath(snapshot_root))
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = _open_directory(snapshot_root.parent)
    os.close(parent_fd)
    snapshot_root.mkdir(mode=0o700, exist_ok=True)
    snapshot_root.chmod(0o700)
    root_fd = _open_directory(snapshot_root)
    os.close(root_fd)
    source_manifest = build_snapshot_manifest(model_root, data_file)
    digest = manifest_sha256(source_manifest)
    destination = snapshot_root / digest
    if destination.exists():
        verify_snapshot(destination)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=snapshot_root))
    try:
        _copy_model_tree(model_root, staging / "Qwen3-4B")
        data_destination = staging / "dapo-math-17k"
        data_destination.mkdir(mode=0o700)
        _copy_with_reflink(data_file, data_destination / "dapo-math-17k.jsonl")
        (data_destination / "dapo-math-17k.jsonl").chmod(stat.S_IMODE(data_file.stat().st_mode))
        copied_manifest = build_snapshot_manifest(
            staging / "Qwen3-4B",
            data_destination / "dapo-math-17k.jsonl",
        )
        if copied_manifest != source_manifest:
            raise ValueError("source changed while creating snapshot")
        _exclusive_json(staging / "MANIFEST.json", copied_manifest)
        marker_fd = os.open(staging / "COMPLETE", os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
        try:
            os.write(marker_fd, f"{digest}\n".encode())
            os.fsync(marker_fd)
        finally:
            os.close(marker_fd)
        _make_read_only(staging, include_root=False)
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.exists():
                raise
            for current, directories, files in os.walk(staging):
                Path(current).chmod(0o700)
                for name in files:
                    (Path(current) / name).chmod(0o600)
            shutil.rmtree(staging)
        else:
            destination.chmod(stat.S_IMODE(destination.stat().st_mode) & ~0o222)
        verify_snapshot(destination)
        return destination
    except BaseException:
        if staging.exists():
            for current, directories, files in os.walk(staging):
                Path(current).chmod(0o700)
                for name in files:
                    (Path(current) / name).chmod(0o600)
            shutil.rmtree(staging)
        raise


def load_json_nofollow(path: Path) -> Any:
    absolute = Path(os.path.abspath(path))
    parent_fd = _open_directory(absolute.parent)
    try:
        fd = os.open(absolute.name, os.O_RDONLY | NOFOLLOW, dir_fd=parent_fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"input is not a regular file: {absolute}")
            chunks = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(fd)
            if _stable_identity(before) != _stable_identity(after):
                raise ValueError(f"file changed while reading: {absolute}")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    return json.loads(b"".join(chunks))


def attest_manifest(expected_path: Path, output_path: Path, roots: list[Path]) -> dict[str, Any]:
    expected = load_json_nofollow(expected_path)
    actual = build_manifest(roots)
    if actual != expected:
        raise ValueError("input manifest differs from expected manifest")
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256(actual),
        "manifest": actual,
    }
    _exclusive_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("roots", nargs="+", type=Path)
    attest_parser = subparsers.add_parser("attest")
    attest_parser.add_argument("--expected", type=Path, required=True)
    attest_parser.add_argument("--output", type=Path, required=True)
    attest_parser.add_argument("roots", nargs="+", type=Path)
    create_parser = subparsers.add_parser("snapshot-create")
    create_parser.add_argument("--snapshot-root", type=Path, required=True)
    create_parser.add_argument("--model-root", type=Path, required=True)
    create_parser.add_argument("--data-file", type=Path, required=True)
    verify_parser = subparsers.add_parser("snapshot-verify")
    verify_parser.add_argument("--snapshot", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "manifest":
        _exclusive_json(args.output, build_manifest(args.roots))
    elif args.command == "attest":
        attest_manifest(args.expected, args.output, args.roots)
    elif args.command == "snapshot-create":
        print(create_snapshot(args.snapshot_root, args.model_root, args.data_file))
    else:
        manifest = verify_snapshot(args.snapshot)
        if args.output:
            _exclusive_json(
                args.output,
                {
                    "schema_version": 1,
                    "manifest_sha256": manifest_sha256(manifest),
                    "manifest": manifest,
                },
            )
        print(args.snapshot)


if __name__ == "__main__":
    main()
