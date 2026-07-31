# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import stat
from pathlib import Path

import pytest

from scripts.task22 import input_guard


def test_input_manifest_rejects_symlinks_and_does_not_escape_root(tmp_path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (root / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        input_guard.build_manifest([root])


def test_input_manifest_rejects_symlinked_directory(tmp_path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret", encoding="utf-8")
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        input_guard.build_manifest([root])


def test_input_manifest_rejects_symlinked_root(tmp_path) -> None:
    real_root = tmp_path / "real-input"
    real_root.mkdir()
    (real_root / "data").write_text("payload", encoding="utf-8")
    linked_root = tmp_path / "linked-input"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe directory"):
        input_guard.build_manifest([linked_root])


def test_input_manifest_output_is_exclusive(tmp_path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "data").write_text("payload", encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError):
        input_guard._exclusive_json(output, input_guard.build_manifest([root]))

    assert output.read_text(encoding="utf-8") == "do not replace"


def test_input_manifest_output_rejects_symlink_and_symlinked_parent(tmp_path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "data").write_text("payload", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("untouched", encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.symlink_to(outside)

    with pytest.raises(FileExistsError):
        input_guard._exclusive_json(output, input_guard.build_manifest([root]))
    assert outside.read_text(encoding="utf-8") == "untouched"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe directory"):
        input_guard._exclusive_json(
            linked_parent / "candidate.json",
            input_guard.build_manifest([root]),
        )
    assert not (real_parent / "candidate.json").exists()


def test_input_manifest_detects_file_change_while_hashing(tmp_path, monkeypatch) -> None:
    root = tmp_path / "input"
    root.mkdir()
    path = root / "large.bin"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    original_read = os.read
    changed = False

    def mutating_read(fd, size):
        nonlocal changed
        chunk = original_read(fd, size)
        if chunk and not changed:
            changed = True
            with path.open("ab") as output:
                output.write(b"changed")
        return chunk

    monkeypatch.setattr(input_guard.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed while hashing"):
        input_guard.build_manifest([root])


def test_input_manifest_detects_same_path_inode_replacement(tmp_path, monkeypatch) -> None:
    root = tmp_path / "input"
    root.mkdir()
    path = root / "data"
    path.write_text("payload", encoding="utf-8")
    original_read = os.read
    replaced = False

    def replacing_read(fd, size):
        nonlocal replaced
        chunk = original_read(fd, size)
        if chunk and not replaced:
            replaced = True
            path.unlink()
            path.write_text("payload", encoding="utf-8")
        return chunk

    monkeypatch.setattr(input_guard.os, "read", replacing_read)
    with pytest.raises(ValueError, match="changed while (hashing|scanning)"):
        input_guard.build_manifest([root])


def test_content_addressed_snapshot_copies_only_required_inputs_and_is_read_only(
    tmp_path,
) -> None:
    model = tmp_path / "source/Qwen3-4B"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    nested = model / "weights"
    nested.mkdir()
    (nested / "part.bin").write_bytes(b"weights")
    data_dir = tmp_path / "data/dapo-math-17k"
    data_dir.mkdir(parents=True)
    data = data_dir / "dapo-math-17k.jsonl"
    data.write_text('{"prompt":"fixture"}\n', encoding="utf-8")
    (data_dir / "not-required.jsonl").write_text("exclude me\n", encoding="utf-8")

    snapshot = input_guard.create_snapshot(tmp_path / "snapshots", model, data)
    manifest = input_guard.verify_snapshot(snapshot)

    assert snapshot.name == input_guard.manifest_sha256(manifest)
    assert (snapshot / "MANIFEST.json").is_file()
    assert (snapshot / "COMPLETE").read_text(encoding="utf-8").strip() == snapshot.name
    assert (snapshot / "Qwen3-4B/weights/part.bin").read_bytes() == b"weights"
    assert (snapshot / "dapo-math-17k/dapo-math-17k.jsonl").is_file()
    assert not (snapshot / "dapo-math-17k/not-required.jsonl").exists()
    assert not list(snapshot.parent.glob(".*.tmp-*"))
    for current, directories, files in os.walk(snapshot):
        for name in [*directories, *files]:
            assert not (Path(current) / name).stat().st_mode & stat.S_IWUSR

    same_snapshot = input_guard.create_snapshot(tmp_path / "snapshots", model, data)
    assert same_snapshot == snapshot

    for current, _, files in os.walk(snapshot):
        Path(current).chmod(0o700)
        for name in files:
            (Path(current) / name).chmod(0o600)


def test_snapshot_creation_rejects_symlink_in_model_tree(tmp_path) -> None:
    model = tmp_path / "source/Qwen3-4B"
    model.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (model / "linked").symlink_to(outside)
    data = tmp_path / "data/dapo-math-17k/dapo-math-17k.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"prompt":"fixture"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        input_guard.create_snapshot(tmp_path / "snapshots", model, data)


def test_snapshot_preserves_private_source_permissions_without_write_bits(tmp_path) -> None:
    model = tmp_path / "source/Qwen3-4B"
    private_dir = model / "private"
    private_dir.mkdir(parents=True)
    private_dir.chmod(0o700)
    private_file = private_dir / "weights.bin"
    private_file.write_bytes(b"weights")
    private_file.chmod(0o600)
    executable_file = model / "tokenizer"
    executable_file.write_text("#!/bin/sh\n", encoding="utf-8")
    executable_file.chmod(0o700)
    data = tmp_path / "data/dapo-math-17k/dapo-math-17k.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"prompt":"fixture"}\n', encoding="utf-8")
    data.chmod(0o600)

    snapshot_root = tmp_path / "snapshots"
    snapshot = input_guard.create_snapshot(snapshot_root, model, data)

    assert stat.S_IMODE(snapshot_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((snapshot / "Qwen3-4B/private").stat().st_mode) == 0o500
    assert stat.S_IMODE((snapshot / "Qwen3-4B/private/weights.bin").stat().st_mode) == 0o400
    assert stat.S_IMODE((snapshot / "Qwen3-4B/tokenizer").stat().st_mode) == 0o500
    assert stat.S_IMODE(
        (snapshot / "dapo-math-17k/dapo-math-17k.jsonl").stat().st_mode
    ) == 0o400
