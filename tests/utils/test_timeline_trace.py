# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType


def _load_timeline_trace_module():
    module_path = Path(__file__).resolve().parents[2] / "relax" / "utils" / "metrics" / "timeline_trace.py"
    timer_stub = ModuleType("relax.utils.timer")
    timer_stub.TimelineEvent = object
    original_timer = sys.modules.get("relax.utils.timer")
    sys.modules["relax.utils.timer"] = timer_stub
    try:
        spec = importlib.util.spec_from_file_location("timeline_trace_under_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original_timer is None:
            sys.modules.pop("relax.utils.timer", None)
        else:
            sys.modules["relax.utils.timer"] = original_timer


def test_timeline_dump_quota_counts_unique_steps_and_returns_actual_status(tmp_path) -> None:
    timeline_trace = _load_timeline_trace_module()
    adapter = timeline_trace.TimelineTraceAdapter(str(tmp_path), max_dump=2)
    adapter.add_event_dicts(
        [{"name": "train", "ph": "X", "ts": 1, "dur": 1, "pid": 1, "tid": 1}]
    )

    assert adapter.dump(0)
    assert adapter.dump(0)
    assert adapter.dump(1)
    assert not adapter.dump(2)
    assert sorted(path.name for path in tmp_path.glob("timeline_step_*.json")) == [
        "timeline_step_0.json",
        "timeline_step_1.json",
    ]


def test_timeline_failed_refresh_preserves_previous_atomic_snapshot(tmp_path, monkeypatch) -> None:
    timeline_trace = _load_timeline_trace_module()
    adapter = timeline_trace.TimelineTraceAdapter(str(tmp_path), max_dump=1)
    adapter.add_event_dicts(
        [{"name": "train", "ph": "X", "ts": 1, "dur": 1, "pid": 1, "tid": 1}]
    )
    assert adapter.dump(0)
    path = tmp_path / "timeline_step_0.json"
    before = json.loads(path.read_text(encoding="utf-8"))

    def fail_dump(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(timeline_trace.json, "dump", fail_dump)
    try:
        adapter.dump(0)
    except OSError:
        pass

    assert json.loads(path.read_text(encoding="utf-8")) == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_timeline_memory_and_dump_are_safe_for_concurrent_writers(tmp_path) -> None:
    timeline_trace = _load_timeline_trace_module()
    adapter = timeline_trace.TimelineTraceAdapter(str(tmp_path), max_dump=1)

    def add(index):
        adapter.add_event_dicts(
            [{"name": str(index), "ph": "X", "ts": index, "dur": 1, "pid": 1, "tid": 1}]
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(100)))
        assert adapter.dump(0)

    payload = json.loads((tmp_path / "timeline_step_0.json").read_text(encoding="utf-8"))
    assert len(payload) == 100
    assert adapter.get_event_count() == 100


def test_timeline_dump_does_not_hold_memory_lock_during_io(tmp_path, monkeypatch) -> None:
    timeline_trace = _load_timeline_trace_module()
    adapter = timeline_trace.TimelineTraceAdapter(str(tmp_path), max_dump=1)
    adapter.add_event_dicts([{"name": "first", "ph": "X", "ts": 1}])
    entered_io = threading.Event()
    release_io = threading.Event()
    original_dump = timeline_trace.json.dump

    def blocking_dump(*args, **kwargs):
        entered_io.set()
        assert release_io.wait(timeout=5)
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(timeline_trace.json, "dump", blocking_dump)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dump_future = executor.submit(adapter.dump, 0)
        assert entered_io.wait(timeout=5)
        add_future = executor.submit(
            adapter.add_event_dicts,
            [{"name": "concurrent", "ph": "X", "ts": 2}],
        )
        add_future.result(timeout=1)
        release_io.set()
        assert dump_future.result(timeout=5)

    assert adapter.get_event_count() == 2


def test_timeline_failed_dump_rolls_back_reservation_and_fsyncs_directory(tmp_path, monkeypatch) -> None:
    timeline_trace = _load_timeline_trace_module()
    adapter = timeline_trace.TimelineTraceAdapter(str(tmp_path), max_dump=1)
    adapter.add_event_dicts([{"name": "first", "ph": "X", "ts": 1}])
    original_replace = timeline_trace.os.replace

    monkeypatch.setattr(
        timeline_trace.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected replace failure")),
    )
    try:
        adapter.dump(0)
    except OSError:
        pass

    fsync_calls = []
    monkeypatch.setattr(timeline_trace.os, "replace", original_replace)
    original_fsync = timeline_trace.os.fsync

    def recording_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(timeline_trace.os, "fsync", recording_fsync)
    assert adapter.dump(0)
    assert len(fsync_calls) == 2
