# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
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
