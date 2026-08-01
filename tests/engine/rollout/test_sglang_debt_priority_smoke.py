# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from scripts.task22.smoke_sglang_debt_priority import evaluate


def _record(name: str, dispatched: float, completed: float, tokens: int) -> dict:
    return {
        "name": name,
        "dispatched_at": dispatched,
        "completed_at": completed,
        "max_new_tokens": tokens,
        "output_tokens": tokens,
    }


def test_debt_priority_smoke_requires_waiting_queue_overtake_without_preemption() -> None:
    result = evaluate(
        [
            _record("blocker", 0.0, 3.0, 2048),
            _record("fresh", 1.0, 5.0, 64),
            _record("old_debt", 2.0, 4.0, 64),
        ]
    )

    assert result["verdict"] == "PASS"
    assert result["completion_order"] == ["blocker", "old_debt", "fresh"]


def test_debt_priority_smoke_rejects_fcfs_completion() -> None:
    result = evaluate(
        [
            _record("blocker", 0.0, 3.0, 2048),
            _record("fresh", 1.0, 4.0, 64),
            _record("old_debt", 2.0, 5.0, 64),
        ]
    )

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["queued_old_debt_overtakes_fresh"]
