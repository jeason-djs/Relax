# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rollout.sync_intent import (
    ADAPTIVE_OVERSAMPLING_GROUPS_ENV,
    SYNC_INTENT_TTL_ENV,
    SyncIntentController,
    SyncIntentSnapshot,
    mark_work_origin,
    plan_adaptive_window_fetch,
    plan_debt_early_flush,
    plan_intent_guard_fetch,
    resolve_partition_request_priority,
)


def test_sync_intent_identity_is_idempotent_and_monotonic() -> None:
    controller = SyncIntentController()

    first = controller.begin(sync_id=7, actor_rollout_id=6)
    assert controller.begin(sync_id=7, actor_rollout_id=6) == first
    assert controller.begin(sync_id=6, actor_rollout_id=5) == first
    with pytest.raises(ValueError, match="identity mismatch"):
        controller.begin(sync_id=7, actor_rollout_id=5)

    newer = controller.begin(sync_id=8, actor_rollout_id=7)
    assert newer.sync_id == 8
    assert controller.end(7) == newer
    assert not controller.end(8).active


def test_sync_intent_expires_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNC_INTENT_TTL_ENV, "0.001")
    timestamps = iter((100.0, 102.0))
    monkeypatch.setattr("relax.engine.rollout.sync_intent.time.monotonic", lambda: next(timestamps))
    controller = SyncIntentController()
    controller.begin(sync_id=3, actor_rollout_id=2)

    snapshot = controller.snapshot()

    assert not snapshot.active
    assert snapshot.expired


def test_intent_guard_limits_next_debt_rollout_then_parks_fresh() -> None:
    snapshot = SyncIntentSnapshot(active=True, sync_id=6, actor_rollout_id=5)

    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=5,
            old_debt_groups=4,
            completed_debt_groups=0,
            inflight_debt_groups=0,
            default_fetch_groups=12,
        )
        == 12
    )
    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=6,
            old_debt_groups=4,
            completed_debt_groups=0,
            inflight_debt_groups=0,
            default_fetch_groups=12,
        )
        == 4
    )
    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=6,
            old_debt_groups=4,
            completed_debt_groups=0,
            inflight_debt_groups=4,
            default_fetch_groups=12,
        )
        == 0
    )


def test_intent_guard_preserves_overlap_without_debt() -> None:
    snapshot = SyncIntentSnapshot(active=True, sync_id=6, actor_rollout_id=5)

    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=6,
            old_debt_groups=0,
            completed_debt_groups=0,
            inflight_debt_groups=0,
            default_fetch_groups=8,
        )
        == 8
    )


def test_intent_guard_refills_failed_debt_group() -> None:
    snapshot = SyncIntentSnapshot(active=True, sync_id=6, actor_rollout_id=5)

    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=6,
            old_debt_groups=4,
            completed_debt_groups=2,
            inflight_debt_groups=1,
            default_fetch_groups=12,
        )
        == 1
    )


def test_adaptive_window_is_total_resident_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTIVE_OVERSAMPLING_GROUPS_ENV, "16")

    assert (
        plan_adaptive_window_fetch(
            cumulative_submitted_groups=0,
            baseline_fetch_groups=8,
            commit_target_groups=8,
        )
        == 16
    )
    assert (
        plan_adaptive_window_fetch(
            cumulative_submitted_groups=6,
            baseline_fetch_groups=12,
            commit_target_groups=12,
        )
        == 10
    )
    assert (
        plan_adaptive_window_fetch(
            cumulative_submitted_groups=16,
            baseline_fetch_groups=8,
            commit_target_groups=8,
        )
        == 0
    )


def test_mark_work_origin_marks_buffer_first_debt() -> None:
    groups = [
        [SimpleNamespace(metadata={})],
        [SimpleNamespace(metadata={})],
        [SimpleNamespace(metadata={})],
    ]

    mark_work_origin(groups, old_debt_groups=2)

    assert [group[0].metadata["work_origin"] for group in groups] == ["old_debt", "old_debt", "fresh"]


def test_debt_early_flush_requires_active_intent_and_full_debt() -> None:
    active = SyncIntentSnapshot(active=True, sync_id=4, actor_rollout_id=3)
    inactive = SyncIntentSnapshot(active=False)

    assert (
        plan_debt_early_flush(
            snapshot=active,
            old_debt_groups=3,
            committed_debt_groups=0,
            total_completed_groups=3,
            staged_groups=3,
        )
        == 3
    )
    assert (
        plan_debt_early_flush(
            snapshot=active,
            old_debt_groups=3,
            committed_debt_groups=0,
            total_completed_groups=2,
            staged_groups=2,
        )
        == 0
    )
    assert (
        plan_debt_early_flush(
            snapshot=inactive,
            old_debt_groups=3,
            committed_debt_groups=0,
            total_completed_groups=3,
            staged_groups=3,
        )
        == 0
    )


@pytest.mark.parametrize(
    ("enabled", "origin", "expected"),
    (
        (False, "old_debt", None),
        (True, "old_debt", 1),
        (True, "fresh", 0),
    ),
)
def test_partition_priority_is_opt_in(enabled: bool, origin: str, expected: int | None) -> None:
    args = SimpleNamespace(sglang_enable_priority_scheduling=enabled)
    sample = SimpleNamespace(metadata={"work_origin": origin})

    assert resolve_partition_request_priority(args, sample) == expected
