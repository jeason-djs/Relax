# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 22 synchronization-intent state and fresh-work admission policy."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


SYNC_INTENT_GUARD_ENV = "RELAX_TASK22_SYNC_INTENT_GUARD"
SYNC_INTENT_TTL_ENV = "RELAX_TASK22_SYNC_INTENT_TTL_SECONDS"
ADAPTIVE_OVERSAMPLING_GROUPS_ENV = "RELAX_TASK22_ADAPTIVE_OVERSAMPLING_GROUPS"
DEFAULT_ROLLOUT_REQUEST_PRIORITY = 0
OLD_DEBT_REQUEST_PRIORITY = 1


def sync_intent_guard_enabled() -> bool:
    return os.environ.get(SYNC_INTENT_GUARD_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def sync_intent_ttl_seconds() -> float:
    raw_value = os.environ.get(SYNC_INTENT_TTL_ENV, "600")
    try:
        ttl_seconds = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{SYNC_INTENT_TTL_ENV} must be a positive number, got {raw_value!r}") from error
    if ttl_seconds <= 0:
        raise ValueError(f"{SYNC_INTENT_TTL_ENV} must be positive, got {ttl_seconds}")
    return ttl_seconds


def adaptive_oversampling_groups() -> int | None:
    raw_value = os.environ.get(ADAPTIVE_OVERSAMPLING_GROUPS_ENV)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        groups = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{ADAPTIVE_OVERSAMPLING_GROUPS_ENV} must be a positive integer") from error
    if groups <= 0:
        raise ValueError(f"{ADAPTIVE_OVERSAMPLING_GROUPS_ENV} must be positive, got {groups}")
    return groups


@dataclass(frozen=True)
class SyncIntentSnapshot:
    active: bool
    sync_id: int | None = None
    actor_rollout_id: int | None = None
    started_at: float | None = None
    expired: bool = False


class SyncIntentController:
    """Process-local, cross-thread synchronization intent.

    ``RolloutManager`` receives control RPCs on the Ray actor event loop while
    the rollout function executes on Relax's long-lived generation thread.
    A small lock-protected controller keeps the handoff linearizable without
    binding the state to either asyncio loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sync_id: int | None = None
        self._actor_rollout_id: int | None = None
        self._started_at: float | None = None

    def begin(self, sync_id: int, actor_rollout_id: int) -> SyncIntentSnapshot:
        if sync_id < 0:
            raise ValueError(f"sync_id must be non-negative, got {sync_id}")
        if actor_rollout_id < 0:
            raise ValueError(f"actor_rollout_id must be non-negative, got {actor_rollout_id}")
        with self._lock:
            if self._sync_id is not None:
                if sync_id < self._sync_id:
                    return self._snapshot_locked()
                if sync_id == self._sync_id:
                    if actor_rollout_id != self._actor_rollout_id:
                        raise ValueError(
                            "sync intent identity mismatch: "
                            f"sync_id={sync_id}, existing_actor_rollout_id={self._actor_rollout_id}, "
                            f"requested_actor_rollout_id={actor_rollout_id}"
                        )
                    return self._snapshot_locked()
            self._sync_id = sync_id
            self._actor_rollout_id = actor_rollout_id
            self._started_at = time.monotonic()
            return self._snapshot_locked()

    def end(self, sync_id: int) -> SyncIntentSnapshot:
        with self._lock:
            if self._sync_id is None:
                return SyncIntentSnapshot(active=False)
            if sync_id < self._sync_id:
                return self._snapshot_locked()
            if sync_id > self._sync_id:
                raise ValueError(f"cannot end future sync intent {sync_id}; active sync_id={self._sync_id}")
            self._clear_locked()
            return SyncIntentSnapshot(active=False)

    def snapshot(self) -> SyncIntentSnapshot:
        with self._lock:
            if self._sync_id is None:
                return SyncIntentSnapshot(active=False)
            assert self._started_at is not None
            if time.monotonic() - self._started_at > sync_intent_ttl_seconds():
                self._clear_locked()
                return SyncIntentSnapshot(active=False, expired=True)
            return self._snapshot_locked()

    def reset(self) -> None:
        with self._lock:
            self._clear_locked()

    def wait_until_inactive(self, sync_id: int, *, poll_interval_seconds: float = 0.01) -> SyncIntentSnapshot:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        while True:
            snapshot = self.snapshot()
            if not snapshot.active or snapshot.sync_id != sync_id:
                return snapshot
            time.sleep(poll_interval_seconds)

    def _snapshot_locked(self) -> SyncIntentSnapshot:
        return SyncIntentSnapshot(
            active=self._sync_id is not None,
            sync_id=self._sync_id,
            actor_rollout_id=self._actor_rollout_id,
            started_at=self._started_at,
        )

    def _clear_locked(self) -> None:
        self._sync_id = None
        self._actor_rollout_id = None
        self._started_at = None


_SYNC_INTENT = SyncIntentController()


def begin_sync_intent(sync_id: int, actor_rollout_id: int) -> SyncIntentSnapshot:
    return _SYNC_INTENT.begin(sync_id, actor_rollout_id)


def end_sync_intent(sync_id: int) -> SyncIntentSnapshot:
    return _SYNC_INTENT.end(sync_id)


def get_sync_intent() -> SyncIntentSnapshot:
    return _SYNC_INTENT.snapshot()


def reset_sync_intent() -> None:
    _SYNC_INTENT.reset()


def wait_for_sync_intent_end(sync_id: int, *, poll_interval_seconds: float = 0.01) -> SyncIntentSnapshot:
    return _SYNC_INTENT.wait_until_inactive(sync_id, poll_interval_seconds=poll_interval_seconds)


def plan_intent_guard_fetch(
    *,
    snapshot: SyncIntentSnapshot,
    physical_rollout_id: int,
    old_debt_groups: int,
    completed_debt_groups: int,
    inflight_debt_groups: int,
    default_fetch_groups: int,
) -> int:
    """Return the next group fetch count under the intent guard.

    Only a debt-bearing physical rollout is constrained. Its first admission
    releases exactly the known old-debt groups. Once those groups close the
    Actor's target partition, fresh work resumes after publication.
    """

    if default_fetch_groups <= 0:
        raise ValueError(f"default_fetch_groups must be positive, got {default_fetch_groups}")
    if min(old_debt_groups, completed_debt_groups, inflight_debt_groups) < 0:
        raise ValueError("debt group counts must be non-negative")
    if not snapshot.active or old_debt_groups == 0:
        return default_fetch_groups
    if snapshot.actor_rollout_id is not None and physical_rollout_id <= snapshot.actor_rollout_id:
        return default_fetch_groups
    missing_debt_groups = max(old_debt_groups - completed_debt_groups - inflight_debt_groups, 0)
    if missing_debt_groups == 0:
        return 0
    return min(default_fetch_groups, missing_debt_groups)


def plan_adaptive_window_fetch(
    *,
    cumulative_submitted_groups: int,
    baseline_fetch_groups: int,
    commit_target_groups: int,
) -> int:
    """Size a total oversampling window without counting old debt twice."""

    if cumulative_submitted_groups < 0:
        raise ValueError("cumulative_submitted_groups must be non-negative")
    if baseline_fetch_groups <= 0 or commit_target_groups <= 0:
        raise ValueError("baseline_fetch_groups and commit_target_groups must be positive")
    configured_window = adaptive_oversampling_groups()
    if configured_window is None:
        return baseline_fetch_groups
    desired_window = max(configured_window, commit_target_groups)
    return max(desired_window - cumulative_submitted_groups, 0)


def plan_debt_early_flush(
    *,
    snapshot: SyncIntentSnapshot,
    old_debt_groups: int,
    committed_debt_groups: int,
    total_completed_groups: int,
    staged_groups: int,
) -> int:
    """Return how many staged groups can close the previous partition now."""

    if min(old_debt_groups, committed_debt_groups, total_completed_groups, staged_groups) < 0:
        raise ValueError("debt flush counts must be non-negative")
    remaining_debt = max(old_debt_groups - committed_debt_groups, 0)
    if not snapshot.active or remaining_debt == 0:
        return 0
    if total_completed_groups < old_debt_groups or staged_groups < remaining_debt:
        return 0
    return remaining_debt


def mark_work_origin(samples: list[list[object]], old_debt_groups: int) -> None:
    """Mark buffer-first debt groups for request priority routing."""

    for group_index, group in enumerate(samples):
        origin = "old_debt" if group_index < old_debt_groups else "fresh"
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                setattr(sample, "metadata", metadata)
            metadata["work_origin"] = origin


def resolve_partition_request_priority(args: object, sample: object) -> int | None:
    if not getattr(args, "sglang_enable_priority_scheduling", False):
        return None
    metadata = getattr(sample, "metadata", None)
    work_origin = metadata.get("work_origin") if isinstance(metadata, dict) else None
    if work_origin == "old_debt":
        return OLD_DEBT_REQUEST_PRIORITY
    return DEFAULT_ROLLOUT_REQUEST_PRIORITY
