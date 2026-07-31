# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Partition-critical admission primitives for rollout generation.

The rollout data source remains the owner of groups that have not been
admitted. This module decides how many groups may be fetched and submitted
immediately; it does not introduce another durable queue.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Generic, TypeVar


class AdmissionMode(str, Enum):
    """Runtime behavior of the debt-aware admission controller."""

    OFF = "off"
    SHADOW = "shadow"
    ON = "on"


class UnrecoverableFinalBackfillError(RuntimeError):
    """The durable partition is incomplete but volatile debt state is gone."""


def require_final_backfill_deficit(*, rollout_id: int, deficit_groups: int) -> int:
    """Fail closed when a restarted process cannot reconstruct backfill debt."""

    if deficit_groups <= 0:
        raise UnrecoverableFinalBackfillError(
            "Unrecoverable final backfill: durable partition "
            f"train_{rollout_id - 1} is incomplete, but the in-memory deficit was lost "
            "and the data-system API cannot reconstruct the missing group count"
        )
    return deficit_groups


@dataclass(frozen=True)
class DebtAwareAdmissionConfig:
    """Bounded admission policy for one physical rollout."""

    mode: AdmissionMode = AdmissionMode.OFF
    min_inflight_groups: int = 1
    max_inflight_groups: int = 1
    slack_groups: int = 0

    def validate(self) -> None:
        if self.min_inflight_groups <= 0:
            raise ValueError("min_inflight_groups must be positive")
        if self.max_inflight_groups < self.min_inflight_groups:
            raise ValueError("max_inflight_groups must be >= min_inflight_groups")
        if self.slack_groups < 0:
            raise ValueError("slack_groups must be non-negative")


@dataclass(frozen=True)
class AdmissionDecision:
    """One admission decision and its instantaneous shadow counterfactual."""

    decision_id: str
    decision_sequence: int
    mode: AdmissionMode
    inflight_groups: int
    debt_remaining: int
    available_groups: int
    eager_admit_groups: int
    desired_inflight_groups: int
    bounded_admit_groups: int
    actual_admit_groups: int
    bypass_reason: str | None = None


def previous_partition_debt_remaining(*, previous_debt_groups: int, completed_groups: int) -> int:
    """Return successful group completions still owed to the previous partition."""

    for name, value in (
        ("previous_debt_groups", previous_debt_groups),
        ("completed_groups", completed_groups),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    return max(previous_debt_groups - min(completed_groups, previous_debt_groups), 0)


def previous_partition_release_remaining(
    *, previous_debt_groups: int, completed_groups: int, transfer_batch_groups: int
) -> int:
    """Compatibility wrapper for previous-partition logical debt.

    Transfer batching is a data-plane preference and must not delay the
    producer's ``is_last`` partition close.
    """

    if transfer_batch_groups <= 0:
        raise ValueError("transfer_batch_groups must be positive")
    return previous_partition_debt_remaining(
        previous_debt_groups=previous_debt_groups,
        completed_groups=completed_groups,
    )


def split_transfer_counts(batch_groups: int, remaining_previous_debt: int) -> tuple[int, int]:
    """Split a transfer batch between the previous and current partitions."""

    if batch_groups < 0:
        raise ValueError("batch_groups must be non-negative")
    if remaining_previous_debt < 0:
        raise ValueError("remaining_previous_debt must be non-negative")
    previous_groups = min(batch_groups, remaining_previous_debt)
    return previous_groups, batch_groups - previous_groups


GroupT = TypeVar("GroupT")


@dataclass(frozen=True)
class PartitionTransferBatch(Generic[GroupT]):
    """One partition-homogeneous transfer selected by the rollout planner."""

    partition_kind: str
    groups: tuple[GroupT, ...]
    flush_reason: str
    is_last: bool
    buffer_enter_abs: float
    flush_trigger_abs: float


class PartitionTransferPlanner(Generic[GroupT]):
    """Separate semantic previous closes from current throughput batching."""

    def __init__(
        self,
        *,
        previous_quota_groups: int,
        current_quota_groups: int,
        preferred_batch_groups: int,
    ) -> None:
        for name, value in (
            ("previous_quota_groups", previous_quota_groups),
            ("current_quota_groups", current_quota_groups),
            ("preferred_batch_groups", preferred_batch_groups),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if preferred_batch_groups == 0:
            raise ValueError("preferred_batch_groups must be positive")
        self.previous_quota_groups = previous_quota_groups
        self.current_quota_groups = current_quota_groups
        self.preferred_batch_groups = preferred_batch_groups
        self.previous_groups: list[GroupT] = []
        self.current_groups: list[GroupT] = []
        self._previous_buffer_enter_abs: float | None = None
        self._current_buffer_enter_abs: float | None = None
        self.committed_previous_groups = 0
        self.committed_current_groups = 0

    @property
    def previous_remaining_groups(self) -> int:
        return max(self.previous_quota_groups - self.committed_previous_groups, 0)

    def _pop_previous(self, *, now: float, reason: str, is_last: bool) -> PartitionTransferBatch[GroupT]:
        groups = tuple(self.previous_groups)
        assert groups and self._previous_buffer_enter_abs is not None
        self.previous_groups.clear()
        buffer_enter_abs = self._previous_buffer_enter_abs
        self._previous_buffer_enter_abs = None
        return PartitionTransferBatch(
            partition_kind="previous",
            groups=groups,
            flush_reason=reason,
            is_last=is_last,
            buffer_enter_abs=buffer_enter_abs,
            flush_trigger_abs=now,
        )

    def _pop_current(
        self,
        count: int,
        *,
        now: float,
        reason: str,
        is_last: bool,
    ) -> PartitionTransferBatch[GroupT]:
        groups = tuple(self.current_groups[:count])
        assert groups and self._current_buffer_enter_abs is not None
        del self.current_groups[:count]
        buffer_enter_abs = self._current_buffer_enter_abs
        self._current_buffer_enter_abs = now if self.current_groups else None
        return PartitionTransferBatch(
            partition_kind="current",
            groups=groups,
            flush_reason=reason,
            is_last=is_last,
            buffer_enter_abs=buffer_enter_abs,
            flush_trigger_abs=now,
        )

    def add_completed(self, group: GroupT, *, now: float) -> list[PartitionTransferBatch[GroupT]]:
        """Route one successful group and return immediately flushable batches."""

        if self.committed_previous_groups < self.previous_quota_groups:
            if not self.previous_groups:
                self._previous_buffer_enter_abs = now
            self.previous_groups.append(group)
            self.committed_previous_groups += 1
            if self.committed_previous_groups == self.previous_quota_groups:
                return [
                    self._pop_previous(
                        now=now,
                        reason="logical_debt_closed",
                        is_last=True,
                    )
                ]
            return []

        if self.committed_current_groups >= self.current_quota_groups:
            raise ValueError("successful group exceeds the current partition quota")
        if not self.current_groups:
            self._current_buffer_enter_abs = now
        self.current_groups.append(group)
        self.committed_current_groups += 1
        if len(self.current_groups) < self.preferred_batch_groups:
            return []
        is_last = (
            self.committed_current_groups == self.current_quota_groups
            and len(self.current_groups) == self.preferred_batch_groups
        )
        return [
            self._pop_current(
                self.preferred_batch_groups,
                now=now,
                reason="preferred_size",
                is_last=is_last,
            )
        ]

    def flush_tail(self, *, now: float, reason: str = "physical_close") -> list[PartitionTransferBatch[GroupT]]:
        """Flush partition-homogeneous tails without fabricating completion."""

        batches: list[PartitionTransferBatch[GroupT]] = []
        if self.previous_groups:
            batches.append(
                self._pop_previous(
                    now=now,
                    reason=reason,
                    is_last=self.committed_previous_groups >= self.previous_quota_groups,
                )
            )
        if self.current_groups:
            batches.append(
                self._pop_current(
                    len(self.current_groups),
                    now=now,
                    reason=reason,
                    is_last=self.committed_current_groups >= self.current_quota_groups,
                )
            )
        return batches


def config_from_namespace(args: Any) -> tuple[DebtAwareAdmissionConfig, str | None]:
    """Build a config from either the public CLI or a custom namespace.

    The attributes remain optional so custom rollout callers created before the
    public CLI was added preserve eager behavior.
    """

    raw_mode = getattr(args, "partition_critical_admission_mode", AdmissionMode.OFF.value)
    try:
        mode = raw_mode if isinstance(raw_mode, AdmissionMode) else AdmissionMode(str(raw_mode).lower())
        if mode is AdmissionMode.OFF:
            return DebtAwareAdmissionConfig(), None
        required_names = (
            "partition_critical_admission_min_inflight_groups",
            "partition_critical_admission_max_inflight_groups",
            "partition_critical_admission_slack_groups",
        )
        missing = [name for name in required_names if getattr(args, name, None) is None]
        if missing:
            raise ValueError(f"missing admission settings: {', '.join(missing)}")
        config = DebtAwareAdmissionConfig(
            mode=mode,
            min_inflight_groups=int(args.partition_critical_admission_min_inflight_groups),
            max_inflight_groups=int(args.partition_critical_admission_max_inflight_groups),
            slack_groups=int(args.partition_critical_admission_slack_groups),
        )
        config.validate()
    except (TypeError, ValueError) as exc:
        return DebtAwareAdmissionConfig(), str(exc)
    return config, None


def validate_admission_namespace(args: Any) -> DebtAwareAdmissionConfig:
    """Validate the public CLI contract and return its admission config."""

    config, config_error = config_from_namespace(args)
    raw_mode = getattr(args, "partition_critical_admission_mode", AdmissionMode.OFF.value)
    mode = raw_mode if isinstance(raw_mode, AdmissionMode) else AdmissionMode(str(raw_mode).lower())
    if mode is AdmissionMode.OFF:
        return config
    if not (getattr(args, "fully_async", False) or getattr(args, "hybrid", False)):
        raise ValueError("--partition-critical-admission-mode shadow/on requires --fully-async or --hybrid.")
    if config_error is not None:
        raise ValueError(f"Invalid partition-critical admission settings: {config_error}")
    return config


class DebtAwareAdmissionController:
    """Choose how many groups to fetch and submit immediately.

    ``available_groups`` is an upper bound on useful new work for the current
    target. ``eager_admit_groups`` is what the existing rollout path would
    fetch now. ON uses the bounded decision; OFF and SHADOW preserve eager
    behavior.
    """

    def __init__(
        self,
        config: DebtAwareAdmissionConfig,
        *,
        final_backfill: bool = False,
        physical_rollout_id: int | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.final_backfill = final_backfill
        self.physical_rollout_id = physical_rollout_id
        self._failed_open_reason: str | None = None
        self._decisions = 0
        self._actual_admitted = 0
        self._shadow_bounded_admitted = 0
        self._max_inflight_observed = 0

    def decide(
        self,
        *,
        inflight_groups: int,
        debt_remaining: int,
        available_groups: int,
        eager_admit_groups: int,
    ) -> AdmissionDecision:
        for name, value in (
            ("inflight_groups", inflight_groups),
            ("debt_remaining", debt_remaining),
            ("available_groups", available_groups),
            ("eager_admit_groups", eager_admit_groups),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        bypass_reason = None
        if self.config.mode is AdmissionMode.OFF:
            bypass_reason = "disabled"
        elif self.final_backfill:
            bypass_reason = "final_backfill"
        elif self._failed_open_reason is not None:
            bypass_reason = f"fail_open:{self._failed_open_reason}"

        desired_inflight = self.config.max_inflight_groups
        if debt_remaining > 0:
            desired_inflight = max(
                self.config.min_inflight_groups,
                min(self.config.max_inflight_groups, debt_remaining + self.config.slack_groups),
            )
        bounded_admit = min(available_groups, max(desired_inflight - inflight_groups, 0))
        actual_admit = bounded_admit if self.config.mode is AdmissionMode.ON else eager_admit_groups
        if bypass_reason is not None:
            actual_admit = eager_admit_groups
        if self.final_backfill:
            # Final backfill has no current partition that could own surplus
            # work. Preserve eager scheduling, but never submit beyond the
            # remaining previous-partition debt.
            actual_admit = min(actual_admit, debt_remaining, available_groups)

        return AdmissionDecision(
            decision_id="unrecorded",
            decision_sequence=0,
            mode=self.config.mode,
            inflight_groups=inflight_groups,
            debt_remaining=debt_remaining,
            available_groups=available_groups,
            eager_admit_groups=eager_admit_groups,
            desired_inflight_groups=desired_inflight,
            bounded_admit_groups=bounded_admit,
            actual_admit_groups=actual_admit,
            bypass_reason=bypass_reason,
        )

    def admit_count(
        self,
        *,
        inflight_groups: int,
        debt_remaining: int,
        available_groups: int,
        eager_admit_groups: int,
    ) -> AdmissionDecision:
        decision = self.decide(
            inflight_groups=inflight_groups,
            debt_remaining=debt_remaining,
            available_groups=available_groups,
            eager_admit_groups=eager_admit_groups,
        )
        self._decisions += 1
        decision = replace(
            decision,
            decision_id=f"admission:{self.physical_rollout_id if self.physical_rollout_id is not None else 'na'}:{self._decisions}",
            decision_sequence=self._decisions,
        )
        self._actual_admitted += decision.actual_admit_groups
        self._shadow_bounded_admitted += decision.bounded_admit_groups
        self._max_inflight_observed = max(
            self._max_inflight_observed,
            inflight_groups + decision.actual_admit_groups,
        )
        return decision

    def fail_open(self, reason: str) -> None:
        """Disable bounding for the rest of this physical rollout."""

        self._failed_open_reason = reason or "unspecified"

    def metrics(self) -> dict[str, float]:
        return {
            "rollout/admission/decisions": float(self._decisions),
            "rollout/admission/actual_admitted_groups": float(self._actual_admitted),
            # This is a sum of per-decision instantaneous suggestions. In SHADOW
            # mode it is not a replay of the full ON trajectory.
            "rollout/admission/instant_shadow_bounded_groups": float(self._shadow_bounded_admitted),
            "rollout/admission/max_inflight_observed": float(self._max_inflight_observed),
            "rollout/admission/failed_open": float(self._failed_open_reason is not None),
        }


def plan_next_admission(
    controller: DebtAwareAdmissionController,
    *,
    target_groups: int,
    progress_groups: int,
    transferred_groups: int,
    inflight_groups: int,
    cumulative_submitted_groups: int,
    previous_debt_groups: int,
    transfer_batch_groups: int,
    eager_fetch_groups: int,
) -> AdmissionDecision:
    """Translate rollout counters into one fetch-and-submit decision."""

    for name, value in (
        ("target_groups", target_groups),
        ("progress_groups", progress_groups),
        ("transferred_groups", transferred_groups),
        ("cumulative_submitted_groups", cumulative_submitted_groups),
        ("eager_fetch_groups", eager_fetch_groups),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    # ``progress_groups`` follows generate_rollout_async's stopping rule. In a
    # normal fully-async physical rollout that includes aborted groups already
    # counted in ``data``; in final backfill it is transferred groups only.
    # Debt closure, however, must always be based on actually transferred groups.
    available_groups = max(target_groups - progress_groups - inflight_groups, 0)
    eager_admit_groups = eager_fetch_groups if cumulative_submitted_groups < target_groups else 0
    logical_debt_remaining = previous_partition_debt_remaining(
        previous_debt_groups=previous_debt_groups,
        completed_groups=transferred_groups,
    )
    return controller.admit_count(
        inflight_groups=inflight_groups,
        debt_remaining=logical_debt_remaining,
        available_groups=available_groups,
        eager_admit_groups=eager_admit_groups,
    )
