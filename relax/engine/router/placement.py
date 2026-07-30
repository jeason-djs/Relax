# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared request-placement policy contract for online routing and replay."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


PLACEMENT_POLICY_VERSION = "v1"


class PlacementMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ON = "on"


class PlacementPolicy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_ACTIVE_REQUESTS = "least_active_requests"
    LEAST_PREDICTED_WORK = "least_predicted_work"


@dataclass(frozen=True)
class PlacementConfig:
    mode: PlacementMode = PlacementMode.OFF
    policy: PlacementPolicy = PlacementPolicy.LEAST_PREDICTED_WORK


@dataclass(frozen=True)
class PlacementRequest:
    decision_id: str
    predicted_work: int
    sequence: int


@dataclass(frozen=True)
class WorkerSnapshot:
    engine_id: str
    active_requests: int
    predicted_work: int
    queued_requests: int | None = None
    running_context_tokens: int | None = None
    running_output_tokens: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "running_context_tokens": self.running_context_tokens,
            "running_output_tokens": self.running_output_tokens,
            "predicted_work": self.predicted_work,
            "snapshot_age_s": 0.0,
            "snapshot_source": "router_decision",
        }


@dataclass(frozen=True)
class PlacementDecision:
    decision_id: str
    decision_sequence: int
    mode: PlacementMode
    policy: PlacementPolicy
    candidates: tuple[WorkerSnapshot, ...]
    request_predicted_work: int
    baseline_engine_id: str
    selected_engine_id: str
    actual_engine_id: str
    routing_reason: str
    fallback_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "placement_policy_version": PLACEMENT_POLICY_VERSION,
            "placement_decision_id": self.decision_id,
            "placement_decision_sequence": self.decision_sequence,
            "placement_mode": self.mode.value,
            "placement_policy": self.policy.value,
            "placement_candidate_engines": [candidate.to_record() for candidate in self.candidates],
            "placement_request_predicted_work": self.request_predicted_work,
            "placement_baseline_engine_id": self.baseline_engine_id,
            "placement_shadow_engine_id": (
                self.selected_engine_id if self.mode is PlacementMode.SHADOW else None
            ),
            "placement_selected_engine_id": self.selected_engine_id,
            "placement_actual_engine_id": self.actual_engine_id,
            "placement_routing_reason": self.routing_reason,
            "placement_fallback_reason": self.fallback_reason,
        }


def placement_config_from_namespace(args: Any) -> PlacementConfig:
    """Read placement config without requiring public CLI argument changes."""

    raw_mode = getattr(args, "request_placement_mode", None)
    if raw_mode is None:
        raw_mode = os.environ.get("RELAX_REQUEST_PLACEMENT_MODE", PlacementMode.OFF.value)
    raw_policy = getattr(args, "request_placement_policy", None)
    if raw_policy is None:
        raw_policy = os.environ.get(
            "RELAX_REQUEST_PLACEMENT_POLICY",
            PlacementPolicy.LEAST_PREDICTED_WORK.value,
        )
    return PlacementConfig(mode=PlacementMode(raw_mode), policy=PlacementPolicy(raw_policy))


def engine_id_from_url(worker_url: str) -> str:
    """Return a stable opaque engine id without exposing the worker endpoint."""

    digest = hashlib.sha256(worker_url.encode("utf-8")).hexdigest()[:12]
    return f"engine-{digest}"


def estimate_request_work(*, logical_prefix_tokens: int, max_new_tokens: int) -> int:
    """Estimate schedulable token work using fields available before dispatch."""

    return max(int(logical_prefix_tokens), 0) + max(int(max_new_tokens), 0)


def select_worker(
    policy: PlacementPolicy,
    candidates: tuple[WorkerSnapshot, ...],
    *,
    sequence: int,
) -> str:
    """Select one candidate with deterministic, replayable tie breaking."""

    if not candidates:
        raise RuntimeError("No healthy workers available for request placement")

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.engine_id))
    if policy is PlacementPolicy.ROUND_ROBIN:
        return ordered[sequence % len(ordered)].engine_id
    if policy is PlacementPolicy.LEAST_ACTIVE_REQUESTS:
        minimum = min(candidate.active_requests for candidate in ordered)
        tied = tuple(candidate for candidate in ordered if candidate.active_requests == minimum)
        return tied[sequence % len(tied)].engine_id
    if policy is PlacementPolicy.LEAST_PREDICTED_WORK:
        return min(
            ordered,
            key=lambda candidate: (
                candidate.predicted_work,
                candidate.active_requests,
                candidate.engine_id,
            ),
        ).engine_id
    raise ValueError(f"Unsupported request placement policy: {policy}")


def decide_placement(
    config: PlacementConfig,
    request: PlacementRequest,
    candidates: tuple[WorkerSnapshot, ...],
    *,
    baseline_engine_id: str,
) -> PlacementDecision:
    """Compute shadow and actual choices while preserving OFF semantics."""

    if config.mode is PlacementMode.OFF:
        selected_engine_id = baseline_engine_id
        actual_engine_id = baseline_engine_id
        routing_reason = "baseline_off"
    else:
        selected_engine_id = select_worker(config.policy, candidates, sequence=request.sequence)
    if config.mode is PlacementMode.ON:
        actual_engine_id = selected_engine_id
        routing_reason = "policy_on"
    elif config.mode is PlacementMode.SHADOW:
        actual_engine_id = baseline_engine_id
        routing_reason = "shadow_only"

    return PlacementDecision(
        decision_id=request.decision_id,
        decision_sequence=request.sequence,
        mode=config.mode,
        policy=config.policy,
        candidates=candidates,
        request_predicted_work=request.predicted_work,
        baseline_engine_id=baseline_engine_id,
        selected_engine_id=selected_engine_id,
        actual_engine_id=actual_engine_id,
        routing_reason=routing_reason,
    )


def fail_open_placement(
    config: PlacementConfig,
    request: PlacementRequest,
    candidates: tuple[WorkerSnapshot, ...],
    *,
    baseline_engine_id: str,
    reason: str,
) -> PlacementDecision:
    """Return a baseline decision when placement cannot be evaluated safely."""

    return PlacementDecision(
        decision_id=request.decision_id,
        decision_sequence=request.sequence,
        mode=config.mode,
        policy=config.policy,
        candidates=candidates,
        request_predicted_work=request.predicted_work,
        baseline_engine_id=baseline_engine_id,
        selected_engine_id=baseline_engine_id,
        actual_engine_id=baseline_engine_id,
        routing_reason="fail_open_router",
        fallback_reason=reason,
    )
