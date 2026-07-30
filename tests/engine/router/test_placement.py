# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.engine.router.placement import (
    PlacementConfig,
    PlacementMode,
    PlacementPolicy,
    PlacementRequest,
    WorkerSnapshot,
    decide_placement,
    select_worker,
)


def _candidate(engine_id: str, *, active: int, work: int) -> WorkerSnapshot:
    return WorkerSnapshot(
        engine_id=engine_id,
        active_requests=active,
        predicted_work=work,
    )


def test_placement_off_preserves_baseline_without_candidates() -> None:
    decision = decide_placement(
        PlacementConfig(mode=PlacementMode.OFF),
        PlacementRequest(decision_id="rid-1", predicted_work=100, sequence=0),
        (),
        baseline_engine_id="engine-baseline",
    )

    assert decision.actual_engine_id == "engine-baseline"
    assert decision.selected_engine_id == "engine-baseline"
    assert decision.routing_reason == "baseline_off"


def test_placement_shadow_records_policy_choice_but_preserves_baseline() -> None:
    candidates = (
        _candidate("engine-a", active=0, work=100),
        _candidate("engine-b", active=1, work=0),
    )

    decision = decide_placement(
        PlacementConfig(
            mode=PlacementMode.SHADOW,
            policy=PlacementPolicy.LEAST_PREDICTED_WORK,
        ),
        PlacementRequest(decision_id="rid-2", predicted_work=50, sequence=0),
        candidates,
        baseline_engine_id="engine-a",
    )
    record = decision.to_record()

    assert decision.selected_engine_id == "engine-b"
    assert decision.actual_engine_id == "engine-a"
    assert record["placement_shadow_engine_id"] == "engine-b"
    assert record["placement_actual_engine_id"] == "engine-a"


def test_placement_policies_use_deterministic_tie_breaking() -> None:
    candidates = (
        _candidate("engine-c", active=1, work=10),
        _candidate("engine-a", active=1, work=10),
        _candidate("engine-b", active=1, work=10),
    )

    assert select_worker(PlacementPolicy.ROUND_ROBIN, candidates, sequence=4) == "engine-b"
    assert select_worker(PlacementPolicy.LEAST_ACTIVE_REQUESTS, candidates, sequence=2) == "engine-c"
    assert select_worker(PlacementPolicy.LEAST_PREDICTED_WORK, candidates, sequence=99) == "engine-a"
