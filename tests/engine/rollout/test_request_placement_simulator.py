# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from scripts.task22.simulate_request_placement import simulate_trace


def _row(block: int, index: int, actual_engine_id: str, *, work: int = 100) -> dict:
    dispatch = float(block * 100 + index)
    return {
        "schema_version": 1,
        "record_type": "placement_replay_request",
        "rid": f"relax:p{block}:s{index}",
        "physical_rollout_id": block,
        "dispatch_abs": dispatch,
        "request_end_abs": dispatch + 10.0,
        "observed_request_wall": 10.0,
        "predicted_work": work,
        "observed_work": work,
        "actual_engine_id": actual_engine_id,
        "candidate_engine_ids": ["engine-a", "engine-b"],
    }


def test_placement_simulator_promotes_only_stable_shadow_candidate() -> None:
    block_work = [10, 10, 10, 10, 100, 30, 10]
    rows = [
        _row(
            block,
            index,
            "engine-a",
            work=work,
        )
        for block in range(5)
        for index, work in enumerate(block_work)
    ]

    result = simulate_trace(
        rows,
        bootstrap_samples=200,
        minimum_blocks=5,
        minimum_direction_fraction=0.7,
        minimum_relative_gain=0.01,
    )

    assert result["recommendation"] == "STABLE_SHADOW_CANDIDATE"
    assert result["stable_winner"] == "least_predicted_work"
    assert result["policies"][result["stable_winner"]]["bootstrap_ci95"][0] > 0
    assert all(comparison["stable"] for comparison in result["winner_comparisons"].values())
    assert not result["online_on_eligible"]


def test_placement_simulator_keeps_off_without_stable_gain() -> None:
    rows = [
        _row(block, index, "engine-a" if index % 2 == 0 else "engine-b")
        for block in range(5)
        for index in range(4)
    ]

    result = simulate_trace(
        rows,
        bootstrap_samples=200,
        minimum_blocks=5,
        minimum_direction_fraction=0.7,
        minimum_relative_gain=0.01,
    )

    assert result["recommendation"] == "KEEP_OFF"
    assert result["stable_winner"] is None
    assert not any(policy["stable"] for policy in result["policies"].values())


def test_placement_simulator_keeps_off_when_stable_candidates_are_indistinguishable() -> None:
    rows = [
        _row(block, index, "engine-a")
        for block in range(5)
        for index in range(4)
    ]

    result = simulate_trace(
        rows,
        bootstrap_samples=200,
        minimum_blocks=5,
        minimum_direction_fraction=0.7,
        minimum_relative_gain=0.01,
    )

    assert len(result["stable_candidates"]) > 1
    assert result["stable_winner"] is None
    assert result["recommendation"] == "KEEP_OFF"
