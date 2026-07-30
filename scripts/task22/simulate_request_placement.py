#!/usr/bin/env python3
"""Replay prompt-free request traces through request-placement policies."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relax.engine.router.placement import PlacementPolicy, WorkerSnapshot, select_worker


POLICIES = tuple(policy.value for policy in PlacementPolicy)


@dataclass
class SimulatedJob:
    finish: float
    predicted_work: int


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") != "placement_replay_request":
            raise ValueError(f"{path}:{line_no}: unexpected record_type={row.get('record_type')!r}")
        for key in ("rid", "physical_rollout_id", "dispatch_abs", "actual_engine_id"):
            if row.get(key) is None:
                raise ValueError(f"{path}:{line_no}: missing required field {key}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: placement trace is empty")
    return rows


def _calibrate_service_rate(rows: list[dict[str, Any]]) -> float:
    cohorts: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cohorts[(int(row["physical_rollout_id"]), str(row["actual_engine_id"]))].append(row)

    cohort_rates = []
    for cohort_rows in cohorts.values():
        work = sum(int(row.get("observed_work", 0) or 0) for row in cohort_rows)
        start = min(float(row["dispatch_abs"]) for row in cohort_rows)
        end = max(float(row["request_end_abs"]) for row in cohort_rows)
        if work > 0 and end > start:
            cohort_rates.append(work / (end - start))
    if not cohort_rates:
        raise ValueError("trace has no positive engine-cohort work/wall pairs")
    return float(median(cohort_rates))


def _engine_ids(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    engine_ids = {
        str(engine_id)
        for row in rows
        for engine_id in row.get("candidate_engine_ids", [])
        if engine_id is not None
    }
    engine_ids.update(str(row["actual_engine_id"]) for row in rows)
    if len(engine_ids) < 2:
        raise ValueError("placement replay requires at least two engines")
    return tuple(sorted(engine_ids))


def _prune_jobs(jobs: list[SimulatedJob], dispatch: float) -> list[SimulatedJob]:
    return [job for job in jobs if job.finish > dispatch]


def _simulate(
    rows: list[dict[str, Any]],
    *,
    engine_ids: tuple[str, ...],
    service_rate: float,
    policy: PlacementPolicy | None,
) -> dict[str, Any]:
    jobs_by_engine: dict[str, list[SimulatedJob]] = {engine_id: [] for engine_id in engine_ids}
    work_by_engine = {engine_id: 0 for engine_id in engine_ids}
    completion_latencies = []
    decisions = []
    first_dispatch = min(float(row["dispatch_abs"]) for row in rows)
    last_finish = first_dispatch

    for sequence, row in enumerate(sorted(rows, key=lambda item: (float(item["dispatch_abs"]), str(item["rid"])))):
        dispatch = float(row["dispatch_abs"])
        for engine_id in engine_ids:
            jobs_by_engine[engine_id] = _prune_jobs(jobs_by_engine[engine_id], dispatch)

        candidates = tuple(
            WorkerSnapshot(
                engine_id=engine_id,
                active_requests=len(jobs_by_engine[engine_id]),
                predicted_work=sum(job.predicted_work for job in jobs_by_engine[engine_id]),
            )
            for engine_id in engine_ids
        )
        if policy is None:
            selected = str(row["actual_engine_id"])
            policy_name = "observed"
        else:
            selected = select_worker(policy, candidates, sequence=sequence)
            policy_name = policy.value

        predicted_work = max(int(row.get("predicted_work", 0) or 0), 1)
        observed_work = max(int(row.get("observed_work", 0) or 0), 1)
        engine_jobs = jobs_by_engine[selected]
        start = max(dispatch, max((job.finish for job in engine_jobs), default=dispatch))
        finish = start + observed_work / service_rate
        engine_jobs.append(SimulatedJob(finish=finish, predicted_work=predicted_work))
        work_by_engine[selected] += observed_work
        completion_latencies.append(finish - dispatch)
        last_finish = max(last_finish, finish)
        decisions.append(
            {
                "rid": row["rid"],
                "policy": policy_name,
                "selected_engine_id": selected,
                "dispatch_abs": dispatch,
                "simulated_finish_abs": finish,
            }
        )

    engine_work_values = list(work_by_engine.values())
    return {
        "policy": "observed" if policy is None else policy.value,
        "requests": len(rows),
        "completion_p50_s": float(median(completion_latencies)),
        "completion_p95_s": _percentile(completion_latencies, 0.95),
        "makespan_s": float(last_finish - first_dispatch),
        "max_engine_work": max(engine_work_values),
        "engine_work_variance": float(
            mean((value - mean(engine_work_values)) ** 2 for value in engine_work_values)
        ),
        "engine_work": work_by_engine,
        "decisions": decisions,
    }


def _bootstrap_interval(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    bootstrapped = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return _percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)


def simulate_trace(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    minimum_blocks: int = 5,
    minimum_direction_fraction: float = 0.7,
    minimum_relative_gain: float = 0.01,
    minimum_winner_margin: float = 0.002,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")
    service_rate = _calibrate_service_rate(rows)
    engine_ids = _engine_ids(rows)
    blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[int(row["physical_rollout_id"])].append(row)

    policy_block_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block_id, block_rows in sorted(blocks.items()):
        baseline = _simulate(block_rows, engine_ids=engine_ids, service_rate=service_rate, policy=None)
        for policy in PlacementPolicy:
            candidate = _simulate(block_rows, engine_ids=engine_ids, service_rate=service_rate, policy=policy)
            relative_gain = (
                (baseline["makespan_s"] - candidate["makespan_s"]) / baseline["makespan_s"]
                if baseline["makespan_s"] > 0
                else 0.0
            )
            policy_block_metrics[policy.value].append(
                {
                    "physical_rollout_id": block_id,
                    "relative_makespan_gain": relative_gain,
                    "baseline": {key: value for key, value in baseline.items() if key != "decisions"},
                    "candidate": {key: value for key, value in candidate.items() if key != "decisions"},
                }
            )

    policies = {}
    for policy_name, metrics in policy_block_metrics.items():
        gains = [float(metric["relative_makespan_gain"]) for metric in metrics]
        ci_low, ci_high = _bootstrap_interval(
            gains,
            samples=bootstrap_samples,
            seed=seed + POLICIES.index(policy_name),
        )
        direction_fraction = sum(gain > 0 for gain in gains) / len(gains)
        mean_gain = float(mean(gains))
        policies[policy_name] = {
            "blocks": len(gains),
            "mean_relative_makespan_gain": mean_gain,
            "median_relative_makespan_gain": float(median(gains)),
            "positive_block_fraction": direction_fraction,
            "bootstrap_ci95": [ci_low, ci_high],
            "stable": (
                len(gains) >= minimum_blocks
                and direction_fraction >= minimum_direction_fraction
                and ci_low > 0
                and mean_gain >= minimum_relative_gain
            ),
            "block_metrics": metrics,
        }

    ranked = sorted(
        policies.items(),
        key=lambda item: item[1]["mean_relative_makespan_gain"],
        reverse=True,
    )
    stable_candidates = [item[0] for item in ranked if item[1]["stable"]]
    winner_comparisons = {}
    winner = None
    if len(stable_candidates) == 1:
        winner = stable_candidates[0]
    elif stable_candidates:
        leading_policy = stable_candidates[0]
        leading_gains = [
            float(metric["relative_makespan_gain"])
            for metric in policy_block_metrics[leading_policy]
        ]
        uniquely_better = True
        for comparison_index, other_policy in enumerate(stable_candidates[1:], 1):
            other_gains = [
                float(metric["relative_makespan_gain"])
                for metric in policy_block_metrics[other_policy]
            ]
            paired_deltas = [
                leading - other
                for leading, other in zip(leading_gains, other_gains, strict=True)
            ]
            ci_low, ci_high = _bootstrap_interval(
                paired_deltas,
                samples=bootstrap_samples,
                seed=seed + len(POLICIES) + comparison_index,
            )
            mean_delta = float(mean(paired_deltas))
            comparison_stable = ci_low > 0 and mean_delta >= minimum_winner_margin
            winner_comparisons[other_policy] = {
                "mean_paired_gain_delta": mean_delta,
                "bootstrap_ci95": [ci_low, ci_high],
                "stable": comparison_stable,
            }
            uniquely_better = uniquely_better and comparison_stable
        if uniquely_better:
            winner = leading_policy

    return {
        "schema_version": 1,
        "model": {
            "name": "fifo_fluid_work_proxy",
            "service_rate_tokens_per_second": service_rate,
            "limitations": [
                "Does not reproduce SGLang continuous batching or KV-cache eviction.",
                "Placement decisions use pre-dispatch predicted work; service uses observed request work.",
                "Cache-hit and recompute effects are not counterfactually modeled.",
            ],
        },
        "trace": {
            "requests": len(rows),
            "physical_rollout_blocks": len(blocks),
            "engine_ids": list(engine_ids),
        },
        "stability_gate": {
            "minimum_blocks": minimum_blocks,
            "minimum_direction_fraction": minimum_direction_fraction,
            "minimum_relative_gain": minimum_relative_gain,
            "minimum_winner_margin": minimum_winner_margin,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
        "policies": policies,
        "stable_candidates": stable_candidates,
        "winner_comparisons": winner_comparisons,
        "stable_winner": winner,
        "recommendation": "STABLE_SHADOW_CANDIDATE" if winner else "KEEP_OFF",
        "online_on_eligible": False,
        "online_on_blockers": [
            "Counterfactual cache-hit/recompute behavior is unknown.",
            "A separate matched placement SHADOW/ON run has not passed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-blocks", type=int, default=5)
    parser.add_argument("--minimum-direction-fraction", type=float, default=0.7)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.01)
    parser.add_argument("--minimum-winner-margin", type=float, default=0.002)
    args = parser.parse_args()

    result = simulate_trace(
        _load_trace(args.trace_jsonl),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_blocks=args.minimum_blocks,
        minimum_direction_fraction=args.minimum_direction_fraction,
        minimum_relative_gain=args.minimum_relative_gain,
        minimum_winner_margin=args.minimum_winner_margin,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
