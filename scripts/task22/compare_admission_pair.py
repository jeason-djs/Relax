#!/usr/bin/env python3
"""Verify and summarize a matched Task 22 SHADOW/ON artifact pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


PAIR_METRICS = (
    "perf/step_time",
    "perf/samples_per_second",
    "rollout/raw_reward",
    "rollout/response_lengths",
    "rollout/rollout_log_probs",
    "rollout/total_lengths",
    "train/ppo_kl",
    "train/mismatch_kl",
    "train/tis",
    "train/tis_clipfrac",
)
ALLOWED_CONTRACT_DIFFS = {"admission_mode"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def compare_pair(
    shadow_dir: Path,
    on_dir: Path,
    *,
    allowed_contract_diffs: set[str] | None = None,
) -> dict[str, Any]:
    allowed = ALLOWED_CONTRACT_DIFFS if allowed_contract_diffs is None else allowed_contract_diffs
    failures: list[dict[str, Any]] = []

    try:
        shadow_contract = _read_json(shadow_dir / "run_contract.json")
        on_contract = _read_json(on_dir / "run_contract.json")
        shadow_validation = _read_json(shadow_dir / "validation.json")
        on_validation = _read_json(on_dir / "validation.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "verdict": "FAIL",
            "checks": {"pair_artifacts_readable": False},
            "failures": [{"check": "pair_artifacts_readable", "detail": str(exc)}],
        }

    checks: dict[str, bool] = {"pair_artifacts_readable": True}

    def check(name: str, condition: bool, detail: Any) -> None:
        checks[name] = bool(condition)
        if not condition:
            failures.append({"check": name, "detail": detail})

    contract_keys = set(shadow_contract) | set(on_contract)
    contract_diffs = {
        key: {"shadow": shadow_contract.get(key), "on": on_contract.get(key)}
        for key in sorted(contract_keys)
        if shadow_contract.get(key) != on_contract.get(key)
    }
    check(
        "only_allowlisted_contract_fields_differ",
        set(contract_diffs) == allowed,
        {"observed_diffs": contract_diffs, "allowed_diffs": sorted(allowed)},
    )
    check(
        "mode_order_is_shadow_then_on",
        shadow_contract.get("admission_mode") == "shadow" and on_contract.get("admission_mode") == "on",
        {
            "shadow_mode": shadow_contract.get("admission_mode"),
            "on_mode": on_contract.get("admission_mode"),
        },
    )
    check(
        "same_nonempty_git_commit",
        bool(shadow_contract.get("git_commit"))
        and shadow_contract.get("git_commit") == on_contract.get("git_commit"),
        {
            "shadow_commit": shadow_contract.get("git_commit"),
            "on_commit": on_contract.get("git_commit"),
        },
    )
    check(
        "shadow_run_validation_passes",
        shadow_validation.get("verdict") == "PASS",
        shadow_validation.get("failures"),
    )
    check(
        "on_run_validation_passes",
        on_validation.get("verdict") == "PASS",
        on_validation.get("failures"),
    )

    shadow_rows = shadow_validation.get("quality_metrics")
    on_rows = on_validation.get("quality_metrics")
    if not isinstance(shadow_rows, list):
        shadow_rows = []
    if not isinstance(on_rows, list):
        on_rows = []
    shadow_by_step = {
        row.get("step"): row for row in shadow_rows if isinstance(row, dict) and isinstance(row.get("step"), int)
    }
    on_by_step = {row.get("step"): row for row in on_rows if isinstance(row, dict) and isinstance(row.get("step"), int)}
    check(
        "headline_steps_match",
        bool(shadow_by_step) and set(shadow_by_step) == set(on_by_step),
        {
            "shadow_steps": sorted(shadow_by_step),
            "on_steps": sorted(on_by_step),
        },
    )

    summary: dict[str, dict[str, float]] = {}
    common_steps = sorted(set(shadow_by_step) & set(on_by_step))
    metrics_complete = bool(common_steps)
    for metric in PAIR_METRICS:
        shadow_values = [shadow_by_step[step].get(metric) for step in common_steps]
        on_values = [on_by_step[step].get(metric) for step in common_steps]
        if not all(_finite(value) for value in shadow_values + on_values):
            metrics_complete = False
            continue
        shadow_mean = mean(float(value) for value in shadow_values)
        on_mean = mean(float(value) for value in on_values)
        summary[metric] = {
            "shadow_mean": shadow_mean,
            "on_mean": on_mean,
            "delta_on_minus_shadow": on_mean - shadow_mean,
            "relative_delta": ((on_mean / shadow_mean) - 1.0 if shadow_mean != 0 else None),
        }
    check(
        "paired_metrics_complete_and_finite",
        metrics_complete and set(summary) == set(PAIR_METRICS),
        {"missing_metrics": sorted(set(PAIR_METRICS) - set(summary))},
    )

    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failures": failures,
        "contract_diffs": contract_diffs,
        "headline_steps": common_steps,
        "paired_metrics": summary,
        "optimization_decision": "NOT_AUTOMATED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = compare_pair(args.shadow_dir, args.on_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
