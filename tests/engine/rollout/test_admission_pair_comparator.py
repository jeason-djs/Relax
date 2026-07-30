# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

from scripts.task22.compare_admission_pair import PAIR_METRICS, compare_pair


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build_pair(tmp_path):
    shadow_dir = tmp_path / "shadow"
    on_dir = tmp_path / "on"
    shadow_dir.mkdir()
    on_dir.mkdir()
    common_contract = {
        "git_commit": "abc123",
        "num_rollout": 15,
        "train_seed": 1234,
        "rollout_seed": 42,
        "admission_min": 4,
        "admission_max": 8,
        "admission_slack": 2,
    }
    _write_json(shadow_dir / "run_contract.json", {**common_contract, "admission_mode": "shadow"})
    _write_json(on_dir / "run_contract.json", {**common_contract, "admission_mode": "on"})
    shadow_metrics = {"step": 5}
    on_metrics = {"step": 5}
    for index, metric in enumerate(PAIR_METRICS, 1):
        shadow_metrics[metric] = float(index)
        on_metrics[metric] = float(index) + 0.5
    _write_json(
        shadow_dir / "validation.json",
        {"verdict": "PASS", "failures": {}, "quality_metrics": [shadow_metrics]},
    )
    _write_json(
        on_dir / "validation.json",
        {"verdict": "PASS", "failures": {}, "quality_metrics": [on_metrics]},
    )
    return shadow_dir, on_dir


def test_admission_pair_comparator_accepts_mode_only_contract_delta(tmp_path) -> None:
    shadow_dir, on_dir = _build_pair(tmp_path)

    result = compare_pair(shadow_dir, on_dir)

    assert result["verdict"] == "PASS"
    assert result["contract_diffs"] == {
        "admission_mode": {"shadow": "shadow", "on": "on"}
    }
    assert result["paired_metrics"]["perf/step_time"]["delta_on_minus_shadow"] == 0.5
    assert result["optimization_decision"] == "NOT_AUTOMATED"


def test_admission_pair_comparator_rejects_non_mode_contract_drift(tmp_path) -> None:
    shadow_dir, on_dir = _build_pair(tmp_path)
    on_contract = json.loads((on_dir / "run_contract.json").read_text(encoding="utf-8"))
    on_contract["rollout_seed"] = 43
    _write_json(on_dir / "run_contract.json", on_contract)

    result = compare_pair(shadow_dir, on_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["only_allowlisted_contract_fields_differ"]
    assert result["contract_diffs"]["rollout_seed"] == {"shadow": 42, "on": 43}
