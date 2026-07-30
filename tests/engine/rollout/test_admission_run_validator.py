# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from pathlib import Path

from relax.engine.rollout.request_observability import attempt_token_from_id
from scripts.task22.validate_admission_run import validate_run


RID = "relax:p0:kfresh:g0:s0:a0:00000000000a"
DECISION_ID = "admission:0:1"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _server_event(pid: int, event: dict) -> str:
    payload = dict(event)
    payload.setdefault(
        "timestamp",
        {
            "request.received": 100.2,
            "request.finished": 101.8,
        }.get(event.get("event"), 100.5),
    )
    return f"(SGLangEngine pid={pid}) {json.dumps(payload, sort_keys=True)}"


def _request_row() -> dict:
    return {
        "schema_version": 2,
        "record_type": "attempt",
        "rid": RID,
        "attempt_id": RID,
        "parent_attempt_id": None,
        "attempt_sequence": 1,
        "physical_rollout_id": 0,
        "group_index": 0,
        "sample_index": 0,
        "abort_count": 0,
        "attempt_kind": "fresh",
        "work_origin": "fresh",
        "dispatch_abs": 100.0,
        "request_end_abs": 102.0,
        "client_status": "finished",
        "rid_match": True,
        "generated_tokens_this_attempt": 10,
        "call_prompt_tokens": 100,
        "call_cached_tokens": 50,
        "call_new_prompt_tokens": 50,
        "forward_entry_time": 100.1,
        "prefill_finished_time": 100.4,
        "queue_time": 0.1,
        "outcome": "committed",
        "target_partition": "train_0",
        "admission_decision_id": DECISION_ID,
        "admission_decision_sequence": 1,
    }


def _decision_row() -> dict:
    return {
        "schema_version": 2,
        "record_type": "admission_decision",
        "decision_id": DECISION_ID,
        "decision_sequence": 1,
        "physical_rollout_id": 0,
        "mode": "shadow",
        "release_remaining": 2,
        "inflight_groups": 3,
        "available_groups": 5,
        "eager_admit_groups": 2,
        "bounded_admit_groups": 1,
        "desired_inflight_groups": 4,
        "actual_admit_groups": 2,
        "bypass_reason": None,
    }


def _commit_row() -> dict:
    return {
        "schema_version": 2,
        "record_type": "partition_outcome",
        "attempt_id": RID,
        "attempt_token": attempt_token_from_id(RID),
        "attempt_sequence": 1,
        "attempt_decision_id": DECISION_ID,
        "decision_id": DECISION_ID,
        "decision_sequence": 1,
        "physical_rollout_id": 0,
        "generation_physical_rollout_id": 0,
        "target_partition": "train_0",
        "sample_index": 0,
        "group_index": 0,
        "abort_count": 0,
        "work_origin": "fresh",
        "outcome": "committed",
        "error_type": None,
    }


def _consume_row() -> dict:
    return {
        "schema_version": 2,
        "record_type": "consume_outcome",
        "rollout_id": 0,
        "target_partition": "train_0",
        "sample_index": 0,
        "group_index": 0,
        "abort_count": 0,
        "attempt_sequence": 1,
        "attempt_token": attempt_token_from_id(RID),
        "decision_sequence": 1,
        "decision_id": DECISION_ID,
        "decision_physical_rollout_id": 0,
        "generation_physical_rollout_id": 0,
        "work_origin": "fresh",
        "generation_start_version": 1,
        "generation_end_version": 1,
        "generation_version_span": 0,
        "consume_version": 2,
        "actual_staleness": 1,
        "consume_outcome": "consumed",
    }


def _driver_lines() -> list[str]:
    scheduler = {
        "event": "scheduler.status",
        "forward_mode": "ForwardMode.DECODE",
        "running_rids": [RID],
        "running_seq_lens": [100],
        "running_origin_input_lens": [90],
        "running_output_lens": [10],
        "queued_rids": [],
        "queued_origin_input_lens": [],
        "queued_output_lens": [],
    }
    lines = [
        "(SGLangEngine pid=100) server_args=ServerArgs(base_gpu_id=2, tp_size=1)",
        _server_event(100, {"event": "request.received", "rid": RID, "obj": {"rid": RID}}),
        _server_event(100, {"event": "request.finished", "rid": RID}),
        _server_event(100, scheduler),
    ]
    for index, phase in enumerate(("gate", "pause", "flush", "transfer", "continue"), 1):
        lines.append(
            f"TASK22_EVENT phase={phase} sync_id=1 "
            f"t_begin={index:.6f} t_end={index + 1:.6f} dur=1.000000"
        )
    lines.extend(
        [
            "TASK22_EVENT phase=abort rollout_id=0 t_begin=6.000000 t_end=7.000000 dur=1.000000",
            (
                "TASK22_FLOW phase=gate_ready sync_id=1 candidate_partitions=train_0 "
                "ready_partition=train_0 t=1.500000"
            ),
            "TASK22_FLOW phase=physical_start physical_rollout_id=0 t=0.500000",
            "TASK22_FLOW phase=partition_close physical_rollout_id=0 target_partition=train_0 groups=1 t=5.500000",
            (
                "TASK22_FLOW phase=physical_end physical_rollout_id=0 "
                "t_begin=0.500000 t_end=7.500000 dur=7.000000"
            ),
            (
                "rollout 0: {'rollout/raw_reward': 1.0, 'rollout/response_lengths': 10.0, "
                "'rollout/rollout_log_probs': -0.1, 'rollout/total_lengths': 20.0}"
            ),
            (
                "step 0: {'train/ppo_kl': 0.0, 'train/mismatch_kl': 0.01, "
                "'train/tis': 1.0, 'train/tis_clipfrac': 0.0}"
            ),
            "perf 0: {'perf/step_time': 2.0}",
        ]
    )
    return lines


def _build_valid_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    observability = run_dir / "observability"
    timeline = run_dir / "timeline"
    logs = run_dir / "logs"
    observability.mkdir(parents=True)
    timeline.mkdir()
    logs.mkdir()

    request = _request_row()
    _jsonl(observability / "request_lifecycle_rollout_0.jsonl", [request])
    _jsonl(observability / "admission_ledger_rollout_0.jsonl", [_decision_row(), request])
    _jsonl(observability / "admission_outcomes_rollout_0.jsonl", [_commit_row()])
    _jsonl(observability / "consumption_ledger_rollout_0_rank_0.jsonl", [_consume_row()])
    (run_dir / "driver.log").write_text("\n".join(_driver_lines()) + "\n", encoding="utf-8")
    (run_dir / "EXIT_CODE").write_text("0\n", encoding="utf-8")
    (run_dir / "run_contract.json").write_text(
        json.dumps(
            {
                "admission_mode": "shadow",
                "num_rollout": 1,
                "expected_samples_per_partition": 1,
                "expected_engines": 1,
                "max_staleness": 2,
                "headline_lo": 0,
                "headline_hi": 0,
                "admission_min": 4,
                "admission_max": 8,
                "admission_slack": 2,
                "request_placement_mode": "off",
                "use_slime_router": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (timeline / "timeline_step_0.json").write_text(
        '[{"name":"train","ph":"X","ts":1000,"dur":10,"pid":1,"tid":2}]\n',
        encoding="utf-8",
    )
    (logs / "nvidia_smi_1s.csv").write_text(
        "2026-07-30T10:00:00.000000000+0800\n"
        "0, 1024 MiB, 75 %, 10 %, 200 W\n"
        "1, 1010 MiB, 70 %, 9 %, 195 W\n"
        "2026-07-30T10:00:01.000000000+0800\n"
        "0, 1030 MiB, 80 %, 12 %, 205 W\n"
        "1, 1020 MiB, 78 %, 11 %, 202 W\n",
        encoding="utf-8",
    )
    return run_dir


def _validate(run_dir: Path) -> dict:
    return validate_run(
        run_dir,
        expected_mode="shadow",
        expected_rollouts=1,
        expected_samples_per_partition=1,
        expected_engines=1,
        max_staleness=2,
        headline_lo=0,
        headline_hi=0,
        require_resume=False,
    )


def test_admission_run_validator_accepts_complete_evidence(tmp_path) -> None:
    result = _validate(_build_valid_run(tmp_path))

    assert result["verdict"] == "PASS", result["failures"]
    assert all(result["checks"].values())
    assert result["counts"]["consumption_rows"] == 1


def test_admission_run_validator_rejects_missing_admission_ledger(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    (run_dir / "observability" / "admission_ledger_rollout_0.jsonl").unlink()

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["expected_admission_ledgers_exact"]
    assert not result["checks"]["attempt_decision_exists"]


def test_admission_run_validator_rejects_duplicate_consume(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    consume_path = run_dir / "observability" / "consumption_ledger_rollout_0_rank_0.jsonl"
    _jsonl(consume_path, [_consume_row(), _consume_row()])

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["commit_has_exactly_one_consume"]
    assert not result["checks"]["partition_samples_unique"]
    assert not result["checks"]["attempt_tokens_globally_unique_in_consumes"]


def test_admission_run_validator_rejects_extra_consumed_partition(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    consume = _consume_row()
    consume["rollout_id"] = 1
    consume["target_partition"] = "train_1"
    _jsonl(
        run_dir / "observability" / "consumption_ledger_rollout_1_rank_0.jsonl",
        [consume],
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["consumption_partition_files_exact"]
    assert not result["checks"]["expected_partitions_consumed_exact"]


def test_admission_run_validator_rejects_attempt_committed_across_partitions(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    observability = run_dir / "observability"
    second_commit = _commit_row()
    second_commit["target_partition"] = "train_1"
    _jsonl(
        observability / "admission_outcomes_rollout_0.jsonl",
        [_commit_row(), second_commit],
    )
    second_consume = _consume_row()
    second_consume["rollout_id"] = 1
    second_consume["target_partition"] = "train_1"
    _jsonl(
        observability / "consumption_ledger_rollout_1_rank_0.jsonl",
        [second_consume],
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert result["checks"]["commit_has_exactly_one_consume"]
    assert not result["checks"]["attempt_tokens_globally_unique_in_commits"]
    assert not result["checks"]["attempt_tokens_globally_unique_in_consumes"]


def test_admission_run_validator_rejects_orphan_timeline_keys(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    driver_log = run_dir / "driver.log"
    driver_log.write_text(
        driver_log.read_text(encoding="utf-8")
        + "TASK22_EVENT phase=pause sync_id=orphan t_begin=8.000000 t_end=9.000000 dur=1.000000\n"
        + "TASK22_FLOW phase=gate_ready sync_id=orphan candidate_partitions=train_0 "
        + "ready_partition=train_0 t=8.500000\n"
        + "TASK22_FLOW phase=partition_close physical_rollout_id=0 "
        + "target_partition=train_extra groups=1 t=5.750000\n",
        encoding="utf-8",
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["keyed_sync_id_sets_match"]
    assert not result["checks"]["gate_flow_sync_ids_closed"]
    assert not result["checks"]["partition_close_partitions_exact"]


def test_admission_run_validator_rejects_extra_complete_sync_cycle(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    extra_lines = []
    for index, phase in enumerate(("gate", "pause", "flush", "transfer", "continue"), 8):
        extra_lines.append(
            f"TASK22_EVENT phase={phase} sync_id=2 "
            f"t_begin={index:.6f} t_end={index + 1:.6f} dur=1.000000"
        )
    extra_lines.append(
        "TASK22_FLOW phase=gate_ready sync_id=2 candidate_partitions=train_0 "
        "ready_partition=train_0 t=8.500000"
    )
    driver_log = run_dir / "driver.log"
    driver_log.write_text(
        driver_log.read_text(encoding="utf-8") + "\n".join(extra_lines) + "\n",
        encoding="utf-8",
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert result["checks"]["keyed_sync_id_sets_match"]
    assert result["checks"]["gate_flow_sync_ids_closed"]
    assert not result["checks"]["gate_cycle_count_exact"]


def test_admission_run_validator_rejects_excessive_staleness(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    consume = _consume_row()
    consume["consume_version"] = 4
    consume["actual_staleness"] = 3
    _jsonl(
        run_dir / "observability" / "consumption_ledger_rollout_0_rank_0.jsonl",
        [consume],
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["actual_staleness_in_bounds"]
    assert result["checks"]["actual_staleness_matches_versions"]


def test_admission_run_validator_requires_declared_final_backfill_artifacts(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    driver_log = run_dir / "driver.log"
    driver_log.write_text(
        driver_log.read_text(encoding="utf-8")
        + "TASK22_EVENT phase=abort rollout_id=1 t_begin=8.000000 t_end=9.000000 dur=1.000000\n"
        + "TASK22_FLOW phase=physical_start physical_rollout_id=1 t=7.500000\n"
        + "TASK22_FLOW phase=physical_end physical_rollout_id=1 t_begin=7.500000 t_end=9.500000 dur=2.000000\n",
        encoding="utf-8",
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["expected_request_files_exact"]
    assert not result["checks"]["expected_admission_ledgers_exact"]


def test_admission_run_validator_rejects_contract_drift(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    contract_path = run_dir / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        admission_min=3,
        request_placement_mode="shadow",
        use_slime_router=True,
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["admission_contract_is_4_8_2"]
    assert not result["checks"]["request_placement_is_off"]
    assert not result["checks"]["slime_router_is_disabled"]


def test_admission_run_validator_recomputes_bounded_admission(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    ledger_path = run_dir / "observability" / "admission_ledger_rollout_0.jsonl"
    decision = _decision_row()
    decision["bounded_admit_groups"] = 2
    _jsonl(ledger_path, [decision, _request_row()])

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["admission_bounded_recomputes"]


def test_admission_run_validator_strictly_parses_timeline_and_gpu_snapshots(tmp_path) -> None:
    run_dir = _build_valid_run(tmp_path)
    (run_dir / "timeline" / "timeline_step_0.json").write_text(
        '[{"name":"train","ts":1000}]\n',
        encoding="utf-8",
    )
    (run_dir / "logs" / "nvidia_smi_1s.csv").write_text(
        "2026-07-30T10:00:00+0800\nnot,a,gpu,row\n",
        encoding="utf-8",
    )

    result = _validate(run_dir)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["headline_timeline_files_complete"]
    assert not result["checks"]["gpu_snapshots_parse_strictly"]
