# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import json
from pathlib import Path

import pytest

from scripts.task22.monitor_admission_run import (
    STRICT_CHECK_COVERAGE,
    MonitorFailure,
    _scan,
    _stop_process,
    _validate_closed_lifecycle,
    _validate_contract,
    _validate_gpu_snapshots,
    _validate_headline_metrics,
    _validate_physical_flow,
    _validate_sync,
    _validate_timelines,
)


def _prepare_monitor_inputs(run_dir) -> None:
    contract = {
        "admission_mode": "shadow",
        "num_rollout": 1,
        "expected_samples_per_partition": 64,
        "expected_engines": 2,
        "max_staleness": 2,
        "headline_lo": 5,
        "headline_hi": 6,
        "admission_min": 4,
        "admission_max": 8,
        "admission_slack": 2,
        "request_placement_mode": "off",
        "use_slime_router": False,
    }
    (run_dir / "run_contract.json").write_text(json.dumps(contract) + "\n")
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    snapshot = (
        "2026-07-30T10:00:00.000000000+0800\n"
        "0, 100 MiB, 10 %, 20 %, 30 W\n"
        "1, 100 MiB, 10 %, 20 %, 30 W\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "3, 100 MiB, 10 %, 20 %, 30 W\n"
        "2026-07-30T10:00:01.000000000+0800\n"
        "0, 100 MiB, 10 %, 20 %, 30 W\n"
        "1, 100 MiB, 10 %, 20 %, 30 W\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "3, 100 MiB, 10 %, 20 %, 30 W\n"
    )
    (logs / "nvidia_smi_1s.csv").write_text(snapshot)


def _scan_once(run_dir, *, final=False) -> None:
    _prepare_monitor_inputs(run_dir)
    _scan(
        run_dir,
        expected_mode="shadow",
        expected_rollouts=1,
        expected_samples_per_partition=64,
        expected_engines=2,
        max_staleness=2,
        admission_min=4,
        admission_max=8,
        admission_slack=2,
        headline_lo=5,
        headline_hi=6,
        final=final,
        reported_lifecycle=set(),
        event_log=run_dir / "online_monitor.jsonl",
    )


def _write_physical_flow(run_dir, rollout_ids=(0,)) -> None:
    lines = []
    for rollout_id in rollout_ids:
        lines.extend(
            [
                f"TASK22_FLOW phase=physical_start physical_rollout_id={rollout_id} t={rollout_id}.0",
                (
                    f"TASK22_FLOW phase=physical_end physical_rollout_id={rollout_id} "
                    f"t_begin={rollout_id}.0 t_end={rollout_id}.5 dur=0.5"
                ),
            ]
        )
    (run_dir / "driver.log").write_text("\n".join(lines) + "\n")


def test_online_monitor_rejects_malformed_jsonl(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    (observability / "admission_ledger_rollout_0.jsonl").write_text('{"record_type":\n')
    (tmp_path / "driver.log").write_text(
        "TASK22_FLOW phase=physical_end physical_rollout_id=0 next_debt_groups=0\n"
    )

    with pytest.raises(MonitorFailure, match="malformed_jsonl"):
        _scan_once(tmp_path)


def test_online_monitor_rejects_admission_fail_open(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    row = {
        "record_type": "admission_decision",
        "physical_rollout_id": 0,
        "bypass_reason": "fail_open:no_progress",
    }
    (observability / "admission_ledger_rollout_0.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "driver.log").write_text(
        "TASK22_FLOW phase=physical_end physical_rollout_id=0 next_debt_groups=0\n"
    )

    with pytest.raises(MonitorFailure, match="admission_fail_open"):
        _scan_once(tmp_path)


def test_online_monitor_real_metric_log_can_arrive_before_timeline_file(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    (tmp_path / "timeline").mkdir()
    driver_log = tmp_path / "driver.log"
    evidence_due_since = {}
    rollout = (
        "{'rollout/raw_reward': 1.0, 'rollout/response_lengths': 1.0, "
        "'rollout/rollout_log_probs': -1.0, 'rollout/total_lengths': 2.0}"
    )
    train = (
        "{'train/ppo_kl': 0.0, 'train/mismatch_kl': 0.0, "
        "'train/tis': 1.0, 'train/tis_clipfrac': 0.0}"
    )
    driver_log.write_text(
        f"rollout 6: {rollout}\nstep 6: {train}\nperf 6: {{'perf/step_time': 1.0}}\n"
    )
    (tmp_path / "timeline" / "timeline_step_6.json").write_text(
        '[{"name":"train","ph":"X","ts":1,"dur":1,"pid":1,"tid":1}]\n'
    )

    # A complete later step does not make an earlier async step's timeline due.
    _scan(
        tmp_path,
        expected_mode="shadow",
        expected_rollouts=1,
        expected_samples_per_partition=64,
        expected_engines=2,
        max_staleness=2,
        admission_min=4,
        admission_max=8,
        admission_slack=2,
        headline_lo=5,
        headline_hi=6,
        final=False,
        reported_lifecycle=set(),
        event_log=tmp_path / "online_monitor.jsonl",
        evidence_grace=5.0,
        timeline_evidence_due_since=evidence_due_since,
        now_monotonic=100.0,
    )

    driver_log.write_text(
        driver_log.read_text()
        + f"rollout 5: {rollout}\n"
        + "perf 5: {'perf/step_time': 1.0}\n"
    )
    _scan_once(tmp_path)

    driver_log.write_text(driver_log.read_text() + f"step 5: {train}\n")
    _validate_timelines(
        driver_log,
        tmp_path / "timeline",
        headline_lo=5,
        headline_hi=6,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=evidence_due_since,
        now_monotonic=101.0,
    )
    assert evidence_due_since == {5: 101.0}

    (tmp_path / "timeline" / "timeline_step_5.json").write_text(
        '[{"name":"train","ph":"X","ts":1,"dur":1,"pid":1,"tid":1}]\n'
    )
    _validate_timelines(
        driver_log,
        tmp_path / "timeline",
        headline_lo=5,
        headline_hi=6,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=evidence_due_since,
        now_monotonic=104.0,
    )
    assert not evidence_due_since


def test_online_monitor_timeline_grace_expires_but_final_is_immediate(tmp_path) -> None:
    timeline_dir = tmp_path / "timeline"
    timeline_dir.mkdir()
    driver_log = tmp_path / "driver.log"
    driver_log.write_text("rollout 5: {}\nstep 5: {}\nperf 5: {}\n")
    evidence_due_since = {}

    _validate_timelines(
        driver_log,
        timeline_dir,
        headline_lo=5,
        headline_hi=5,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=evidence_due_since,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="missing_headline_timeline:step=5"):
        _validate_timelines(
            driver_log,
            timeline_dir,
            headline_lo=5,
            headline_hi=5,
            final=False,
            evidence_grace=5.0,
            evidence_due_since=evidence_due_since,
            now_monotonic=15.0,
        )

    with pytest.raises(MonitorFailure, match="missing_headline_timeline:step=6"):
        _validate_timelines(
            driver_log,
            timeline_dir,
            headline_lo=6,
            headline_hi=6,
            final=True,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
        )


def test_online_monitor_defers_multiple_bootstrap_candidates_until_higher_gate() -> None:
    lines = [
        (
            f"TASK22_EVENT phase={phase} sync_id={sync_id} "
            f"t_begin={index}.0 t_end={index}.1 dur=0.1"
        )
        for sync_id in (1, 2)
        for index, phase in enumerate(("pause", "flush", "transfer", "continue"), 1)
    ]

    _validate_sync("\n".join(lines), final=False, expected_rollouts=2)

    lines.append("TASK22_EVENT phase=gate sync_id=3 t_begin=10.0 t_end=10.1 dur=0.1")
    with pytest.raises(MonitorFailure, match="multiple_bootstrap_sync_ids"):
        _validate_sync("\n".join(lines), final=False, expected_rollouts=2)


def test_online_monitor_final_strictly_rejects_multiple_bootstrap_candidates() -> None:
    driver_text = "\n".join(
        (
            f"TASK22_EVENT phase={phase} sync_id={sync_id} "
            f"t_begin={index}.0 t_end={index}.1 dur=0.1"
        )
        for sync_id in (1, 2)
        for index, phase in enumerate(("pause", "flush", "transfer", "continue"), 1)
    )

    with pytest.raises(MonitorFailure, match="multiple_bootstrap_sync_ids"):
        _validate_sync(driver_text, final=True, expected_rollouts=2)


@pytest.mark.parametrize(
    "fatal_line",
    [
        "CUDA out of memory",
        "torch.cuda.OutOfMemoryError: CUDA allocation failed",
        "worker OOM",
        "NVRM: Xid 79, GPU has fallen off the bus",
        "NCCL watchdog timeout",
        "RayActorError: actor died",
        "ActorDiedError: worker exited",
        "Ray actor worker failed",
        "engine failed during startup",
        "Traceback (most recent call last):",
        "Can not initialize distributed backend",
        "Job failed",
    ],
)
def test_online_monitor_rejects_required_fatal_signatures(tmp_path, fatal_line) -> None:
    (tmp_path / "driver.log").write_text(fatal_line + "\n")

    with pytest.raises(MonitorFailure, match="fatal_driver_log"):
        _scan_once(tmp_path)


def test_online_monitor_ignores_in_progress_outcome_snapshot_until_physical_end(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    (observability / "admission_outcomes_rollout_0.jsonl").write_text('{"record_type":\n')

    _scan_once(tmp_path)


def test_online_monitor_rejects_replayed_decision_after_physical_end(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    decision = {
        "record_type": "admission_decision",
        "physical_rollout_id": 0,
        "decision_id": "admission:0:1",
        "mode": "shadow",
        "release_remaining": 0,
        "logical_debt_groups": 0,
        "inflight_groups": 0,
        "available_groups": 8,
        "eager_admit_groups": 8,
        "desired_inflight_groups": 8,
        "bounded_admit_groups": 8,
        "actual_admit_groups": 8,
        "bypass_reason": None,
    }
    attempt = {
        "record_type": "attempt",
        "attempt_id": "attempt:1",
        "physical_rollout_id": 0,
        "outcome": "committed",
    }
    rows = [decision, decision, attempt]
    (observability / "admission_ledger_rollout_0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (observability / "request_lifecycle_rollout_0.jsonl").write_text(json.dumps(attempt) + "\n")
    (tmp_path / "driver.log").write_text(
        "TASK22_FLOW phase=physical_end physical_rollout_id=0 next_debt_groups=0\n"
    )

    with pytest.raises(MonitorFailure, match="duplicate_decision_id"):
        _scan_once(tmp_path)


def test_online_monitor_rejects_invalid_staleness_before_run_finishes(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    consume = {
        "record_type": "consume_outcome",
        "rollout_id": 0,
        "consume_outcome": "consumed",
        "target_partition": "train_0",
        "attempt_token": 1,
        "generation_end_version": 2,
        "consume_version": 5,
        "actual_staleness": 3,
    }
    (observability / "consumption_ledger_rollout_0_rank_0.jsonl").write_text(
        json.dumps(consume) + "\n"
    )

    with pytest.raises(MonitorFailure, match="invalid_staleness"):
        _scan_once(tmp_path)


def test_online_monitor_final_requires_exact_partition_set(tmp_path) -> None:
    _write_physical_flow(tmp_path)
    with pytest.raises(MonitorFailure, match="partition_set_mismatch"):
        _scan_once(tmp_path, final=True)


def test_online_monitor_final_requires_64_samples_per_partition(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    commit = {
        "record_type": "partition_outcome",
        "physical_rollout_id": 0,
        "outcome": "committed",
        "target_partition": "train_0",
        "attempt_token": 1,
    }
    consume = {
        "record_type": "consume_outcome",
        "rollout_id": 0,
        "consume_outcome": "consumed",
        "target_partition": "train_0",
        "attempt_token": 1,
        "generation_end_version": 0,
        "consume_version": 0,
        "actual_staleness": 0,
    }
    (observability / "admission_outcomes_rollout_0.jsonl").write_text(json.dumps(commit) + "\n")
    (observability / "consumption_ledger_rollout_0_rank_0.jsonl").write_text(
        json.dumps(consume) + "\n"
    )
    _write_physical_flow(tmp_path)

    with pytest.raises(MonitorFailure, match="partition_count_mismatch"):
        _scan_once(tmp_path, final=True)


def test_online_monitor_final_requires_bounded_path_to_trigger(tmp_path) -> None:
    observability = tmp_path / "observability"
    observability.mkdir()
    decision = {
        "record_type": "admission_decision",
        "physical_rollout_id": 0,
        "decision_id": "admission:0:1",
        "mode": "shadow",
        "release_remaining": 0,
        "inflight_groups": 0,
        "available_groups": 8,
        "eager_admit_groups": 8,
        "desired_inflight_groups": 8,
        "bounded_admit_groups": 8,
        "actual_admit_groups": 8,
        "bypass_reason": None,
    }
    (observability / "admission_ledger_rollout_0.jsonl").write_text(json.dumps(decision) + "\n")
    commits = []
    consumes = []
    for token in range(64):
        commits.append(
            {
                "record_type": "partition_outcome",
                "physical_rollout_id": 0,
                "outcome": "committed",
                "target_partition": "train_0",
                "attempt_token": token,
                "sample_index": token,
            }
        )
        consumes.append(
            {
                "record_type": "consume_outcome",
                "rollout_id": 0,
                "consume_outcome": "consumed",
                "target_partition": "train_0",
                "attempt_token": token,
                "sample_index": token,
                "generation_end_version": 0,
                "consume_version": 0,
                "actual_staleness": 0,
            }
        )
    (observability / "admission_outcomes_rollout_0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in commits)
    )
    (observability / "consumption_ledger_rollout_0_rank_0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in consumes)
    )
    _write_physical_flow(tmp_path)

    with pytest.raises(MonitorFailure, match="bounded_path_not_exercised"):
        _scan_once(tmp_path, final=True)


def test_process_group_stop_rejects_unrelated_group(monkeypatch) -> None:
    killed = []
    monkeypatch.setattr("os.getpgid", lambda pid: pid + 1)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append((pgid, sig)))

    with pytest.raises(MonitorFailure, match="unsafe_process_group"):
        _stop_process(123, 123)

    assert not killed


def test_strict_check_coverage_mapping_has_no_validator_omissions() -> None:
    validator_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "task22" / "validate_admission_run.py"
    )
    tree = ast.parse(validator_path.read_text(encoding="utf-8"))
    strict_checks = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "check"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }

    assert set(STRICT_CHECK_COVERAGE) == strict_checks
    assert set(STRICT_CHECK_COVERAGE.values()) == {"online", "final-only"}


def test_online_monitor_rejects_run_contract_fixed_field_drift(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    contract_path = tmp_path / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["expected_engines"] = 1
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(MonitorFailure, match="run_contract_mismatch"):
        _validate_contract(
            tmp_path,
            expected_mode="shadow",
            expected_rollouts=1,
            expected_samples_per_partition=64,
            expected_engines=2,
            max_staleness=2,
            admission_min=4,
            admission_max=8,
            admission_slack=2,
            headline_lo=5,
            headline_hi=6,
        )


def test_online_monitor_validates_closed_lifecycle_timing_rid_and_engine() -> None:
    rid = "relax:p0:kfresh:g0:s0:a0:00000000000a"
    row = {
        "rid": rid,
        "client_status": "finished",
        "rid_match": True,
        "dispatch_abs": 100.0,
        "request_end_abs": 102.0,
        "forward_entry_time": 100.1,
        "prefill_finished_time": 100.4,
        "queue_time": 0.1,
    }
    driver = "\n".join(
        [
            "(SGLangEngine pid=10) server_args=ServerArgs(base_gpu_id=2, tp_size=1)",
            "(SGLangEngine pid=11) server_args=ServerArgs(base_gpu_id=3, tp_size=1)",
            f'(SGLangEngine pid=10) {{"event":"request.received","rid":"{rid}","timestamp":100.2}}',
            f'(SGLangEngine pid=11) {{"event":"request.finished","rid":"{rid}","timestamp":101.8}}',
        ]
    )

    with pytest.raises(MonitorFailure, match="invalid_lifecycle_engine_interval"):
        _validate_closed_lifecycle([row], driver, expected_engines=2)


def test_online_monitor_headline_metrics_are_unique_and_finite() -> None:
    text = "\n".join(
        [
            "rollout 5: {'rollout/raw_reward': 1.0, 'rollout/response_lengths': 2.0, "
            "'rollout/rollout_log_probs': -1.0, 'rollout/total_lengths': 3.0}",
            "step 5: {'train/ppo_kl': 0.0, 'train/mismatch_kl': 0.0, "
            "'train/tis': 1.0, 'train/tis_clipfrac': 0.0}",
            "perf 5: {'perf/step_time': 1.0}",
            "perf 5: {'perf/step_time': 2.0}",
        ]
    )

    with pytest.raises(MonitorFailure, match="duplicate_headline_perf"):
        _validate_headline_metrics(text, headline_lo=5, headline_hi=5, final=False)


def test_online_monitor_gpu_snapshot_uses_grace_for_partial_write(tmp_path) -> None:
    gpu_log = tmp_path / "nvidia.csv"
    gpu_log.write_text(
        "2026-07-30T10:00:00.000000000+0800\n"
        "0, 100 MiB, 10 %, 20 %, 30 W\n"
    )
    due_since = {}

    _validate_gpu_snapshots(
        gpu_log,
        expected_engines=2,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=due_since,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="incomplete GPU snapshot"):
        _validate_gpu_snapshots(
            gpu_log,
            expected_engines=2,
            final=False,
            evidence_grace=5.0,
            evidence_due_since=due_since,
            now_monotonic=15.0,
        )


def test_online_monitor_physical_start_end_are_unique_and_ordered_with_grace() -> None:
    rows = [
        {
            "phase": "physical_end",
            "physical_rollout_id": "0",
            "t_begin": "1.0",
            "t_end": "2.0",
            "dur": "1.0",
        }
    ]
    due_since = {}
    _validate_physical_flow(
        rows,
        final=False,
        expected_rollouts=1,
        evidence_grace=5.0,
        evidence_due_since=due_since,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="invalid_physical_interval"):
        _validate_physical_flow(
            rows,
            final=False,
            expected_rollouts=1,
            evidence_grace=5.0,
            evidence_due_since=due_since,
            now_monotonic=15.0,
        )


def test_online_monitor_sync_grace_handles_aggregation_reordering() -> None:
    due_since = {}
    driver = "TASK22_EVENT phase=continue sync_id=1 t_begin=4 t_end=5 dur=1"
    _validate_sync(
        driver,
        final=False,
        expected_rollouts=1,
        evidence_grace=5.0,
        evidence_due_since=due_since,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="incomplete_sync_cycle"):
        _validate_sync(
            driver,
            final=False,
            expected_rollouts=1,
            evidence_grace=5.0,
            evidence_due_since=due_since,
            now_monotonic=15.0,
        )


def test_online_monitor_sync_rejects_bootstrap_after_runtime_after_grace() -> None:
    lines = [
        (
            f"TASK22_EVENT phase={phase} sync_id=bootstrap "
            f"t_begin={10 + index} t_end={11 + index} dur=1"
        )
        for index, phase in enumerate(("pause", "flush", "transfer", "continue"))
    ]
    lines.append("TASK22_EVENT phase=gate sync_id=1 t_begin=1 t_end=2 dur=1")
    due_since = {}
    _validate_sync(
        "\n".join(lines),
        final=False,
        expected_rollouts=1,
        evidence_grace=5.0,
        evidence_due_since=due_since,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="bootstrap_sync_does_not_precede_runtime"):
        _validate_sync(
            "\n".join(lines),
            final=False,
            expected_rollouts=1,
            evidence_grace=5.0,
            evidence_due_since=due_since,
            now_monotonic=15.0,
        )


def test_online_monitor_final_rejects_empty_or_incomplete_physical_flow() -> None:
    with pytest.raises(MonitorFailure, match="physical_flow_id_mismatch"):
        _validate_physical_flow(
            [],
            final=True,
            expected_rollouts=2,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
        )

    rows = [
        {"phase": "physical_start", "physical_rollout_id": "0", "t": "0"},
        {
            "phase": "physical_end",
            "physical_rollout_id": "0",
            "t_begin": "0",
            "t_end": "1",
            "dur": "1",
        },
    ]
    with pytest.raises(MonitorFailure, match=r"expected=\[0, 1\]"):
        _validate_physical_flow(
            rows,
            final=True,
            expected_rollouts=2,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
        )


def test_online_monitor_final_requires_flow_for_declared_final_artifact(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    observability = tmp_path / "observability"
    observability.mkdir()
    _write_physical_flow(tmp_path)
    (observability / "admission_ledger_rollout_1.jsonl").write_text("")

    with pytest.raises(MonitorFailure, match=r"expected=\[0, 1\].*completed=\[0\]"):
        _scan(
            tmp_path,
            expected_mode="shadow",
            expected_rollouts=1,
            expected_samples_per_partition=64,
            expected_engines=2,
            max_staleness=2,
            admission_min=4,
            admission_max=8,
            admission_slack=2,
            headline_lo=5,
            headline_hi=6,
            final=True,
            reported_lifecycle=set(),
            event_log=tmp_path / "online_monitor.jsonl",
        )


def test_online_monitor_ignores_trailing_incomplete_driver_line(tmp_path) -> None:
    (tmp_path / "driver.log").write_text("healthy complete line\nTraceback (most recent call last):")

    _scan_once(tmp_path)


def test_online_monitor_ledger_only_set_uses_grace_and_final_is_strict(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    observability = tmp_path / "observability"
    observability.mkdir()
    _write_physical_flow(tmp_path)
    ledger = {
        "record_type": "attempt",
        "physical_rollout_id": 0,
        "attempt_id": "ledger-only",
        "outcome": "committed",
    }
    (observability / "admission_ledger_rollout_0.jsonl").write_text(json.dumps(ledger) + "\n")
    due_since = {}
    scan_kwargs = {
        "expected_mode": "shadow",
        "expected_rollouts": 1,
        "expected_samples_per_partition": 64,
        "expected_engines": 2,
        "max_staleness": 2,
        "admission_min": 4,
        "admission_max": 8,
        "admission_slack": 2,
        "headline_lo": 5,
        "headline_hi": 6,
        "reported_lifecycle": set(),
        "event_log": tmp_path / "online_monitor.jsonl",
        "evidence_grace": 5.0,
        "strict_evidence_due_since": due_since,
    }

    _scan(tmp_path, final=False, now_monotonic=10.0, **scan_kwargs)
    with pytest.raises(MonitorFailure, match="ledger_only=.*ledger-only"):
        _scan(tmp_path, final=False, now_monotonic=15.0, **scan_kwargs)
    with pytest.raises(MonitorFailure, match="ledger_only=.*ledger-only"):
        _scan(tmp_path, final=True, now_monotonic=10.0, **scan_kwargs)


def test_online_monitor_closed_lifecycle_uses_grace_and_final_is_strict(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    observability = tmp_path / "observability"
    observability.mkdir()
    _write_physical_flow(tmp_path)
    attempt = {
        "record_type": "attempt",
        "physical_rollout_id": 0,
        "attempt_id": "attempt:1",
        "rid": "relax:p0:kfresh:g0:s0:a0:000000000001",
        "outcome": "committed",
        "client_status": "finished",
        "rid_match": True,
    }
    for name in ("request_lifecycle_rollout_0.jsonl", "admission_ledger_rollout_0.jsonl"):
        (observability / name).write_text(json.dumps(attempt) + "\n")
    due_since = {}
    scan_kwargs = {
        "expected_mode": "shadow",
        "expected_rollouts": 1,
        "expected_samples_per_partition": 64,
        "expected_engines": 2,
        "max_staleness": 2,
        "admission_min": 4,
        "admission_max": 8,
        "admission_slack": 2,
        "headline_lo": 5,
        "headline_hi": 6,
        "reported_lifecycle": set(),
        "event_log": tmp_path / "online_monitor.jsonl",
        "evidence_grace": 5.0,
        "strict_evidence_due_since": due_since,
    }

    _scan(tmp_path, final=False, now_monotonic=10.0, **scan_kwargs)
    with pytest.raises(MonitorFailure, match="engine_mapping_mismatch"):
        _scan(tmp_path, final=False, now_monotonic=15.0, **scan_kwargs)
    with pytest.raises(MonitorFailure, match="engine_mapping_mismatch"):
        _scan(tmp_path, final=True, now_monotonic=10.0, **scan_kwargs)


def test_online_monitor_gpu_indices_need_only_be_unique_nonnegative(tmp_path) -> None:
    gpu_log = tmp_path / "nvidia.csv"
    snapshot = (
        "2026-07-30T10:00:00+0800\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "3, 100 MiB, 10 %, 20 %, 30 W\n"
        "6, 100 MiB, 10 %, 20 %, 30 W\n"
        "7, 100 MiB, 10 %, 20 %, 30 W\n"
        "2026-07-30T10:00:01+0800\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "3, 100 MiB, 10 %, 20 %, 30 W\n"
        "6, 100 MiB, 10 %, 20 %, 30 W\n"
        "7, 100 MiB, 10 %, 20 %, 30 W\n"
    )
    gpu_log.write_text(snapshot)

    _validate_gpu_snapshots(
        gpu_log,
        expected_engines=2,
        final=True,
        evidence_grace=5.0,
        evidence_due_since={},
        now_monotonic=0.0,
    )
