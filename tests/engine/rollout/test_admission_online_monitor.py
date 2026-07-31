# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import json
import os
import signal
from datetime import datetime
from pathlib import Path

import pytest

from scripts.task22.monitor_admission_run import (
    STRICT_CHECK_COVERAGE,
    FileCursor,
    MonitorFailure,
    MonitorScanState,
    _complete_driver_text,
    _evidence_progress_token,
    _process_exists,
    _read_incremental_bytes,
    _read_retained_jsonl,
    _scan,
    _stop_process,
    _structured_rows,
    _validate_closed_lifecycle,
    _validate_contract,
    _validate_gpu_snapshots,
    _validate_headline_metrics,
    _validate_physical_flow,
    _validate_sync,
    _validate_timelines,
    _valid_interval,
    monitor,
)


def _prepare_monitor_inputs(run_dir) -> None:
    digest = "a" * 64
    contract = {
        "schema_version": 6,
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
        "working_dir": str(run_dir.resolve()),
        "working_dir_content_sha256": digest,
        "runtime_env_json_sha256": digest,
        "task22_env_sha256": digest,
        "input_manifest_sha256": digest,
        "monitor_poll_interval_s": 1.0,
        "monitor_evidence_grace_s": 5.0,
        "monitor_no_progress_timeout_s": 600.0,
        "gpu_max_snapshot_age_s": 5.0,
        "gpu_max_snapshot_interval_s": 2.0,
        "monitor_timeout_s": 5700,
        "monitor_term_grace_s": 1.0,
        "training_term_timeout_s": 10.0,
        "num_gpus": 4,
        "cuda_visible_devices": "",
        "training_python": {},
        "sglang_source_sha256": {},
    }
    (run_dir / "run_contract.json").write_text(json.dumps(contract) + "\n")
    attestation_dir = run_dir / "runtime_attestation"
    attestation_dir.mkdir(exist_ok=True)
    common = {
        "working_dir": contract["working_dir"],
        "working_dir_content_sha256": digest,
        "runtime_env_json_sha256": digest,
        "task22_env_sha256": digest,
        "python": {},
        "sglang_source_sha256": {},
        "input_manifest": {"sha256": digest},
    }
    for role, rank in (
        ("driver", None),
        ("actor", 0),
        ("actor", 1),
        ("rollout_engine", 0),
        ("rollout_engine", 1),
    ):
        suffix = role if rank is None else f"{role}_{rank}"
        (attestation_dir / f"runtime_attestation_{suffix}.json").write_text(
            json.dumps({**common, "role": role, "rank": rank}) + "\n"
        )
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


def test_online_monitor_uses_content_hash_not_ray_unpack_path(tmp_path) -> None:
    digest = "a" * 64
    contract = {
        "schema_version": 6,
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
        "working_dir": "/source/repo",
        "working_dir_content_sha256": digest,
        "runtime_env_json_sha256": digest,
        "task22_env_sha256": digest,
        "input_manifest_sha256": digest,
        "monitor_poll_interval_s": 1.0,
        "monitor_evidence_grace_s": 5.0,
        "monitor_no_progress_timeout_s": 600.0,
        "gpu_max_snapshot_age_s": 5.0,
        "gpu_max_snapshot_interval_s": 2.0,
        "monitor_timeout_s": 5700,
        "monitor_term_grace_s": 1.0,
        "training_term_timeout_s": 10.0,
        "num_gpus": 4,
        "cuda_visible_devices": "",
        "training_python": {},
        "sglang_source_sha256": {},
    }
    (tmp_path / "run_contract.json").write_text(json.dumps(contract) + "\n")
    attestation_dir = tmp_path / "runtime_attestation"
    attestation_dir.mkdir()
    for role, rank in (("driver", None), ("actor", 0), ("actor", 1)):
        attestation = {
            "role": role,
            "rank": rank,
            "working_dir": f"/tmp/ray/session/{role}",
            "working_dir_content_sha256": digest,
            "runtime_env_json_sha256": digest,
            "task22_env_sha256": digest,
            "python": {},
            "sglang_source_sha256": {},
            "input_manifest": {"sha256": digest},
        }
        suffix = role if rank is None else f"{role}_{rank}"
        (attestation_dir / f"runtime_attestation_{suffix}.json").write_text(
            json.dumps(attestation) + "\n"
        )
    for rank in range(2):
        attestation = {
            "role": "rollout_engine",
            "rank": rank,
            "working_dir": f"/tmp/ray/session/rollout_engine_{rank}",
            "working_dir_content_sha256": digest,
            "runtime_env_json_sha256": digest,
            "task22_env_sha256": digest,
            "python": {},
            "sglang_source_sha256": {},
            "input_manifest": {"sha256": digest},
        }
        (attestation_dir / f"runtime_attestation_rollout_engine_{rank}.json").write_text(
            json.dumps(attestation) + "\n"
        )

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
        final=True,
    )

    worker_path = attestation_dir / "runtime_attestation_actor_0.json"
    worker = json.loads(worker_path.read_text())
    worker["working_dir_content_sha256"] = "b" * 64
    worker_path.write_text(json.dumps(worker) + "\n")
    with pytest.raises(MonitorFailure, match="runtime_attestation_contract_mismatch"):
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
            final=True,
        )


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


def test_process_group_stop_escalates_to_kill_after_bounded_term(monkeypatch) -> None:
    signals = []
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_start_identity",
        lambda pid: "start-1",
    )

    _stop_process(
        123,
        123,
        expected_identity="start-1",
        term_timeout=0,
        poll_interval=0.001,
    )

    assert signals[0] == signal.SIGTERM
    assert signals[-1] == signal.SIGKILL


def test_monitor_kills_even_when_failure_audit_cannot_be_written(tmp_path, monkeypatch) -> None:
    stopped = []

    def fail_scan(*args, **kwargs):
        raise MonitorFailure("fixture failure")

    monkeypatch.setattr("scripts.task22.monitor_admission_run._scan", fail_scan)
    monkeypatch.setattr("scripts.task22.monitor_admission_run._process_exists", lambda pid: True)
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_start_identity",
        lambda pid: "start-1",
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._append_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._stop_process",
        lambda *args, **kwargs: stopped.append(args[0]),
    )

    rc = monitor(
        tmp_path,
        pid=123,
        process_group_id=123,
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
        poll_interval=0.01,
        evidence_grace=5.0,
        pid_start_identity="start-1",
    )

    assert rc == 4
    assert stopped == [123]


def test_monitor_fails_closed_when_running_evidence_makes_no_progress(tmp_path, monkeypatch) -> None:
    times = iter((0.0, 0.0, 601.0))
    stopped = []
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.monotonic", lambda: next(times))
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.sleep", lambda delay: None)
    monkeypatch.setattr("scripts.task22.monitor_admission_run._scan", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.task22.monitor_admission_run._process_exists", lambda pid: True)
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_start_identity",
        lambda pid: "start-1",
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._stop_process",
        lambda *args, **kwargs: stopped.append(args[0]),
    )

    rc = monitor(
        tmp_path,
        pid=123,
        process_group_id=123,
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
        poll_interval=0.01,
        evidence_grace=5.0,
        pid_start_identity="start-1",
        no_progress_timeout=600.0,
    )

    assert rc == 4
    assert stopped == [123]
    rows = [json.loads(line) for line in (tmp_path / "online_monitor.jsonl").read_text().splitlines()]
    assert any(row.get("reason") == "no_evidence_progress:600s" for row in rows)


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


@pytest.mark.parametrize(
    "field",
    (
        "monitor_poll_interval_s",
        "monitor_evidence_grace_s",
        "monitor_no_progress_timeout_s",
        "gpu_max_snapshot_age_s",
        "gpu_max_snapshot_interval_s",
        "monitor_timeout_s",
        "monitor_term_grace_s",
        "training_term_timeout_s",
    ),
)
def test_online_monitor_rejects_supervision_argument_contract_drift_999(
    tmp_path, field
) -> None:
    _prepare_monitor_inputs(tmp_path)
    contract_path = tmp_path / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract[field] = 999
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


def test_online_monitor_accepts_backend_aborted_terminal_lifecycle() -> None:
    rid = "relax:p1:kfresh:g9:s72:a0:00000000000b"
    row = {
        "rid": rid,
        "outcome": "aborted",
        "client_status": "request_aborted",
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
            f'(SGLangEngine pid=10) {{"event":"request.received","rid":"{rid}","timestamp":100.2}}',
            f'(SGLangEngine pid=10) {{"event":"request.finished","rid":"{rid}","timestamp":101.8}}',
        ]
    )

    _validate_closed_lifecycle([row], driver, expected_engines=1)

    row["rid_match"] = False
    with pytest.raises(MonitorFailure, match="invalid_lifecycle_terminal_rid"):
        _validate_closed_lifecycle([row], driver, expected_engines=1)


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


def test_online_monitor_ignores_trailing_partial_gpu_snapshot_until_final(tmp_path) -> None:
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
    _validate_gpu_snapshots(
        gpu_log,
        expected_engines=2,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=due_since,
        now_monotonic=15.0,
    )
    with pytest.raises(MonitorFailure, match="incomplete GPU snapshot"):
        _validate_gpu_snapshots(
            gpu_log,
            expected_engines=2,
            final=True,
            evidence_grace=5.0,
            evidence_due_since=due_since,
            now_monotonic=15.0,
        )


def test_online_monitor_partial_gpu_tail_keeps_last_complete_snapshot_for_freshness(tmp_path) -> None:
    gpu_log = tmp_path / "nvidia.csv"
    gpu_log.write_text(
        "2026-07-30T10:00:00.000000000+0800\n"
        "0, 100 MiB, 10 %, 20 %, 30 W\n"
        "1, 100 MiB, 10 %, 20 %, 30 W\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "3, 100 MiB, 10 %, 20 %, 30 W\n"
        "2026-07-30T10:00:01.000000000+0800\n"
        "0, 101 MiB, 11 %, 21 %, 31 W\n"
        "1, 101 MiB, 11 %, 21 %, 31 W\n"
        "2, 101 MiB, 11 %, 21 %, 31 W\n"
    )
    last_complete = datetime.fromisoformat("2026-07-30T10:00:00+08:00").timestamp()

    _validate_gpu_snapshots(
        gpu_log,
        expected_engines=2,
        final=False,
        evidence_grace=5.0,
        evidence_due_since={},
        now_monotonic=10.0,
        wall_time=last_complete + 4,
        max_snapshot_age=5.0,
    )
    with pytest.raises(MonitorFailure, match="stale_gpu_snapshot"):
        _validate_gpu_snapshots(
            gpu_log,
            expected_engines=2,
            final=False,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=15.0,
            wall_time=last_complete + 6,
            max_snapshot_age=5.0,
        )


def test_online_monitor_does_not_ignore_invalid_complete_gpu_snapshot(tmp_path) -> None:
    gpu_log = tmp_path / "nvidia.csv"
    gpu_log.write_text(
        "2026-07-30T10:00:00.000000000+0800\n"
        "0, 100 MiB, 10 %, 20 %, 30 W\n"
        "1, 100 MiB, 10 %, 20 %, 30 W\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "2, 100 MiB, 10 %, 20 %, 30 W\n"
        "2026-07-30T10:00:01.000000000+0800\n"
        "0, 101 MiB, 11 %, 21 %, 31 W\n"
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
    with pytest.raises(MonitorFailure, match="GPU indices must be unique"):
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


def test_online_monitor_preserves_complete_generic_traceback_for_exit_diagnosis(tmp_path) -> None:
    (tmp_path / "driver.log").write_text(
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in <module>\n'
        "    raise RuntimeError('diagnostic')\n"
        "RuntimeError: diagnostic\n"
    )

    _scan_once(tmp_path)


def test_structured_event_interval_ignores_ray_ansi_suffix() -> None:
    rows = _structured_rows(
        "TASK22_EVENT phase=pause sync_id=1 "
        "t_begin=1785483054.045624 t_end=1785483054.053882 dur=0.008258\x1b[0m\n",
        "TASK22_EVENT",
    )

    assert len(rows) == 1
    assert _valid_interval(rows[0])


def test_online_monitor_final_reads_valid_unterminated_driver_tail() -> None:
    assert _complete_driver_text("healthy\nFATAL", final=False) == "healthy\n"
    assert _complete_driver_text("healthy\nFATAL", final=True) == "healthy\nFATAL"


def test_online_monitor_retains_driver_evidence_across_truncation(tmp_path) -> None:
    _prepare_monitor_inputs(tmp_path)
    state = MonitorScanState()
    driver = tmp_path / "driver.log"
    driver.write_text("healthy first segment\n")
    kwargs = {
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
        "final": False,
        "reported_lifecycle": set(),
        "event_log": tmp_path / "online_monitor.jsonl",
        "scan_state": state,
    }
    _scan(tmp_path, **kwargs)
    driver.write_text("FATAL after rotation\n")

    with pytest.raises(MonitorFailure, match="fatal_driver_log"):
        _scan(tmp_path, **kwargs)
    assert "healthy first segment" in state.driver_text


def test_consumption_half_write_uses_grace_and_keeps_previous_rows(tmp_path) -> None:
    path = tmp_path / "consumption_ledger_rollout_0_rank_0.jsonl"
    row = {"record_type": "consume_outcome", "attempt_token": 1}
    path.write_text(json.dumps(row) + "\n")
    state = MonitorScanState()
    due = {}
    first = _read_retained_jsonl(
        path,
        final=False,
        state=state,
        evidence_grace=5.0,
        evidence_due_since=due,
        now_monotonic=10.0,
    )
    path.write_text(json.dumps(row) + '\n{"record_type":')

    retained = _read_retained_jsonl(
        path,
        final=False,
        state=state,
        evidence_grace=5.0,
        evidence_due_since=due,
        now_monotonic=11.0,
    )
    assert retained == first
    with pytest.raises(MonitorFailure, match="malformed_jsonl"):
        _read_retained_jsonl(
            path,
            final=False,
            state=state,
            evidence_grace=5.0,
            evidence_due_since=due,
            now_monotonic=16.0,
        )


def test_timeline_half_write_uses_grace(tmp_path) -> None:
    driver = tmp_path / "driver.log"
    driver.write_text("rollout 5: {}\nstep 5: {}\nperf 5: {}\n")
    timeline = tmp_path / "timeline"
    timeline.mkdir()
    (timeline / "timeline_step_5.json").write_text('[{"name":')
    due = {}

    _validate_timelines(
        driver,
        timeline,
        headline_lo=5,
        headline_hi=5,
        final=False,
        evidence_grace=5.0,
        evidence_due_since=due,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="invalid_headline_timeline"):
        _validate_timelines(
            driver,
            timeline,
            headline_lo=5,
            headline_hi=5,
            final=False,
            evidence_grace=5.0,
            evidence_due_since=due,
            now_monotonic=15.0,
        )


def test_process_identity_mismatch_is_not_considered_same_process(monkeypatch) -> None:
    monkeypatch.setattr("os.kill", lambda pid, sig: None)
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_start_identity",
        lambda pid: "new-start",
    )

    assert not _process_exists(123, "old-start")


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


def test_retained_jsonl_same_record_after_truncate_is_a_visible_replay(tmp_path) -> None:
    path = tmp_path / "admission_ledger_rollout_0.jsonl"
    row = {"record_type": "admission_decision", "decision_id": "same"}
    encoded = json.dumps(row) + "\n"
    path.write_text(encoded)
    state = MonitorScanState()
    kwargs = {
        "final": False,
        "state": state,
        "evidence_grace": 5.0,
        "evidence_due_since": {},
    }

    first = _read_retained_jsonl(path, now_monotonic=1.0, **kwargs)
    path.write_text(encoded)
    state.jsonl_cursors[str(path)].ctime_ns = -1
    replayed = _read_retained_jsonl(path, now_monotonic=2.0, **kwargs)

    assert len(first) == 2  # the retained list is updated in place
    assert len(replayed) == 2
    assert [item["_epoch"] for item in replayed] == [0, 1]
    with pytest.raises(MonitorFailure, match="duplicate_decision_id"):
        from scripts.task22.monitor_admission_run import _require_unique

        _require_unique(replayed, "decision_id", "duplicate_decision_id")


def test_mtime_only_change_is_not_semantic_progress(tmp_path) -> None:
    state = MonitorScanState()
    artifact = tmp_path / "observability" / "rows.jsonl"
    artifact.parent.mkdir()
    artifact.write_text("")
    before = _evidence_progress_token(tmp_path, state)

    os.utime(artifact, None)

    assert _evidence_progress_token(tmp_path, state) == before


def test_complete_nonsemantic_driver_noise_is_not_progress(tmp_path) -> None:
    state = MonitorScanState()
    driver = tmp_path / "driver.log"
    driver.write_text("ordinary library chatter\n")
    before = _evidence_progress_token(tmp_path, state)

    from scripts.task22.monitor_admission_run import _read_incremental_driver

    _read_incremental_driver(driver, state, final=False)

    assert _evidence_progress_token(tmp_path, state) == before


def test_incremental_jsonl_poll_does_not_replay_unchanged_bytes(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value": 1}\n')
    state = MonitorScanState()
    kwargs = {
        "final": False,
        "state": state,
        "evidence_grace": 5.0,
        "evidence_due_since": {},
    }

    rows = _read_retained_jsonl(path, now_monotonic=1.0, **kwargs)
    offset = state.jsonl_cursors[str(path)].offset
    revision = state.semantic_revision
    again = _read_retained_jsonl(path, now_monotonic=2.0, **kwargs)

    assert again == rows
    assert state.jsonl_cursors[str(path)].offset == offset
    assert state.semantic_revision == revision


def test_incremental_reader_does_not_treat_concurrent_append_as_new_epoch(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "rows.jsonl"
    first = b'{"value": 1}\n'
    appended = b'{"value": 2}\n'
    path.write_bytes(first)
    cursor = FileCursor()
    real_fstat = os.fstat
    calls = 0

    def append_after_first_fstat(fd):
        nonlocal calls
        calls += 1
        observed = real_fstat(fd)
        if calls == 1:
            with path.open("ab") as output:
                output.write(appended)
        return observed

    monkeypatch.setattr("scripts.task22.monitor_admission_run.os.fstat", append_after_first_fstat)

    first_poll = _read_incremental_bytes(path, cursor, final=False)
    assert [raw for _, _, raw in first_poll] == [first.rstrip()]
    assert cursor.offset == len(first)
    assert cursor.epoch == 0

    second_poll = _read_incremental_bytes(path, cursor, final=False)
    assert [raw for _, _, raw in second_poll] == [appended.rstrip()]
    assert cursor.offset == len(first) + len(appended)
    assert cursor.epoch == 0


def test_complete_malformed_jsonl_remains_due_across_incremental_polls(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"broken":}\n')
    state = MonitorScanState()
    due = {}

    _read_retained_jsonl(
        path,
        final=False,
        state=state,
        evidence_grace=5.0,
        evidence_due_since=due,
        now_monotonic=10.0,
    )
    with pytest.raises(MonitorFailure, match="malformed_jsonl"):
        _read_retained_jsonl(
            path,
            final=False,
            state=state,
            evidence_grace=5.0,
            evidence_due_since=due,
            now_monotonic=15.0,
        )


def test_monitor_waits_for_process_group_and_then_stable_final_scan(tmp_path, monkeypatch) -> None:
    scans = []
    groups = iter((True, False))
    times = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.monotonic", lambda: next(times))
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.sleep", lambda _: None)
    monkeypatch.setattr("scripts.task22.monitor_admission_run._process_exists", lambda *args: False)
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_group_exists",
        lambda pgid: next(groups),
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._scan",
        lambda *args, **kwargs: scans.append(kwargs["final"]),
    )

    rc = monitor(
        tmp_path,
        pid=123,
        process_group_id=123,
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
        poll_interval=0.01,
        evidence_grace=5.0,
        pid_start_identity="start-1",
        post_exit_grace=0,
    )

    assert rc == 0
    assert scans == [False, False, True]


def test_monitor_restarts_post_exit_grace_when_non_final_scan_finds_evidence(tmp_path, monkeypatch) -> None:
    scans = []
    times = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.monotonic", lambda: next(times))
    monkeypatch.setattr("scripts.task22.monitor_admission_run.time.sleep", lambda _: None)
    monkeypatch.setattr("scripts.task22.monitor_admission_run._process_exists", lambda *args: False)

    def scan(*args, **kwargs):
        scans.append(kwargs["final"])
        if len(scans) == 1:
            kwargs["scan_state"].semantic_revision += 1

    monkeypatch.setattr("scripts.task22.monitor_admission_run._scan", scan)

    rc = monitor(
        tmp_path,
        pid=123,
        process_group_id=None,
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
        poll_interval=0.01,
        evidence_grace=5.0,
        pid_start_identity="start-1",
        post_exit_grace=0,
    )

    assert rc == 0
    assert scans == [False, False, True]


def test_monitor_fails_closed_when_gpu_sampler_identity_disappears(tmp_path, monkeypatch) -> None:
    stopped = []
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_exists",
        lambda pid, expected_identity=None: pid == 123,
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._process_start_identity",
        lambda pid: "training-start" if pid == 123 else None,
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_run._stop_process",
        lambda *args, **kwargs: stopped.append(args[0]),
    )

    rc = monitor(
        tmp_path,
        pid=123,
        process_group_id=123,
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
        poll_interval=0.01,
        evidence_grace=5.0,
        pid_start_identity="training-start",
        sampler_pid=456,
        sampler_start_identity="sampler-start",
    )

    assert rc == 4
    assert stopped == [123]


def test_gpu_snapshots_must_be_fresh_and_cover_run_interval(tmp_path) -> None:
    path = tmp_path / "nvidia.csv"
    path.write_text(
        "2026-07-30T10:00:00+0800\n"
        "0, 1 MiB, 1 %, 1 %, 1 W\n"
        "1, 1 MiB, 1 %, 1 %, 1 W\n"
        "2, 1 MiB, 1 %, 1 %, 1 W\n"
        "3, 1 MiB, 1 %, 1 %, 1 W\n"
        "2026-07-30T10:00:01+0800\n"
        "0, 1 MiB, 1 %, 1 %, 1 W\n"
        "1, 1 MiB, 1 %, 1 %, 1 W\n"
        "2, 1 MiB, 1 %, 1 %, 1 W\n"
        "3, 1 MiB, 1 %, 1 %, 1 W\n"
    )
    timestamp = datetime.fromisoformat("2026-07-30T10:00:01+08:00").timestamp()

    with pytest.raises(MonitorFailure, match="stale_gpu_snapshot"):
        _validate_gpu_snapshots(
            path,
            expected_engines=2,
            final=False,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
            wall_time=timestamp + 6,
            max_snapshot_age=5.0,
        )
    with pytest.raises(MonitorFailure, match="gpu_coverage_ends_early"):
        _validate_gpu_snapshots(
            path,
            expected_engines=2,
            final=True,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
            run_ended_at=timestamp + 4,
        )


def test_gpu_snapshots_reject_excessive_internal_sampling_gap(tmp_path) -> None:
    path = tmp_path / "nvidia.csv"
    snapshots = []
    for timestamp in ("2026-07-30T10:00:00+0800", "2026-07-30T10:00:04+0800"):
        snapshots.append(timestamp)
        snapshots.extend(f"{index}, 1 MiB, 1 %, 1 %, 1 W" for index in range(4))
    path.write_text("\n".join(snapshots) + "\n")

    _validate_gpu_snapshots(
        path,
        expected_engines=2,
        final=False,
        evidence_grace=5.0,
        evidence_due_since={},
        now_monotonic=0.0,
        wall_time=datetime.fromisoformat("2026-07-30T10:00:04+08:00").timestamp() + 1,
        max_snapshot_age=5.0,
        max_snapshot_interval=2.0,
    )
    with pytest.raises(MonitorFailure, match="gpu_snapshot_gap"):
        _validate_gpu_snapshots(
            path,
            expected_engines=2,
            final=True,
            evidence_grace=5.0,
            evidence_due_since={},
            now_monotonic=0.0,
            max_snapshot_interval=2.0,
        )
