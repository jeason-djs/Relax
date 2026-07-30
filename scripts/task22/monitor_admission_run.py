#!/usr/bin/env python3
"""Fail-closed online evidence monitor for a Task 22 admission run."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shlex
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TERMINAL_OUTCOMES = {
    "aborted",
    "carried_with_aborted_group",
    "committed",
    "filtered",
    "surplus",
    "uncommitted_protected",
}
METRIC_STEP_RE = re.compile(r"\b(rollout|step|perf) (\d+):")
ARTIFACT_ID_RE = re.compile(r"_(\d+)(?:_rank_\d+)?\.jsonl$")
FATAL_RE = re.compile(
    r"(?:"
    r"\bFATAL\b|"
    r"Traceback \(most recent call last\)|"
    r"\b(?:out[ -]of[ -]memory|OutOfMemoryError|OOM)\b|"
    r"\bXid(?:\s+\d+)?\b|"
    r"\bNCCL\b[^\n]*(?:error|abort|fail|timeout)|"
    r"\b(?:RayTaskError|RayActorError|ActorDiedError)\b|"
    r"\bRay actor\b[^\n]*(?:died|error|fail)|"
    r"\bengine failed\b|"
    r"\bCan not initialize\b|"
    r"\bJob failed\b|"
    r"\bCUDA error\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
SYNC_PHASES = ("pause", "flush", "transfer", "continue")
SYNC_ALL_PHASES = ("gate", *SYNC_PHASES)
EVENT_PHASES = {*SYNC_ALL_PHASES, "abort"}
FLOW_PHASES = {"gate_blocked", "gate_ready", "physical_start", "physical_end", "partition_close"}
GPU_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?([+-]\d{4})$"
)
ROLLOUT_METRICS_RE = re.compile(r"\brollout (\d+): (\{.*\})")
TRAIN_METRICS_RE = re.compile(r"\bstep (\d+): (\{.*\})")
PERF_METRICS_RE = re.compile(r"\bperf (\d+): (\{.*\})")
ENGINE_PID_RE = re.compile(r"\(SGLangEngine pid=(\d+)\)")
BASE_GPU_RE = re.compile(r"\bbase_gpu_id=(\d+)")
SERVER_EVENT_RE = re.compile(r"(\{.*\})\s*$")
SERVER_ENVELOPE_TOLERANCE_S = 1.0
REQUIRED_ROLLOUT_METRICS = {
    "rollout/raw_reward",
    "rollout/response_lengths",
    "rollout/rollout_log_probs",
    "rollout/total_lengths",
}
REQUIRED_TRAIN_METRICS = {
    "train/ppo_kl",
    "train/mismatch_kl",
    "train/tis",
    "train/tis_clipfrac",
}

# Every literal Validation.check() in validate_admission_run.py must be classified
# here. "online" means the monitor enforces the condition as soon as its evidence
# is stable; "final-only" means absence/cardinality can only be decided after exit.
ONLINE_STRICT_CHECKS = {
    "actual_staleness_in_bounds",
    "actual_staleness_matches_versions",
    "admission_bounded_recomputes",
    "admission_contract_is_4_8_2",
    "admission_decision_ids_unique",
    "admission_has_no_fail_open",
    "admission_mode_matches_contract",
    "admission_mode_semantics",
    "artifact_filenames_parseable",
    "attempt_rows_have_terminal_business_outcome",
    "attempt_tokens_globally_unique_in_commits",
    "attempt_tokens_globally_unique_in_consumes",
    "bootstrap_sync_cycle_complete",
    "bootstrap_sync_phase_order_valid",
    "bootstrap_sync_precedes_runtime",
    "commit_and_consume_decisions_match",
    "commit_and_consume_identity_match",
    "consume_outcomes_terminal",
    "consume_record_types_valid",
    "final_backfill_actual_within_debt",
    "gpu_snapshots_parse_strictly",
    "headline_perf_metrics_unique",
    "headline_quality_fields_present",
    "headline_quality_values_finite",
    "headline_rollout_metrics_unique",
    "headline_step_time_positive",
    "headline_timeline_files_complete",
    "headline_train_metrics_unique",
    "jsonl_parseable",
    "jsonl_readable",
    "jsonl_rows_are_objects",
    "keyed_sync_cycles_complete",
    "keyed_sync_phase_order_valid",
    "ledger_attempts_match_request_rows",
    "ledger_file_physical_ids_match",
    "partition_outcome_record_types_valid",
    "partition_outcomes_committed",
    "partition_samples_unique",
    "physical_flow_complete",
    "physical_flow_interval_valid",
    "physical_rollout_ids_allowed",
    "request_attempt_ids_unique",
    "request_placement_is_off",
    "run_contract_matches_validator_arguments",
    "run_contract_valid",
    "runtime_keyed_sync_id_sets_match",
    "slime_router_is_disabled",
    "timeline_event_intervals_valid",
    "timeline_event_phases_valid",
    "timeline_flow_phases_valid",
    "unique_bootstrap_sync_cycle",
}
FINAL_ONLY_STRICT_CHECKS = {
    "abort_within_physical_interval",
    "admission_bounded_path_exercised",
    "attempt_decision_exists",
    "attempt_parent_chains_valid",
    "commit_attempt_exists",
    "commit_decision_exists",
    "commit_decision_physical_id_matches",
    "commit_has_exactly_one_consume",
    "commit_matches_attempt_identity",
    "committed_and_consumed_attempt_tokens_match",
    "consume_attempt_exists",
    "consume_decision_exists",
    "consume_decision_physical_id_matches",
    "consume_has_exactly_one_commit",
    "consumption_file_partition_ids_match",
    "consumption_ledgers_present",
    "consumption_partition_files_exact",
    "driver_log_present",
    "expected_admission_ledgers_exact",
    "expected_partitions_consumed_exact",
    "expected_request_files_exact",
    "gate_blocked_flow_not_duplicated",
    "gate_blocked_precedes_ready",
    "gate_cycle_count_exact",
    "gate_flow_sync_ids_closed",
    "gate_flow_within_event_interval",
    "gate_ready_flow_complete",
    "gate_ready_partition_is_candidate",
    "gpu_sample_present",
    "gpu_sampling_has_multiple_snapshots",
    "observability_directory_present",
    "outcome_file_decision_physical_ids_match",
    "partition_close_flow_complete",
    "partition_close_partitions_exact",
    "partition_close_within_physical_interval",
    "partition_outcomes_present",
    "partition_sample_count_exact",
    "physical_abort_events_complete",
    "process_exit_zero",
    "quality_metrics_written",
    "request_attempt_tokens_derivable",
    "request_attempt_tokens_unique",
    "request_file_physical_ids_match",
    "request_observability_passes",
    "run_directory_exists",
    "timeline_directory_present",
}
STRICT_CHECK_COVERAGE = {
    **{name: "online" for name in ONLINE_STRICT_CHECKS},
    **{name: "final-only" for name in FINAL_ONLY_STRICT_CHECKS},
}


class MonitorFailure(RuntimeError):
    pass


def _append_event(path: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MonitorFailure(f"malformed_jsonl:{path.name}:{line_no}:{exc.msg}") from exc
            if not isinstance(row, dict):
                raise MonitorFailure(f"non_object_jsonl:{path.name}:{line_no}")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _complete_driver_text(text: str) -> str:
    """Discard the trailing fragment while another process may still append it."""
    if text and not text.endswith(("\n", "\r")):
        text = text.rsplit("\n", 1)[0] + ("\n" if "\n" in text else "")
    return text


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process(pid: int, process_group_id: int | None) -> None:
    try:
        if process_group_id is not None:
            actual_group = os.getpgid(pid)
            if actual_group != process_group_id or actual_group != pid:
                raise MonitorFailure(
                    f"unsafe_process_group:pid={pid}:expected={process_group_id}:actual={actual_group}"
                )
            os.killpg(process_group_id, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _completed_metric_steps(driver_log: Path) -> set[int]:
    if not driver_log.is_file():
        return set()
    text = _complete_driver_text(driver_log.read_text(encoding="utf-8", errors="replace"))
    categories_by_step: dict[int, set[str]] = {}
    for match in METRIC_STEP_RE.finditer(text):
        category = "train" if match.group(1) == "step" else match.group(1)
        categories_by_step.setdefault(int(match.group(2)), set()).add(category)
    required = {"rollout", "train", "perf"}
    return {step for step, categories in categories_by_step.items() if required <= categories}


def _structured_rows(text: str, prefix: str) -> list[dict[str, str]]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        marker = line.find(prefix)
        if marker < 0:
            continue
        row = {"_line": str(line_no)}
        for item in shlex.split(line[marker + len(prefix) :].strip()):
            if "=" in item:
                key, value = item.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def _artifact_id(path: Path) -> int | None:
    match = ARTIFACT_ID_RE.search(path.name)
    return int(match.group(1)) if match else None


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    )


def _finite_field(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _valid_interval(row: dict[str, Any]) -> bool:
    begin = _finite_field(row, "t_begin")
    end = _finite_field(row, "t_end")
    duration = _finite_field(row, "dur")
    return (
        begin is not None
        and end is not None
        and duration is not None
        and begin <= end
        and duration >= 0
        and math.isclose(duration, end - begin, rel_tol=1e-4, abs_tol=1e-3)
    )


def _defer_or_raise(
    condition: bool,
    *,
    key: str,
    reason: str,
    final: bool,
    evidence_grace: float,
    evidence_due_since: dict[str, float],
    now_monotonic: float,
) -> None:
    if condition:
        evidence_due_since.pop(key, None)
        return
    if final:
        raise MonitorFailure(reason)
    first_due = evidence_due_since.setdefault(key, now_monotonic)
    if now_monotonic - first_due >= evidence_grace:
        raise MonitorFailure(reason)


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitorFailure(f"invalid_run_contract:{exc}") from exc
    if not isinstance(payload, dict):
        raise MonitorFailure("invalid_run_contract:not_object")
    return payload


def _validate_contract(
    run_dir: Path,
    *,
    expected_mode: str,
    expected_rollouts: int,
    expected_samples_per_partition: int,
    expected_engines: int,
    max_staleness: int,
    admission_min: int,
    admission_max: int,
    admission_slack: int,
    headline_lo: int,
    headline_hi: int,
) -> None:
    contract = _load_contract(run_dir / "run_contract.json")
    expected = {
        "admission_mode": expected_mode,
        "num_rollout": expected_rollouts,
        "expected_samples_per_partition": expected_samples_per_partition,
        "expected_engines": expected_engines,
        "max_staleness": max_staleness,
        "headline_lo": headline_lo,
        "headline_hi": headline_hi,
        "admission_min": admission_min,
        "admission_max": admission_max,
        "admission_slack": admission_slack,
        "request_placement_mode": "off",
        "use_slime_router": False,
    }
    mismatches = {
        key: (value, contract.get(key))
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise MonitorFailure(f"run_contract_mismatch:{mismatches}")


def _parse_metric_rows(text: str, pattern: re.Pattern[str]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError) as exc:
            raise MonitorFailure(f"invalid_headline_metric:step={match.group(1)}") from exc
        if not isinstance(payload, dict):
            raise MonitorFailure(f"invalid_headline_metric:step={match.group(1)}")
        rows.setdefault(int(match.group(1)), []).append(payload)
    return rows


def _validate_headline_metrics(text: str, *, headline_lo: int, headline_hi: int, final: bool) -> None:
    categories = (
        ("rollout", _parse_metric_rows(text, ROLLOUT_METRICS_RE), REQUIRED_ROLLOUT_METRICS),
        ("train", _parse_metric_rows(text, TRAIN_METRICS_RE), REQUIRED_TRAIN_METRICS),
        ("perf", _parse_metric_rows(text, PERF_METRICS_RE), {"perf/step_time"}),
    )
    for label, rows_by_step, required in categories:
        for step in range(headline_lo, headline_hi + 1):
            rows = rows_by_step.get(step, [])
            if len(rows) > 1:
                raise MonitorFailure(f"duplicate_headline_{label}:step={step}")
            if final and len(rows) != 1:
                raise MonitorFailure(f"missing_headline_{label}:step={step}")
            if not rows:
                continue
            row = rows[0]
            missing = sorted(required - set(row))
            if missing:
                raise MonitorFailure(
                    f"missing_headline_{label}_fields:step={step}:{','.join(missing)}"
                )
            if not all(_finite_number(row[key]) for key in required):
                raise MonitorFailure(f"nonfinite_headline_{label}:step={step}")
            if label == "perf" and float(row["perf/step_time"]) <= 0:
                raise MonitorFailure(f"nonpositive_headline_perf:step={step}")


def _gpu_number(value: str, suffix: str) -> float:
    value = value.strip()
    if suffix and value.endswith(suffix):
        value = value[: -len(suffix)].strip()
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def _gpu_snapshot_status(path: Path, expected_gpu_count: int) -> tuple[int, str | None]:
    if not path.is_file():
        return 0, "missing GPU sampler output"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = 0
    offset = 0
    while offset < len(lines):
        match = GPU_TIMESTAMP_RE.fullmatch(lines[offset].strip())
        if match is None:
            return snapshots, f"line={offset + 1}:invalid timestamp"
        fraction = (match.group(2) or "0")[:6].ljust(6, "0")
        try:
            datetime.strptime(
                f"{match.group(1)}.{fraction}{match.group(3)}",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            )
        except ValueError as exc:
            return snapshots, f"line={offset + 1}:invalid timestamp:{exc}"
        offset += 1
        if len(lines) - offset < expected_gpu_count:
            return snapshots, f"line={offset + 1}:incomplete GPU snapshot"
        indices = set()
        for _ in range(expected_gpu_count):
            try:
                fields = next(csv.reader([lines[offset]], skipinitialspace=True))
                if len(fields) != 5:
                    raise ValueError(f"expected 5 CSV fields, got {len(fields)}")
                index = int(fields[0].strip())
                memory_used = _gpu_number(fields[1], "MiB")
                gpu_util = _gpu_number(fields[2], "%")
                memory_util = _gpu_number(fields[3], "%")
                power_draw = _gpu_number(fields[4], "W")
                if index < 0 or index in indices:
                    raise ValueError("GPU indices must be unique and non-negative")
                if memory_used < 0 or power_draw < 0:
                    raise ValueError("negative memory or power")
                if not (0 <= gpu_util <= 100 and 0 <= memory_util <= 100):
                    raise ValueError("utilization outside [0,100]")
                indices.add(index)
            except (ValueError, csv.Error) as exc:
                return snapshots, f"line={offset + 1}:{exc}"
            offset += 1
        if len(indices) != expected_gpu_count:
            return snapshots, f"snapshot={snapshots}:GPU count={len(indices)}"
        snapshots += 1
    return snapshots, None


def _validate_gpu_snapshots(
    path: Path,
    *,
    expected_engines: int,
    final: bool,
    evidence_grace: float,
    evidence_due_since: dict[str, float],
    now_monotonic: float,
) -> None:
    snapshots, error = _gpu_snapshot_status(path, expected_engines * 2)
    _defer_or_raise(
        error is None,
        key="gpu_snapshot",
        reason=f"invalid_gpu_snapshot:{error}",
        final=final,
        evidence_grace=evidence_grace,
        evidence_due_since=evidence_due_since,
        now_monotonic=now_monotonic,
    )
    if final and snapshots < 2:
        raise MonitorFailure(f"insufficient_gpu_snapshots:{snapshots}")


def _valid_timeline(payload: Any) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    previous_ts = -1.0
    for event in payload:
        if not isinstance(event, dict) or not isinstance(event.get("name"), str) or not event["name"]:
            return False
        if event.get("ph") != "X":
            return False
        timestamp = event.get("ts")
        duration = event.get("dur")
        if not _finite_number(timestamp) or float(timestamp) < previous_ts or float(timestamp) < 0:
            return False
        if not _finite_number(duration) or float(duration) < 0:
            return False
        if any(
            not isinstance(event.get(key), int) or isinstance(event.get(key), bool)
            for key in ("pid", "tid")
        ):
            return False
        if "args" in event and not isinstance(event["args"], dict):
            return False
        previous_ts = float(timestamp)
    return True


def _validate_timelines(
    driver_log: Path,
    timeline_dir: Path,
    *,
    headline_lo: int,
    headline_hi: int,
    final: bool,
    evidence_grace: float,
    evidence_due_since: dict[int, float],
    now_monotonic: float,
) -> None:
    completed = _completed_metric_steps(driver_log)
    for step in range(headline_lo, headline_hi + 1):
        # Async steps may log out of order. A timeline becomes due only once
        # rollout, train, and perf metrics for that same step have all arrived.
        due = final or step in completed
        path = timeline_dir / f"timeline_step_{step}.json"
        if not due:
            evidence_due_since.pop(step, None)
            continue
        if not path.is_file():
            if final:
                raise MonitorFailure(f"missing_headline_timeline:step={step}")
            first_due = evidence_due_since.setdefault(step, now_monotonic)
            if now_monotonic - first_due >= evidence_grace:
                raise MonitorFailure(f"missing_headline_timeline:step={step}")
            continue
        evidence_due_since.pop(step, None)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MonitorFailure(f"invalid_headline_timeline:step={step}:{exc}") from exc
        if not _valid_timeline(payload):
            raise MonitorFailure(f"invalid_headline_timeline:step={step}:format")


def _require_unique(rows: list[dict[str, Any]], key: str, label: str) -> None:
    counts = Counter(str(row.get(key)) for row in rows)
    invalid = sorted(value for value, count in counts.items() if value == "None" or count != 1)
    if invalid:
        raise MonitorFailure(f"{label}:{','.join(invalid[:10])}")


def _numeric_sync_id(sync_id: str) -> int | None:
    try:
        return int(sync_id)
    except ValueError:
        return None


def _server_observability(
    driver_text: str,
) -> tuple[dict[str, int], dict[str, dict[str, list[tuple[str, float | None]]]]]:
    engine_to_gpu: dict[str, int] = {}
    events: dict[str, dict[str, list[tuple[str, float | None]]]] = {}
    for line in driver_text.splitlines():
        pid_match = ENGINE_PID_RE.search(line)
        if pid_match is None:
            continue
        pid = pid_match.group(1)
        if "server_args=ServerArgs(" in line and (gpu_match := BASE_GPU_RE.search(line)):
            engine_to_gpu[pid] = int(gpu_match.group(1))
        event_match = SERVER_EVENT_RE.search(line)
        if event_match is None:
            continue
        try:
            event = json.loads(event_match.group(1))
        except json.JSONDecodeError:
            continue
        name = event.get("event")
        rid = event.get("rid")
        if name not in {"request.received", "request.finished"} or not isinstance(rid, str):
            continue
        timestamp = event.get("timestamp")
        parsed_timestamp: float | None
        if _finite_number(timestamp):
            parsed_timestamp = float(timestamp)
        elif isinstance(timestamp, str):
            try:
                parsed_timestamp = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, OverflowError):
                parsed_timestamp = None
        else:
            parsed_timestamp = None
        events.setdefault(rid, {}).setdefault(name, []).append((pid, parsed_timestamp))
    return engine_to_gpu, events


def _validate_closed_lifecycle(
    request_rows: list[dict[str, Any]],
    driver_text: str,
    *,
    expected_engines: int,
) -> None:
    if not request_rows:
        return
    engine_to_gpu, server_events = _server_observability(driver_text)
    if len(engine_to_gpu) != expected_engines or len(set(engine_to_gpu.values())) != expected_engines:
        raise MonitorFailure(
            f"engine_mapping_mismatch:engines={len(engine_to_gpu)}:"
            f"gpus={len(set(engine_to_gpu.values()))}:expected={expected_engines}"
        )
    for row in request_rows:
        source = f"{row.get('_line_no')}"
        rid = row.get("rid")
        if not isinstance(rid, str) or not rid.startswith("relax:"):
            raise MonitorFailure(f"invalid_lifecycle_rid:line={source}")
        if row.get("client_status") != "finished" or row.get("rid_match") is not True:
            raise MonitorFailure(f"invalid_lifecycle_terminal_rid:rid={rid}")
        dispatch = _finite_field(row, "dispatch_abs")
        request_end = _finite_field(row, "request_end_abs")
        forward_entry = _finite_field(row, "forward_entry_time")
        prefill_finished = _finite_field(row, "prefill_finished_time")
        queue_time = _finite_field(row, "queue_time")
        if (
            dispatch is None
            or request_end is None
            or dispatch > request_end
            or forward_entry is None
            or prefill_finished is None
            or forward_entry > prefill_finished
            or queue_time is None
            or queue_time < 0
        ):
            raise MonitorFailure(f"invalid_lifecycle_timing:rid={rid}")
        lifecycle = server_events.get(rid, {})
        received = lifecycle.get("request.received", [])
        finished = lifecycle.get("request.finished", [])
        if len(received) != 1 or len(finished) != 1:
            raise MonitorFailure(
                f"invalid_server_lifecycle:rid={rid}:received={len(received)}:finished={len(finished)}"
            )
        receive_pid, server_begin = received[0]
        finish_pid, server_end = finished[0]
        if (
            receive_pid != finish_pid
            or receive_pid not in engine_to_gpu
            or server_begin is None
            or server_end is None
            or server_begin > server_end
            or server_begin < dispatch - SERVER_ENVELOPE_TOLERANCE_S
            or server_end > request_end + SERVER_ENVELOPE_TOLERANCE_S
        ):
            raise MonitorFailure(f"invalid_lifecycle_engine_interval:rid={rid}")


def _validate_physical_flow(
    flow_rows: list[dict[str, str]],
    *,
    final: bool,
    expected_rollouts: int,
    evidence_grace: float,
    evidence_due_since: dict[str, float],
    now_monotonic: float,
) -> set[int]:
    starts = [row for row in flow_rows if row.get("phase") == "physical_start"]
    ends = [row for row in flow_rows if row.get("phase") == "physical_end"]
    for phase, rows in (("start", starts), ("end", ends)):
        counts = Counter(row.get("physical_rollout_id") for row in rows)
        duplicates = sorted(str(key) for key, count in counts.items() if key is None or count > 1)
        if duplicates:
            raise MonitorFailure(f"duplicate_physical_{phase}:{','.join(duplicates)}")
    invalid_ids = [
        row.get("physical_rollout_id")
        for row in starts + ends
        if not str(row.get("physical_rollout_id", "")).isdigit()
    ]
    if invalid_ids:
        raise MonitorFailure(f"invalid_physical_rollout_ids:{invalid_ids}")
    allowed_ids = set(range(expected_rollouts + 1))
    observed_ids = {
        int(row["physical_rollout_id"])
        for row in starts + ends
        if str(row.get("physical_rollout_id", "")).isdigit()
    }
    if observed_ids - allowed_ids:
        raise MonitorFailure(f"invalid_physical_rollout_ids:{sorted(observed_ids - allowed_ids)}")
    starts_by_id = {int(row["physical_rollout_id"]): row for row in starts}
    ends_by_id = {int(row["physical_rollout_id"]): row for row in ends}
    if final:
        expected_ids = set(range(expected_rollouts))
        if expected_rollouts in observed_ids:
            expected_ids.add(expected_rollouts)
        if set(starts_by_id) != expected_ids or set(ends_by_id) != expected_ids:
            raise MonitorFailure(
                "physical_flow_id_mismatch:"
                f"expected={sorted(expected_ids)}:"
                f"starts={sorted(starts_by_id)}:"
                f"ends={sorted(ends_by_id)}"
            )
    completed = set(ends_by_id)
    for rollout_id, end_row in ends_by_id.items():
        start_row = starts_by_id.get(rollout_id)
        start_time = _finite_field(start_row or {}, "t")
        end_begin = _finite_field(end_row, "t_begin")
        end_time = _finite_field(end_row, "t_end")
        valid = (
            start_row is not None
            and start_time is not None
            and _valid_interval(end_row)
            and end_begin is not None
            and end_time is not None
            and math.isclose(start_time, end_begin, rel_tol=1e-4, abs_tol=1e-3)
        )
        _defer_or_raise(
            valid,
            key=f"physical:{rollout_id}",
            reason=f"invalid_physical_interval:{rollout_id}",
            final=final,
            evidence_grace=evidence_grace,
            evidence_due_since=evidence_due_since,
            now_monotonic=now_monotonic,
        )
    ordered = []
    for rollout_id, row in starts_by_id.items():
        timestamp = _finite_field(row, "t")
        if timestamp is None:
            raise MonitorFailure(f"invalid_physical_start_time:{rollout_id}")
        ordered.append((timestamp, rollout_id))
    if [rollout_id for _, rollout_id in sorted(ordered)] != sorted(starts_by_id):
        raise MonitorFailure("physical_start_order_invalid")
    ordered_ends = [
        (_finite_field(row, "t_end"), rollout_id) for rollout_id, row in ends_by_id.items()
    ]
    valid_ordered_ends = [
        (timestamp, rollout_id) for timestamp, rollout_id in ordered_ends if timestamp is not None
    ]
    if len(valid_ordered_ends) == len(ordered_ends) and [
        rollout_id for _, rollout_id in sorted(valid_ordered_ends)
    ] != sorted(ends_by_id):
        raise MonitorFailure("physical_end_order_invalid")
    return completed


def _validate_sync(
    driver_text: str,
    *,
    final: bool,
    expected_rollouts: int,
    evidence_grace: float = 5.0,
    evidence_due_since: dict[str, float] | None = None,
    now_monotonic: float | None = None,
) -> None:
    events = _structured_rows(driver_text, "TASK22_EVENT")
    due_since = evidence_due_since if evidence_due_since is not None else {}
    scan_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    sync_events = [row for row in events if row.get("phase") in {"gate", *SYNC_PHASES}]
    for row in sync_events:
        if row.get("sync_id") is None or not _valid_interval(row):
            raise MonitorFailure(
                f"invalid_sync_interval:{row.get('phase')}:{row.get('sync_id')}"
            )
    keyed: dict[tuple[str, str], int] = Counter(
        (str(row.get("phase")), str(row.get("sync_id")))
        for row in sync_events
    )
    duplicates = [f"{phase}:{sync_id}" for (phase, sync_id), count in keyed.items() if count > 1]
    if duplicates:
        raise MonitorFailure(f"duplicate_sync_phase:{','.join(sorted(duplicates)[:10])}")

    gate_ids = {sync_id for phase, sync_id in keyed if phase == "gate"}
    closed_ids = {sync_id for phase, sync_id in keyed if phase == "continue"}
    bootstrap_candidates = closed_ids - gate_ids
    if final:
        bootstrap_ids = bootstrap_candidates
    else:
        numeric_gate_ids = [
            numeric_id for sync_id in gate_ids if (numeric_id := _numeric_sync_id(sync_id)) is not None
        ]
        # A gate-less closed cycle is stable bootstrap evidence only after a
        # higher gate is visible. Until then its gate line may merely be late.
        bootstrap_ids = {
            sync_id
            for sync_id in bootstrap_candidates
            if (numeric_id := _numeric_sync_id(sync_id)) is not None
            and any(gate_id > numeric_id for gate_id in numeric_gate_ids)
        }
        for sync_id in bootstrap_candidates:
            key = f"bootstrap_candidate:{sync_id}"
            if sync_id in bootstrap_ids:
                due_since.pop(key, None)
                continue
            first_seen = due_since.setdefault(key, scan_monotonic)
            if gate_ids and scan_monotonic - first_seen >= evidence_grace:
                bootstrap_ids.add(sync_id)
        for sync_id in set(gate_ids):
            due_since.pop(f"bootstrap_candidate:{sync_id}", None)
    if len(bootstrap_ids) > 1:
        raise MonitorFailure(f"multiple_bootstrap_sync_ids:{','.join(sorted(bootstrap_ids))}")

    # A continue event closes a sync cycle, so all preceding phases are due
    # after a grace window for cross-process log aggregation reordering.
    for sync_id in closed_ids:
        expected_phases = ("gate", *SYNC_PHASES) if sync_id in gate_ids else SYNC_PHASES
        missing = [phase for phase in expected_phases if keyed[(phase, sync_id)] != 1]
        _defer_or_raise(
            not missing,
            key=f"sync:{sync_id}",
            reason=f"incomplete_sync_cycle:{sync_id}:{','.join(missing)}",
            final=final,
            evidence_grace=evidence_grace,
            evidence_due_since=due_since,
            now_monotonic=scan_monotonic,
        )
        if missing:
            continue
        rows_by_phase = {
            phase: next(
                row
                for row in sync_events
                if row.get("phase") == phase and str(row.get("sync_id")) == sync_id
            )
            for phase in expected_phases
        }
        intervals = [
            (_finite_field(rows_by_phase[phase], "t_begin"), _finite_field(rows_by_phase[phase], "t_end"))
            for phase in expected_phases
        ]
        if any(
            begin is None
            or end is None
            or (index > 0 and intervals[index - 1][1] > begin + 1e-3)
            for index, (begin, end) in enumerate(intervals)
        ):
            raise MonitorFailure(f"sync_phase_order_invalid:{sync_id}")

    if bootstrap_ids and gate_ids:
        bootstrap_ends = [
            _finite_field(row, "t_end")
            for row in sync_events
            if str(row.get("sync_id")) in bootstrap_ids
        ]
        runtime_begins = [
            _finite_field(row, "t_begin")
            for row in sync_events
            if str(row.get("sync_id")) in gate_ids
        ]
        valid_precedence = (
            bootstrap_ends
            and runtime_begins
            and all(value is not None for value in bootstrap_ends + runtime_begins)
            and max(value for value in bootstrap_ends if value is not None)
            <= min(value for value in runtime_begins if value is not None) + 1e-3
        )
        if not valid_precedence:
            raise MonitorFailure("bootstrap_sync_does_not_precede_runtime")

    if final:
        if len(bootstrap_ids) != 1:
            raise MonitorFailure(f"bootstrap_sync_count:{len(bootstrap_ids)}")
        if len(gate_ids) != expected_rollouts:
            raise MonitorFailure(f"runtime_sync_count:{len(gate_ids)}:expected={expected_rollouts}")
        for sync_id in gate_ids | bootstrap_ids:
            expected_phases = SYNC_PHASES if sync_id in bootstrap_ids else ("gate", *SYNC_PHASES)
            if any(keyed[(phase, sync_id)] != 1 for phase in expected_phases):
                raise MonitorFailure(f"incomplete_sync_cycle:{sync_id}")


def _scan(
    run_dir: Path,
    *,
    expected_mode: str,
    expected_rollouts: int,
    expected_samples_per_partition: int,
    expected_engines: int,
    max_staleness: int,
    admission_min: int,
    admission_max: int,
    admission_slack: int,
    headline_lo: int,
    headline_hi: int,
    final: bool,
    reported_lifecycle: set[tuple[str, int]],
    event_log: Path,
    evidence_grace: float = 5.0,
    timeline_evidence_due_since: dict[int, float] | None = None,
    strict_evidence_due_since: dict[str, float] | None = None,
    now_monotonic: float | None = None,
) -> None:
    if (admission_min, admission_max, admission_slack) != (4, 8, 2):
        raise MonitorFailure(
            f"invalid_4_8_2_contract:{admission_min}:{admission_max}:{admission_slack}"
        )
    if expected_samples_per_partition != 64:
        raise MonitorFailure(
            f"invalid_partition_contract:{expected_samples_per_partition}:expected=64"
        )
    if max_staleness != 2:
        raise MonitorFailure(f"invalid_staleness_contract:{max_staleness}:expected=2")
    if expected_engines != 2:
        raise MonitorFailure(f"invalid_engine_contract:{expected_engines}:expected=2")

    _validate_contract(
        run_dir,
        expected_mode=expected_mode,
        expected_rollouts=expected_rollouts,
        expected_samples_per_partition=expected_samples_per_partition,
        expected_engines=expected_engines,
        max_staleness=max_staleness,
        admission_min=admission_min,
        admission_max=admission_max,
        admission_slack=admission_slack,
        headline_lo=headline_lo,
        headline_hi=headline_hi,
    )

    driver_log = run_dir / "driver.log"
    driver_text = (
        _complete_driver_text(driver_log.read_text(encoding="utf-8", errors="replace"))
        if driver_log.is_file()
        else ""
    )
    fatal = FATAL_RE.search(driver_text)
    if fatal:
        raise MonitorFailure(f"fatal_driver_log:{fatal.group(0).strip()}")

    scan_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    strict_due_since = strict_evidence_due_since if strict_evidence_due_since is not None else {}
    event_rows = _structured_rows(driver_text, "TASK22_EVENT")
    for row in event_rows:
        if row.get("phase") not in EVENT_PHASES:
            raise MonitorFailure(f"invalid_event_phase:{row.get('phase')}")
        if not _valid_interval(row):
            raise MonitorFailure(f"invalid_event_interval:{row.get('phase')}")
    flow_rows = _structured_rows(driver_text, "TASK22_FLOW")
    for row in flow_rows:
        if row.get("phase") not in FLOW_PHASES:
            raise MonitorFailure(f"invalid_flow_phase:{row.get('phase')}")
    completed_physical = _validate_physical_flow(
        flow_rows,
        final=final,
        expected_rollouts=expected_rollouts,
        evidence_grace=evidence_grace,
        evidence_due_since=strict_due_since,
        now_monotonic=scan_monotonic,
    )
    physical_end_rows = [row for row in flow_rows if row.get("phase") == "physical_end"]

    observability = run_dir / "observability"
    request_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    ledger_attempt_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    consumption_rows: list[dict[str, Any]] = []
    artifact_physical_ids: set[int] = set()
    if observability.is_dir():
        for path in sorted(observability.glob("*.jsonl")):
            artifact_id = _artifact_id(path)
            if path.name.startswith(
                (
                    "request_lifecycle_rollout_",
                    "admission_ledger_rollout_",
                    "admission_outcomes_rollout_",
                    "consumption_ledger_rollout_",
                )
            ) and artifact_id is None:
                raise MonitorFailure(f"unparseable_artifact_filename:{path.name}")
            if artifact_id is not None and artifact_id not in set(range(expected_rollouts + 1)):
                raise MonitorFailure(f"invalid_artifact_physical_id:{path.name}:{artifact_id}")
            if artifact_id is not None and path.name.startswith(
                (
                    "request_lifecycle_rollout_",
                    "admission_ledger_rollout_",
                    "admission_outcomes_rollout_",
                )
            ):
                artifact_physical_ids.add(artifact_id)
            # Outcome files are atomically extended while a rollout is live.
            # Delay all semantic reads until physical_end makes the snapshot final.
            if path.name.startswith("admission_outcomes_rollout_") and not (
                final or artifact_id in completed_physical
            ):
                continue
            # Lifecycle and admission files are whole-rollout snapshots published
            # immediately before physical_end. Ignoring early files prevents a
            # scheduler poll from treating a not-yet-closed attempt as terminal.
            if (
                path.name.startswith(("request_lifecycle_rollout_", "admission_ledger_rollout_"))
                and not (final or artifact_id in completed_physical)
            ):
                continue
            rows = _read_jsonl(path)
            if path.name.startswith("request_lifecycle_rollout_"):
                if any(row.get("physical_rollout_id") != artifact_id for row in rows):
                    raise MonitorFailure(f"request_file_physical_id_mismatch:{path.name}")
                request_rows.extend(rows)
                for row in rows:
                    key = (path.name, int(row["_line_no"]))
                    outcome = row.get("outcome")
                    if outcome not in TERMINAL_OUTCOMES:
                        raise MonitorFailure(
                            f"non_terminal_lifecycle:{path.name}:{row['_line_no']}:{outcome!r}"
                        )
                    if key not in reported_lifecycle:
                        reported_lifecycle.add(key)
                        _append_event(
                            event_log,
                            "lifecycle_terminal",
                            source=path.name,
                            line=row["_line_no"],
                            attempt_id=row.get("attempt_id") or row.get("rid"),
                            outcome=outcome,
                        )
            if path.name.startswith("admission_ledger_rollout_"):
                if any(row.get("physical_rollout_id") != artifact_id for row in rows):
                    raise MonitorFailure(f"ledger_file_physical_id_mismatch:{path.name}")
                for row in rows:
                    bypass = str(row.get("bypass_reason") or "")
                    if bypass.startswith("fail_open:"):
                        raise MonitorFailure(
                            f"admission_fail_open:{path.name}:{row['_line_no']}:{bypass}"
                        )
                    if row.get("record_type") == "admission_decision":
                        decision_rows.append(row)
                    elif row.get("record_type") == "attempt":
                        ledger_attempt_rows.append(row)
            elif path.name.startswith("admission_outcomes_rollout_"):
                if any(row.get("physical_rollout_id") != artifact_id for row in rows):
                    raise MonitorFailure(f"outcome_file_physical_id_mismatch:{path.name}")
                outcome_rows.extend(rows)
            elif path.name.startswith("consumption_ledger_rollout_"):
                if any(
                    row.get("rollout_id") != artifact_id
                    or row.get("target_partition") != f"train_{artifact_id}"
                    for row in rows
                ):
                    raise MonitorFailure(f"consumption_file_partition_mismatch:{path.name}")
                consumption_rows.extend(rows)

    if final:
        expected_physical_ids = set(range(expected_rollouts))
        if expected_rollouts in artifact_physical_ids:
            expected_physical_ids.add(expected_rollouts)
        if completed_physical != expected_physical_ids:
            raise MonitorFailure(
                "physical_flow_artifact_id_mismatch:"
                f"expected={sorted(expected_physical_ids)}:"
                f"completed={sorted(completed_physical)}"
            )

    if decision_rows:
        _require_unique(decision_rows, "decision_id", "duplicate_decision_id")
    for row in decision_rows:
        values = (
            row.get("release_remaining"),
            row.get("inflight_groups"),
            row.get("available_groups"),
        )
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
            raise MonitorFailure(f"invalid_admission_inputs:{row.get('decision_id')}")
        debt, inflight, available = values
        desired = admission_max
        if debt > 0:
            desired = max(admission_min, min(admission_max, debt + admission_slack))
        bounded = min(available, max(desired - inflight, 0))
        if row.get("desired_inflight_groups") != desired or row.get("bounded_admit_groups") != bounded:
            raise MonitorFailure(f"admission_recompute_mismatch:{row.get('decision_id')}")
        if row.get("mode") != expected_mode:
            raise MonitorFailure(f"admission_mode_mismatch:{row.get('decision_id')}")
        bypass = row.get("bypass_reason")
        actual = row.get("actual_admit_groups")
        expected_actual = row.get("eager_admit_groups") if expected_mode == "shadow" else bounded
        if bypass is None and actual != expected_actual:
            raise MonitorFailure(f"admission_mode_mismatch:{row.get('decision_id')}")
        if bypass == "final_backfill":
            logical_debt = row.get("logical_debt_groups")
            if not (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and isinstance(logical_debt, int)
                and not isinstance(logical_debt, bool)
                and 0 <= actual <= min(debt, logical_debt, available)
            ):
                raise MonitorFailure(f"final_backfill_past_debt:{row.get('decision_id')}")

    if request_rows:
        _require_unique(request_rows, "attempt_id", "duplicate_attempt_id")
        try:
            _validate_closed_lifecycle(request_rows, driver_text, expected_engines=expected_engines)
        except MonitorFailure as exc:
            _defer_or_raise(
                False,
                key="closed_lifecycle",
                reason=str(exc),
                final=final,
                evidence_grace=evidence_grace,
                evidence_due_since=strict_due_since,
                now_monotonic=scan_monotonic,
            )
        else:
            strict_due_since.pop("closed_lifecycle", None)
    request_ids = {str(row.get("attempt_id") or row.get("rid")) for row in request_rows}
    ledger_attempt_ids = {
        str(row.get("attempt_id") or row.get("rid")) for row in ledger_attempt_rows
    }
    _defer_or_raise(
        request_ids == ledger_attempt_ids,
        key="ledger_request_set",
        reason=(
            "ledger_attempt_mismatch:"
            f"request_only={sorted(request_ids - ledger_attempt_ids)}:"
            f"ledger_only={sorted(ledger_attempt_ids - request_ids)}"
        ),
        final=final,
        evidence_grace=evidence_grace,
        evidence_due_since=strict_due_since,
        now_monotonic=scan_monotonic,
    )

    commit_keys = []
    for row in outcome_rows:
        if row.get("record_type") != "partition_outcome" or row.get("outcome") != "committed":
            raise MonitorFailure(f"invalid_commit_outcome:{row.get('_line_no')}")
        commit_keys.append((str(row.get("target_partition")), row.get("attempt_token")))
    if len(commit_keys) != len(set(commit_keys)):
        raise MonitorFailure("duplicate_commit_association")
    commit_tokens = [row.get("attempt_token") for row in outcome_rows]
    if any(token is None for token in commit_tokens) or len(commit_tokens) != len(set(commit_tokens)):
        raise MonitorFailure("duplicate_commit_attempt_token")

    consume_keys = []
    consume_sample_keys = []
    partition_counts: Counter[str] = Counter()
    for row in consumption_rows:
        if row.get("record_type") != "consume_outcome" or row.get("consume_outcome") != "consumed":
            raise MonitorFailure(f"invalid_consume_outcome:{row.get('_line_no')}")
        partition = str(row.get("target_partition"))
        partition_counts[partition] += 1
        consume_keys.append((partition, row.get("attempt_token")))
        consume_sample_keys.append((partition, row.get("group_index"), row.get("sample_index")))
        generation_end = row.get("generation_end_version")
        consume_version = row.get("consume_version")
        actual_staleness = row.get("actual_staleness")
        if not (
            isinstance(generation_end, int)
            and not isinstance(generation_end, bool)
            and isinstance(consume_version, int)
            and not isinstance(consume_version, bool)
            and isinstance(actual_staleness, int)
            and not isinstance(actual_staleness, bool)
            and actual_staleness == consume_version - generation_end
            and 0 <= actual_staleness <= max_staleness
        ):
            raise MonitorFailure(f"invalid_staleness:{row.get('_line_no')}")
    if len(consume_keys) != len(set(consume_keys)):
        raise MonitorFailure("duplicate_consume_association")
    if len(consume_sample_keys) != len(set(consume_sample_keys)):
        raise MonitorFailure("duplicate_partition_sample")
    consume_tokens = [row.get("attempt_token") for row in consumption_rows]
    if any(token is None for token in consume_tokens) or len(consume_tokens) != len(set(consume_tokens)):
        raise MonitorFailure("duplicate_consume_attempt_token")
    overfull = {
        partition: count
        for partition, count in partition_counts.items()
        if count > expected_samples_per_partition
    }
    if overfull:
        raise MonitorFailure(f"partition_overflow:{overfull}")

    commit_by_key = {
        (str(row.get("target_partition")), row.get("attempt_token")): row for row in outcome_rows
    }
    consume_by_key = {
        (str(row.get("target_partition")), row.get("attempt_token")): row
        for row in consumption_rows
    }
    for key in set(commit_by_key) & set(consume_by_key):
        commit = commit_by_key[key]
        consume = consume_by_key[key]
        if (
            commit.get("decision_id") != consume.get("decision_id")
            or commit.get("sample_index") != consume.get("sample_index")
            or commit.get("group_index") != consume.get("group_index")
            or commit.get("generation_physical_rollout_id")
            != consume.get("generation_physical_rollout_id")
        ):
            raise MonitorFailure(f"commit_consume_identity_mismatch:{key}")
    if final and set(commit_by_key) != set(consume_by_key):
        raise MonitorFailure(
            f"commit_consume_set_mismatch:commits={len(commit_by_key)}:consumes={len(consume_by_key)}"
        )
    if final:
        expected_partitions = {f"train_{rollout_id}" for rollout_id in range(expected_rollouts)}
        observed_partitions = set(partition_counts)
        if observed_partitions != expected_partitions:
            raise MonitorFailure(
                "partition_set_mismatch:"
                f"missing={sorted(expected_partitions - observed_partitions)}:"
                f"extra={sorted(observed_partitions - expected_partitions)}"
            )
        incomplete = {
            partition: partition_counts[partition]
            for partition in sorted(expected_partitions)
            if partition_counts[partition] != expected_samples_per_partition
        }
        if incomplete:
            raise MonitorFailure(f"partition_count_mismatch:{incomplete}")
        bounded_triggered = any(
            row.get("bypass_reason") is None
            and row.get("bounded_admit_groups") != row.get("eager_admit_groups")
            for row in decision_rows
        )
        if not bounded_triggered:
            raise MonitorFailure("bounded_path_not_exercised")

    if final and physical_end_rows:
        final_rows = [
            row
            for row in physical_end_rows
            if row.get("physical_rollout_id") == str(expected_rollouts)
        ]
        if final_rows and final_rows[0].get("next_debt_groups") != "0":
            raise MonitorFailure(
                f"final_debt_not_zero:{final_rows[0].get('next_debt_groups')}"
            )

    _validate_headline_metrics(
        driver_text,
        headline_lo=headline_lo,
        headline_hi=headline_hi,
        final=final,
    )
    _validate_sync(
        driver_text,
        final=final,
        expected_rollouts=expected_rollouts,
        evidence_grace=evidence_grace,
        evidence_due_since=strict_due_since,
        now_monotonic=scan_monotonic,
    )
    _validate_gpu_snapshots(
        run_dir / "logs" / "nvidia_smi_1s.csv",
        expected_engines=expected_engines,
        final=final,
        evidence_grace=evidence_grace,
        evidence_due_since=strict_due_since,
        now_monotonic=scan_monotonic,
    )

    due_since = timeline_evidence_due_since if timeline_evidence_due_since is not None else {}
    _validate_timelines(
        driver_log,
        run_dir / "timeline",
        headline_lo=headline_lo,
        headline_hi=headline_hi,
        final=final,
        evidence_grace=evidence_grace,
        evidence_due_since=due_since,
        now_monotonic=scan_monotonic,
    )


def monitor(
    run_dir: Path,
    *,
    pid: int,
    process_group_id: int | None,
    expected_mode: str,
    expected_rollouts: int,
    expected_samples_per_partition: int,
    expected_engines: int,
    max_staleness: int,
    admission_min: int,
    admission_max: int,
    admission_slack: int,
    headline_lo: int,
    headline_hi: int,
    poll_interval: float,
    evidence_grace: float,
) -> int:
    event_log = run_dir / "online_monitor.jsonl"
    reported_lifecycle: set[tuple[str, int]] = set()
    timeline_evidence_due_since: dict[int, float] = {}
    strict_evidence_due_since: dict[str, float] = {}
    _append_event(event_log, "monitor_started", pid=pid)
    while True:
        running = _process_exists(pid)
        try:
            _scan(
                run_dir,
                expected_mode=expected_mode,
                expected_rollouts=expected_rollouts,
                expected_samples_per_partition=expected_samples_per_partition,
                expected_engines=expected_engines,
                max_staleness=max_staleness,
                admission_min=admission_min,
                admission_max=admission_max,
                admission_slack=admission_slack,
                headline_lo=headline_lo,
                headline_hi=headline_hi,
                final=not running,
                reported_lifecycle=reported_lifecycle,
                event_log=event_log,
                evidence_grace=evidence_grace,
                timeline_evidence_due_since=timeline_evidence_due_since,
                strict_evidence_due_since=strict_evidence_due_since,
            )
        except (MonitorFailure, OSError) as exc:
            reason = str(exc)
            _append_event(event_log, "monitor_failed", reason=reason)
            if running:
                try:
                    _stop_process(pid, process_group_id)
                except MonitorFailure as stop_exc:
                    _append_event(event_log, "training_stop_rejected", pid=pid, reason=str(stop_exc))
                else:
                    _append_event(event_log, "training_stop_requested", pid=pid, reason=reason)
            return 4
        if not running:
            _append_event(event_log, "monitor_passed")
            return 0
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--process-group-id", type=int)
    parser.add_argument("--expected-mode", choices=("shadow", "on"), required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--expected-samples-per-partition", type=int, required=True)
    parser.add_argument("--expected-engines", type=int, required=True)
    parser.add_argument("--max-staleness", type=int, required=True)
    parser.add_argument("--admission-min", type=int, required=True)
    parser.add_argument("--admission-max", type=int, required=True)
    parser.add_argument("--admission-slack", type=int, required=True)
    parser.add_argument("--headline-lo", type=int, required=True)
    parser.add_argument("--headline-hi", type=int, required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--evidence-grace", type=float, default=5.0)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.evidence_grace < 5.0:
        parser.error("--evidence-grace must be at least 5 seconds")
    if args.headline_lo < 0 or args.headline_hi < args.headline_lo:
        parser.error("invalid headline range")
    if (
        args.expected_rollouts <= 0
        or args.expected_samples_per_partition <= 0
        or args.expected_engines <= 0
    ):
        parser.error("expected rollout, partition, and engine sizes must be positive")
    if args.max_staleness < 0:
        parser.error("--max-staleness must be non-negative")
    if not (0 < args.admission_min <= args.admission_max) or args.admission_slack < 0:
        parser.error("invalid admission contract")
    if args.process_group_id is not None and args.process_group_id <= 0:
        parser.error("--process-group-id must be positive")
    raise SystemExit(
        monitor(
            args.run_dir,
            pid=args.pid,
            process_group_id=args.process_group_id,
            expected_mode=args.expected_mode,
            expected_rollouts=args.expected_rollouts,
            expected_samples_per_partition=args.expected_samples_per_partition,
            expected_engines=args.expected_engines,
            max_staleness=args.max_staleness,
            admission_min=args.admission_min,
            admission_max=args.admission_max,
            admission_slack=args.admission_slack,
            headline_lo=args.headline_lo,
            headline_hi=args.headline_hi,
            poll_interval=args.poll_interval,
            evidence_grace=args.evidence_grace,
        )
    )


if __name__ == "__main__":
    main()
