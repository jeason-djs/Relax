#!/usr/bin/env python3
"""Fail-closed online evidence monitor for a Task 22 admission run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from relax.utils.task22_runtime_attestation import attestation_matches_contract


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
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FATAL_RE = re.compile(
    r"(?:"
    r"\bFATAL\b|"
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
MAX_RETAINED_DRIVER_BYTES = 32 * 1024 * 1024
MAX_RETAINED_JSONL_ROWS = 1_000_000
GPU_COVERAGE_TOLERANCE_S = 2.0
DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S = 2.0
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
    "run_contract_schema_is_current_v6",
    "run_contract_valid",
    "runtime_attestations_match_contract",
    "runtime_attestations_parseable",
    "runtime_contract_content_hashes_valid",
    "runtime_contract_fields_valid",
    "qualification_monitor_contract_valid",
    "qualification_gpu_contract_valid",
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
    "input_manifest_attestations_parseable",
    "input_manifest_before_after_match_contract",
    "input_manifest_three_point_attestation_consistent",
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
    "runtime_attestation_roles_complete",
    "timeline_directory_present",
}
STRICT_CHECK_COVERAGE = {
    **{name: "online" for name in ONLINE_STRICT_CHECKS},
    **{name: "final-only" for name in FINAL_ONLY_STRICT_CHECKS},
}


class MonitorFailure(RuntimeError):
    pass


@dataclass
class FileCursor:
    identity: tuple[int, int] | None = None
    offset: int = 0
    epoch: int = 0
    partial: bytes = b""
    ctime_ns: int = 0


@dataclass
class MonitorScanState:
    """Incrementally retained evidence; epochs make every replay observable."""

    driver_text: str = ""
    driver_cursor: FileCursor = field(default_factory=FileCursor)
    driver_bytes: int = 0
    jsonl_cursors: dict[str, FileCursor] = field(default_factory=dict)
    jsonl_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    semantic_revision: int = 0
    timeline_fingerprints: dict[int, str] = field(default_factory=dict)


def _append_event(path: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()


def _safe_append_event(path: Path, event: str, **fields: Any) -> bool:
    try:
        _append_event(path, event, **fields)
    except OSError:
        return False
    return True


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


def _read_incremental_bytes(path: Path, cursor: FileCursor, *, final: bool) -> list[tuple[int, int, bytes]]:
    """Read only new complete records, starting a new epoch after replacement/truncation."""

    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        identity = (before.st_dev, before.st_ino)
        reset = cursor.identity is not None and (
            identity != cursor.identity
            or before.st_size < cursor.offset
            # Detect truncate+rewrite-to-the-same-size between polls. ctime is
            # not semantic progress; it only establishes a replay epoch when
            # the opened file is otherwise exactly at the committed cursor.
            or (
                identity == cursor.identity
                and before.st_size == cursor.offset
                and before.st_ctime_ns != cursor.ctime_ns
            )
        )
        if cursor.identity is None:
            cursor.identity = identity
        elif reset:
            cursor.identity = identity
            cursor.offset = 0
            cursor.partial = b""
            cursor.epoch += 1

        start = cursor.offset
        source.seek(start)
        # Bound the read to the size observed on this opened descriptor.
        # Bytes appended concurrently remain for the next poll instead of
        # advancing the cursor beyond the metadata snapshot and looking like
        # a same-size rewrite on that poll.
        chunk = source.read(max(before.st_size - start, 0))
        after = os.fstat(source.fileno())
    cursor.offset = start + len(chunk)
    cursor.ctime_ns = after.st_ctime_ns
    data = cursor.partial + chunk
    base_offset = start - len(cursor.partial)
    if final:
        complete, cursor.partial = data, b""
    else:
        newline = data.rfind(b"\n")
        if newline < 0:
            cursor.partial = data
            return []
        complete, cursor.partial = data[: newline + 1], data[newline + 1 :]

    records: list[tuple[int, int, bytes]] = []
    relative = 0
    for raw in complete.splitlines(keepends=True):
        payload = raw.rstrip(b"\r\n")
        if payload.strip():
            records.append((cursor.epoch, base_offset + relative, payload))
        relative += len(raw)
    return records


def _read_retained_jsonl(
    path: Path,
    *,
    final: bool,
    state: MonitorScanState | None,
    evidence_grace: float,
    evidence_due_since: dict[str, float],
    now_monotonic: float,
) -> list[dict[str, Any]]:
    if state is None:
        return _read_jsonl(path)
    key = str(path)
    retained = state.jsonl_rows.setdefault(key, [])
    cursor = state.jsonl_cursors.setdefault(key, FileCursor())
    current: list[dict[str, Any]] = []
    bad_offset: int | None = None
    try:
        records = _read_incremental_bytes(path, cursor, final=final)
        for epoch, offset, raw in records:
            bad_offset = offset
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise MonitorFailure(f"non_object_jsonl:{path.name}:epoch={epoch}:offset={offset}")
            row["_epoch"] = epoch
            row["_offset"] = offset
            row["_line_no"] = len(retained) + len(current) + 1
            current.append(row)
            bad_offset = None
    except (UnicodeDecodeError, json.JSONDecodeError, MonitorFailure) as exc:
        if current:
            retained.extend(current)
            state.semantic_revision += len(current)
        if bad_offset is not None:
            cursor.offset = bad_offset
            cursor.partial = b""
        reason = (
            str(exc)
            if isinstance(exc, MonitorFailure)
            else f"malformed_jsonl:{path.name}:epoch={cursor.epoch}:offset={cursor.offset}:{exc}"
        )
        _defer_or_raise(
            False,
            key=f"jsonl:{key}",
            reason=reason,
            final=final,
            evidence_grace=evidence_grace,
            evidence_due_since=evidence_due_since,
            now_monotonic=now_monotonic,
        )
        return retained
    if current:
        retained.extend(current)
        state.semantic_revision += len(current)
    if len(retained) > MAX_RETAINED_JSONL_ROWS:
        raise MonitorFailure(f"jsonl_memory_limit:{path.name}:{len(retained)}")
    if cursor.partial:
        _defer_or_raise(
            False,
            key=f"jsonl:{key}",
            reason=(
                f"malformed_jsonl:{path.name}:epoch={cursor.epoch}:"
                f"offset={cursor.offset - len(cursor.partial)}:incomplete_record"
            ),
            final=final,
            evidence_grace=evidence_grace,
            evidence_due_since=evidence_due_since,
            now_monotonic=now_monotonic,
        )
        return retained
    evidence_due_since.pop(f"jsonl:{key}", None)
    return retained


def _complete_driver_text(text: str, *, final: bool = False) -> str:
    """Discard the trailing fragment while another process may still append it."""
    if not final and text and not text.endswith(("\n", "\r")):
        text = text.rsplit("\n", 1)[0] + ("\n" if "\n" in text else "")
    return text


def _process_start_identity(pid: int) -> str | None:
    """Return an immutable per-process start identity (Linux procfs)."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        # On non-procfs platforms, use the kernel start-time rendering. It is
        # still tied to this PID incarnation and is only used for comparison.
        try:
            import subprocess

            value = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except OSError:
            return None
        return value or None
    # comm may contain spaces and ')'; fields after its final ')' start at #3.
    suffix = stat.rsplit(")", 1)[1].split()
    return suffix[19] if len(suffix) > 19 else None


def _evidence_progress_token(run_dir: Path, state: MonitorScanState) -> tuple[Any, ...]:
    # Deliberately excludes mtime/size. Only newly parsed complete evidence can
    # reset the no-progress or post-exit stability clocks.
    return (state.semantic_revision,)


def _process_exists(pid: int, expected_identity: str | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        exists = True
    else:
        exists = True
    if exists and expected_identity is not None:
        return _process_start_identity(pid) == expected_identity
    return exists


def _process_group_exists(process_group_id: int | None) -> bool:
    if process_group_id is None:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process(
    pid: int,
    process_group_id: int | None,
    *,
    expected_identity: str | None = None,
    term_timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> None:
    leader_exists = _process_exists(pid)
    if (
        leader_exists
        and expected_identity is not None
        and _process_start_identity(pid) != expected_identity
    ):
        raise MonitorFailure(f"pid_identity_mismatch:pid={pid}")

    def target_exists() -> bool:
        try:
            if process_group_id is not None:
                os.killpg(process_group_id, 0)
                return True
            return _process_exists(pid, expected_identity)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        if process_group_id is not None:
            try:
                actual_group = os.getpgid(pid)
            except ProcessLookupError:
                actual_group = None
            if actual_group is not None:
                if actual_group != process_group_id or actual_group != pid:
                    raise MonitorFailure(
                        f"unsafe_process_group:pid={pid}:expected={process_group_id}:actual={actual_group}"
                    )
            elif not _process_group_exists(process_group_id):
                return
            os.killpg(process_group_id, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(term_timeout, 0.0)
    while target_exists() and time.monotonic() < deadline:
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0.0)))
    if not target_exists():
        return
    # Re-verify the leader incarnation immediately before escalation. If the
    # leader is gone, the still-existing group itself prevents PGID reuse.
    if expected_identity is not None and _process_exists(pid):
        if _process_start_identity(pid) != expected_identity:
            raise MonitorFailure(f"pid_identity_mismatch_before_kill:pid={pid}")
        if process_group_id is not None and os.getpgid(pid) != process_group_id:
            raise MonitorFailure(f"process_group_changed_before_kill:pid={pid}")
    try:
        if process_group_id is not None:
            os.killpg(process_group_id, signal.SIGKILL)
        else:
            if expected_identity is not None and _process_start_identity(pid) != expected_identity:
                raise MonitorFailure(f"pid_identity_mismatch:pid={pid}")
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _read_incremental_driver(path: Path, state: MonitorScanState, *, final: bool) -> str:
    if not path.is_file():
        return state.driver_text
    records = _read_incremental_bytes(path, state.driver_cursor, final=final)
    if not records:
        return state.driver_text
    decoded = []
    for _, _, raw in records:
        decoded.append(raw.decode("utf-8", errors="replace") + "\n")
    addition = "".join(decoded)
    state.driver_bytes += len(addition.encode("utf-8"))
    if state.driver_bytes > MAX_RETAINED_DRIVER_BYTES:
        raise MonitorFailure(f"driver_memory_limit:{state.driver_bytes}")
    state.driver_text += addition
    state.semantic_revision += sum(
        1
        for line in decoded
        if (
            "TASK22_EVENT " in line
            or "TASK22_FLOW " in line
            or METRIC_STEP_RE.search(line)
            or (
                ENGINE_PID_RE.search(line)
                and (BASE_GPU_RE.search(line) or SERVER_EVENT_RE.search(line))
            )
        )
    )
    return state.driver_text


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
        line = ANSI_ESCAPE_RE.sub("", line)
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
    poll_interval: float = 1.0,
    evidence_grace: float = 5.0,
    no_progress_timeout: float = 600.0,
    gpu_max_snapshot_age: float = 5.0,
    gpu_max_snapshot_interval: float = DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S,
    monitor_timeout: float = 5700.0,
    monitor_term_grace: float = 1.0,
    training_term_timeout: float = 10.0,
    num_gpus: int = 4,
    cuda_visible_devices: str = "",
    final: bool = False,
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
        "monitor_poll_interval_s": poll_interval,
        "monitor_evidence_grace_s": evidence_grace,
        "monitor_no_progress_timeout_s": no_progress_timeout,
        "gpu_max_snapshot_age_s": gpu_max_snapshot_age,
        "gpu_max_snapshot_interval_s": gpu_max_snapshot_interval,
        "monitor_timeout_s": monitor_timeout,
        "monitor_term_grace_s": monitor_term_grace,
        "training_term_timeout_s": training_term_timeout,
        "num_gpus": num_gpus,
        "cuda_visible_devices": cuda_visible_devices,
    }
    mismatches = {
        key: (value, contract.get(key))
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise MonitorFailure(f"run_contract_mismatch:{mismatches}")
    if contract.get("schema_version") != 6:
        raise MonitorFailure(
            f"run_contract_schema_mismatch:expected=6:actual={contract.get('schema_version')}"
        )
    working_dir = contract.get("working_dir")
    runtime_hash = contract.get("runtime_env_json_sha256")
    if (
        not isinstance(working_dir, str)
        or not Path(working_dir).is_absolute()
        or not isinstance(runtime_hash, str)
        or len(runtime_hash) != 64
        or not isinstance(contract.get("task22_env_sha256"), str)
        or len(contract["task22_env_sha256"]) != 64
    ):
        raise MonitorFailure("invalid_runtime_contract_fields")
    positive_monitor_fields = (
        "monitor_poll_interval_s",
        "monitor_evidence_grace_s",
        "gpu_max_snapshot_age_s",
        "gpu_max_snapshot_interval_s",
        "monitor_timeout_s",
        "training_term_timeout_s",
    )
    if (
        any(
            not isinstance(contract.get(name), (int, float))
            or isinstance(contract.get(name), bool)
            or float(contract[name]) <= 0
            for name in positive_monitor_fields
        )
        or not isinstance(contract.get("monitor_term_grace_s"), (int, float))
        or isinstance(contract.get("monitor_term_grace_s"), bool)
        or float(contract["monitor_term_grace_s"]) < 0
        or contract.get("monitor_no_progress_timeout_s") != 600.0
    ):
        raise MonitorFailure("invalid_qualification_monitor_contract")
    visible_devices = cuda_visible_devices.split(",") if cuda_visible_devices else []
    if (
        num_gpus != 4
        or (visible_devices and (len(visible_devices) != 4 or len(set(visible_devices)) != 4))
        or contract.get("num_gpus") != 4
        or contract.get("cuda_visible_devices") != cuda_visible_devices
    ):
        raise MonitorFailure("invalid_qualification_gpu_contract")
    attestation_dir = Path(
        os.environ.get(
            "TASK22_RUNTIME_ATTESTATION_DIR",
            str(run_dir / "runtime_attestation"),
        )
    )
    roles = Counter()
    attestations = []
    for path in sorted(attestation_dir.glob("runtime_attestation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MonitorFailure(f"invalid_runtime_attestation:{path.name}:{exc}") from exc
        if not isinstance(payload, dict):
            raise MonitorFailure(f"invalid_runtime_attestation:{path.name}:not_object")
        attestations.append(payload)
        if not attestation_matches_contract(payload, contract):
            raise MonitorFailure(
                "runtime_attestation_contract_mismatch:"
                f"{path.name}:submitted_working_dir={working_dir}:"
                f"runtime_working_dir={payload.get('working_dir')}"
            )
        roles[payload.get("role")] += 1
    if final:
        actor_ranks = {
            payload.get("rank") for payload in attestations if payload.get("role") == "actor"
        }
        engine_ranks = {
            payload.get("rank")
            for payload in attestations
            if payload.get("role") == "rollout_engine"
        }
        if not (
            roles["driver"] == 1
            and roles["actor"] == 2
            and actor_ranks == {0, 1}
            and roles["rollout_engine"] == expected_engines
            and engine_ranks == set(range(expected_engines))
        ):
            raise MonitorFailure(
                "incomplete_runtime_attestation_roles:"
                f"roles={dict(roles)}:actors={sorted(map(str, actor_ranks))}:"
                f"engines={sorted(map(str, engine_ranks))}"
            )


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


def _gpu_snapshot_status(
    path: Path, expected_gpu_count: int
) -> tuple[int, float | None, float | None, float, str | None]:
    if not path.is_file():
        return 0, None, None, 0.0, "missing GPU sampler output"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = 0
    first_timestamp = None
    last_timestamp = None
    max_interval = 0.0
    offset = 0
    while offset < len(lines):
        match = GPU_TIMESTAMP_RE.fullmatch(lines[offset].strip())
        if match is None:
            return (
                snapshots,
                first_timestamp,
                last_timestamp,
                max_interval,
                f"line={offset + 1}:invalid timestamp",
            )
        fraction = (match.group(2) or "0")[:6].ljust(6, "0")
        try:
            parsed_timestamp = datetime.strptime(
                f"{match.group(1)}.{fraction}{match.group(3)}",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ).timestamp()
        except ValueError as exc:
            return (
                snapshots,
                first_timestamp,
                last_timestamp,
                max_interval,
                f"line={offset + 1}:invalid timestamp:{exc}",
            )
        if last_timestamp is not None:
            interval = parsed_timestamp - last_timestamp
            if interval <= 0:
                return (
                    snapshots,
                    first_timestamp,
                    last_timestamp,
                    max_interval,
                    f"line={offset + 1}:timestamps not strictly increasing",
                )
            max_interval = max(max_interval, interval)
        first_timestamp = parsed_timestamp if first_timestamp is None else first_timestamp
        last_timestamp = parsed_timestamp
        offset += 1
        if len(lines) - offset < expected_gpu_count:
            return (
                snapshots,
                first_timestamp,
                last_timestamp,
                max_interval,
                f"line={offset + 1}:incomplete GPU snapshot",
            )
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
                return (
                    snapshots,
                    first_timestamp,
                    last_timestamp,
                    max_interval,
                    f"line={offset + 1}:{exc}",
                )
            offset += 1
        if len(indices) != expected_gpu_count:
            return (
                snapshots,
                first_timestamp,
                last_timestamp,
                max_interval,
                f"snapshot={snapshots}:GPU count={len(indices)}",
            )
        snapshots += 1
    return snapshots, first_timestamp, last_timestamp, max_interval, None


def _validate_gpu_snapshots(
    path: Path,
    *,
    expected_engines: int,
    final: bool,
    evidence_grace: float,
    evidence_due_since: dict[str, float],
    now_monotonic: float,
    run_started_at: float | None = None,
    run_ended_at: float | None = None,
    wall_time: float | None = None,
    max_snapshot_age: float | None = None,
    max_snapshot_interval: float = DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S,
) -> None:
    snapshots, first_timestamp, last_timestamp, observed_max_interval, error = _gpu_snapshot_status(
        path, expected_engines * 2
    )
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
    if error is not None or last_timestamp is None:
        return
    if observed_max_interval > max_snapshot_interval:
        raise MonitorFailure(
            f"gpu_snapshot_gap:observed={observed_max_interval:.3f}:"
            f"max={max_snapshot_interval:.3f}"
        )
    if (
        run_started_at is not None
        and first_timestamp is not None
        and first_timestamp > run_started_at + GPU_COVERAGE_TOLERANCE_S
    ):
        raise MonitorFailure(
            f"gpu_coverage_starts_late:first={first_timestamp}:run={run_started_at}"
        )
    if run_ended_at is not None and last_timestamp < run_ended_at - GPU_COVERAGE_TOLERANCE_S:
        raise MonitorFailure(
            f"gpu_coverage_ends_early:last={last_timestamp}:run={run_ended_at}"
        )
    if (
        not final
        and wall_time is not None
        and max_snapshot_age is not None
        and wall_time - last_timestamp > max_snapshot_age
    ):
        raise MonitorFailure(
            f"stale_gpu_snapshot:age={wall_time - last_timestamp:.3f}:"
            f"max={max_snapshot_age:.3f}"
        )


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
    driver_text: str | None = None,
    scan_state: MonitorScanState | None = None,
) -> None:
    if driver_text is None:
        completed = _completed_metric_steps(driver_log)
    else:
        categories_by_step: dict[int, set[str]] = {}
        for match in METRIC_STEP_RE.finditer(driver_text):
            category = "train" if match.group(1) == "step" else match.group(1)
            categories_by_step.setdefault(int(match.group(2)), set()).add(category)
        completed = {
            step
            for step, categories in categories_by_step.items()
            if {"rollout", "train", "perf"} <= categories
        }
    for step in range(headline_lo, headline_hi + 1):
        # Async steps may log out of order. A timeline becomes due only once
        # rollout, train, and perf metrics for that same step have all arrived.
        due = final or step in completed
        path = timeline_dir / f"timeline_step_{step}.json"
        if not due:
            continue
        if not path.is_file():
            if final:
                raise MonitorFailure(f"missing_headline_timeline:step={step}")
            first_due = evidence_due_since.setdefault(step, now_monotonic)
            if now_monotonic - first_due >= evidence_grace:
                raise MonitorFailure(f"missing_headline_timeline:step={step}")
            continue
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            _defer_or_raise(
                False,
                key=step,
                reason=f"invalid_headline_timeline:step={step}:{exc}",
                final=final,
                evidence_grace=evidence_grace,
                evidence_due_since=evidence_due_since,
                now_monotonic=now_monotonic,
            )
            continue
        if not _valid_timeline(payload):
            _defer_or_raise(
                False,
                key=step,
                reason=f"invalid_headline_timeline:step={step}:format",
                final=final,
                evidence_grace=evidence_grace,
                evidence_due_since=evidence_due_since,
                now_monotonic=now_monotonic,
            )
            continue
        evidence_due_since.pop(step, None)
        if scan_state is not None:
            fingerprint = hashlib.sha256(raw).hexdigest()
            if scan_state.timeline_fingerprints.get(step) != fingerprint:
                scan_state.timeline_fingerprints[step] = fingerprint
                scan_state.semantic_revision += 1


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
    poll_interval: float = 1.0,
    evidence_grace: float = 5.0,
    no_progress_timeout: float = 600.0,
    timeline_evidence_due_since: dict[int, float] | None = None,
    strict_evidence_due_since: dict[str, float] | None = None,
    now_monotonic: float | None = None,
    scan_state: MonitorScanState | None = None,
    run_started_at: float | None = None,
    run_ended_at: float | None = None,
    wall_time: float | None = None,
    gpu_max_snapshot_age: float | None = None,
    gpu_max_snapshot_interval: float = DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S,
    monitor_timeout: float = 5700.0,
    monitor_term_grace: float = 1.0,
    training_term_timeout: float = 10.0,
    num_gpus: int = 4,
    cuda_visible_devices: str = "",
) -> None:
    contract_gpu_max_snapshot_age = (
        5.0 if gpu_max_snapshot_age is None else gpu_max_snapshot_age
    )
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
        poll_interval=poll_interval,
        evidence_grace=evidence_grace,
        no_progress_timeout=no_progress_timeout,
        gpu_max_snapshot_age=contract_gpu_max_snapshot_age,
        gpu_max_snapshot_interval=gpu_max_snapshot_interval,
        monitor_timeout=monitor_timeout,
        monitor_term_grace=monitor_term_grace,
        training_term_timeout=training_term_timeout,
        num_gpus=num_gpus,
        cuda_visible_devices=cuda_visible_devices,
        final=final,
    )

    driver_log = run_dir / "driver.log"
    if scan_state is None:
        driver_text = (
            _complete_driver_text(
                driver_log.read_text(encoding="utf-8", errors="replace"),
                final=final,
            )
            if driver_log.is_file()
            else ""
        )
    else:
        driver_text = _read_incremental_driver(driver_log, scan_state, final=final)
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
            rows = _read_retained_jsonl(
                path,
                final=final,
                state=scan_state,
                evidence_grace=evidence_grace,
                evidence_due_since=strict_due_since,
                now_monotonic=scan_monotonic,
            )
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
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        wall_time=wall_time,
        max_snapshot_age=gpu_max_snapshot_age,
        max_snapshot_interval=gpu_max_snapshot_interval,
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
        driver_text=driver_text,
        scan_state=scan_state,
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
    pid_start_identity: str | None = None,
    no_progress_timeout: float = 600.0,
    post_exit_grace: float | None = None,
    term_timeout: float = 10.0,
    sampler_pid: int | None = None,
    sampler_start_identity: str | None = None,
    run_started_at: float | None = None,
    gpu_max_snapshot_age: float = 5.0,
    gpu_max_snapshot_interval: float = DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S,
    monitor_timeout: float = 5700.0,
    monitor_term_grace: float = 1.0,
    num_gpus: int = 4,
    cuda_visible_devices: str = "",
) -> int:
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
    if num_gpus != 4:
        raise MonitorFailure(f"invalid_gpu_contract:{num_gpus}:expected=4")
    if no_progress_timeout != 600.0:
        raise MonitorFailure(
            f"invalid_no_progress_timeout:{no_progress_timeout:g}:expected=600"
        )

    event_log = run_dir / "online_monitor.jsonl"
    reported_lifecycle: set[tuple[str, int]] = set()
    timeline_evidence_due_since: dict[int, float] = {}
    strict_evidence_due_since: dict[str, float] = {}
    scan_state = MonitorScanState()
    expected_identity = pid_start_identity or _process_start_identity(pid)
    expected_sampler_identity = (
        sampler_start_identity
        or (_process_start_identity(sampler_pid) if sampler_pid is not None else None)
    )
    stable_grace = evidence_grace if post_exit_grace is None else post_exit_grace
    last_progress_at = time.monotonic()
    last_progress_token: tuple[Any, ...] | None = None
    exit_stable_since: float | None = None
    exit_quiet_scans = 0
    run_ended_at: float | None = None
    _safe_append_event(
        event_log,
        "monitor_started",
        pid=pid,
        pid_start_identity=expected_identity,
        sampler_pid=sampler_pid,
        sampler_start_identity=expected_sampler_identity,
    )
    while True:
        now = time.monotonic()
        pid_present = _process_exists(pid)
        identity = _process_start_identity(pid) if pid_present else None
        identity_changed = (
            pid_present
            and expected_identity is not None
            and identity != expected_identity
        )
        group_present = pid_present or (
            _process_group_exists(process_group_id)
            if process_group_id is not None
            else False
        )
        running = group_present and not identity_changed
        sampler_running = (
            True
            if sampler_pid is None
            else _process_exists(sampler_pid, expected_sampler_identity)
        )
        progress_token = _evidence_progress_token(run_dir, scan_state)
        if progress_token != last_progress_token:
            last_progress_token = progress_token
            last_progress_at = now
            if exit_stable_since is not None:
                exit_stable_since = now
        if running:
            exit_stable_since = None
            exit_quiet_scans = 0
            run_ended_at = None
        elif exit_stable_since is None:
            exit_stable_since = now
            run_ended_at = time.time()
        try:
            if identity_changed:
                raise MonitorFailure(
                    f"pid_identity_changed:pid={pid}:expected={expected_identity}:actual={identity}"
                )
            if not sampler_running:
                raise MonitorFailure(f"gpu_sampler_not_running:pid={sampler_pid}")
            revision_before_scan = scan_state.semantic_revision
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
                final=False,
                reported_lifecycle=reported_lifecycle,
                event_log=event_log,
                poll_interval=poll_interval,
                evidence_grace=evidence_grace,
                no_progress_timeout=no_progress_timeout,
                timeline_evidence_due_since=timeline_evidence_due_since,
                strict_evidence_due_since=strict_evidence_due_since,
                now_monotonic=now,
                scan_state=scan_state,
                run_started_at=run_started_at,
                run_ended_at=run_ended_at,
                wall_time=time.time(),
                gpu_max_snapshot_age=gpu_max_snapshot_age,
                gpu_max_snapshot_interval=gpu_max_snapshot_interval,
                monitor_timeout=monitor_timeout,
                monitor_term_grace=monitor_term_grace,
                training_term_timeout=term_timeout,
                num_gpus=num_gpus,
                cuda_visible_devices=cuda_visible_devices,
            )
            # Scan before enforcing the deadline so a complete semantic record
            # already on disk at the boundary gets counted.
            scanned_token = _evidence_progress_token(run_dir, scan_state)
            if scanned_token != last_progress_token:
                last_progress_token = scanned_token
                last_progress_at = now
                if exit_stable_since is not None:
                    exit_stable_since = now
            semantic_progress = scan_state.semantic_revision != revision_before_scan
            final = False
            if not running:
                if semantic_progress:
                    exit_quiet_scans = 0
                else:
                    exit_quiet_scans += 1
                    if exit_quiet_scans >= 1 and now - exit_stable_since >= stable_grace:
                        final_revision = scan_state.semantic_revision
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
                            final=True,
                            reported_lifecycle=reported_lifecycle,
                            event_log=event_log,
                            poll_interval=poll_interval,
                            evidence_grace=evidence_grace,
                            no_progress_timeout=no_progress_timeout,
                            timeline_evidence_due_since=timeline_evidence_due_since,
                            strict_evidence_due_since=strict_evidence_due_since,
                            now_monotonic=now,
                            scan_state=scan_state,
                            run_started_at=run_started_at,
                            run_ended_at=run_ended_at,
                            wall_time=time.time(),
                            gpu_max_snapshot_age=gpu_max_snapshot_age,
                            gpu_max_snapshot_interval=gpu_max_snapshot_interval,
                            monitor_timeout=monitor_timeout,
                            monitor_term_grace=monitor_term_grace,
                            training_term_timeout=term_timeout,
                            num_gpus=num_gpus,
                            cuda_visible_devices=cuda_visible_devices,
                        )
                        if scan_state.semantic_revision != final_revision:
                            last_progress_token = _evidence_progress_token(run_dir, scan_state)
                            last_progress_at = now
                            exit_stable_since = now
                            exit_quiet_scans = 0
                        else:
                            final = True
            if running and now - last_progress_at >= no_progress_timeout:
                raise MonitorFailure(f"no_evidence_progress:{no_progress_timeout:g}s")
        except (MonitorFailure, OSError) as exc:
            reason = str(exc)
            audit_written = _safe_append_event(event_log, "monitor_failed", reason=reason)
            if running:
                try:
                    _stop_process(
                        pid,
                        process_group_id,
                        expected_identity=expected_identity,
                        term_timeout=term_timeout,
                        poll_interval=min(poll_interval, 0.1),
                    )
                except MonitorFailure as stop_exc:
                    _safe_append_event(
                        event_log,
                        "training_stop_rejected",
                        pid=pid,
                        reason=str(stop_exc),
                    )
                else:
                    _safe_append_event(
                        event_log,
                        "training_stop_requested",
                        pid=pid,
                        reason=reason,
                        failure_audit_written=audit_written,
                    )
            return 4
        if final:
            _safe_append_event(event_log, "monitor_passed")
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
    parser.add_argument("--pid-start-identity")
    parser.add_argument("--sampler-pid", type=int)
    parser.add_argument("--sampler-start-identity")
    parser.add_argument("--run-started-at", type=float)
    parser.add_argument("--gpu-max-snapshot-age", type=float, default=5.0)
    parser.add_argument(
        "--gpu-max-snapshot-interval",
        type=float,
        default=DEFAULT_GPU_MAX_SNAPSHOT_INTERVAL_S,
    )
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument("--post-exit-grace", type=float)
    parser.add_argument("--term-timeout", type=float, default=10.0)
    parser.add_argument("--monitor-timeout", type=float, default=5700.0)
    parser.add_argument("--monitor-term-grace", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.evidence_grace < 5.0:
        parser.error("--evidence-grace must be at least 5 seconds")
    if args.no_progress_timeout <= 0:
        parser.error("--no-progress-timeout must be positive")
    if args.post_exit_grace is not None and args.post_exit_grace < 0:
        parser.error("--post-exit-grace must be non-negative")
    if args.term_timeout < 0:
        parser.error("--term-timeout must be non-negative")
    if args.monitor_timeout <= 0:
        parser.error("--monitor-timeout must be positive")
    if args.monitor_term_grace < 0:
        parser.error("--monitor-term-grace must be non-negative")
    if args.sampler_pid is not None and args.sampler_pid <= 0:
        parser.error("--sampler-pid must be positive")
    if args.gpu_max_snapshot_age <= 0:
        parser.error("--gpu-max-snapshot-age must be positive")
    if args.gpu_max_snapshot_interval <= 0:
        parser.error("--gpu-max-snapshot-interval must be positive")
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
            pid_start_identity=args.pid_start_identity,
            no_progress_timeout=args.no_progress_timeout,
            post_exit_grace=args.post_exit_grace,
            term_timeout=args.term_timeout,
            sampler_pid=args.sampler_pid,
            sampler_start_identity=args.sampler_start_identity,
            run_started_at=args.run_started_at,
            gpu_max_snapshot_age=args.gpu_max_snapshot_age,
            gpu_max_snapshot_interval=args.gpu_max_snapshot_interval,
            monitor_timeout=args.monitor_timeout,
            monitor_term_grace=args.monitor_term_grace,
            num_gpus=args.num_gpus,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    )


if __name__ == "__main__":
    main()
