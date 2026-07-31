#!/usr/bin/env python3
"""Strict artifact validator for one Task 22 admission run."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from relax.engine.rollout.request_observability import attempt_token_from_id
from relax.utils.task22_runtime_attestation import attestation_matches_contract
from scripts.task22.analyze_rollout_observability import analyze as analyze_requests


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ROLLOUT_METRICS_RE = re.compile(r"\brollout (\d+): (\{.*\})")
TRAIN_METRICS_RE = re.compile(r"\bstep (\d+): (\{.*\})")
PERF_METRICS_RE = re.compile(r"\bperf (\d+): (\{.*\})")
ID_FROM_FILENAME_RE = re.compile(r"_(\d+)(?:_rank_\d+)?\.jsonl$")
GPU_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?([+-]\d{4})$"
)
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
ATTEMPT_OUTCOMES = {
    "aborted",
    "carried_with_aborted_group",
    "committed",
    "filtered",
    "surplus",
    "uncommitted_protected",
}
SYNC_EVENT_PHASES = {"gate", "pause", "flush", "transfer", "continue"}
EVENT_PHASES = SYNC_EVENT_PHASES | {"abort"}
FLOW_PHASES = {"gate_blocked", "gate_ready", "physical_start", "physical_end", "partition_close"}
RUN_CONTRACT_SCHEMA_VERSION = 6


class Validation:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.failures: dict[str, list[Any]] = defaultdict(list)

    def check(self, name: str, condition: bool, detail: Any | None = None) -> None:
        passed = bool(condition)
        self.checks[name] = self.checks.get(name, True) and passed
        if not passed and detail is not None and len(self.failures[name]) < 50:
            self.failures[name].append(detail)

    def result(self, **extra: Any) -> dict[str, Any]:
        return {
            "verdict": "PASS" if self.checks and all(self.checks.values()) else "FAIL",
            "checks": self.checks,
            "failures": dict(self.failures),
            **extra,
        }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _read_jsonl(path: Path, validation: Validation) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        validation.check("jsonl_readable", False, {"path": str(path), "error": str(exc)})
        return rows
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            validation.check(
                "jsonl_parseable",
                False,
                {"path": str(path), "line": line_no, "error": str(exc)},
            )
            continue
        if not isinstance(row, dict):
            validation.check(
                "jsonl_rows_are_objects",
                False,
                {"path": str(path), "line": line_no},
            )
            continue
        row["_source"] = f"{path.name}:{line_no}"
        rows.append(row)
    return rows


def _complete_driver_text(text: str) -> str:
    if text and not text.endswith(("\n", "\r")):
        text = text.rsplit("\n", 1)[0] + ("\n" if "\n" in text else "")
    return text


def _load_glob(directory: Path, pattern: str, validation: Validation) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(directory.glob(pattern))
    rows = [row for path in paths for row in _read_jsonl(path, validation)]
    return paths, rows


def _artifact_id(path: Path) -> int | None:
    match = ID_FROM_FILENAME_RE.search(path.name)
    return int(match.group(1)) if match else None


def _parse_structured_lines(driver_text: str, prefix: str) -> list[dict[str, str]]:
    rows = []
    for line_no, raw_line in enumerate(driver_text.splitlines(), 1):
        clean = ANSI_RE.sub("", raw_line)
        marker = clean.find(prefix)
        if marker < 0:
            continue
        row = {"_line": str(line_no)}
        for item in shlex.split(clean[marker + len(prefix) :].strip()):
            if "=" in item:
                key, value = item.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def _parse_metric_rows(driver_text: str, pattern: re.Pattern[str]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_line in driver_text.splitlines():
        clean = ANSI_RE.sub("", raw_line)
        match = pattern.search(clean)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            rows[int(match.group(1))].append(payload)
    return rows


def _validate_event_interval(row: dict[str, str]) -> bool:
    try:
        begin = float(row["t_begin"])
        end = float(row["t_end"])
        duration = float(row["dur"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(begin)
        and math.isfinite(end)
        and math.isfinite(duration)
        and begin <= end
        and duration >= 0
        and math.isclose(duration, end - begin, rel_tol=1e-4, abs_tol=1e-3)
    )


def _finite_field(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _valid_chrome_timeline(payload: Any) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    previous_ts = -1.0
    for event in payload:
        if not isinstance(event, dict):
            return False
        if not isinstance(event.get("name"), str) or not event["name"]:
            return False
        if event.get("ph") != "X":
            return False
        ts = event.get("ts")
        duration = event.get("dur")
        if not _finite_number(ts) or float(ts) < 0 or float(ts) < previous_ts:
            return False
        if not _finite_number(duration) or float(duration) < 0:
            return False
        if any(not isinstance(event.get(key), int) or isinstance(event.get(key), bool) for key in ("pid", "tid")):
            return False
        if "args" in event and not isinstance(event["args"], dict):
            return False
        previous_ts = float(ts)
    return True


def _gpu_number(value: str, suffix: str) -> float:
    value = value.strip()
    if suffix and value.endswith(suffix):
        value = value[: -len(suffix)].strip()
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite GPU value: {value!r}")
    return number


def _parse_gpu_snapshots(path: Path, expected_engines: int) -> tuple[int, list[dict[str, Any]]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_gpu_count = expected_engines * 2
    snapshots = 0
    failures: list[dict[str, Any]] = []
    offset = 0
    while offset < len(lines):
        timestamp_line = lines[offset].strip()
        try:
            match = GPU_TIMESTAMP_RE.fullmatch(timestamp_line)
            if match is None:
                raise ValueError("timestamp does not match ISO-8601 sampler format")
            fraction = (match.group(2) or "0")[:6].ljust(6, "0")
            datetime.strptime(
                f"{match.group(1)}.{fraction}{match.group(3)}",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            )
        except ValueError as exc:
            failures.append({"line": offset + 1, "error": f"invalid timestamp: {exc}"})
            break
        offset += 1
        gpu_indices = set()
        for _ in range(expected_gpu_count):
            if offset >= len(lines):
                failures.append({"line": offset + 1, "error": "incomplete GPU snapshot"})
                break
            try:
                fields = next(csv.reader([lines[offset]], skipinitialspace=True))
                if len(fields) != 5:
                    raise ValueError(f"expected 5 CSV fields, got {len(fields)}")
                gpu_index = int(fields[0].strip())
                memory_used = _gpu_number(fields[1], "MiB")
                gpu_util = _gpu_number(fields[2], "%")
                memory_util = _gpu_number(fields[3], "%")
                power_draw = _gpu_number(fields[4], "W")
                if gpu_index < 0 or gpu_index in gpu_indices:
                    raise ValueError("GPU indices must be unique and non-negative per snapshot")
                if memory_used < 0 or power_draw < 0:
                    raise ValueError("memory and power values must be non-negative")
                if not (0 <= gpu_util <= 100 and 0 <= memory_util <= 100):
                    raise ValueError("utilization must be between 0 and 100")
                gpu_indices.add(gpu_index)
            except (ValueError, csv.Error) as exc:
                failures.append({"line": offset + 1, "error": str(exc)})
            offset += 1
        if len(gpu_indices) != expected_gpu_count:
            break
        snapshots += 1
    if offset != len(lines):
        failures.append({"line": offset + 1, "error": "unexpected trailing GPU sampler output"})
    return snapshots, failures


def validate_run(
    run_dir: Path,
    *,
    expected_mode: str,
    expected_rollouts: int,
    expected_samples_per_partition: int,
    expected_engines: int,
    max_staleness: int,
    headline_lo: int,
    headline_hi: int,
    require_resume: bool,
    monitor_poll_interval: float = 1.0,
    monitor_evidence_grace: float = 5.0,
    monitor_no_progress_timeout: float = 600.0,
    gpu_max_snapshot_age: float = 5.0,
    gpu_max_snapshot_interval: float = 2.0,
    monitor_timeout: float = 5700.0,
    monitor_term_grace: float = 1.0,
    training_term_timeout: float = 10.0,
    num_gpus: int = 4,
    cuda_visible_devices: str = "",
) -> dict[str, Any]:
    validation = Validation()
    driver_log = run_dir / "driver.log"
    observability_dir = run_dir / "observability"
    timeline_dir = run_dir / "timeline"
    gpu_log = run_dir / "logs" / "nvidia_smi_1s.csv"
    contract_path = run_dir / "run_contract.json"

    validation.check("run_directory_exists", run_dir.is_dir(), str(run_dir))
    validation.check("driver_log_present", driver_log.is_file() and driver_log.stat().st_size > 0, str(driver_log))
    validation.check("observability_directory_present", observability_dir.is_dir(), str(observability_dir))
    validation.check("timeline_directory_present", timeline_dir.is_dir(), str(timeline_dir))
    validation.check("gpu_sample_present", gpu_log.is_file() and gpu_log.stat().st_size > 0, str(gpu_log))
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("run contract must be a JSON object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        contract = {}
        validation.check("run_contract_valid", False, {"path": str(contract_path), "error": str(exc)})
    else:
        validation.check("run_contract_valid", True)
    validation.check(
        "run_contract_schema_is_current_v6",
        contract.get("schema_version") == RUN_CONTRACT_SCHEMA_VERSION,
        {
            "expected": RUN_CONTRACT_SCHEMA_VERSION,
            "actual": contract.get("schema_version"),
        },
    )
    expected_contract = {
        "admission_mode": expected_mode,
        "num_rollout": expected_rollouts,
        "expected_samples_per_partition": expected_samples_per_partition,
        "expected_engines": expected_engines,
        "max_staleness": max_staleness,
        "headline_lo": headline_lo,
        "headline_hi": headline_hi,
        "monitor_poll_interval_s": monitor_poll_interval,
        "monitor_evidence_grace_s": monitor_evidence_grace,
        "monitor_no_progress_timeout_s": monitor_no_progress_timeout,
        "gpu_max_snapshot_age_s": gpu_max_snapshot_age,
        "gpu_max_snapshot_interval_s": gpu_max_snapshot_interval,
        "monitor_timeout_s": monitor_timeout,
        "monitor_term_grace_s": monitor_term_grace,
        "training_term_timeout_s": training_term_timeout,
        "num_gpus": num_gpus,
        "cuda_visible_devices": cuda_visible_devices,
    }
    validation.check(
        "run_contract_matches_validator_arguments",
        all(contract.get(key) == value for key, value in expected_contract.items()),
        {key: {"expected": value, "actual": contract.get(key)} for key, value in expected_contract.items()},
    )
    validation.check(
        "admission_contract_is_4_8_2",
        (
            contract.get("admission_min"),
            contract.get("admission_max"),
            contract.get("admission_slack"),
        )
        == (4, 8, 2),
        {
            "admission_min": contract.get("admission_min"),
            "admission_max": contract.get("admission_max"),
            "admission_slack": contract.get("admission_slack"),
        },
    )
    validation.check(
        "request_placement_is_off",
        contract.get("request_placement_mode") == "off",
        contract.get("request_placement_mode"),
    )
    validation.check(
        "slime_router_is_disabled",
        contract.get("use_slime_router") is False,
        contract.get("use_slime_router"),
    )
    if contract.get("schema_version") == RUN_CONTRACT_SCHEMA_VERSION:
        validation.check(
            "runtime_contract_fields_valid",
            isinstance(contract.get("working_dir"), str)
            and Path(contract["working_dir"]).is_absolute()
            and isinstance(contract.get("runtime_env_json_sha256"), str)
            and len(contract["runtime_env_json_sha256"]) == 64
            and isinstance(contract.get("task22_env_sha256"), str)
            and len(contract["task22_env_sha256"]) == 64,
        )
        validation.check(
            "qualification_monitor_contract_valid",
            all(
                isinstance(contract.get(name), (int, float))
                and not isinstance(contract.get(name), bool)
                and float(contract[name]) > 0
                for name in (
                    "monitor_poll_interval_s",
                    "monitor_evidence_grace_s",
                    "gpu_max_snapshot_age_s",
                    "gpu_max_snapshot_interval_s",
                    "monitor_timeout_s",
                    "training_term_timeout_s",
                )
            )
            and isinstance(contract.get("monitor_term_grace_s"), (int, float))
            and not isinstance(contract.get("monitor_term_grace_s"), bool)
            and float(contract["monitor_term_grace_s"]) >= 0
            and contract.get("monitor_no_progress_timeout_s") == 600.0,
        )
        visible_devices = cuda_visible_devices.split(",") if cuda_visible_devices else []
        validation.check(
            "qualification_gpu_contract_valid",
            num_gpus == 4
            and contract.get("num_gpus") == 4
            and contract.get("cuda_visible_devices") == cuda_visible_devices
            and (
                not visible_devices
                or (len(visible_devices) == 4 and len(set(visible_devices)) == 4)
            ),
            {
                "expected_num_gpus": num_gpus,
                "contract_num_gpus": contract.get("num_gpus"),
                "expected_cuda_visible_devices": cuda_visible_devices,
                "contract_cuda_visible_devices": contract.get("cuda_visible_devices"),
            },
        )
        if contract.get("schema_version") == RUN_CONTRACT_SCHEMA_VERSION:
            content_hashes_valid = all(
                isinstance(contract.get(name), str) and len(contract[name]) == 64
                for name in ("working_dir_content_sha256", "input_manifest_sha256")
            )
            validation.check("runtime_contract_content_hashes_valid", content_hashes_valid)
            phase_hashes = {}
            for phase in ("before", "after"):
                path = run_dir / f"input_manifest_{phase}.json"
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    phase_hashes[phase] = payload["manifest_sha256"]
                except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    validation.check(
                        "input_manifest_attestations_parseable",
                        False,
                        {"path": str(path), "error": str(exc)},
                    )
                else:
                    validation.check("input_manifest_attestations_parseable", True)
            validation.check(
                "input_manifest_before_after_match_contract",
                phase_hashes.get("before")
                == phase_hashes.get("after")
                == contract.get("input_manifest_sha256"),
                {
                    "before": phase_hashes.get("before"),
                    "training_contract": contract.get("input_manifest_sha256"),
                    "after": phase_hashes.get("after"),
                },
            )
        attestation_dir = run_dir / "runtime_attestation"
        attestation_paths = sorted(attestation_dir.glob("runtime_attestation_*.json"))
        attestations = []
        for path in attestation_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("attestation must be a JSON object")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                validation.check(
                    "runtime_attestations_parseable",
                    False,
                    {"path": str(path), "error": str(exc)},
                )
                continue
            attestations.append(payload)
            validation.check("runtime_attestations_parseable", True)
            validation.check(
                "runtime_attestations_match_contract",
                attestation_matches_contract(payload, contract),
                {
                    "path": str(path),
                    "role": payload.get("role"),
                    "submitted_working_dir": contract.get("working_dir"),
                    "runtime_working_dir": payload.get("working_dir"),
                },
            )
        roles = Counter(item.get("role") for item in attestations)
        actor_ranks = {
            item.get("rank") for item in attestations if item.get("role") == "actor"
        }
        engine_ranks = {
            item.get("rank")
            for item in attestations
            if item.get("role") == "rollout_engine"
        }
        validation.check(
            "runtime_attestation_roles_complete",
            roles["driver"] == 1
            and roles["actor"] == 2
            and actor_ranks == {0, 1}
            and roles["rollout_engine"] == expected_engines
            and engine_ranks == set(range(expected_engines)),
            {
                "roles": dict(roles),
                "actor_ranks": sorted(value for value in actor_ranks if isinstance(value, int)),
                "engine_ranks": sorted(value for value in engine_ranks if isinstance(value, int)),
            },
        )
        if contract.get("schema_version") == RUN_CONTRACT_SCHEMA_VERSION:
            runtime_input_hashes = {
                item.get("input_manifest", {}).get("sha256") for item in attestations
            }
            validation.check(
                "input_manifest_three_point_attestation_consistent",
                runtime_input_hashes == {contract.get("input_manifest_sha256")}
                and phase_hashes.get("before")
                == phase_hashes.get("after")
                == contract.get("input_manifest_sha256"),
                {
                    "before": phase_hashes.get("before"),
                    "runtime": sorted(value for value in runtime_input_hashes if value),
                    "after": phase_hashes.get("after"),
                },
            )

    exit_code_path = run_dir / "EXIT_CODE"
    try:
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        exit_code = None
    validation.check("process_exit_zero", exit_code == 0, {"exit_code": exit_code})

    if not driver_log.is_file() or not observability_dir.is_dir():
        return validation.result(counts={})
    driver_text = _complete_driver_text(driver_log.read_text(encoding="utf-8", errors="replace"))

    try:
        request_analysis = analyze_requests(
            driver_log,
            observability_dir,
            expected_engines=expected_engines,
            require_resume=require_resume,
        )
    except Exception as exc:  # noqa: BLE001
        request_analysis = {"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    validation.check(
        "request_observability_passes",
        request_analysis.get("verdict") == "PASS",
        request_analysis,
    )

    request_paths, request_rows = _load_glob(
        observability_dir,
        "request_lifecycle_rollout_*.jsonl",
        validation,
    )
    ledger_paths, ledger_rows = _load_glob(
        observability_dir,
        "admission_ledger_rollout_*.jsonl",
        validation,
    )
    outcome_paths, outcome_rows = _load_glob(
        observability_dir,
        "admission_outcomes_rollout_*.jsonl",
        validation,
    )
    consumption_paths, consumption_rows = _load_glob(
        observability_dir,
        "consumption_ledger_rollout_*_rank_*.jsonl",
        validation,
    )
    event_rows = _parse_structured_lines(driver_text, "TASK22_EVENT")
    flow_rows = _parse_structured_lines(driver_text, "TASK22_FLOW")
    required_physical = set(range(expected_rollouts))
    allowed_physical = required_physical | {expected_rollouts}
    expected_partitions = {f"train_{rollout_id}" for rollout_id in range(expected_rollouts)}

    request_file_ids = {_artifact_id(path) for path in request_paths}
    ledger_file_ids = {_artifact_id(path) for path in ledger_paths}
    outcome_file_ids = {_artifact_id(path) for path in outcome_paths}
    consumption_file_ids = {_artifact_id(path) for path in consumption_paths}
    lifecycle_physical_ids = {
        int(row["physical_rollout_id"])
        for row in flow_rows
        if row.get("phase") in {"physical_start", "physical_end"}
        and str(row.get("physical_rollout_id", "")).isdigit()
    }
    lifecycle_physical_ids.update(
        int(row["rollout_id"])
        for row in event_rows
        if row.get("phase") == "abort" and str(row.get("rollout_id", "")).isdigit()
    )
    validation.check(
        "artifact_filenames_parseable",
        None not in request_file_ids | ledger_file_ids | outcome_file_ids | consumption_file_ids,
        {
            "request_files": [path.name for path in request_paths if _artifact_id(path) is None],
            "ledger_files": [path.name for path in ledger_paths if _artifact_id(path) is None],
            "outcome_files": [path.name for path in outcome_paths if _artifact_id(path) is None],
            "consumption_files": [path.name for path in consumption_paths if _artifact_id(path) is None],
        },
    )
    request_file_ids.discard(None)
    ledger_file_ids.discard(None)
    outcome_file_ids.discard(None)
    consumption_file_ids.discard(None)
    observed_physical = request_file_ids | ledger_file_ids | outcome_file_ids | lifecycle_physical_ids
    expected_physical = required_physical | ({expected_rollouts} if expected_rollouts in observed_physical else set())
    validation.check(
        "physical_rollout_ids_allowed",
        observed_physical.issubset(allowed_physical),
        {"unexpected": sorted(observed_physical - allowed_physical), "allowed": sorted(allowed_physical)},
    )
    validation.check(
        "expected_request_files_exact",
        request_file_ids == expected_physical,
        {
            "missing": sorted(expected_physical - request_file_ids),
            "extra": sorted(request_file_ids - expected_physical),
        },
    )
    validation.check(
        "expected_admission_ledgers_exact",
        ledger_file_ids == expected_physical,
        {
            "missing": sorted(expected_physical - ledger_file_ids),
            "extra": sorted(ledger_file_ids - expected_physical),
        },
    )
    validation.check(
        "consumption_partition_files_exact",
        consumption_file_ids == required_physical,
        {
            "missing": sorted(required_physical - consumption_file_ids),
            "extra": sorted(consumption_file_ids - required_physical),
        },
    )
    validation.check("partition_outcomes_present", bool(outcome_paths and outcome_rows))
    validation.check("consumption_ledgers_present", bool(consumption_paths and consumption_rows))

    for path in request_paths:
        artifact_id = _artifact_id(path)
        for row in _read_jsonl(path, validation):
            validation.check(
                "request_file_physical_ids_match",
                row.get("physical_rollout_id") == artifact_id,
                {"source": row["_source"], "physical_rollout_id": row.get("physical_rollout_id")},
            )
    for path in ledger_paths:
        artifact_id = _artifact_id(path)
        for row in _read_jsonl(path, validation):
            validation.check(
                "ledger_file_physical_ids_match",
                row.get("physical_rollout_id") == artifact_id,
                {"source": row["_source"], "physical_rollout_id": row.get("physical_rollout_id")},
            )
    for path in outcome_paths:
        artifact_id = _artifact_id(path)
        for row in _read_jsonl(path, validation):
            validation.check(
                "outcome_file_decision_physical_ids_match",
                row.get("physical_rollout_id") == artifact_id,
                {
                    "source": row["_source"],
                    "physical_rollout_id": row.get("physical_rollout_id"),
                },
            )
    for path in consumption_paths:
        artifact_id = _artifact_id(path)
        for row in _read_jsonl(path, validation):
            validation.check(
                "consumption_file_partition_ids_match",
                row.get("rollout_id") == artifact_id
                and row.get("target_partition") == f"train_{artifact_id}",
                {
                    "source": row["_source"],
                    "rollout_id": row.get("rollout_id"),
                    "target_partition": row.get("target_partition"),
                },
            )

    decisions = [row for row in ledger_rows if row.get("record_type") == "admission_decision"]
    ledger_attempts = [row for row in ledger_rows if row.get("record_type") == "attempt"]
    decision_counts = Counter(str(row.get("decision_id")) for row in decisions)
    decision_by_id = {
        str(row["decision_id"]): row
        for row in decisions
        if row.get("decision_id") is not None and decision_counts[str(row["decision_id"])] == 1
    }
    validation.check(
        "admission_decision_ids_unique",
        bool(decisions) and all(count == 1 for count in decision_counts.values()),
        {"duplicates": sorted(key for key, count in decision_counts.items() if count != 1)},
    )
    validation.check(
        "admission_mode_matches_contract",
        bool(decisions) and all(row.get("mode") == expected_mode for row in decisions),
        sorted({row.get("mode") for row in decisions}),
    )
    validation.check(
        "admission_has_no_fail_open",
        all(not str(row.get("bypass_reason") or "").startswith("fail_open:") for row in decisions),
        [
            row.get("decision_id")
            for row in decisions
            if str(row.get("bypass_reason") or "").startswith("fail_open:")
        ],
    )
    mode_semantics_valid = True
    bounded_triggered = False
    for row in decisions:
        bypass = row.get("bypass_reason")
        actual = row.get("actual_admit_groups")
        eager = row.get("eager_admit_groups")
        bounded = row.get("bounded_admit_groups")
        if bypass is None:
            bounded_triggered = bounded_triggered or bounded != eager
            expected_actual = eager if expected_mode == "shadow" else bounded
            if actual != expected_actual:
                mode_semantics_valid = False
                validation.failures["admission_mode_semantics"].append(row.get("decision_id"))
        debt = row.get("release_remaining")
        available = row.get("available_groups")
        if bypass == "final_backfill":
            logical_debt = row.get("logical_debt_groups")
            validation.check(
                "final_backfill_actual_within_debt",
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and isinstance(debt, int)
                and not isinstance(debt, bool)
                and isinstance(logical_debt, int)
                and not isinstance(logical_debt, bool)
                and isinstance(available, int)
                and not isinstance(available, bool)
                and 0 <= actual <= min(debt, logical_debt, available),
                {
                    "decision_id": row.get("decision_id"),
                    "actual_admit_groups": actual,
                    "release_remaining": debt,
                    "logical_debt_groups": logical_debt,
                    "available_groups": available,
                },
            )
        inflight = row.get("inflight_groups")
        valid_inputs = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (debt, inflight, available)
        )
        expected_desired = 8
        if valid_inputs and debt > 0:
            expected_desired = max(4, min(8, debt + 2))
        expected_bounded = min(available, max(expected_desired - inflight, 0)) if valid_inputs else None
        validation.check(
            "admission_bounded_recomputes",
            valid_inputs
            and row.get("desired_inflight_groups") == expected_desired
            and bounded == expected_bounded,
            {
                "decision_id": row.get("decision_id"),
                "expected_desired": expected_desired if valid_inputs else None,
                "actual_desired": row.get("desired_inflight_groups"),
                "expected_bounded": expected_bounded,
                "actual_bounded": bounded,
            },
        )
    validation.check("admission_mode_semantics", mode_semantics_valid)
    validation.check("admission_bounded_path_exercised", bounded_triggered)

    attempt_counts = Counter(str(row.get("attempt_id") or row.get("rid")) for row in request_rows)
    request_by_token: dict[int, dict[str, Any]] = {}
    duplicate_attempt_tokens = []
    for row in request_rows:
        attempt_id = row.get("attempt_id") or row.get("rid")
        try:
            token = attempt_token_from_id(str(attempt_id))
        except (TypeError, ValueError):
            validation.check("request_attempt_tokens_derivable", False, row.get("_source"))
            continue
        if token in request_by_token:
            duplicate_attempt_tokens.append(token)
        else:
            request_by_token[token] = row
        validation.check(
            "attempt_rows_have_terminal_business_outcome",
            row.get("outcome") in ATTEMPT_OUTCOMES,
            {"source": row.get("_source"), "outcome": row.get("outcome")},
        )
    validation.check(
        "request_attempt_ids_unique",
        bool(attempt_counts) and all(count == 1 for count in attempt_counts.values()),
        {"duplicates": sorted(key for key, count in attempt_counts.items() if count != 1)},
    )
    validation.check("request_attempt_tokens_unique", not duplicate_attempt_tokens, duplicate_attempt_tokens)
    validation.check(
        "ledger_attempts_match_request_rows",
        Counter(str(row.get("attempt_id") or row.get("rid")) for row in ledger_attempts) == attempt_counts,
    )
    for row in request_rows:
        validation.check(
            "attempt_decision_exists",
            row.get("admission_decision_id") in decision_by_id,
            {"source": row.get("_source"), "decision_id": row.get("admission_decision_id")},
        )

    attempt_by_id = {
        str(row.get("attempt_id") or row.get("rid")): row
        for row in request_rows
        if row.get("attempt_id") or row.get("rid")
    }
    for attempt_id, row in attempt_by_id.items():
        parent_id = row.get("parent_attempt_id")
        sequence = row.get("attempt_sequence")
        if parent_id is None:
            valid_parent = sequence == 1 and row.get("attempt_kind") == "fresh"
        else:
            parent = attempt_by_id.get(str(parent_id))
            valid_parent = (
                parent is not None
                and sequence == int(parent.get("attempt_sequence", -1)) + 1
                and row.get("attempt_kind") == "resume"
                and row.get("sample_index") == parent.get("sample_index")
                and row.get("group_index") == parent.get("group_index")
            )
        validation.check(
            "attempt_parent_chains_valid",
            valid_parent,
            {"attempt_id": attempt_id, "parent_attempt_id": parent_id},
        )

    committed_by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    committed_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        validation.check(
            "partition_outcome_record_types_valid",
            row.get("record_type") == "partition_outcome",
            {"source": row.get("_source"), "record_type": row.get("record_type")},
        )
        validation.check(
            "partition_outcomes_committed",
            row.get("outcome") == "committed",
            {"source": row.get("_source"), "outcome": row.get("outcome")},
        )
        token = row.get("attempt_token")
        if isinstance(token, int):
            committed_by_token[token].append(row)
            committed_by_key[(str(row.get("target_partition")), token)].append(row)
        validation.check(
            "commit_attempt_exists",
            token in request_by_token,
            {"source": row.get("_source"), "attempt_token": token},
        )
        validation.check(
            "commit_decision_exists",
            row.get("decision_id") in decision_by_id,
            {"source": row.get("_source"), "decision_id": row.get("decision_id")},
        )
        decision = decision_by_id.get(str(row.get("decision_id")))
        if decision is not None:
            validation.check(
                "commit_decision_physical_id_matches",
                row.get("physical_rollout_id") == decision.get("physical_rollout_id"),
                {
                    "source": row.get("_source"),
                    "commit_physical_rollout_id": row.get("physical_rollout_id"),
                    "decision_physical_rollout_id": decision.get("physical_rollout_id"),
                },
            )
        request_row = request_by_token.get(token)
        if request_row is not None:
            validation.check(
                "commit_matches_attempt_identity",
                row.get("attempt_id") in {request_row.get("attempt_id"), request_row.get("rid")}
                and row.get("attempt_decision_id") == request_row.get("admission_decision_id")
                and row.get("generation_physical_rollout_id") == request_row.get("physical_rollout_id")
                and row.get("sample_index") == request_row.get("sample_index")
                and row.get("group_index") == request_row.get("group_index"),
                {"source": row.get("_source"), "attempt_source": request_row.get("_source")},
            )

    partition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    consumed_by_key: Counter[tuple[str, int]] = Counter()
    for row in consumption_rows:
        partition_rows[str(row.get("target_partition"))].append(row)
        validation.check(
            "consume_record_types_valid",
            row.get("record_type") == "consume_outcome",
            {"source": row.get("_source"), "record_type": row.get("record_type")},
        )
        validation.check(
            "consume_outcomes_terminal",
            row.get("consume_outcome") == "consumed",
            {"source": row.get("_source"), "consume_outcome": row.get("consume_outcome")},
        )
        staleness = row.get("actual_staleness")
        validation.check(
            "actual_staleness_in_bounds",
            isinstance(staleness, int) and 0 <= staleness <= max_staleness,
            {"source": row.get("_source"), "actual_staleness": staleness},
        )
        token = row.get("attempt_token")
        if isinstance(token, int):
            consumed_by_key[(str(row.get("target_partition")), token)] += 1
        matching_commits = [
            outcome
            for outcome in committed_by_token.get(token, [])
            if outcome.get("target_partition") == row.get("target_partition")
        ]
        validation.check(
            "consume_has_exactly_one_commit",
            len(matching_commits) == 1,
            {"source": row.get("_source"), "matching_commits": len(matching_commits)},
        )
        validation.check(
            "consume_attempt_exists",
            token in request_by_token,
            {"source": row.get("_source"), "attempt_token": token},
        )
        validation.check(
            "consume_decision_exists",
            row.get("decision_id") in decision_by_id,
            {"source": row.get("_source"), "decision_id": row.get("decision_id")},
        )
        decision = decision_by_id.get(str(row.get("decision_id")))
        if decision is not None:
            validation.check(
                "consume_decision_physical_id_matches",
                row.get("decision_physical_rollout_id") == decision.get("physical_rollout_id"),
                {
                    "source": row.get("_source"),
                    "consume_decision_physical_rollout_id": row.get("decision_physical_rollout_id"),
                    "decision_physical_rollout_id": decision.get("physical_rollout_id"),
                },
            )
        generation_end = row.get("generation_end_version")
        consume_version = row.get("consume_version")
        validation.check(
            "actual_staleness_matches_versions",
            isinstance(generation_end, int)
            and isinstance(consume_version, int)
            and row.get("actual_staleness") == consume_version - generation_end,
            {
                "source": row.get("_source"),
                "generation_end_version": generation_end,
                "consume_version": consume_version,
                "actual_staleness": row.get("actual_staleness"),
            },
        )
        if len(matching_commits) == 1:
            validation.check(
                "commit_and_consume_decisions_match",
                matching_commits[0].get("decision_id") == row.get("decision_id"),
                {
                    "source": row.get("_source"),
                    "commit_decision_id": matching_commits[0].get("decision_id"),
                    "consume_decision_id": row.get("decision_id"),
                },
            )
            validation.check(
                "commit_and_consume_identity_match",
                matching_commits[0].get("sample_index") == row.get("sample_index")
                and matching_commits[0].get("group_index") == row.get("group_index")
                and matching_commits[0].get("generation_physical_rollout_id")
                == row.get("generation_physical_rollout_id"),
                {"source": row.get("_source"), "commit_source": matching_commits[0].get("_source")},
            )

    for key, commits in committed_by_key.items():
        validation.check(
            "commit_has_exactly_one_consume",
            len(commits) == 1 and consumed_by_key[key] == 1,
            {
                "target_partition": key[0],
                "attempt_token": key[1],
                "commit_count": len(commits),
                "consume_count": consumed_by_key[key],
            },
        )

    commit_token_counts = {token: len(rows) for token, rows in committed_by_token.items()}
    consume_token_counts = Counter(
        row.get("attempt_token") for row in consumption_rows if isinstance(row.get("attempt_token"), int)
    )
    validation.check(
        "attempt_tokens_globally_unique_in_commits",
        bool(commit_token_counts) and all(count == 1 for count in commit_token_counts.values()),
        {"duplicates": sorted(token for token, count in commit_token_counts.items() if count != 1)},
    )
    validation.check(
        "attempt_tokens_globally_unique_in_consumes",
        bool(consume_token_counts) and all(count == 1 for count in consume_token_counts.values()),
        {"duplicates": sorted(token for token, count in consume_token_counts.items() if count != 1)},
    )
    validation.check(
        "committed_and_consumed_attempt_tokens_match",
        set(commit_token_counts) == set(consume_token_counts),
        {
            "commit_only": sorted(set(commit_token_counts) - set(consume_token_counts)),
            "consume_only": sorted(set(consume_token_counts) - set(commit_token_counts)),
        },
    )

    validation.check(
        "expected_partitions_consumed_exact",
        set(partition_rows) == expected_partitions,
        {
            "missing": sorted(expected_partitions - set(partition_rows)),
            "extra": sorted(set(partition_rows) - expected_partitions),
        },
    )
    for partition in sorted(expected_partitions):
        rows = partition_rows.get(partition, [])
        validation.check(
            "partition_sample_count_exact",
            len(rows) == expected_samples_per_partition,
            {"partition": partition, "count": len(rows), "expected": expected_samples_per_partition},
        )
        sample_keys = [(row.get("group_index"), row.get("sample_index")) for row in rows]
        validation.check(
            "partition_samples_unique",
            len(sample_keys) == len(set(sample_keys)),
            {"partition": partition},
        )

    event_by_phase_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        phase = row.get("phase")
        key = row.get("sync_id") if phase != "abort" else row.get("rollout_id")
        event_by_phase_key[(str(phase), str(key))].append(row)
        validation.check(
            "timeline_event_phases_valid",
            phase in EVENT_PHASES,
            {"line": row.get("_line"), "phase": phase},
        )
        validation.check("timeline_event_intervals_valid", _validate_event_interval(row), row)

    sync_ids_by_phase = {
        phase: {
            str(row["sync_id"])
            for row in event_rows
            if row.get("phase") == phase and row.get("sync_id") is not None
        }
        for phase in SYNC_EVENT_PHASES
    }
    gate_sync_ids = sync_ids_by_phase["gate"]
    bootstrap_sync_ids = (
        set.intersection(*(sync_ids_by_phase[phase] for phase in ("pause", "flush", "transfer", "continue")))
        - gate_sync_ids
    )
    expected_non_gate_sync_ids = gate_sync_ids | bootstrap_sync_ids
    validation.check(
        "unique_bootstrap_sync_cycle",
        len(bootstrap_sync_ids) == 1
        and all(
            sync_ids_by_phase[phase] == expected_non_gate_sync_ids
            for phase in ("pause", "flush", "transfer", "continue")
        ),
        {phase: sorted(ids) for phase, ids in sync_ids_by_phase.items()},
    )
    bootstrap_rows_for_order = [
        row
        for row in event_rows
        if row.get("sync_id") in bootstrap_sync_ids
        and row.get("phase") in {"pause", "flush", "transfer", "continue"}
    ]
    runtime_rows_for_order = [
        row
        for row in event_rows
        if row.get("sync_id") in gate_sync_ids and row.get("phase") in SYNC_EVENT_PHASES
    ]
    bootstrap_ends = [_finite_field(row, "t_end") for row in bootstrap_rows_for_order]
    runtime_begins = [_finite_field(row, "t_begin") for row in runtime_rows_for_order]
    validation.check(
        "bootstrap_sync_precedes_runtime",
        len(bootstrap_sync_ids) == 1
        and bool(bootstrap_ends)
        and bool(runtime_begins)
        and all(value is not None for value in bootstrap_ends + runtime_begins)
        and max(value for value in bootstrap_ends if value is not None)
        <= min(value for value in runtime_begins if value is not None) + 1e-3,
        {
            "bootstrap_sync_ids": sorted(bootstrap_sync_ids),
            "bootstrap_ends": bootstrap_ends,
            "runtime_begins": runtime_begins,
        },
    )
    validation.check(
        "runtime_keyed_sync_id_sets_match",
        all(
            sync_ids_by_phase[phase] - bootstrap_sync_ids == gate_sync_ids
            for phase in SYNC_EVENT_PHASES
        )
        and gate_sync_ids.isdisjoint(bootstrap_sync_ids),
        {
            "runtime": sorted(gate_sync_ids),
            "bootstrap": sorted(bootstrap_sync_ids),
        },
    )
    validation.check(
        "gate_cycle_count_exact",
        len(gate_sync_ids) == expected_rollouts,
        {"count": len(gate_sync_ids), "expected": expected_rollouts},
    )
    for sync_id in sorted(gate_sync_ids):
        cycle_rows = []
        for phase in ("gate", "pause", "flush", "transfer", "continue"):
            rows = event_by_phase_key[(phase, str(sync_id))]
            validation.check(
                "keyed_sync_cycles_complete",
                len(rows) == 1,
                {"sync_id": sync_id, "phase": phase, "count": len(rows)},
            )
            if len(rows) == 1:
                cycle_rows.append(rows[0])
        if len(cycle_rows) == 5:
            intervals = [
                (_finite_field(row, "t_begin"), _finite_field(row, "t_end")) for row in cycle_rows
            ]
            validation.check(
                "keyed_sync_phase_order_valid",
                all(
                    begin is not None
                    and end is not None
                    and (index == 0 or intervals[index - 1][1] <= begin + 1e-3)
                    for index, (begin, end) in enumerate(intervals)
                ),
                {"sync_id": sync_id, "intervals": intervals},
            )
    for sync_id in sorted(bootstrap_sync_ids):
        bootstrap_rows = [
            event_by_phase_key[(phase, str(sync_id))][0]
            for phase in ("pause", "flush", "transfer", "continue")
            if len(event_by_phase_key[(phase, str(sync_id))]) == 1
        ]
        validation.check(
            "bootstrap_sync_cycle_complete",
            len(bootstrap_rows) == 4,
            {"sync_id": sync_id},
        )
        if len(bootstrap_rows) == 4:
            intervals = [
                (_finite_field(row, "t_begin"), _finite_field(row, "t_end")) for row in bootstrap_rows
            ]
            validation.check(
                "bootstrap_sync_phase_order_valid",
                all(
                    begin is not None
                    and end is not None
                    and (index == 0 or intervals[index - 1][1] <= begin + 1e-3)
                    for index, (begin, end) in enumerate(intervals)
                ),
                {"sync_id": sync_id, "intervals": intervals},
            )

    for rollout_id in sorted(expected_physical):
        validation.check(
            "physical_abort_events_complete",
            len(event_by_phase_key[("abort", str(rollout_id))]) == 1,
            {"rollout_id": rollout_id},
        )

    flow_by_phase_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in flow_rows:
        phase = str(row.get("phase"))
        validation.check(
            "timeline_flow_phases_valid",
            phase in FLOW_PHASES,
            {"line": row.get("_line"), "phase": phase},
        )
        if phase.startswith("gate_"):
            key = str(row.get("sync_id"))
        elif phase in {"physical_start", "physical_end"}:
            key = str(row.get("physical_rollout_id"))
        elif phase == "partition_close":
            key = str(row.get("target_partition"))
        else:
            key = "unknown"
        flow_by_phase_key[(phase, key)].append(row)

    gate_blocked_sync_ids = {
        str(row["sync_id"])
        for row in flow_rows
        if row.get("phase") == "gate_blocked" and row.get("sync_id") is not None
    }
    gate_ready_sync_ids = {
        str(row["sync_id"])
        for row in flow_rows
        if row.get("phase") == "gate_ready" and row.get("sync_id") is not None
    }
    validation.check(
        "gate_flow_sync_ids_closed",
        gate_ready_sync_ids == gate_sync_ids and gate_blocked_sync_ids.issubset(gate_sync_ids),
        {
            "event_gate_sync_ids": sorted(gate_sync_ids),
            "ready_sync_ids": sorted(gate_ready_sync_ids),
            "blocked_sync_ids": sorted(gate_blocked_sync_ids),
        },
    )
    partition_close_ids = {
        str(row["target_partition"])
        for row in flow_rows
        if row.get("phase") == "partition_close" and row.get("target_partition") is not None
    }
    validation.check(
        "partition_close_partitions_exact",
        partition_close_ids == expected_partitions,
        {
            "missing": sorted(expected_partitions - partition_close_ids),
            "extra": sorted(partition_close_ids - expected_partitions),
        },
    )

    for sync_id in sorted(gate_sync_ids):
        blocked_rows = flow_by_phase_key[("gate_blocked", str(sync_id))]
        ready_rows = flow_by_phase_key[("gate_ready", str(sync_id))]
        validation.check(
            "gate_blocked_flow_not_duplicated",
            len(blocked_rows) <= 1,
            {"sync_id": sync_id, "count": len(blocked_rows)},
        )
        validation.check(
            "gate_ready_flow_complete",
            len(ready_rows) == 1,
            {"sync_id": sync_id, "count": len(ready_rows)},
        )
        if len(ready_rows) == 1:
            ready = ready_rows[0]
            candidates = str(ready.get("candidate_partitions", "")).split(",")
            ready_time = _finite_field(ready, "t")
            validation.check(
                "gate_ready_partition_is_candidate",
                ready.get("ready_partition") in candidates,
                ready,
            )
            gate_rows = event_by_phase_key[("gate", str(sync_id))]
            gate_begin = _finite_field(gate_rows[0], "t_begin") if len(gate_rows) == 1 else None
            gate_end = _finite_field(gate_rows[0], "t_end") if len(gate_rows) == 1 else None
            validation.check(
                "gate_flow_within_event_interval",
                ready_time is not None
                and gate_begin is not None
                and gate_end is not None
                and gate_begin - 1e-3 <= ready_time <= gate_end + 1e-3,
                {
                    "sync_id": sync_id,
                    "gate_begin": gate_begin,
                    "ready_time": ready_time,
                    "gate_end": gate_end,
                },
            )
            if len(blocked_rows) == 1:
                blocked_time = _finite_field(blocked_rows[0], "t")
                validation.check(
                    "gate_blocked_precedes_ready",
                    blocked_time is not None
                    and ready_time is not None
                    and gate_begin is not None
                    and gate_begin - 1e-3 <= blocked_time <= ready_time,
                    {
                        "sync_id": sync_id,
                        "gate_begin": gate_begin,
                        "blocked_time": blocked_time,
                        "ready_time": ready_time,
                    },
                )

    physical_intervals: dict[int, tuple[float, float]] = {}
    for rollout_id in sorted(expected_physical):
        start_rows = flow_by_phase_key[("physical_start", str(rollout_id))]
        end_rows = flow_by_phase_key[("physical_end", str(rollout_id))]
        for phase, rows in (("physical_start", start_rows), ("physical_end", end_rows)):
            validation.check(
                "physical_flow_complete",
                len(rows) == 1,
                {"phase": phase, "rollout_id": rollout_id},
            )
        if len(start_rows) == 1 and len(end_rows) == 1:
            start_time = _finite_field(start_rows[0], "t")
            end_begin = _finite_field(end_rows[0], "t_begin")
            end_time = _finite_field(end_rows[0], "t_end")
            validation.check(
                "physical_flow_interval_valid",
                start_time is not None
                and end_begin is not None
                and end_time is not None
                and _validate_event_interval(end_rows[0])
                and math.isclose(start_time, end_begin, rel_tol=1e-4, abs_tol=1e-3),
                {
                    "rollout_id": rollout_id,
                    "start_time": start_time,
                    "end_begin": end_begin,
                    "end_time": end_time,
                },
            )
            if start_time is not None and end_time is not None:
                physical_intervals[rollout_id] = (start_time, end_time)
        abort_rows = event_by_phase_key[("abort", str(rollout_id))]
        if len(abort_rows) == 1 and rollout_id in physical_intervals:
            abort_begin = _finite_field(abort_rows[0], "t_begin")
            abort_end = _finite_field(abort_rows[0], "t_end")
            physical_begin, physical_end = physical_intervals[rollout_id]
            validation.check(
                "abort_within_physical_interval",
                abort_begin is not None
                and abort_end is not None
                and physical_begin - 1e-3 <= abort_begin <= abort_end <= physical_end + 1e-3,
                {
                    "rollout_id": rollout_id,
                    "physical_interval": [physical_begin, physical_end],
                    "abort_interval": [abort_begin, abort_end],
                },
            )
    for partition in sorted(expected_partitions):
        close_rows = flow_by_phase_key[("partition_close", partition)]
        validation.check(
            "partition_close_flow_complete",
            len(close_rows) == 1,
            {"partition": partition},
        )
        if len(close_rows) == 1:
            close_time = _finite_field(close_rows[0], "t")
            try:
                close_physical = int(close_rows[0]["physical_rollout_id"])
            except (KeyError, TypeError, ValueError):
                close_physical = None
            interval = physical_intervals.get(close_physical) if close_physical is not None else None
            validation.check(
                "partition_close_within_physical_interval",
                close_time is not None
                and interval is not None
                and interval[0] - 1e-3 <= close_time <= interval[1] + 1e-3,
                {
                    "partition": partition,
                    "physical_rollout_id": close_physical,
                    "close_time": close_time,
                    "physical_interval": interval,
                },
            )

    headline_steps = list(range(headline_lo, headline_hi + 1))
    rollout_metrics = _parse_metric_rows(driver_text, ROLLOUT_METRICS_RE)
    train_metrics = _parse_metric_rows(driver_text, TRAIN_METRICS_RE)
    perf_metrics = _parse_metric_rows(driver_text, PERF_METRICS_RE)
    quality_rows = []
    for step in headline_steps:
        validation.check(
            "headline_rollout_metrics_unique",
            len(rollout_metrics.get(step, [])) == 1,
            {"step": step, "count": len(rollout_metrics.get(step, []))},
        )
        validation.check(
            "headline_train_metrics_unique",
            len(train_metrics.get(step, [])) == 1,
            {"step": step, "count": len(train_metrics.get(step, []))},
        )
        validation.check(
            "headline_perf_metrics_unique",
            len(perf_metrics.get(step, [])) == 1,
            {"step": step, "count": len(perf_metrics.get(step, []))},
        )
        if (
            len(rollout_metrics.get(step, [])) != 1
            or len(train_metrics.get(step, [])) != 1
            or len(perf_metrics.get(step, [])) != 1
        ):
            continue
        rollout_row = rollout_metrics[step][0]
        train_row = train_metrics[step][0]
        perf_row = perf_metrics[step][0]
        missing_rollout = sorted(REQUIRED_ROLLOUT_METRICS - set(rollout_row))
        missing_train = sorted(REQUIRED_TRAIN_METRICS - set(train_row))
        validation.check(
            "headline_quality_fields_present",
            not missing_rollout and not missing_train,
            {"step": step, "missing_rollout": missing_rollout, "missing_train": missing_train},
        )
        quality_values = {
            key: value
            for key, value in {**rollout_row, **train_row}.items()
            if key in REQUIRED_ROLLOUT_METRICS | REQUIRED_TRAIN_METRICS
        }
        validation.check(
            "headline_quality_values_finite",
            all(_finite_number(value) for value in quality_values.values()),
            {"step": step, "values": quality_values},
        )
        step_time = perf_row.get("perf/step_time")
        validation.check(
            "headline_step_time_positive",
            _finite_number(step_time) and float(step_time) > 0,
            {"step": step, "perf/step_time": step_time},
        )
        quality_rows.append(
            {
                "step": step,
                **quality_values,
                "perf/step_time": step_time,
                "perf/samples_per_second": (
                    expected_samples_per_partition / float(step_time)
                    if _finite_number(step_time) and float(step_time) > 0
                    else None
                ),
            }
        )

    timeline_failures = []
    for step in headline_steps:
        path = timeline_dir / f"timeline_step_{step}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _valid_chrome_timeline(payload):
                raise ValueError("invalid Chrome complete-event timeline")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            timeline_failures.append({"step": step, "path": str(path), "error": str(exc)})
    validation.check("headline_timeline_files_complete", not timeline_failures, timeline_failures)

    try:
        gpu_snapshots, gpu_failures = _parse_gpu_snapshots(gpu_log, expected_engines)
    except OSError as exc:
        gpu_snapshots, gpu_failures = 0, [{"path": str(gpu_log), "error": str(exc)}]
    validation.check(
        "gpu_sampling_has_multiple_snapshots",
        gpu_snapshots >= 2,
        {"snapshot_count": gpu_snapshots},
    )
    validation.check("gpu_snapshots_parse_strictly", not gpu_failures, gpu_failures)

    quality_output = run_dir / "quality_metrics.json"
    try:
        quality_output.write_text(json.dumps(quality_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validation.check("quality_metrics_written", True)
    except OSError as exc:
        validation.check("quality_metrics_written", False, str(exc))

    counts = {
        "request_files": len(request_paths),
        "request_rows": len(request_rows),
        "admission_ledger_files": len(ledger_paths),
        "admission_decisions": len(decisions),
        "partition_outcome_files": len(outcome_paths),
        "partition_outcomes": len(outcome_rows),
        "consumption_files": len(consumption_paths),
        "consumption_rows": len(consumption_rows),
        "gate_cycles": len(gate_sync_ids),
        "event_rows": len(event_rows),
        "flow_rows": len(flow_rows),
        "headline_quality_rows": len(quality_rows),
    }
    return validation.result(
        counts=counts,
        request_analysis=request_analysis,
        quality_metrics=quality_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=("shadow", "on"), required=True)
    parser.add_argument("--expected-rollouts", type=int, default=15)
    parser.add_argument("--expected-samples-per-partition", type=int, default=64)
    parser.add_argument("--expected-engines", type=int, default=2)
    parser.add_argument("--max-staleness", type=int, default=2)
    parser.add_argument("--headline-lo", type=int, default=5)
    parser.add_argument("--headline-hi", type=int, default=14)
    parser.add_argument("--monitor-poll-interval", type=float, default=1.0)
    parser.add_argument("--monitor-evidence-grace", type=float, default=5.0)
    parser.add_argument("--monitor-no-progress-timeout", type=float, default=600.0)
    parser.add_argument("--gpu-max-snapshot-age", type=float, default=5.0)
    parser.add_argument("--gpu-max-snapshot-interval", type=float, default=2.0)
    parser.add_argument("--monitor-timeout", type=float, default=5700.0)
    parser.add_argument("--monitor-term-grace", type=float, default=1.0)
    parser.add_argument("--training-term-timeout", type=float, default=10.0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.expected_rollouts <= 0:
        parser.error("--expected-rollouts must be positive")
    if not (0 <= args.headline_lo <= args.headline_hi < args.expected_rollouts):
        parser.error("headline range must fall within expected rollouts")

    result = validate_run(
        args.run_dir,
        expected_mode=args.expected_mode,
        expected_rollouts=args.expected_rollouts,
        expected_samples_per_partition=args.expected_samples_per_partition,
        expected_engines=args.expected_engines,
        max_staleness=args.max_staleness,
        headline_lo=args.headline_lo,
        headline_hi=args.headline_hi,
        monitor_poll_interval=args.monitor_poll_interval,
        monitor_evidence_grace=args.monitor_evidence_grace,
        monitor_no_progress_timeout=args.monitor_no_progress_timeout,
        gpu_max_snapshot_age=args.gpu_max_snapshot_age,
        gpu_max_snapshot_interval=args.gpu_max_snapshot_interval,
        monitor_timeout=args.monitor_timeout,
        monitor_term_grace=args.monitor_term_grace,
        training_term_timeout=args.training_term_timeout,
        num_gpus=args.num_gpus,
        cuda_visible_devices=args.cuda_visible_devices,
        require_resume=args.require_resume,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
