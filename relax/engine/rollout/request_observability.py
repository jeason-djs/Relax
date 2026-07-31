# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Low-overhead request lifecycle records for rollout scheduling analysis."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from relax.utils.types import Sample


def request_observability_enabled(args: Any) -> bool:
    return bool(getattr(args, "rollout_request_observability_dir", None))


def _rid_component(value: Any) -> str:
    if value is None:
        return "na"
    return str(value).replace(":", "_").replace("/", "_")


def attempt_token_from_id(attempt_id: str) -> int:
    """Return the numeric token carried through TransferQueue for one RID."""

    return int(attempt_id.rsplit(":", 1)[-1], 16)


def _response_meta_field(meta_info: dict[str, Any], key: str) -> Any:
    if key in meta_info:
        return meta_info.get(key)
    for nested_key in ("time_stats", "time_info", "time_cost", "request_time_stats"):
        nested = meta_info.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _flatten_int_values(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_int_values(item))
        return flattened
    if hasattr(value, "item"):
        value = value.item()
    return [int(value)]


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish a complete JSONL snapshot without exposing a partial file."""

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_consumption_records(
    rollout_data: dict[str, Any],
    *,
    rollout_id: int,
    consume_version: int,
) -> list[dict[str, Any]]:
    """Join transferred generation fields with the Actor consume version."""

    fields = {
        key: _flatten_int_values(rollout_data.get(key))
        for key in (
            "sample_indices",
            "group_indices",
            "abort_counts",
            "request_attempt_sequences",
            "request_attempt_tokens",
            "admission_decision_sequences",
            "admission_decision_physical_rollout_ids",
            "generation_physical_rollout_ids",
            "work_origin_codes",
            "generation_start_version",
            "generation_end_version",
            "generation_version_span",
        )
    }
    row_count = len(fields["generation_end_version"])
    if row_count == 0:
        return []
    mismatched = {key: len(values) for key, values in fields.items() if len(values) != row_count}
    if mismatched:
        raise ValueError(f"consumption ledger field length mismatch: expected {row_count}, got {mismatched}")

    origin_names = {0: "fresh", 1: "old_debt", 2: "surplus"}
    rows = []
    for index in range(row_count):
        generation_end = fields["generation_end_version"][index]
        decision_physical = fields["admission_decision_physical_rollout_ids"][index]
        generation_physical = fields["generation_physical_rollout_ids"][index]
        decision_sequence = fields["admission_decision_sequences"][index]
        rows.append(
            {
                "schema_version": 2,
                "record_type": "consume_outcome",
                "rollout_id": int(rollout_id),
                "target_partition": f"train_{rollout_id}",
                "sample_index": fields["sample_indices"][index],
                "group_index": fields["group_indices"][index],
                "abort_count": fields["abort_counts"][index],
                "attempt_sequence": fields["request_attempt_sequences"][index],
                "attempt_token": fields["request_attempt_tokens"][index],
                "decision_sequence": decision_sequence,
                "decision_id": f"admission:{decision_physical}:{decision_sequence}",
                "decision_physical_rollout_id": decision_physical,
                "generation_physical_rollout_id": generation_physical,
                "work_origin": origin_names.get(fields["work_origin_codes"][index], "unknown"),
                "generation_start_version": fields["generation_start_version"][index],
                "generation_end_version": generation_end,
                "generation_version_span": fields["generation_version_span"][index],
                "consume_version": int(consume_version),
                "actual_staleness": int(consume_version) - generation_end if generation_end >= 0 else None,
                "consume_outcome": "consumed",
                "consume_abs": time.time(),
            }
        )
    return rows


def export_consumption_records(
    rows: list[dict[str, Any]],
    *,
    output_dir: str,
    rollout_id: int,
    rank: int,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"consumption_ledger_rollout_{rollout_id}_rank_{rank}.jsonl"
    _atomic_write_jsonl(output_path, rows)
    return output_path


def begin_request_trace(
    *,
    sample: Sample,
    payload: dict[str, Any],
    physical_rollout_id: int | None,
    admission_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], float]:
    """Attach a request id and create a prompt-content-free lifecycle row."""

    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    attempt_sequence = int(sample.metadata.get("_request_attempt_sequence", 0) or 0) + 1
    parent_attempt_id = sample.metadata.get("_last_request_attempt_id")
    attempt_kind = "resume" if int(getattr(sample, "response_length", 0) or 0) > 0 else "fresh"
    rid = ":".join(
        (
            "relax",
            f"p{_rid_component(physical_rollout_id)}",
            f"k{attempt_kind}",
            f"g{_rid_component(getattr(sample, 'group_index', None))}",
            f"s{_rid_component(getattr(sample, 'index', None))}",
            f"a{_rid_component(getattr(sample, 'abort_count', 0))}",
            uuid.uuid4().hex[:12],
        )
    )
    payload["rid"] = rid
    request_start_monotonic = monotonic()
    request_start_abs = time.time()
    row = {
        "schema_version": 2,
        "record_type": "attempt",
        "rid": rid,
        "attempt_id": rid,
        "parent_attempt_id": parent_attempt_id,
        "attempt_sequence": attempt_sequence,
        "physical_rollout_id": physical_rollout_id,
        "group_index": getattr(sample, "group_index", None),
        "sample_index": getattr(sample, "index", None),
        "abort_count": int(getattr(sample, "abort_count", 0) or 0),
        "attempt_kind": attempt_kind,
        "work_origin": sample.metadata.get("work_origin", "fresh"),
        "logical_prefix_tokens": len(payload.get("input_ids", [])),
        "partial_response_tokens": int(getattr(sample, "response_length", 0) or 0),
        "max_new_tokens": int(payload.get("sampling_params", {}).get("max_new_tokens", 0) or 0),
        "dispatch_abs": request_start_abs,
        "diff_realtime_monotonic": request_start_abs - request_start_monotonic,
        "client_status": "dispatched",
    }
    if admission_context:
        row.update({f"admission_{key}": value for key, value in admission_context.items()})
    sample.metadata["_generation_physical_rollout_id"] = physical_rollout_id
    sample.metadata["_attempt_admission_decision_id"] = row.get("admission_decision_id")
    sample.metadata["_request_attempt_sequence"] = attempt_sequence
    sample.metadata["_last_request_attempt_id"] = rid
    sample.metadata["_last_request_attempt_token"] = attempt_token_from_id(rid)
    sample.metadata["_active_request_trace"] = row
    return row, request_start_monotonic


def finish_request_trace(
    row: dict[str, Any],
    *,
    output: Any,
    request_start_monotonic: float,
) -> None:
    """Complete a lifecycle row from one SGLang response."""

    request_end_abs = time.time()
    request_wall = monotonic() - request_start_monotonic
    meta_info = output.get("meta_info", {}) if isinstance(output, dict) else {}
    placement = meta_info.get("relax_placement")
    if isinstance(placement, dict):
        row.update(
            {
                key: value
                for key, value in placement.items()
                if isinstance(key, str) and key.startswith("placement_")
            }
        )
    returned_rid = meta_info.get("id")
    prompt_tokens = int(meta_info.get("prompt_tokens", 0) or 0)
    cached_tokens = int(meta_info.get("cached_tokens", 0) or 0)
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if isinstance(output_token_logprobs, list):
        generated_tokens = len(output_token_logprobs)
    elif isinstance(output, dict):
        generated_tokens = len(output.get("output_ids", []))
    else:
        generated_tokens = 0
    forward_entry_time = _response_meta_field(meta_info, "forward_entry_time")
    prefill_finished_time = _response_meta_field(meta_info, "prefill_finished_time")
    queue_time = _response_meta_field(meta_info, "queue_time")
    if queue_time is None:
        forward = _as_float(forward_entry_time)
        wait_queue_entry = _as_float(_response_meta_field(meta_info, "wait_queue_entry_time"))
        if forward is not None and wait_queue_entry is not None:
            queue_time = forward - wait_queue_entry
    row.update(
        {
            "request_end_abs": request_end_abs,
            "request_wall": request_wall,
            "returned_rid": returned_rid,
            "rid_match": returned_rid == row["rid"],
            "call_prompt_tokens": prompt_tokens,
            "call_cached_tokens": cached_tokens,
            "call_new_prompt_tokens": max(prompt_tokens - cached_tokens, 0),
            "generated_tokens_this_attempt": generated_tokens,
            "finish_reason": meta_info.get("finish_reason"),
            "forward_entry_time": forward_entry_time,
            "prefill_finished_time": prefill_finished_time,
            "queue_time": queue_time,
            "dp_rank": _response_meta_field(meta_info, "dp_rank"),
            "worker_id": (_response_meta_field(meta_info, "worker_id") or _response_meta_field(meta_info, "worker")),
            "client_status": "finished",
        }
    )


def fail_request_trace(
    row: dict[str, Any],
    error: BaseException,
    *,
    client_status: str = "generation_exception",
) -> None:
    row.update(
        {
            "request_end_abs": time.time(),
            "client_status": client_status,
            "exception_type": type(error).__name__,
        }
    )


def abort_request_trace(row: dict[str, Any]) -> None:
    """Mark a request that returned only because the backend was aborted."""

    row.update(
        {
            "request_end_abs": time.time(),
            "client_status": "request_aborted",
        }
    )


def record_request_outcome(
    sample: Sample,
    outcome: str,
    *,
    target_partition: str | None = None,
) -> None:
    """Close the business outcome for the sample's latest request attempt."""

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return
    row = metadata.get("_active_request_trace")
    if not isinstance(row, dict):
        return
    row["outcome"] = outcome
    row["target_partition"] = target_partition
    row["outcome_abs"] = time.time()


def close_discarded_abort_outcomes(groups: list[list[Sample]]) -> None:
    """Close attempts discarded by a non-partial rollout abort."""

    for group in groups:
        for sample in group:
            record_request_outcome(sample, "aborted")


def export_partition_outcomes(
    samples: list[Sample],
    *,
    output_dir: str,
    physical_rollout_id: int | None,
    target_partition: str,
    outcome: str,
    error_type: str | None = None,
) -> Path | None:
    """Atomically extend partition outcomes after async_put is known."""

    rows = []
    outcome_abs = time.time()
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        attempt_id = metadata.get("_last_request_attempt_id")
        if attempt_id is None:
            continue
        generation_physical_rollout_id = metadata.get("_generation_physical_rollout_id")
        rows.append(
            {
                "schema_version": 2,
                "record_type": "partition_outcome",
                "attempt_id": attempt_id,
                "attempt_token": metadata.get("_last_request_attempt_token"),
                "attempt_sequence": metadata.get("_request_attempt_sequence"),
                "attempt_decision_id": metadata.get("_attempt_admission_decision_id"),
                "decision_id": metadata.get("_admission_decision_id"),
                "decision_sequence": metadata.get("_admission_decision_sequence"),
                "physical_rollout_id": metadata.get(
                    "_admission_decision_physical_rollout_id", physical_rollout_id
                ),
                "generation_physical_rollout_id": generation_physical_rollout_id,
                "target_partition": target_partition,
                "sample_index": getattr(sample, "index", None),
                "group_index": getattr(sample, "group_index", None),
                "abort_count": int(getattr(sample, "abort_count", 0) or 0),
                "work_origin": metadata.get("work_origin"),
                "outcome": outcome,
                "error_type": error_type,
                "outcome_abs": outcome_abs,
            }
        )
    if not rows:
        return None

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rollout_component = physical_rollout_id if physical_rollout_id is not None else "unknown"
    output_path = destination / f"admission_outcomes_rollout_{rollout_component}.jsonl"
    lock_path = destination / f".{output_path.name}.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            existing_rows = []
            if output_path.exists():
                with output_path.open(encoding="utf-8") as source:
                    existing_rows = [json.loads(line) for line in source if line.strip()]
            _atomic_write_jsonl(output_path, [*existing_rows, *rows])
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return output_path


def admission_decision_record(
    decision: Any,
    *,
    physical_rollout_id: int,
    logical_debt_groups: int,
) -> dict[str, Any]:
    """Build one prompt-free admission-decision ledger row."""

    return {
        "schema_version": 2,
        "record_type": "admission_decision",
        "decision_id": decision.decision_id,
        "decision_sequence": decision.decision_sequence,
        "physical_rollout_id": physical_rollout_id,
        "mode": decision.mode.value,
        "logical_debt_groups": logical_debt_groups,
        "release_remaining": decision.debt_remaining,
        "inflight_groups": decision.inflight_groups,
        "available_groups": decision.available_groups,
        "eager_admit_groups": decision.eager_admit_groups,
        "desired_inflight_groups": decision.desired_inflight_groups,
        "bounded_admit_groups": decision.bounded_admit_groups,
        "actual_admit_groups": decision.actual_admit_groups,
        "bypass_reason": decision.bypass_reason,
        "decision_abs": time.time(),
    }


def export_request_traces(
    rows: list[dict[str, Any]],
    *,
    output_dir: str,
    physical_rollout_id: int,
) -> Path:
    """Write one bounded JSONL file after a physical rollout finishes."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"request_lifecycle_rollout_{physical_rollout_id}.jsonl"
    _atomic_write_jsonl(output_path, rows)
    return output_path


def export_admission_ledger(
    decision_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    *,
    output_dir: str,
    physical_rollout_id: int,
) -> Path:
    """Write decision and attempt rows for one physical rollout."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"admission_ledger_rollout_{physical_rollout_id}.jsonl"
    _atomic_write_jsonl(output_path, [*decision_rows, *request_rows])
    return output_path
