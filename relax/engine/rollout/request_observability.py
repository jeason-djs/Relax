# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Low-overhead request lifecycle records for rollout scheduling analysis."""

from __future__ import annotations

import json
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


def begin_request_trace(
    *,
    sample: Sample,
    payload: dict[str, Any],
    physical_rollout_id: int | None,
    admission_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], float]:
    """Attach a request id and create a prompt-content-free lifecycle row."""

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
        "rid": rid,
        "physical_rollout_id": physical_rollout_id,
        "group_index": getattr(sample, "group_index", None),
        "sample_index": getattr(sample, "index", None),
        "abort_count": int(getattr(sample, "abort_count", 0) or 0),
        "attempt_kind": attempt_kind,
        "logical_prefix_tokens": len(payload.get("input_ids", [])),
        "partial_response_tokens": int(getattr(sample, "response_length", 0) or 0),
        "max_new_tokens": int(payload.get("sampling_params", {}).get("max_new_tokens", 0) or 0),
        "dispatch_abs": request_start_abs,
        "diff_realtime_monotonic": request_start_abs - request_start_monotonic,
        "client_status": "dispatched",
    }
    if admission_context:
        row.update({f"admission_{key}": value for key, value in admission_context.items()})
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


def fail_request_trace(row: dict[str, Any], error: BaseException) -> None:
    row.update(
        {
            "request_end_abs": time.time(),
            "client_status": "exception",
            "exception_type": type(error).__name__,
        }
    )


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
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return output_path
