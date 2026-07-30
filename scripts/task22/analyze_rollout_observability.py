#!/usr/bin/env python3
"""Validate and summarize rollout request-to-engine observability."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENGINE_PID_RE = re.compile(r"\(SGLangEngine pid=(\d+)\)")
BASE_GPU_RE = re.compile(r"\bbase_gpu_id=(\d+)")
FORBIDDEN_FIELDS = {
    "text",
    "prompt",
    "input_ids",
    "input_embeds",
    "image_data",
    "audio_data",
    "video_data",
    "sampling_params",
    "token_ids_logprob",
    "input_token_logprobs",
    "output_token_logprobs",
    "input_top_logprobs",
    "output_top_logprobs",
    "output_ids",
}
SERVER_ENVELOPE_TOLERANCE_S = 1.0


def _complete_driver_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if text and not text.endswith(("\n", "\r")):
        lines.pop()
    return lines


def _load_client_rows(request_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(request_dir.glob("request_lifecycle_rollout_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source"] = f"{path.name}:{line_no}"
            rows.append(row)
    return rows


def _extract_event(line: str) -> dict[str, Any] | None:
    clean = ANSI_RE.sub("", line)
    marker = clean.find("{")
    if marker < 0:
        return None
    try:
        event = json.loads(clean[marker:])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and "event" in event else None


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "p50": float(median(values)),
        "max": float(max(values)),
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_timestamp(event: dict[str, Any]) -> float | None:
    value = event.get("timestamp")
    numeric = _finite_float(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None


def _client_timestamp_error(row: dict[str, Any]) -> str | None:
    dispatch = _finite_float(row.get("dispatch_abs"))
    request_end = _finite_float(row.get("request_end_abs"))
    forward_entry = _finite_float(row.get("forward_entry_time"))
    prefill_finished = _finite_float(row.get("prefill_finished_time"))
    queue_time = _finite_float(row.get("queue_time"))
    if dispatch is None or request_end is None or dispatch > request_end:
        return "invalid request envelope"
    if forward_entry is None or prefill_finished is None or forward_entry > prefill_finished:
        return "invalid forward/prefill timestamps"
    if queue_time is None or queue_time < 0:
        return "invalid queue_time"
    return None


def _find_forbidden_paths(value: Any, *, path: str = "$") -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if key in FORBIDDEN_FIELDS:
                paths.append(nested_path)
            paths.extend(_find_forbidden_paths(nested, path=nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_find_forbidden_paths(nested, path=f"{path}[{index}]"))
    return paths


def _placement_trace_rows(
    client_rows: list[dict[str, Any]],
    received_engines: dict[str, set[str]],
    engine_pids: list[str],
    scheduler_snapshots: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    fallback_candidates = [f"engine-pid-{engine_pid}" for engine_pid in engine_pids]
    rows = []
    for row in client_rows:
        rid = str(row.get("rid") or "")
        if not rid or row.get("client_status") != "finished":
            continue
        mapped = received_engines.get(rid, set())
        actual_engine_id = row.get("placement_actual_engine_id")
        if actual_engine_id is None and len(mapped) == 1:
            actual_engine_id = f"engine-pid-{next(iter(mapped))}"

        candidate_snapshots = row.get("placement_candidate_engines")
        if isinstance(candidate_snapshots, list) and candidate_snapshots:
            candidate_engine_ids = [
                candidate.get("engine_id")
                for candidate in candidate_snapshots
                if isinstance(candidate, dict) and candidate.get("engine_id")
            ]
        else:
            candidate_snapshots = []
            dispatch = _finite_float(row.get("dispatch_abs"))
            if dispatch is not None:
                for engine_pid in engine_pids:
                    snapshots = scheduler_snapshots.get(engine_pid, [])
                    index = bisect_right(
                        [snapshot["snapshot_abs"] for snapshot in snapshots],
                        dispatch,
                    )
                    if index == 0:
                        continue
                    snapshot = dict(snapshots[index - 1])
                    snapshot["snapshot_age_s"] = dispatch - snapshot["snapshot_abs"]
                    candidate_snapshots.append(snapshot)
            candidate_engine_ids = (
                [snapshot["engine_id"] for snapshot in candidate_snapshots]
                if candidate_snapshots
                else fallback_candidates
            )

        rows.append(
            {
                "schema_version": 1,
                "record_type": "placement_replay_request",
                "rid": rid,
                "physical_rollout_id": row.get("physical_rollout_id"),
                "attempt_kind": row.get("attempt_kind"),
                "work_origin": row.get("work_origin", "unknown"),
                "dispatch_abs": row.get("dispatch_abs"),
                "request_end_abs": row.get("request_end_abs"),
                "observed_request_wall": row.get("request_wall"),
                "logical_prefix_tokens": int(row.get("logical_prefix_tokens", 0) or 0),
                "max_new_tokens": int(row.get("max_new_tokens", 0) or 0),
                "predicted_work": int(row.get("placement_request_predicted_work", 0) or 0)
                or int(row.get("logical_prefix_tokens", 0) or 0)
                + int(row.get("max_new_tokens", 0) or 0),
                "observed_work": int(row.get("call_new_prompt_tokens", 0) or 0)
                + int(row.get("generated_tokens_this_attempt", 0) or 0),
                "call_prompt_tokens": int(row.get("call_prompt_tokens", 0) or 0),
                "call_cached_tokens": int(row.get("call_cached_tokens", 0) or 0),
                "actual_engine_id": actual_engine_id,
                "candidate_engine_ids": candidate_engine_ids,
                "candidate_snapshots": candidate_snapshots,
                "placement_mode": row.get("placement_mode"),
                "placement_policy": row.get("placement_policy"),
                "placement_policy_version": row.get("placement_policy_version"),
                "placement_decision_id": row.get("placement_decision_id"),
                "placement_shadow_engine_id": row.get("placement_shadow_engine_id"),
                "placement_routing_reason": row.get("placement_routing_reason"),
                "placement_fallback_reason": row.get("placement_fallback_reason"),
            }
        )
    return rows


def export_placement_trace(driver_log: Path, request_dir: Path, output_path: Path) -> dict[str, Any]:
    """Export prompt-free request rows suitable for placement replay."""

    client_rows = _load_client_rows(request_dir)
    received_engines: dict[str, set[str]] = defaultdict(set)
    scheduler_snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    engine_pids = set()
    for raw_line in _complete_driver_lines(driver_log):
        clean = ANSI_RE.sub("", raw_line)
        pid_match = ENGINE_PID_RE.search(clean)
        if not pid_match:
            continue
        engine_pid = pid_match.group(1)
        engine_pids.add(engine_pid)
        event = _extract_event(raw_line)
        if event is None:
            continue
        if event.get("event") == "scheduler.status":
            snapshot_abs = _event_timestamp(event)
            running_rids = event.get("running_rids")
            running_seq_lens = event.get("running_seq_lens")
            running_output_lens = event.get("running_output_lens")
            queued_rids = event.get("queued_rids")
            queued_origin_input_lens = event.get("queued_origin_input_lens")
            queued_output_lens = event.get("queued_output_lens")
            if (
                snapshot_abs is not None
                and isinstance(running_rids, list)
                and isinstance(running_seq_lens, list)
                and isinstance(running_output_lens, list)
                and isinstance(queued_rids, list)
                and isinstance(queued_origin_input_lens, list)
                and isinstance(queued_output_lens, list)
            ):
                running_context = sum(int(value) for value in running_seq_lens)
                queued_context = sum(int(value) for value in queued_origin_input_lens) + sum(
                    int(value) for value in queued_output_lens
                )
                scheduler_snapshots[engine_pid].append(
                    {
                        "engine_id": f"engine-pid-{engine_pid}",
                        "active_requests": len(running_rids),
                        "queued_requests": len(queued_rids),
                        "running_context_tokens": running_context,
                        "running_output_tokens": sum(int(value) for value in running_output_lens),
                        "predicted_work": running_context + queued_context,
                        "snapshot_abs": snapshot_abs,
                        "snapshot_source": "server_scheduler",
                    }
                )
            continue
        if event.get("event") != "request.received":
            continue
        rid = event.get("rid")
        if isinstance(rid, str) and rid.startswith("relax:"):
            received_engines[rid].add(engine_pid)

    for snapshots in scheduler_snapshots.values():
        snapshots.sort(key=lambda snapshot: snapshot["snapshot_abs"])
    rows = _placement_trace_rows(
        client_rows,
        received_engines,
        sorted(engine_pids),
        scheduler_snapshots,
    )
    snapshot_ages = [
        float(snapshot["snapshot_age_s"])
        for row in rows
        for snapshot in row["candidate_snapshots"]
        if snapshot.get("snapshot_age_s") is not None
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "rows": len(rows),
        "physical_rollouts": len({row["physical_rollout_id"] for row in rows}),
        "rows_with_candidate_snapshots": sum(bool(row["candidate_snapshots"]) for row in rows),
        "scheduler_snapshot_age_s": {
            "min": min(snapshot_ages) if snapshot_ages else None,
            "p50": median(snapshot_ages) if snapshot_ages else None,
            "max": max(snapshot_ages) if snapshot_ages else None,
        },
        "engine_ids": sorted(
            {
                engine_id
                for row in rows
                for engine_id in row["candidate_engine_ids"]
                if isinstance(engine_id, str)
            }
        ),
        "output_path": str(output_path),
    }


def analyze(
    driver_log: Path,
    request_dir: Path,
    *,
    expected_engines: int,
    require_resume: bool,
) -> dict[str, Any]:
    client_rows = _load_client_rows(request_dir)
    engine_to_gpu: dict[str, int] = {}
    received_engines: dict[str, set[str]] = defaultdict(set)
    server_events: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    server_request_rids: set[str] = set()
    scheduler_rows = Counter()
    scheduler_task_rids = Counter()
    scheduler_shape_errors = []
    forbidden_fields = []
    active_requests: dict[str, list[int]] = defaultdict(list)
    queued_requests: dict[str, list[int]] = defaultdict(list)
    running_context_tokens: dict[str, list[int]] = defaultdict(list)
    running_output_tokens: dict[str, list[int]] = defaultdict(list)

    for line_no, raw_line in enumerate(_complete_driver_lines(driver_log), 1):
        clean = ANSI_RE.sub("", raw_line)
        pid_match = ENGINE_PID_RE.search(clean)
        if not pid_match:
            continue
        engine_pid = pid_match.group(1)
        if "server_args=ServerArgs(" in clean:
            gpu_match = BASE_GPU_RE.search(clean)
            if gpu_match:
                engine_to_gpu[engine_pid] = int(gpu_match.group(1))

        event = _extract_event(raw_line)
        if event is None:
            continue
        event_name = event.get("event")
        rid = event.get("rid")
        if isinstance(event_name, str) and event_name.startswith("request."):
            leaked_paths = sorted(set(_find_forbidden_paths(event)))
            if leaked_paths:
                forbidden_fields.append({"line": line_no, "rid": rid, "paths": leaked_paths})
            if isinstance(rid, str) and rid.startswith("relax:"):
                server_request_rids.add(rid)
                if event_name in {"request.received", "request.finished"}:
                    server_events[rid][event_name].append(
                        {
                            "engine_pid": engine_pid,
                            "line": line_no,
                            "timestamp": _event_timestamp(event),
                        }
                    )
                    if event_name == "request.received":
                        received_engines[rid].add(engine_pid)
        elif event_name == "scheduler.status":
            scheduler_rows[engine_pid] += 1
            running_rids = event.get("running_rids")
            running_seq_lens = event.get("running_seq_lens")
            running_origin_input_lens = event.get("running_origin_input_lens")
            running_output_lens = event.get("running_output_lens")
            queued_rids = event.get("queued_rids")
            queued_origin_input_lens = event.get("queued_origin_input_lens")
            queued_output_lens = event.get("queued_output_lens")
            running_lengths = {
                len(value)
                for value in (running_rids, running_seq_lens, running_origin_input_lens, running_output_lens)
                if isinstance(value, list)
            }
            queued_lengths = {
                len(value)
                for value in (queued_rids, queued_origin_input_lens, queued_output_lens)
                if isinstance(value, list)
            }
            if not isinstance(event.get("forward_mode"), str) or len(running_lengths) != 1 or len(queued_lengths) != 1:
                scheduler_shape_errors.append({"line": line_no, "engine_pid": engine_pid})
                continue
            active_requests[engine_pid].append(len(running_rids))
            queued_requests[engine_pid].append(len(queued_rids))
            running_context_tokens[engine_pid].append(sum(int(value) for value in running_seq_lens))
            running_output_tokens[engine_pid].append(sum(int(value) for value in running_output_lens))
            for scheduler_rid in running_rids + queued_rids:
                if isinstance(scheduler_rid, str) and scheduler_rid.startswith("relax:"):
                    scheduler_task_rids[engine_pid] += 1

    client_rid_counts = Counter(str(row["rid"]) for row in client_rows if row.get("rid"))
    client_rids = set(client_rid_counts)
    duplicate_client_rids = sorted(rid for rid, count in client_rid_counts.items() if count != 1)
    resume_rids = {str(row["rid"]) for row in client_rows if row.get("attempt_kind") == "resume" and row.get("rid")}
    mapped_rids = client_rids.intersection(received_engines)
    mapped_resume_rids = resume_rids.intersection(received_engines)
    ambiguous_rids = sorted(rid for rid, engines in received_engines.items() if len(engines) != 1)
    nonterminal_rows = [
        {"source": row["_source"], "rid": row.get("rid"), "client_status": row.get("client_status")}
        for row in client_rows
        if row.get("client_status") != "finished"
    ]
    invalid_client_timestamps = [
        {"source": row["_source"], "rid": row.get("rid"), "error": error}
        for row in client_rows
        if (error := _client_timestamp_error(row)) is not None
    ]
    rid_mismatches = [
        row["_source"]
        for row in client_rows
        if row.get("client_status") == "finished" and row.get("rid_match") is not True
    ]
    engine_pids = sorted(set(engine_to_gpu).intersection(scheduler_rows))
    unique_gpu_ids = sorted({engine_to_gpu[pid] for pid in engine_pids})

    server_lifecycle_errors = []
    server_interval_errors = []
    rows_by_rid = {
        str(row["rid"]): row
        for row in client_rows
        if row.get("rid") and client_rid_counts[str(row["rid"])] == 1
    }
    for rid in sorted(client_rids | server_request_rids):
        lifecycle = server_events.get(rid, {})
        received = lifecycle.get("request.received", [])
        finished = lifecycle.get("request.finished", [])
        if len(received) != 1 or len(finished) != 1:
            server_lifecycle_errors.append(
                {
                    "rid": rid,
                    "received_count": len(received),
                    "finished_count": len(finished),
                }
            )
            continue
        if received[0]["engine_pid"] != finished[0]["engine_pid"]:
            server_lifecycle_errors.append(
                {
                    "rid": rid,
                    "received_engine": received[0]["engine_pid"],
                    "finished_engine": finished[0]["engine_pid"],
                }
            )
            continue

        row = rows_by_rid.get(rid)
        server_begin = received[0]["timestamp"]
        server_end = finished[0]["timestamp"]
        if row is None or server_begin is None or server_end is None:
            server_interval_errors.append({"rid": rid, "error": "missing comparable timestamps"})
            continue
        client_begin = _finite_float(row.get("dispatch_abs"))
        client_end = _finite_float(row.get("request_end_abs"))
        if (
            client_begin is None
            or client_end is None
            or server_begin > server_end
            or server_begin < client_begin - SERVER_ENVELOPE_TOLERANCE_S
            or server_end > client_end + SERVER_ENVELOPE_TOLERANCE_S
        ):
            server_interval_errors.append(
                {
                    "rid": rid,
                    "client_begin": client_begin,
                    "client_end": client_end,
                    "server_begin": server_begin,
                    "server_end": server_end,
                }
            )

    request_rows_by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rid, engines in received_engines.items():
        if len(engines) == 1 and rid in rows_by_rid:
            request_rows_by_engine[next(iter(engines))].append(rows_by_rid[rid])

    engine_summaries = {}
    for engine_pid in engine_pids:
        rows = request_rows_by_engine[engine_pid]
        generated_tokens = sum(int(row.get("generated_tokens_this_attempt", 0) or 0) for row in rows)
        prompt_tokens = sum(int(row.get("call_prompt_tokens", 0) or 0) for row in rows)
        cached_tokens = sum(int(row.get("call_cached_tokens", 0) or 0) for row in rows)
        resume_new_prompt_tokens = sum(
            int(row.get("call_new_prompt_tokens", 0) or 0) for row in rows if row.get("attempt_kind") == "resume"
        )
        starts = [float(row["dispatch_abs"]) for row in rows if row.get("dispatch_abs") is not None]
        ends = [float(row["request_end_abs"]) for row in rows if row.get("request_end_abs") is not None]
        cohort_wall = max(ends) - min(starts) if starts and ends else 0.0
        engine_summaries[engine_pid] = {
            "gpu_id": engine_to_gpu[engine_pid],
            "request_rows": len(rows),
            "resume_rows": sum(row.get("attempt_kind") == "resume" for row in rows),
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_token_hit_ratio": cached_tokens / prompt_tokens if prompt_tokens > 0 else None,
            "resume_new_prompt_tokens": resume_new_prompt_tokens,
            "cohort_generated_tokens_per_wall_second": (generated_tokens / cohort_wall if cohort_wall > 0 else None),
            "active_requests": _summary(active_requests[engine_pid]),
            "queued_requests": _summary(queued_requests[engine_pid]),
            "running_context_tokens": _summary(running_context_tokens[engine_pid]),
            "running_output_tokens": _summary(running_output_tokens[engine_pid]),
        }

    finished_rows = [row for row in client_rows if row.get("client_status") == "finished"]
    timing_metadata_complete = all(
        row.get("forward_entry_time") is not None
        and row.get("prefill_finished_time") is not None
        and row.get("queue_time") is not None
        for row in finished_rows
    )
    checks = {
        "client_rows_present": bool(client_rows),
        "finished_client_rows_present": bool(finished_rows),
        "client_rids_unique": not duplicate_client_rids,
        "client_rows_terminal": not nonterminal_rows,
        "client_timestamps_valid": not invalid_client_timestamps,
        "client_rows_contain_no_prompt_content": all(not _find_forbidden_paths(row) for row in client_rows),
        "client_rid_roundtrip": not rid_mismatches,
        "request_timing_metadata_complete": bool(finished_rows) and timing_metadata_complete,
        "request_mapping_complete": bool(client_rids) and mapped_rids == client_rids,
        "request_mapping_unambiguous": not ambiguous_rids,
        "server_rids_match_client_rids": bool(client_rids) and server_request_rids == client_rids,
        "server_lifecycle_complete": not server_lifecycle_errors,
        "server_intervals_within_client_envelopes": not server_interval_errors,
        "expected_engine_count": len(engine_pids) == expected_engines,
        "unique_engine_gpu_mapping": len(unique_gpu_ids) == expected_engines,
        "scheduler_shape_rows_present": all(scheduler_rows[pid] > 0 for pid in engine_pids),
        "scheduler_shape_schema_valid": not scheduler_shape_errors,
        "scheduler_contains_relax_rids": all(scheduler_task_rids[pid] > 0 for pid in engine_pids),
        "server_request_log_has_no_large_fields": not forbidden_fields,
        "resume_mapping_present": (not require_resume or (bool(resume_rids) and mapped_resume_rids == resume_rids)),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "client_rows": len(client_rows),
            "client_rids": len(client_rids),
            "mapped_rids": len(mapped_rids),
            "resume_rids": len(resume_rids),
            "mapped_resume_rids": len(mapped_resume_rids),
            "rid_mismatches": len(rid_mismatches),
            "duplicate_client_rids": len(duplicate_client_rids),
            "nonterminal_rows": len(nonterminal_rows),
            "invalid_client_timestamps": len(invalid_client_timestamps),
            "ambiguous_rids": len(ambiguous_rids),
            "server_request_rids": len(server_request_rids),
            "server_lifecycle_errors": len(server_lifecycle_errors),
            "server_interval_errors": len(server_interval_errors),
            "scheduler_shape_errors": len(scheduler_shape_errors),
            "forbidden_fields": len(forbidden_fields),
        },
        "engine_summaries": engine_summaries,
        "failures": {
            "rid_mismatches": rid_mismatches[:20],
            "duplicate_client_rids": duplicate_client_rids[:20],
            "nonterminal_rows": nonterminal_rows[:20],
            "invalid_client_timestamps": invalid_client_timestamps[:20],
            "ambiguous_rids": ambiguous_rids[:20],
            "server_only_rids": sorted(server_request_rids - client_rids)[:20],
            "client_only_rids": sorted(client_rids - server_request_rids)[:20],
            "server_lifecycle_errors": server_lifecycle_errors[:20],
            "server_interval_errors": server_interval_errors[:20],
            "scheduler_shape_errors": scheduler_shape_errors[:20],
            "forbidden_fields": forbidden_fields[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--expected-engines", type=int, default=2)
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--placement-trace-jsonl", type=Path)
    args = parser.parse_args()

    result = analyze(
        args.driver_log,
        args.request_dir,
        expected_engines=args.expected_engines,
        require_resume=args.require_resume,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.placement_trace_jsonl:
        summary = export_placement_trace(args.driver_log, args.request_dir, args.placement_trace_jsonl)
        print(json.dumps({"placement_trace": summary}, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
