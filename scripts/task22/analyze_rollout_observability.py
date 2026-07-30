#!/usr/bin/env python3
"""Validate and summarize rollout request-to-engine observability."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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
    scheduler_rows = Counter()
    scheduler_task_rids = Counter()
    scheduler_shape_errors = []
    forbidden_fields = []
    active_requests: dict[str, list[int]] = defaultdict(list)
    queued_requests: dict[str, list[int]] = defaultdict(list)
    running_context_tokens: dict[str, list[int]] = defaultdict(list)
    running_output_tokens: dict[str, list[int]] = defaultdict(list)

    for line_no, raw_line in enumerate(driver_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
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
        if event_name == "request.received" and isinstance(rid, str) and rid.startswith("relax:"):
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

    client_rids = {str(row["rid"]) for row in client_rows if row.get("rid")}
    resume_rids = {str(row["rid"]) for row in client_rows if row.get("attempt_kind") == "resume" and row.get("rid")}
    mapped_rids = client_rids.intersection(received_engines)
    mapped_resume_rids = resume_rids.intersection(received_engines)
    ambiguous_rids = sorted(rid for rid, engines in received_engines.items() if len(engines) != 1)
    rid_mismatches = [
        row["_source"]
        for row in client_rows
        if row.get("client_status") == "finished" and row.get("rid_match") is not True
    ]
    engine_pids = sorted(set(engine_to_gpu).intersection(scheduler_rows))
    unique_gpu_ids = sorted({engine_to_gpu[pid] for pid in engine_pids})

    request_rows_by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_rid = {str(row["rid"]): row for row in client_rows if row.get("rid")}
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
        "client_rows_contain_no_prompt_content": all(not _find_forbidden_paths(row) for row in client_rows),
        "client_rid_roundtrip": not rid_mismatches,
        "request_timing_metadata_complete": bool(finished_rows) and timing_metadata_complete,
        "request_mapping_complete": bool(client_rids) and mapped_rids == client_rids,
        "request_mapping_unambiguous": not ambiguous_rids,
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
            "ambiguous_rids": len(ambiguous_rids),
            "scheduler_shape_errors": len(scheduler_shape_errors),
            "forbidden_fields": len(forbidden_fields),
        },
        "engine_summaries": engine_summaries,
        "failures": {
            "rid_mismatches": rid_mismatches[:20],
            "ambiguous_rids": ambiguous_rids[:20],
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
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
