#!/usr/bin/env python3
"""Validate Task 22 request-to-engine and scheduler-shape observability."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENGINE_PID_RE = re.compile(r"\(SGLangEngine pid=(\d+)\)")
BASE_GPU_RE = re.compile(r"\bbase_gpu_id=(\d+)")
FORBIDDEN_LEVEL0_FIELDS = {
    "text",
    "input_ids",
    "input_embeds",
    "image_data",
    "audio_data",
    "video_data",
    "sampling_params",
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
        return event if isinstance(event, dict) and "event" in event else None
    except json.JSONDecodeError:
        return None


def analyze(driver_log: Path, request_dir: Path, expected_engines: int, require_resume: bool) -> dict[str, Any]:
    client_rows = _load_client_rows(request_dir)
    engine_to_gpu: dict[str, int] = {}
    received_engines: dict[str, set[str]] = defaultdict(set)
    finished_engines: dict[str, set[str]] = defaultdict(set)
    scheduler_rows = Counter()
    scheduler_task22_rids = Counter()
    scheduler_shape_errors = []
    forbidden_request_fields = []

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
        if event_name == "request.received" and isinstance(rid, str) and rid.startswith("task22:"):
            received_engines[rid].add(engine_pid)
            obj = event.get("obj")
            if isinstance(obj, dict):
                leaked = sorted(FORBIDDEN_LEVEL0_FIELDS.intersection(obj))
                if leaked:
                    forbidden_request_fields.append({"line": line_no, "rid": rid, "fields": leaked})
        elif event_name == "request.finished" and isinstance(rid, str) and rid.startswith("task22:"):
            finished_engines[rid].add(engine_pid)
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
            if (
                not isinstance(event.get("forward_mode"), str)
                or len(running_lengths) != 1
                or len(queued_lengths) != 1
            ):
                scheduler_shape_errors.append({"line": line_no, "engine_pid": engine_pid})
            for scheduler_rid in (running_rids or []) + (queued_rids or []):
                if isinstance(scheduler_rid, str) and scheduler_rid.startswith("task22:"):
                    scheduler_task22_rids[engine_pid] += 1

    client_rids = {str(row["rid"]) for row in client_rows if row.get("rid")}
    resumes = {str(row["rid"]) for row in client_rows if row.get("work_class") == "old_debt" and row.get("rid")}
    mapped_rids = client_rids.intersection(received_engines)
    mapped_resumes = resumes.intersection(received_engines)
    ambiguous_rids = sorted(rid for rid, engines in received_engines.items() if len(engines) != 1)
    rid_mismatches = [
        row["_source"]
        for row in client_rows
        if row.get("client_status") == "finished" and not row.get("rid_match")
    ]
    engine_pids = sorted(set(engine_to_gpu).intersection(scheduler_rows))
    unique_gpu_ids = sorted({engine_to_gpu[pid] for pid in engine_pids})

    checks = {
        "client_rows_present": bool(client_rows),
        "client_rid_roundtrip": not rid_mismatches,
        "request_mapping_complete": bool(client_rids) and mapped_rids == client_rids,
        "request_mapping_unambiguous": not ambiguous_rids,
        "expected_engine_count": len(engine_pids) == expected_engines,
        "unique_engine_gpu_mapping": len(unique_gpu_ids) == expected_engines,
        "scheduler_shape_rows_present": all(scheduler_rows[pid] > 0 for pid in engine_pids),
        "scheduler_shape_schema_valid": not scheduler_shape_errors,
        "scheduler_contains_task22_rids": all(scheduler_task22_rids[pid] > 0 for pid in engine_pids),
        "request_level_zero_has_no_large_fields": not forbidden_request_fields,
        "resume_mapping_present": not require_resume or (bool(resumes) and mapped_resumes == resumes),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {
        "verdict": verdict,
        "checks": checks,
        "counts": {
            "client_rows": len(client_rows),
            "client_rids": len(client_rids),
            "mapped_rids": len(mapped_rids),
            "resume_rids": len(resumes),
            "mapped_resume_rids": len(mapped_resumes),
            "request_received_rids": len(received_engines),
            "request_finished_rids": len(finished_engines),
            "rid_mismatches": len(rid_mismatches),
            "ambiguous_rids": len(ambiguous_rids),
            "scheduler_shape_errors": len(scheduler_shape_errors),
            "forbidden_request_fields": len(forbidden_request_fields),
        },
        "engine_to_gpu": {pid: engine_to_gpu[pid] for pid in engine_pids},
        "scheduler_rows_by_engine": dict(scheduler_rows),
        "scheduler_task22_rid_observations_by_engine": dict(scheduler_task22_rids),
        "failures": {
            "rid_mismatches": rid_mismatches[:20],
            "ambiguous_rids": ambiguous_rids[:20],
            "scheduler_shape_errors": scheduler_shape_errors[:20],
            "forbidden_request_fields": forbidden_request_fields[:20],
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

    result = analyze(args.driver_log, args.request_dir, args.expected_engines, args.require_resume)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
