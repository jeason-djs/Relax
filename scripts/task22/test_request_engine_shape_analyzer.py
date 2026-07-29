#!/usr/bin/env python3
"""Synthetic regression test for the Task 22 observability analyzer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from analyze_request_engine_shape import analyze


def _event(pid: int, event: dict) -> str:
    return f"(SGLangEngine pid={pid}) [2026-07-29] {json.dumps(event, sort_keys=True)}"


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        request_dir = root / "requests"
        request_dir.mkdir()
        rows = [
            {
                "rid": "task22:p1:wcurrent:g1:s1:a0:abc",
                "work_class": "current",
                "client_status": "finished",
                "rid_match": True,
            },
            {
                "rid": "task22:p1:wold_debt:g2:s2:a1:def",
                "work_class": "old_debt",
                "client_status": "finished",
                "rid_match": True,
            },
        ]
        lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
        lifecycle.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        log_lines = [
            "(SGLangEngine pid=100) server_args=ServerArgs(base_gpu_id=2, tp_size=1)",
            "(SGLangEngine pid=101) server_args=ServerArgs(base_gpu_id=3, tp_size=1)",
        ]
        for pid, row in zip((100, 101), rows):
            rid = row["rid"]
            log_lines.append(
                _event(
                    pid,
                    {
                        "timestamp": "2026-07-29T00:00:00",
                        "event": "request.received",
                        "rid": rid,
                        "obj": {"rid": rid, "return_logprob": True},
                    },
                )
            )
            log_lines.append(
                _event(
                    pid,
                    {
                        "timestamp": "2026-07-29T00:00:01",
                        "event": "scheduler.status",
                        "rank": 0,
                        "forward_mode": "ForwardMode.DECODE",
                        "running_rids": [rid],
                        "running_seq_lens": [4096],
                        "running_origin_input_lens": [2048],
                        "running_output_lens": [2048],
                        "decoding_rids": [rid],
                        "queued_rids": [],
                        "queued_origin_input_lens": [],
                        "queued_output_lens": [],
                    },
                )
            )
            log_lines.append(
                _event(
                    pid,
                    {
                        "timestamp": "2026-07-29T00:00:02",
                        "event": "request.finished",
                        "rid": rid,
                        "obj": {"rid": rid},
                        "out": {"meta_info": {"id": rid}},
                    },
                )
            )

        driver_log = root / "driver.log"
        driver_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        result = analyze(driver_log, request_dir, expected_engines=2, require_resume=True)
        assert result["verdict"] == "PASS", json.dumps(result, indent=2)
        assert result["engine_to_gpu"] == {"100": 2, "101": 3}
        assert result["counts"]["mapped_resume_rids"] == 1
        print("PASS: synthetic request-to-engine and shape contract")


if __name__ == "__main__":
    main()
