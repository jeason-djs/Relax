# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

from scripts.task22.analyze_rollout_observability import analyze


def _event(pid: int, event: dict) -> str:
    return f"(SGLangEngine pid={pid}) [2026-07-30] {json.dumps(event, sort_keys=True)}"


def test_rollout_observability_analyzer_maps_requests_and_summarizes_engine_shapes(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rows = [
        {
            "rid": "relax:p1:kfresh:g1:s1:a0:abc",
            "attempt_kind": "fresh",
            "client_status": "finished",
            "rid_match": True,
            "dispatch_abs": 100.0,
            "request_end_abs": 102.0,
            "generated_tokens_this_attempt": 100,
            "call_prompt_tokens": 1000,
            "call_cached_tokens": 750,
            "call_new_prompt_tokens": 250,
            "forward_entry_time": 100.1,
            "prefill_finished_time": 100.4,
            "queue_time": 0.1,
        },
        {
            "rid": "relax:p1:kresume:g2:s2:a1:def",
            "attempt_kind": "resume",
            "client_status": "finished",
            "rid_match": True,
            "dispatch_abs": 100.0,
            "request_end_abs": 104.0,
            "generated_tokens_this_attempt": 200,
            "call_prompt_tokens": 2000,
            "call_cached_tokens": 1000,
            "call_new_prompt_tokens": 1000,
            "forward_entry_time": 100.2,
            "prefill_finished_time": 100.8,
            "queue_time": 0.2,
        },
    ]
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    log_lines = [
        "(SGLangEngine pid=100) server_args=ServerArgs(base_gpu_id=2, tp_size=1)",
        "(SGLangEngine pid=101) server_args=ServerArgs(base_gpu_id=3, tp_size=1)",
    ]
    for pid, row in zip((100, 101), rows, strict=True):
        rid = row["rid"]
        log_lines.extend(
            (
                _event(
                    pid,
                    {
                        "event": "request.received",
                        "rid": rid,
                        "obj": {"rid": rid, "return_logprob": True},
                    },
                ),
                _event(
                    pid,
                    {
                        "event": "scheduler.status",
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
                ),
            )
        )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    result = analyze(driver_log, request_dir, expected_engines=2, require_resume=True)

    assert result["verdict"] == "PASS"
    assert result["counts"]["mapped_resume_rids"] == 1
    assert result["engine_summaries"]["100"]["gpu_id"] == 2
    assert result["engine_summaries"]["100"]["active_requests"]["max"] == 1.0
    assert result["engine_summaries"]["100"]["cache_token_hit_ratio"] == 0.75
    assert result["engine_summaries"]["101"]["generated_tokens"] == 200
    assert result["engine_summaries"]["101"]["resume_new_prompt_tokens"] == 1000


def test_rollout_observability_analyzer_rejects_nested_content_in_any_request_event(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:abc"
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(
        json.dumps(
            {
                "rid": rid,
                "attempt_kind": "fresh",
                "client_status": "finished",
                "rid_match": True,
                "dispatch_abs": 100.0,
                "request_end_abs": 102.0,
                "generated_tokens_this_attempt": 10,
                "call_prompt_tokens": 100,
                "call_cached_tokens": 50,
                "forward_entry_time": 100.1,
                "prefill_finished_time": 100.4,
                "queue_time": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(
        "\n".join(
            (
                "(SGLangEngine pid=100) server_args=ServerArgs(base_gpu_id=2, tp_size=1)",
                _event(100, {"event": "request.received", "rid": rid, "obj": {"rid": rid}}),
                _event(
                    100,
                    {
                        "event": "request.finished",
                        "rid": rid,
                        "out": {"debug": {"input_ids": [1, 2, 3]}},
                    },
                ),
                _event(
                    100,
                    {
                        "event": "scheduler.status",
                        "forward_mode": "ForwardMode.DECODE",
                        "running_rids": [rid],
                        "running_seq_lens": [100],
                        "running_origin_input_lens": [90],
                        "running_output_lens": [10],
                        "decoding_rids": [rid],
                        "queued_rids": [],
                        "queued_origin_input_lens": [],
                        "queued_output_lens": [],
                    },
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["server_request_log_has_no_large_fields"]
    assert result["counts"]["forbidden_fields"] == 1
    assert result["failures"]["forbidden_fields"][0]["paths"] == ["$.out.debug.input_ids"]
