# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

from scripts.task22.analyze_rollout_observability import analyze, export_placement_trace


def _event(pid: int, event: dict) -> str:
    payload = dict(event)
    payload.setdefault(
        "timestamp",
        {
            "request.received": 100.2,
            "request.finished": 101.8,
        }.get(event.get("event"), 100.5),
    )
    return f"(SGLangEngine pid={pid}) [2026-07-30] {json.dumps(payload, sort_keys=True)}"


def _finished_row(rid: str) -> dict:
    return {
        "rid": rid,
        "attempt_kind": "fresh",
        "client_status": "finished",
        "rid_match": True,
        "dispatch_abs": 100.0,
        "request_end_abs": 102.0,
        "generated_tokens_this_attempt": 10,
        "call_prompt_tokens": 100,
        "call_cached_tokens": 50,
        "call_new_prompt_tokens": 50,
        "forward_entry_time": 100.1,
        "prefill_finished_time": 100.4,
        "queue_time": 0.1,
    }


def _aborted_row(rid: str) -> dict:
    row = _finished_row(rid)
    row.update(
        {
            "outcome": "aborted",
            "client_status": "request_aborted",
            "finish_reason": {"type": "abort", "message": "Aborted"},
        }
    )
    return row


def _valid_single_engine_log(rids: list[str]) -> str:
    lines = ["(SGLangEngine pid=100) server_args=ServerArgs(base_gpu_id=2, tp_size=1)"]
    for rid in rids:
        lines.append(_event(100, {"event": "request.received", "rid": rid, "obj": {"rid": rid}}))
        lines.append(_event(100, {"event": "request.finished", "rid": rid}))
    lines.append(
        _event(
            100,
            {
                "event": "scheduler.status",
                "forward_mode": "ForwardMode.DECODE",
                "running_rids": rids,
                "running_seq_lens": [100] * len(rids),
                "running_origin_input_lens": [90] * len(rids),
                "running_output_lens": [10] * len(rids),
                "decoding_rids": rids,
                "queued_rids": [],
                "queued_origin_input_lens": [],
                "queued_output_lens": [],
            },
        )
    )
    return "\n".join(lines) + "\n"


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
                _event(pid, {"event": "request.finished", "rid": rid}),
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
                        "out": {"meta_info": {"output_token_logprobs": [[-0.1, 123, None]]}},
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
    assert result["failures"]["forbidden_fields"][0]["paths"] == ["$.out.meta_info.output_token_logprobs"]


def test_rollout_observability_analyzer_rejects_duplicate_client_rids(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:duplicate"
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(
        json.dumps(_finished_row(rid)) + "\n" + json.dumps(_finished_row(rid)) + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([rid]), encoding="utf-8")

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["client_rids_unique"]


def test_rollout_observability_analyzer_rejects_extra_server_rid(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    client_rid = "relax:p1:kfresh:g1:s1:a0:client"
    server_rid = "relax:p1:kfresh:g2:s2:a0:server-only"
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(json.dumps(_finished_row(client_rid)) + "\n", encoding="utf-8")
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([client_rid, server_rid]), encoding="utf-8")

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["server_rids_match_client_rids"]


def test_rollout_observability_analyzer_rejects_duplicate_server_lifecycle_events(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:duplicate-server"
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(json.dumps(_finished_row(rid)) + "\n", encoding="utf-8")
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(
        _valid_single_engine_log([rid])
        + _event(100, {"event": "request.received", "rid": rid, "obj": {"rid": rid}})
        + "\n",
        encoding="utf-8",
    )

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["server_lifecycle_complete"]


def test_rollout_observability_analyzer_rejects_nonterminal_or_invalid_timing_rows(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    nonterminal_rid = "relax:p1:kfresh:g1:s1:a0:nonterminal"
    invalid_time_rid = "relax:p1:kfresh:g2:s2:a0:invalid-time"
    nonterminal = _finished_row(nonterminal_rid)
    nonterminal["client_status"] = "dispatched"
    nonterminal.pop("request_end_abs")
    invalid_time = _finished_row(invalid_time_rid)
    invalid_time["dispatch_abs"] = float("nan")
    invalid_time["request_end_abs"] = 99.0
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(
        json.dumps(nonterminal) + "\n" + json.dumps(invalid_time) + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(
        _valid_single_engine_log([nonterminal_rid, invalid_time_rid]),
        encoding="utf-8",
    )

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "FAIL"
    assert not result["checks"]["client_rows_terminal"]
    assert not result["checks"]["client_timestamps_valid"]


def test_rollout_observability_analyzer_accepts_aborted_terminal_rows(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g9:s72:a0:aborted"
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(json.dumps(_aborted_row(rid)) + "\n", encoding="utf-8")
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([rid]), encoding="utf-8")

    result = analyze(driver_log, request_dir, expected_engines=1, require_resume=False)

    assert result["verdict"] == "PASS"
    assert result["counts"]["nonterminal_rows"] == 0


def test_rollout_observability_analyzer_exports_prompt_free_placement_trace(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:placement"
    row = _finished_row(rid)
    row.update(
        {
            "physical_rollout_id": 1,
            "logical_prefix_tokens": 100,
            "max_new_tokens": 200,
            "request_wall": 2.0,
            "call_new_prompt_tokens": 20,
            "placement_mode": "shadow",
            "placement_policy": "least_predicted_work",
            "placement_candidate_engines": [
                {
                    "engine_id": "engine-a",
                    "active_requests": 1,
                    "predicted_work": 100,
                },
                {
                    "engine_id": "engine-b",
                    "active_requests": 0,
                    "predicted_work": 0,
                },
            ],
            "placement_shadow_engine_id": "engine-b",
            "placement_actual_engine_id": "engine-a",
        }
    )
    (request_dir / "request_lifecycle_rollout_1.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([rid]), encoding="utf-8")
    output_path = tmp_path / "placement_trace.jsonl"

    summary = export_placement_trace(driver_log, request_dir, output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["rows"] == 1
    assert exported["rid"] == rid
    assert exported["predicted_work"] == 300
    assert exported["observed_work"] == 30
    assert exported["candidate_engine_ids"] == ["engine-a", "engine-b"]
    assert exported["actual_engine_id"] == "engine-a"
    assert exported["actual_engine_pid_id"] == "engine-pid-100"
    assert exported["actual_engine_gpu_id"] == 2
    assert not {"prompt", "input_ids", "sampling_params", "output_ids"}.intersection(exported)


def test_placement_trace_uses_latest_scheduler_snapshot_before_dispatch(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:snapshot"
    row = _finished_row(rid)
    row.update(
        {
            "physical_rollout_id": 1,
            "dispatch_abs": 100.0,
            "request_end_abs": 102.0,
            "request_wall": 2.0,
        }
    )
    (request_dir / "request_lifecycle_rollout_1.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(
        "\n".join(
            [
                _event(
                    100,
                    {
                        "event": "scheduler.status",
                        "timestamp": 99.0,
                        "forward_mode": "ForwardMode.DECODE",
                        "running_rids": ["old"],
                        "running_seq_lens": [10],
                        "running_origin_input_lens": [8],
                        "running_output_lens": [2],
                        "queued_rids": ["queued"],
                        "queued_origin_input_lens": [20],
                        "queued_output_lens": [3],
                    },
                ),
                _event(
                    100,
                    {
                        "event": "scheduler.status",
                        "timestamp": 100.5,
                        "forward_mode": "ForwardMode.DECODE",
                        "running_rids": ["future"],
                        "running_seq_lens": [999],
                        "running_origin_input_lens": [999],
                        "running_output_lens": [999],
                        "queued_rids": [],
                        "queued_origin_input_lens": [],
                        "queued_output_lens": [],
                    },
                ),
                _event(100, {"event": "request.received", "rid": rid, "timestamp": 100.2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "placement_trace.jsonl"

    summary = export_placement_trace(driver_log, request_dir, output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    snapshot = exported["candidate_snapshots"][0]

    assert summary["rows_with_candidate_snapshots"] == 1
    assert snapshot["snapshot_abs"] == 99.0
    assert snapshot["snapshot_age_s"] == 1.0
    assert snapshot["active_requests"] == 1
    assert snapshot["queued_requests"] == 1
    assert snapshot["predicted_work"] == 33


def test_placement_trace_maps_server_engine_pid_to_gpu(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g1:s1:a0:engine-gpu"
    row = _finished_row(rid)
    row.update({"physical_rollout_id": 1})
    (request_dir / "request_lifecycle_rollout_1.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([rid]), encoding="utf-8")
    output_path = tmp_path / "placement_trace.jsonl"

    export_placement_trace(driver_log, request_dir, output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["actual_engine_id"] == "engine-pid-100"
    assert exported["actual_engine_pid_id"] == "engine-pid-100"
    assert exported["actual_engine_gpu_id"] == 2


def test_rollout_observability_analyzer_exports_aborted_placement_trace(tmp_path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    rid = "relax:p1:kfresh:g9:s72:a0:aborted-placement"
    row = _aborted_row(rid)
    row.update({"physical_rollout_id": 1, "logical_prefix_tokens": 100, "max_new_tokens": 200})
    lifecycle = request_dir / "request_lifecycle_rollout_1.jsonl"
    lifecycle.write_text(json.dumps(row) + "\n", encoding="utf-8")
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(_valid_single_engine_log([rid]), encoding="utf-8")
    output_path = tmp_path / "placement_trace.jsonl"

    summary = export_placement_trace(driver_log, request_dir, output_path)

    assert summary["rows"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["rid"] == rid
