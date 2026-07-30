# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from argparse import Namespace
from types import SimpleNamespace

from relax.engine.rollout.request_observability import (
    begin_request_trace,
    export_request_traces,
    finish_request_trace,
    request_observability_enabled,
)


def _sample(*, response_length: int = 0):
    return SimpleNamespace(
        response_length=response_length,
        group_index=17,
        index=23,
        abort_count=1,
    )


def test_request_observability_is_disabled_without_output_directory() -> None:
    assert not request_observability_enabled(Namespace(rollout_request_observability_dir=None))
    assert request_observability_enabled(Namespace(rollout_request_observability_dir="/tmp/trace"))


def test_request_trace_contains_counts_and_admission_context_without_prompt_content() -> None:
    payload = {
        "input_ids": [11, 12, 13],
        "sampling_params": {"max_new_tokens": 64},
        "return_logprob": True,
    }

    row, _ = begin_request_trace(
        sample=_sample(response_length=2),
        payload=payload,
        physical_rollout_id=9,
        admission_context={
            "mode": "on",
            "release_remaining": 2,
        },
    )

    assert payload["rid"] == row["rid"]
    assert row["attempt_kind"] == "resume"
    assert row["logical_prefix_tokens"] == 3
    assert row["partial_response_tokens"] == 2
    assert row["admission_mode"] == "on"
    assert row["admission_release_remaining"] == 2
    assert not {"input_ids", "text", "prompt", "sampling_params"}.intersection(row)


def test_finish_request_trace_reads_nested_timing_and_derives_queue_time() -> None:
    payload = {
        "input_ids": [1, 2],
        "sampling_params": {"max_new_tokens": 8},
    }
    row, request_start = begin_request_trace(
        sample=_sample(),
        payload=payload,
        physical_rollout_id=3,
        admission_context=None,
    )
    output = {
        "output_ids": [99],
        "meta_info": {
            "id": row["rid"],
            "prompt_tokens": 10,
            "cached_tokens": 4,
            "output_token_logprobs": [[-0.1, 7], [-0.2, 8], [-0.3, 9]],
            "time_stats": {
                "wait_queue_entry_time": 100.0,
                "forward_entry_time": 100.25,
                "prefill_finished_time": 100.75,
            },
        },
    }

    finish_request_trace(row, output=output, request_start_monotonic=request_start)

    assert row["client_status"] == "finished"
    assert row["rid_match"]
    assert row["call_new_prompt_tokens"] == 6
    assert row["generated_tokens_this_attempt"] == 3
    assert row["queue_time"] == 0.25
    assert row["prefill_finished_time"] == 100.75


def test_export_request_traces_writes_one_json_object_per_line(tmp_path) -> None:
    rows = [{"rid": "relax:test:1", "client_status": "finished"}, {"rid": "relax:test:2"}]

    output_path = export_request_traces(
        rows,
        output_dir=str(tmp_path),
        physical_rollout_id=5,
    )

    loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
