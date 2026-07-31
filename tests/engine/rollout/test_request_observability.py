# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import multiprocessing
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace


try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = type("ndarray", (), {})
    sys.modules["numpy"] = numpy_stub

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from relax.engine.rollout.request_observability import (
    abort_request_trace,
    attempt_token_from_id,
    begin_request_trace,
    build_consumption_records,
    close_discarded_abort_outcomes,
    export_partition_outcomes,
    export_request_traces,
    fail_request_trace,
    finish_request_trace,
    request_observability_enabled,
)
from relax.utils.types import Sample


def _sample(*, response_length: int = 0):
    return SimpleNamespace(
        response_length=response_length,
        group_index=17,
        index=23,
        abort_count=1,
    )


def _export_partition_outcome(output_dir: str, writer_id: int) -> None:
    sample = SimpleNamespace(
        metadata={
            "_last_request_attempt_id": f"attempt:{writer_id}",
            "_last_request_attempt_token": writer_id,
            "_request_attempt_sequence": writer_id,
        },
        group_index=writer_id,
        index=writer_id,
        abort_count=0,
    )
    export_partition_outcomes(
        [sample],
        output_dir=output_dir,
        physical_rollout_id=0,
        target_partition="train_0",
        outcome="committed",
    )


def test_non_partial_abort_closes_each_discarded_attempt_outcome() -> None:
    aborted = Sample(status=Sample.Status.ABORTED)
    completed = Sample(status=Sample.Status.COMPLETED)
    for sample in (aborted, completed):
        sample.metadata["_active_request_trace"] = {"attempt_id": str(id(sample)), "outcome": None}

    close_discarded_abort_outcomes([[aborted, completed]])

    assert aborted.metadata["_active_request_trace"]["outcome"] == "aborted"
    assert completed.metadata["_active_request_trace"]["outcome"] == "aborted"


def test_abort_outcome_does_not_overwrite_failed_trace_terminal_status() -> None:
    sample = Sample(status=Sample.Status.ABORTED)
    trace = {"attempt_id": "failed-attempt", "client_status": "dispatched"}
    sample.metadata["_active_request_trace"] = trace
    fail_request_trace(trace, RuntimeError("generation failed"))

    close_discarded_abort_outcomes([[sample]])

    assert trace["client_status"] == "generation_exception"
    assert trace["exception_type"] == "RuntimeError"
    assert trace["outcome"] == "aborted"


def test_request_trace_distinguishes_cancel_abort_and_generation_failure() -> None:
    cancelled = {"client_status": "dispatched"}
    failed = {"client_status": "dispatched"}
    aborted = {"client_status": "finished"}

    fail_request_trace(cancelled, RuntimeError("cancelled"), client_status="task_cancelled")
    fail_request_trace(failed, ValueError("failed"))
    abort_request_trace(aborted)

    assert cancelled["client_status"] == "task_cancelled"
    assert failed["client_status"] == "generation_exception"
    assert aborted["client_status"] == "request_aborted"


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
    assert attempt_token_from_id(row["rid"]) == int(row["rid"].rsplit(":", 1)[-1], 16)
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
            "relax_placement": {
                "schema_version": 1,
                "placement_policy_version": "v1",
                "placement_decision_id": row["rid"],
                "placement_mode": "shadow",
                "placement_policy": "least_predicted_work",
                "placement_candidate_engines": [
                    {
                        "engine_id": "engine-a",
                        "active_requests": 1,
                        "predicted_work": 128,
                    }
                ],
                "placement_shadow_engine_id": "engine-a",
                "placement_actual_engine_id": "engine-b",
                "untrusted_field": "not-copied",
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
    assert row["placement_mode"] == "shadow"
    assert row["placement_shadow_engine_id"] == "engine-a"
    assert row["placement_actual_engine_id"] == "engine-b"
    assert "untrusted_field" not in row


def test_export_request_traces_writes_one_json_object_per_line(tmp_path) -> None:
    rows = [{"rid": "relax:test:1", "client_status": "finished"}, {"rid": "relax:test:2"}]

    output_path = export_request_traces(
        rows,
        output_dir=str(tmp_path),
        physical_rollout_id=5,
    )

    loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
    assert not list(tmp_path.glob(".*.tmp"))


def test_partition_outcome_extension_republishes_complete_jsonl(tmp_path) -> None:
    sample = _sample()
    row, _ = begin_request_trace(
        sample=sample,
        payload={"input_ids": [1], "sampling_params": {"max_new_tokens": 1}},
        physical_rollout_id=0,
        admission_context={"decision_id": "admission:0:1", "decision_sequence": 1},
    )
    sample.metadata["_admission_decision_id"] = "admission:0:1"
    sample.metadata["_admission_decision_sequence"] = 1
    sample.metadata["_admission_decision_physical_rollout_id"] = 0

    path = export_partition_outcomes(
        [sample],
        output_dir=str(tmp_path),
        physical_rollout_id=0,
        target_partition="train_0",
        outcome="committed",
    )
    export_partition_outcomes(
        [sample],
        output_dir=str(tmp_path),
        physical_rollout_id=0,
        target_partition="train_0",
        outcome="committed",
    )

    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(loaded) == 2
    assert all(item["attempt_id"] == row["attempt_id"] for item in loaded)
    assert not list(tmp_path.glob(".*.tmp"))


def test_partition_outcome_concurrent_threads_do_not_lose_appends(tmp_path) -> None:
    writer_ids = list(range(40))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda writer_id: _export_partition_outcome(str(tmp_path), writer_id),
                writer_ids,
            )
        )

    output_path = tmp_path / "admission_outcomes_rollout_0.jsonl"
    loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {row["attempt_id"] for row in loaded} == {f"attempt:{writer_id}" for writer_id in writer_ids}
    assert len(loaded) == len(writer_ids)


def test_partition_outcome_concurrent_processes_do_not_lose_appends(tmp_path) -> None:
    writer_ids = list(range(12))
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_export_partition_outcome, args=(str(tmp_path), writer_id)) for writer_id in writer_ids
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    output_path = tmp_path / "admission_outcomes_rollout_0.jsonl"
    loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {row["attempt_id"] for row in loaded} == {f"attempt:{writer_id}" for writer_id in writer_ids}
    assert len(loaded) == len(writer_ids)


def test_abort_response_increments_sample_abort_count_once() -> None:
    sample = Sample()
    args = Namespace(sglang_speculative_algorithm=None)

    sample.update_from_meta_info(args, {"finish_reason": {"type": "abort"}})

    assert sample.status is Sample.Status.ABORTED
    assert sample.abort_count == 1


def test_consumption_ledger_preserves_attempt_and_decision_join_keys(tmp_path) -> None:
    sample = _sample()
    payload = {
        "input_ids": [1, 2],
        "sampling_params": {"max_new_tokens": 8},
    }
    row, _ = begin_request_trace(
        sample=sample,
        payload=payload,
        physical_rollout_id=3,
        admission_context={"decision_id": "admission:3:7", "decision_sequence": 7},
    )
    sample.metadata["_generation_physical_rollout_id"] = 3
    sample.metadata["_admission_decision_id"] = "admission:3:7"
    sample.metadata["_admission_decision_sequence"] = 7
    sample.metadata["_admission_decision_physical_rollout_id"] = 3
    sample.metadata["work_origin"] = "old_debt"

    outcome_path = export_partition_outcomes(
        [sample],
        output_dir=str(tmp_path),
        physical_rollout_id=3,
        target_partition="train_2",
        outcome="committed",
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    consume = build_consumption_records(
        {
            "sample_indices": [23],
            "group_indices": [17],
            "abort_counts": [1],
            "request_attempt_sequences": [1],
            "request_attempt_tokens": [attempt_token_from_id(row["rid"])],
            "admission_decision_sequences": [7],
            "admission_decision_physical_rollout_ids": [3],
            "generation_physical_rollout_ids": [3],
            "work_origin_codes": [1],
            "generation_start_version": [4],
            "generation_end_version": [5],
            "generation_version_span": [1],
        },
        rollout_id=2,
        consume_version=6,
    )[0]

    assert outcome["attempt_id"] == row["rid"]
    assert outcome["attempt_token"] == consume["attempt_token"]
    assert outcome["attempt_decision_id"] == "admission:3:7"
    assert outcome["decision_id"] == consume["decision_id"]
    assert consume["decision_id"] == "admission:3:7"
    assert consume["target_partition"] == "train_2"
    assert consume["actual_staleness"] == 1


def test_surplus_commit_keeps_generation_and_current_decision_physical_ids_separate(tmp_path) -> None:
    sample = _sample()
    payload = {
        "input_ids": [1, 2],
        "sampling_params": {"max_new_tokens": 8},
    }
    row, _ = begin_request_trace(
        sample=sample,
        payload=payload,
        physical_rollout_id=3,
        admission_context={"decision_id": "admission:3:7", "decision_sequence": 7},
    )
    sample.metadata["_admission_decision_id"] = "admission:4:2"
    sample.metadata["_admission_decision_sequence"] = 2
    sample.metadata["_admission_decision_physical_rollout_id"] = 4
    sample.metadata["work_origin"] = "surplus"

    outcome_path = export_partition_outcomes(
        [sample],
        output_dir=str(tmp_path),
        physical_rollout_id=4,
        target_partition="train_4",
        outcome="committed",
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    consume = build_consumption_records(
        {
            "sample_indices": [23],
            "group_indices": [17],
            "abort_counts": [1],
            "request_attempt_sequences": [1],
            "request_attempt_tokens": [attempt_token_from_id(row["rid"])],
            "admission_decision_sequences": [2],
            "admission_decision_physical_rollout_ids": [4],
            "generation_physical_rollout_ids": [3],
            "work_origin_codes": [2],
            "generation_start_version": [4],
            "generation_end_version": [5],
            "generation_version_span": [1],
        },
        rollout_id=4,
        consume_version=6,
    )[0]

    assert outcome["attempt_decision_id"] == "admission:3:7"
    assert outcome["decision_id"] == consume["decision_id"] == "admission:4:2"
    assert outcome["physical_rollout_id"] == consume["decision_physical_rollout_id"] == 4
    assert outcome["generation_physical_rollout_id"] == consume["generation_physical_rollout_id"] == 3
