# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import MethodType, SimpleNamespace

import pytest

import relax.engine.rollout.sglang_rollout as sglang_rollout
from relax.engine.rollout.sglang_rollout import (
    GenerateState,
    generate,
    generate_and_rm,
    resolve_partition_request_priority,
)
from relax.utils.types import Sample


def _group(origin: str, index: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(metadata={"work_origin": origin}, index=index)]


def test_partition_request_priority_is_omitted_when_sglang_feature_is_off() -> None:
    args = SimpleNamespace(sglang_enable_priority_scheduling=False)
    sample = _group("old_debt", 0)[0]

    assert resolve_partition_request_priority(args, sample) is None


@pytest.mark.parametrize(
    ("origin", "expected"),
    (("old_debt", 1), ("fresh", 0), ("surplus", 0), ("unknown", 0)),
)
def test_partition_request_priority_maps_old_debt_above_other_work(origin: str, expected: int) -> None:
    args = SimpleNamespace(sglang_enable_priority_scheduling=True)
    sample = _group(origin, 0)[0]

    assert resolve_partition_request_priority(args, sample) == expected


def test_partition_request_priority_treats_missing_metadata_as_fresh() -> None:
    args = SimpleNamespace(sglang_enable_priority_scheduling=True)

    assert resolve_partition_request_priority(args, SimpleNamespace()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluation", "expected_priority"),
    ((False, 1), (True, 0)),
)
async def test_generate_sends_partition_priority_in_http_payload(
    monkeypatch,
    evaluation: bool,
    expected_priority: int,
) -> None:
    payloads = []

    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            del prompt, add_special_tokens
            return [1, 2]

    state = SimpleNamespace(
        tokenizer=Tokenizer(),
        processor=None,
        opd_manager=None,
        current_rollout_id=3,
        request_observability_rows=[],
    )

    async def fake_post(url, payload, headers=None):
        del url, headers
        payloads.append(payload)
        return {"output_ids": [9], "text": "x", "meta_info": {}}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_enable_priority_scheduling=True,
        use_rollout_routing_replay=False,
        lora_rank=0,
        rollout_request_observability_dir=None,
        sglang_router_policy="round_robin",
        slime_router_sticky=False,
        use_slime_router=False,
        slime_router_middleware_paths=[],
    )
    sample = SimpleNamespace(
        status=Sample.Status.PENDING,
        prompt="prompt",
        response="",
        response_length=0,
        tokens=[],
        rollout_tokens=[],
        rollout_log_probs=None,
        loss_mask=None,
        multimodal_inputs=None,
        metadata={"work_origin": "old_debt"},
        session_id=None,
        group_index=1,
        update_from_meta_info=lambda args, meta_info: None,
    )

    await generate(args, sample, {"max_new_tokens": 1}, evaluation=evaluation)

    assert payloads[0]["priority"] == expected_priority


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (Sample.Status.COMPLETED, Sample.Status.TRUNCATED))
async def test_completed_sample_satisfies_dispatch_barrier(status: Sample.Status) -> None:
    dispatch_started = asyncio.Event()
    sample = Sample(status=status, response="done", reward=1.0)
    args = SimpleNamespace(partial_rollout=False, group_rm=False)

    result = await generate_and_rm(
        args,
        sample,
        sampling_params={},
        dispatch_started_event=dispatch_started,
    )

    assert result is sample
    assert dispatch_started.is_set()


@pytest.mark.asyncio
async def test_debt_first_dispatch_barrier_opens_fresh_after_critical_submission() -> None:
    state = object.__new__(GenerateState)
    calls: list[tuple[list[int], bool]] = []

    def fake_submit(
        _self,
        samples,
        *,
        admission_context=None,
        submission_events=None,
    ) -> None:
        del admission_context
        calls.append(([group[0].index for group in samples], submission_events is not None))
        for event in submission_events or []:
            event.set()

    state.submit_generate_tasks = MethodType(fake_submit, state)
    samples = [_group("fresh", 0), _group("old_debt", 1), _group("old_debt", 2), _group("surplus", 3)]

    critical_groups, _ = await state.submit_generate_tasks_debt_first(samples, debt_group_limit=2)

    assert critical_groups == 2
    assert calls == [([1, 2], True), ([0, 3], False)]


@pytest.mark.asyncio
async def test_debt_first_dispatch_limits_priority_without_dropping_groups() -> None:
    state = object.__new__(GenerateState)
    submitted: list[int] = []

    def fake_submit(
        _self,
        samples,
        *,
        admission_context=None,
        submission_events=None,
    ) -> None:
        del admission_context
        submitted.extend(group[0].index for group in samples)
        for event in submission_events or []:
            event.set()

    state.submit_generate_tasks = MethodType(fake_submit, state)
    samples = [_group("old_debt", 0), _group("old_debt", 1), _group("fresh", 2)]

    critical_groups, _ = await state.submit_generate_tasks_debt_first(samples, debt_group_limit=1)

    assert critical_groups == 1
    assert submitted == [0, 1, 2]
