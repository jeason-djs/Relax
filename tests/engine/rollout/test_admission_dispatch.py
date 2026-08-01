# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import MethodType, SimpleNamespace

import pytest

from relax.engine.rollout.sglang_rollout import GenerateState, generate_and_rm
from relax.utils.types import Sample


def _group(origin: str, index: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(metadata={"work_origin": origin}, index=index)]


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
