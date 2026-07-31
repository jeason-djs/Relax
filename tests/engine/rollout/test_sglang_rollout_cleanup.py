# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest


pytest.importorskip("pybase64")
pytest.importorskip("ray")
pytest.importorskip("sglang_router")

from relax.engine.rollout import sglang_rollout


def _sample():
    return SimpleNamespace(
        metadata={
            "_active_request_trace": {
                "client_status": "dispatched",
            }
        }
    )


def test_generation_exception_cleanup_aborts_cancels_and_gathers_all_tasks(monkeypatch) -> None:
    abort_calls = []

    async def fake_abort(_args):
        abort_calls.append(True)

    monkeypatch.setattr(sglang_rollout, "_abort_generation_workers", fake_abort)

    async def scenario():
        blocker = asyncio.Event()

        async def pending_work():
            await blocker.wait()

        async def failed_work():
            raise RuntimeError("injected generation failure")

        pending_task = asyncio.create_task(pending_work())
        failed_task = asyncio.create_task(failed_work())
        pending_task._relax_sample_group = [_sample()]
        failed_task._relax_sample_group = [_sample()]
        await asyncio.sleep(0)

        state = SimpleNamespace(
            pendings={pending_task},
            protected_pendings={failed_task},
        )
        await sglang_rollout._cleanup_failed_generation_tasks(
            SimpleNamespace(),
            state,
            tasks={pending_task, failed_task},
        )

        assert abort_calls == [True]
        assert pending_task.cancelled()
        assert failed_task.done()
        assert not state.pendings
        assert not state.protected_pendings
        assert pending_task._relax_sample_group[0].metadata["_active_request_trace"]["client_status"] == "task_cancelled"

    asyncio.run(scenario())


def test_abort_harvests_protected_and_partial_tasks_before_rethrowing_first_exception(monkeypatch) -> None:
    abort_calls = []

    async def fake_abort(_args):
        abort_calls.append(True)

    monkeypatch.setattr(sglang_rollout, "_abort_generation_workers", fake_abort)

    async def scenario():
        protected_sample = _sample()
        partial_sample = _sample()

        async def fail(message):
            raise RuntimeError(message)

        protected = asyncio.create_task(fail("first protected failure"))
        partial = asyncio.create_task(fail("second partial failure"))
        protected._relax_sample_group = [protected_sample]
        partial._relax_sample_group = [partial_sample]
        state = SimpleNamespace(
            evaluating=0,
            aborted=False,
            protected_pendings={protected},
            pendings={partial},
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)

        with pytest.raises(RuntimeError, match="first protected failure"):
            await sglang_rollout.abort(SimpleNamespace(partial_rollout=True), rollout_id=4)

        assert abort_calls == [True]
        assert protected.done() and partial.done()
        assert not state.protected_pendings
        assert not state.pendings
        assert protected_sample.metadata["_active_request_trace"]["client_status"] == "generation_exception"
        assert partial_sample.metadata["_active_request_trace"]["client_status"] == "generation_exception"

    asyncio.run(scenario())


def test_generation_exception_harvest_does_not_overwrite_aborted_trace() -> None:
    async def scenario():
        sample = _sample()
        sample.metadata["_active_request_trace"]["client_status"] = "request_aborted"

        async def fail_after_abort():
            raise RuntimeError("wrapper failed after abort")

        task = asyncio.create_task(fail_after_abort())
        task._relax_sample_group = [sample]
        _, errors = await sglang_rollout._gather_generation_tasks({task})

        assert len(errors) == 1
        assert sample.metadata["_active_request_trace"]["client_status"] == "request_aborted"

    asyncio.run(scenario())


def test_generate_rollout_lifecycle_preserves_first_exception_and_cleans_everything(monkeypatch) -> None:
    calls = []

    async def start(_args, _rollout_id):
        calls.append("start")

    async def stop(_args, _rollout_id):
        calls.append("stop")
        raise RuntimeError("stop failure")

    async def abort_workers(_args):
        calls.append("abort")

    class Pbar:
        def close(self):
            calls.append("close")

    async def scenario():
        blocker = asyncio.Event()
        created_tasks = []

        async def pending():
            await blocker.wait()

        state = SimpleNamespace(pendings=set(), protected_pendings=set())

        async def fail(_args, _rollout_id, _data_source, _client, lifecycle):
            generation = asyncio.create_task(pending())
            transfer = asyncio.create_task(pending())
            created_tasks.extend((generation, transfer))
            state.pendings.add(generation)
            lifecycle.transfer_tasks.append(transfer)
            lifecycle.pbar = Pbar()
            raise ValueError("first failure")

        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "start_sglang_profile", start)
        monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", stop)
        monkeypatch.setattr(sglang_rollout, "_abort_generation_workers", abort_workers)
        monkeypatch.setattr(sglang_rollout, "_generate_rollout_async_impl", fail)

        with pytest.raises(ValueError, match="first failure"):
            await sglang_rollout.generate_rollout_async(SimpleNamespace(), 3, None, None)

        assert all(task.cancelled() for task in created_tasks)
        assert not state.pendings
        assert calls == ["start", "abort", "stop", "close"]

    asyncio.run(scenario())


def test_generate_rollout_normal_path_only_stops_and_closes_once(monkeypatch) -> None:
    calls = []
    expected = (SimpleNamespace(), [])
    state = SimpleNamespace(pendings=set(), protected_pendings=set())

    async def impl(_args, _rollout_id, _data_source, _client, lifecycle):
        lifecycle.pbar = SimpleNamespace(close=lambda: calls.append("close"))
        return expected

    async def start(_args, _rollout_id):
        calls.append("start")

    async def stop(_args, _rollout_id):
        calls.append("stop")

    async def scenario():
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "start_sglang_profile", start)
        monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", stop)
        monkeypatch.setattr(sglang_rollout, "_generate_rollout_async_impl", impl)
        monkeypatch.setattr(
            sglang_rollout,
            "_cleanup_failed_rollout_lifecycle",
            lambda *_args: pytest.fail("normal path must not run failure cleanup"),
        )

        assert await sglang_rollout.generate_rollout_async(SimpleNamespace(), 4, None, None) is expected
        assert calls == ["start", "stop", "close"]

    asyncio.run(scenario())
