# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("ray")
pytest.importorskip("transfer_queue")

from relax.components.rollout import Rollout


class _AsyncRemote:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _SyncRemote:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _rollout(*, completion_results):
    cls = Rollout.func_or_class
    rollout = cls.__new__(cls)
    rollout.config = SimpleNamespace(
        eval_interval=None,
        skip_eval_before_train=True,
        num_rollout=1,
        fully_async=True,
        offload_rollout=False,
        max_staleness=1,
        save=None,
        save_interval=None,
        use_health_check=True,
    )
    rollout.step = 1
    rollout.status = "running"
    rollout.eval_handler = None
    rollout._stop_event = asyncio.Event()
    rollout._peer_barrier = None
    rollout._logger_instance = MagicMock()
    rollout.rollout_manager = SimpleNamespace(generate=_AsyncRemote())
    rollout.data_system_client = SimpleNamespace(async_get_partition_list=_AsyncRemote([]))
    rollout.healthy = SimpleNamespace(report_error=_SyncRemote(), update_heartbeat=_SyncRemote())
    completion_iter = iter(completion_results)

    async def check_complete(_partition_id):
        return next(completion_iter)

    rollout._async_check_partition_production_complete = check_complete
    return rollout


def test_final_partition_is_validated_before_step_increment_and_failure_is_visible(monkeypatch) -> None:
    monkeypatch.setattr("relax.engine.sft.runtime.is_sft_mode", lambda _config: False)
    rollout = _rollout(completion_results=[False, False])

    with pytest.raises(RuntimeError, match="Final rollout partition train_0 is still incomplete"):
        asyncio.run(rollout._async_run())

    assert rollout.step == 1
    assert rollout.status == "FAILED"
    assert rollout.healthy.report_error.calls


def test_health_check_mode_does_not_turn_generation_error_into_normal_completion(monkeypatch) -> None:
    monkeypatch.setattr("relax.engine.sft.runtime.is_sft_mode", lambda _config: False)
    rollout = _rollout(completion_results=[False])
    rollout.rollout_manager.generate = _AsyncRemote(RuntimeError("generation exploded"))

    with pytest.raises(RuntimeError, match="generation exploded"):
        asyncio.run(rollout._async_run())

    assert rollout.step == 1
    assert rollout.status == "FAILED"
