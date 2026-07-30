# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys
from argparse import Namespace
from types import ModuleType

import pytest


try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ray"] = ModuleType("ray")

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from relax.engine.router.router import SlimeRouter


def _args(*, mode: str, sticky: bool = False) -> Namespace:
    return Namespace(
        request_placement_mode=mode,
        request_placement_policy="least_predicted_work",
        slime_router_sticky=sticky,
        slime_router_sticky_idle_secs=600.0,
        slime_router_max_connections=4,
        slime_router_timeout=None,
        slime_router_middleware_paths=[],
        sglang_server_concurrency=4,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        rollout_health_check_interval=10.0,
        slime_router_health_check_failure_threshold=3,
    )


def _router(*, mode: str) -> SlimeRouter:
    router = SlimeRouter(_args(mode=mode))
    router.worker_request_counts = {
        "http://worker-a": 0,
        "http://worker-b": 1,
        "http://worker-dead": 0,
    }
    router.worker_predicted_work = {
        "http://worker-a": 100,
        "http://worker-b": 0,
        "http://worker-dead": 0,
    }
    router.dead_workers = {"http://worker-dead"}
    return router


def _body(rid: str = "relax:test") -> bytes:
    return json.dumps(
        {
            "rid": rid,
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 7},
        }
    ).encode()


def test_router_placement_shadow_preserves_baseline_and_accounts_work() -> None:
    router = _router(mode="shadow")

    worker_url, predicted_work, decision = router._use_url_with_placement(None, _body())

    assert worker_url == "http://worker-a"
    assert decision.selected_engine_id != decision.actual_engine_id
    assert predicted_work == 10
    assert router.worker_request_counts["http://worker-a"] == 1
    assert router.worker_predicted_work["http://worker-a"] == 110
    assert len(decision.candidates) == 2

    router._finish_url(worker_url, predicted_work=predicted_work)
    assert router.worker_request_counts["http://worker-a"] == 0
    assert router.worker_predicted_work["http://worker-a"] == 100


def test_router_placement_on_routes_to_policy_choice() -> None:
    router = _router(mode="on")

    worker_url, predicted_work, decision = router._use_url_with_placement(None, _body())

    assert worker_url == "http://worker-b"
    assert decision.actual_engine_id == decision.selected_engine_id
    assert router.worker_request_counts["http://worker-b"] == 2
    assert router.worker_predicted_work["http://worker-b"] == 10
    router._finish_url(worker_url, predicted_work=predicted_work)


def test_router_placement_invalid_payload_fails_open_to_baseline() -> None:
    router = _router(mode="on")

    worker_url, predicted_work, decision = router._use_url_with_placement(None, b"{invalid")

    assert worker_url == "http://worker-a"
    assert predicted_work == 0
    assert decision.routing_reason == "fail_open_router"
    assert decision.fallback_reason == "JSONDecodeError"
    router._finish_url(worker_url, predicted_work=predicted_work)


def test_router_placement_rejects_group_sticky_semantics() -> None:
    with pytest.raises(ValueError, match="Request-level placement"):
        SlimeRouter(_args(mode="shadow", sticky=True))
