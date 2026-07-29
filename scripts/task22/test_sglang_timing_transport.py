#!/usr/bin/env python3
"""No-GPU regression test for SGLang's two-hop request timing transport."""

from __future__ import annotations

import math
import pickle

from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats


def _round_trip(value: SchedulerReqTimeStats) -> SchedulerReqTimeStats:
    return pickle.loads(pickle.dumps(value))


def main() -> None:
    scheduler = SchedulerReqTimeStats()
    scheduler.enable_metrics = True
    scheduler.wait_queue_entry_time = 100.0
    scheduler.forward_entry_time = 101.25
    scheduler.prefill_finished_time = 103.5

    first_hop = _round_trip(scheduler)
    assert first_hop.enable_metrics is False
    assert first_hop.metrics_collector is None
    assert getattr(first_hop, "_task22_forward_timing_payload") is True

    second_hop = _round_trip(first_hop)
    assert second_hop.enable_metrics is False
    assert second_hop.metrics_collector is None
    assert getattr(second_hop, "_task22_forward_timing_payload") is True
    assert math.isclose(second_hop.wait_queue_entry_time, 100.0)
    assert math.isclose(second_hop.forward_entry_time, 101.25)
    assert math.isclose(second_hop.prefill_finished_time, 103.5)

    meta = second_hop.convert_to_output_meta_info()
    assert meta["forward_entry_time"] > 1_000_000_000
    assert meta["prefill_finished_time"] > meta["forward_entry_time"]
    assert math.isclose(meta["queue_time"], 1.25)

    disabled = SchedulerReqTimeStats()
    assert disabled.__getstate__() == {}

    state = scheduler.__getstate__()
    assert "metrics_collector" not in state
    assert state["_task22_forward_timing_payload"] is True
    print("PASS: two-hop timing survives without serializing metrics collector")


if __name__ == "__main__":
    main()
