#!/usr/bin/env python3
"""No-GPU smoke test for SGLang timing transport and shape fields."""

from __future__ import annotations

import inspect
import math
import pickle

from sglang.srt.observability import scheduler_metrics_mixin
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.utils import scheduler_status_logger


def _round_trip(value: SchedulerReqTimeStats) -> SchedulerReqTimeStats:
    return pickle.loads(pickle.dumps(value))


def _has_forwarding_marker(value: SchedulerReqTimeStats) -> bool:
    return bool(
        getattr(value, "_relax_forward_timing_payload", False)
        or getattr(value, "_task22_forward_timing_payload", False)
    )


def main() -> None:
    scheduler = SchedulerReqTimeStats()
    scheduler.enable_metrics = True
    scheduler.wait_queue_entry_time = 100.0
    scheduler.forward_entry_time = 101.25
    scheduler.prefill_finished_time = 103.5

    first_hop = _round_trip(scheduler)
    second_hop = _round_trip(first_hop)
    assert _has_forwarding_marker(first_hop)
    assert _has_forwarding_marker(second_hop)
    assert math.isclose(second_hop.wait_queue_entry_time, 100.0)
    assert math.isclose(second_hop.forward_entry_time, 101.25)
    assert math.isclose(second_hop.prefill_finished_time, 103.5)
    output_meta = second_hop.convert_to_output_meta_info()
    assert output_meta["forward_entry_time"] > 1_000_000_000
    assert output_meta["prefill_finished_time"] > output_meta["forward_entry_time"]
    assert math.isclose(output_meta["queue_time"], 1.25)

    status_source = inspect.getsource(scheduler_status_logger.SchedulerStatusLogger)
    metrics_source = inspect.getsource(scheduler_metrics_mixin.SchedulerMetricsMixin)
    for field in (
        "forward_mode",
        "running_seq_lens",
        "running_origin_input_lens",
        "running_output_lens",
        "decoding_rids",
        "queued_origin_input_lens",
        "queued_output_lens",
    ):
        assert field in status_source
    assert (
        "RELAX_REQUEST_SHAPE_OBSERVABILITY" in metrics_source
        or "TASK22_REQUEST_SHAPE_PREFILL_STATUS" in metrics_source
    )
    print("PASS: SGLang request timing and scheduler shape observability")


if __name__ == "__main__":
    main()
