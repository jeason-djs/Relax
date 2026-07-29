#!/usr/bin/env python3
"""No-GPU regression test for Task 22 scheduler shape records."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger


def _request(rid: str, seq_len: int, prompt_len: int, output_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        seqlen=seq_len,
        origin_input_ids=list(range(prompt_len)),
        output_ids=list(range(output_len)),
    )


def main() -> None:
    decode_request = _request("task22:p5:wold_debt:g8:s64:a1:abc", 6100, 2048, 4052)
    extend_request = _request("task22:p5:wcurrent:g9:s72:a0:def", 2100, 2048, 52)
    queued_request = _request("task22:p5:wcurrent:g10:s80:a0:ghi", 2048, 2048, 0)
    batch = SimpleNamespace(
        forward_mode="ForwardMode.MIXED",
        reqs=[decode_request, extend_request],
        decoding_reqs=[decode_request],
    )

    status_logger = SchedulerStatusLogger.__new__(SchedulerStatusLogger)
    status_logger.loggers = []
    status_logger.dump_interval = 0.0
    status_logger.last_dump_time = 0.0
    status_logger.rank = 0
    captured = {}

    def capture(_loggers, event, data):
        captured["event"] = event
        captured["data"] = data

    with patch("sglang.srt.utils.scheduler_status_logger.log_json", capture):
        status_logger.maybe_dump(batch, [queued_request])

    data = captured["data"]
    assert captured["event"] == "scheduler.status"
    assert data["forward_mode"] == "ForwardMode.MIXED"
    assert data["running_rids"] == [decode_request.rid, extend_request.rid]
    assert data["running_seq_lens"] == [6100, 2100]
    assert data["running_origin_input_lens"] == [2048, 2048]
    assert data["running_output_lens"] == [4052, 52]
    assert data["decoding_rids"] == [decode_request.rid]
    assert data["queued_rids"] == [queued_request.rid]
    assert data["queued_origin_input_lens"] == [2048]
    assert data["queued_output_lens"] == [0]
    print("PASS: scheduler status contains request IDs and batch/context shapes")


if __name__ == "__main__":
    main()
