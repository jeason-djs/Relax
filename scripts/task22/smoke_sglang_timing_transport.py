#!/usr/bin/env python3
"""One-request GPU smoke for SGLang timing metadata transport."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import time


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from sglang import Engine

    engine = Engine(
        model_path=args.model_path,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.8,
        skip_server_warmup=True,
        disable_cuda_graph=True,
        show_time_cost=True,
        enable_metrics=True,
    )
    request_id = "task22:smoke:rid-roundtrip"
    try:
        result = engine.generate(
            prompt="Compute 17 + 25. Answer with only the number.",
            sampling_params={"temperature": 0.0, "max_new_tokens": 4},
            rid=request_id,
        )
    finally:
        engine.shutdown()

    meta = result["meta_info"]
    forward = meta.get("forward_entry_time")
    prefill = meta.get("prefill_finished_time")
    queue = meta.get("queue_time")
    record = {
        "captured_at_epoch": time(),
        "forward_entry_time": forward,
        "prefill_finished_time": prefill,
        "queue_time": queue,
        "request_id": request_id,
        "returned_request_id": meta.get("id"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "text": result.get("text"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    assert _finite_number(forward) and float(forward) > 1_000_000_000
    assert _finite_number(prefill) and float(prefill) >= float(forward)
    assert _finite_number(queue) and float(queue) >= 0.0
    assert meta.get("id") == request_id
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    print("PASS: real SGLang response contains valid queue/prefill timing and matching request ID")


if __name__ == "__main__":
    main()
