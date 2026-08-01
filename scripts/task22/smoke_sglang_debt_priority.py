#!/usr/bin/env python3
"""Verify SGLang waiting-queue debt priority without running Relax training."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise TypeError("SGLang response must be a JSON object")
    return result


async def _request(
    base_url: str,
    *,
    name: str,
    input_ids: list[int],
    priority: int,
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "rid": f"task22-debt-priority-{name}",
        "input_ids": input_ids,
        "priority": priority,
        "return_logprob": False,
        "sampling_params": {
            "temperature": 0,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }
    dispatched_at = time.time()
    response = await asyncio.to_thread(
        _post_json,
        f"{base_url.rstrip('/')}/generate",
        payload,
        timeout,
    )
    completed_at = time.time()
    output_ids = response.get("output_ids")
    return {
        "name": name,
        "priority": priority,
        "max_new_tokens": max_new_tokens,
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "request_wall_s": completed_at - dispatched_at,
        "output_tokens": len(output_ids) if isinstance(output_ids, list) else None,
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {record["name"]: record for record in records}
    required = {"blocker", "fresh", "old_debt"}
    if set(by_name) != required:
        raise ValueError(f"expected records {sorted(required)}, got {sorted(by_name)}")
    completion_order = [record["name"] for record in sorted(records, key=lambda record: record["completed_at"])]
    checks = {
        "fresh_dispatched_before_old_debt": (by_name["fresh"]["dispatched_at"] < by_name["old_debt"]["dispatched_at"]),
        "blocker_active_when_old_debt_dispatched": (
            by_name["blocker"]["completed_at"] > by_name["old_debt"]["dispatched_at"]
        ),
        "running_blocker_not_preempted": completion_order[0] == "blocker",
        "queued_old_debt_overtakes_fresh": completion_order.index("old_debt") < completion_order.index("fresh"),
        "all_requests_return_exact_work": all(
            record["output_tokens"] == record["max_new_tokens"] for record in records
        ),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "completion_order": completion_order,
        "records": records,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    blocker = asyncio.create_task(
        _request(
            args.base_url,
            name="blocker",
            input_ids=[100, 101, 102, 103],
            priority=0,
            max_new_tokens=args.blocker_tokens,
            timeout=args.timeout,
        )
    )
    await asyncio.sleep(args.blocker_lead_s)
    fresh = asyncio.create_task(
        _request(
            args.base_url,
            name="fresh",
            input_ids=[110, 111, 112, 113],
            priority=0,
            max_new_tokens=args.follower_tokens,
            timeout=args.timeout,
        )
    )
    await asyncio.sleep(args.queue_gap_s)
    old_debt = asyncio.create_task(
        _request(
            args.base_url,
            name="old_debt",
            input_ids=[120, 121, 122, 123],
            priority=1,
            max_new_tokens=args.follower_tokens,
            timeout=args.timeout,
        )
    )
    return evaluate(await asyncio.gather(blocker, fresh, old_debt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--blocker-tokens", type=int, default=2048)
    parser.add_argument("--follower-tokens", type=int, default=64)
    parser.add_argument("--blocker-lead-s", type=float, default=0.25)
    parser.add_argument("--queue-gap-s", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.blocker_tokens <= 0 or args.follower_tokens <= 0:
        parser.error("token counts must be positive")
    if args.blocker_lead_s <= 0 or args.queue_gap_s <= 0 or args.timeout <= 0:
        parser.error("timing values must be positive")
    try:
        result = asyncio.run(run(args))
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        result = {"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["verdict"] == "PASS" else 4)


if __name__ == "__main__":
    main()
