#!/usr/bin/env python3
"""Coarse paid-run health monitor for clean Task 22 admission A/B runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path


HARD_FAILURE_RE = re.compile(
    r"(?:"
    r"\b(?:out[ -]of[ -]memory|OutOfMemoryError)\b|"
    r"\bNCCL\b[^\n]*(?:error|abort|fail|timeout)|"
    r"\b(?:RayActorError|ActorDiedError)\b|"
    r"\bengine failed\b|"
    r"\bJob failed\b|"
    r"\bCUDA error\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
COARSE_PROGRESS_RE = re.compile(
    r"(?:"
    r"TASK22_FLOW phase=(?:physical_start|physical_end|partition_close)\b|"
    r"\b(?:rollout|step|perf) \d+:"
    r")"
)


def _append_event(path: Path, event: str, **fields: object) -> None:
    payload = {"timestamp": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()


def _process_start_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    suffix = stat.rsplit(")", 1)[1].split()
    return suffix[19] if len(suffix) > 19 else None


def _process_exists(pid: int, expected_identity: str | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if expected_identity is not None:
        return _process_start_identity(pid) == expected_identity
    return True


def _process_group_exists(process_group_id: int | None) -> bool:
    if process_group_id is None:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(
    pid: int,
    process_group_id: int | None,
    *,
    expected_identity: str | None,
    term_timeout: float,
) -> None:
    if expected_identity is not None and _process_exists(pid):
        if _process_start_identity(pid) != expected_identity:
            raise RuntimeError(f"refusing to stop reused pid {pid}")
    target = process_group_id if process_group_id is not None else pid
    try:
        if process_group_id is not None:
            os.killpg(target, signal.SIGTERM)
        else:
            os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline:
        if not (
            _process_group_exists(process_group_id)
            if process_group_id is not None
            else _process_exists(pid, expected_identity)
        ):
            return
        time.sleep(0.1)
    try:
        if process_group_id is not None:
            os.killpg(target, signal.SIGKILL)
        else:
            os.kill(target, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_appended(path: Path, offset: int, identity: tuple[int, int] | None) -> tuple[str, int, tuple[int, int] | None]:
    if not path.is_file():
        return "", offset, identity
    with path.open("rb") as source:
        stat = os.fstat(source.fileno())
        current_identity = (stat.st_dev, stat.st_ino)
        if identity is not None and (current_identity != identity or stat.st_size < offset):
            offset = 0
        source.seek(offset)
        chunk = source.read(max(stat.st_size - offset, 0))
    return chunk.decode("utf-8", errors="replace"), offset + len(chunk), current_identity


def update_failure_state(
    addition: str,
    *,
    now: float,
    hard_failure_since: float | None,
    hard_failure_reason: str | None,
) -> tuple[float | None, str | None, list[tuple[str, str | None]]]:
    events: list[tuple[str, str | None]] = []
    for line in addition.splitlines():
        progress = COARSE_PROGRESS_RE.search(line)
        failure = HARD_FAILURE_RE.search(line)
        if failure is not None:
            if hard_failure_since is None:
                hard_failure_since = now
            hard_failure_reason = failure.group(0).strip()
            events.append(("hard_failure_observed", hard_failure_reason))
        if progress is not None and hard_failure_since is not None:
            events.append(("hard_failure_recovered", hard_failure_reason))
            hard_failure_since = None
            hard_failure_reason = None
    return hard_failure_since, hard_failure_reason, events


def monitor(
    run_dir: Path,
    *,
    pid: int,
    process_group_id: int | None,
    pid_start_identity: str | None,
    sampler_pid: int | None,
    sampler_start_identity: str | None,
    poll_interval: float,
    no_progress_timeout: float,
    hard_failure_grace: float,
    term_timeout: float,
) -> int:
    event_log = run_dir / "online_monitor.jsonl"
    driver_log = run_dir / "driver.log"
    expected_identity = pid_start_identity or _process_start_identity(pid)
    offset = 0
    file_identity: tuple[int, int] | None = None
    last_progress = time.monotonic()
    hard_failure_since: float | None = None
    hard_failure_reason: str | None = None
    sampler_warning_reported = False
    _append_event(
        event_log,
        "health_monitor_started",
        pid=pid,
        pid_start_identity=expected_identity,
        sampler_pid=sampler_pid,
        sampler_start_identity=sampler_start_identity,
    )

    while True:
        now = time.monotonic()
        pid_present = _process_exists(pid)
        identity_changed = (
            pid_present
            and expected_identity is not None
            and _process_start_identity(pid) != expected_identity
        )
        running = (
            pid_present
            or _process_group_exists(process_group_id)
        ) and not identity_changed
        if not running:
            _append_event(event_log, "training_exited")
            return 0
        if identity_changed:
            _append_event(
                event_log,
                "health_monitor_warning",
                reason="pid_identity_changed",
                pid=pid,
            )
            return 0

        if sampler_pid is not None and not _process_exists(sampler_pid, sampler_start_identity):
            if not sampler_warning_reported:
                _append_event(
                    event_log,
                    "health_monitor_warning",
                    reason="gpu_sampler_not_running",
                    sampler_pid=sampler_pid,
                )
                sampler_warning_reported = True

        addition, offset, file_identity = _read_appended(driver_log, offset, file_identity)
        if addition:
            if COARSE_PROGRESS_RE.search(addition):
                last_progress = now
            hard_failure_since, hard_failure_reason, state_events = update_failure_state(
                addition,
                now=now,
                hard_failure_since=hard_failure_since,
                hard_failure_reason=hard_failure_reason,
            )
            for event, reason in state_events:
                _append_event(
                    event_log,
                    event,
                    reason=reason,
                    **(
                        {"grace_seconds": hard_failure_grace}
                        if event == "hard_failure_observed"
                        else {}
                    ),
                )

        if hard_failure_since is not None and now - hard_failure_since >= hard_failure_grace:
            marker = run_dir / "logs" / ".health_monitor_stopped_training"
            marker.write_text(f"{hard_failure_reason}\n", encoding="utf-8")
            _stop_process_group(
                pid,
                process_group_id,
                expected_identity=expected_identity,
                term_timeout=term_timeout,
            )
            _append_event(
                event_log,
                "training_stop_requested",
                reason=hard_failure_reason,
            )
            return 4

        if now - last_progress >= no_progress_timeout:
            marker = run_dir / "logs" / ".health_monitor_stopped_training"
            marker.write_text("no_driver_progress\n", encoding="utf-8")
            _stop_process_group(
                pid,
                process_group_id,
                expected_identity=expected_identity,
                term_timeout=term_timeout,
            )
            _append_event(
                event_log,
                "training_stop_requested",
                reason=f"no_driver_progress:{no_progress_timeout:g}s",
            )
            return 4
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--process-group-id", type=int)
    parser.add_argument("--pid-start-identity")
    parser.add_argument("--sampler-pid", type=int)
    parser.add_argument("--sampler-start-identity")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--no-progress-timeout", type=float, default=1200.0)
    parser.add_argument("--hard-failure-grace", type=float, default=60.0)
    parser.add_argument("--term-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if (
        args.pid <= 0
        or args.poll_interval <= 0
        or args.no_progress_timeout <= 0
        or args.hard_failure_grace < 0
        or args.term_timeout < 0
    ):
        parser.error("invalid health monitor contract")
    raise SystemExit(
        monitor(
            args.run_dir,
            pid=args.pid,
            process_group_id=args.process_group_id,
            pid_start_identity=args.pid_start_identity,
            sampler_pid=args.sampler_pid,
            sampler_start_identity=args.sampler_start_identity,
            poll_interval=args.poll_interval,
            no_progress_timeout=args.no_progress_timeout,
            hard_failure_grace=args.hard_failure_grace,
            term_timeout=args.term_timeout,
        )
    )


if __name__ == "__main__":
    main()
