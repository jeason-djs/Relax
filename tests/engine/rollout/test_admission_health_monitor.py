# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from scripts.task22.monitor_admission_health import (
    COARSE_PROGRESS_RE,
    HARD_FAILURE_RE,
    _read_appended,
    monitor,
    update_failure_state,
)
from scripts.task22 import sample_gpu_state


def test_health_monitor_only_matches_hard_failures() -> None:
    assert HARD_FAILURE_RE.search("ray.exceptions.ActorDiedError: actor exited")
    assert HARD_FAILURE_RE.search("NCCL communicator failed")
    assert HARD_FAILURE_RE.search("CUDA error: illegal memory access")
    assert not HARD_FAILURE_RE.search("duplicate_physical_start:0")
    assert not HARD_FAILURE_RE.search("commit_consume_set_mismatch")
    assert not HARD_FAILURE_RE.search("missing_headline_perf:step=14")


def test_health_monitor_progress_is_coarse() -> None:
    assert COARSE_PROGRESS_RE.search(
        "TASK22_FLOW phase=physical_end physical_rollout_id=3 t_begin=1 t_end=2 dur=1"
    )
    assert COARSE_PROGRESS_RE.search("perf 5: {'perf/step_time': 10.0}")
    assert not COARSE_PROGRESS_RE.search('{"event":"scheduler.status"}')
    assert not COARSE_PROGRESS_RE.search('{"event":"request.finished"}')


def test_health_monitor_does_not_replay_ctime_only_changes(tmp_path) -> None:
    path = tmp_path / "driver.log"
    path.write_text("TASK22_FLOW phase=physical_start physical_rollout_id=0\n")

    first, offset, identity = _read_appended(path, 0, None)
    path.touch()
    second, next_offset, next_identity = _read_appended(path, offset, identity)

    assert "physical_start" in first
    assert second == ""
    assert next_offset == offset
    assert next_identity == identity


def test_failure_then_progress_in_one_chunk_recovers() -> None:
    since, reason, events = update_failure_state(
        "CUDA error: illegal memory access\n"
        "TASK22_FLOW phase=physical_end physical_rollout_id=3\n",
        now=10.0,
        hard_failure_since=None,
        hard_failure_reason=None,
    )

    assert since is None
    assert reason is None
    assert events == [
        ("hard_failure_observed", "CUDA error"),
        ("hard_failure_recovered", "CUDA error"),
    ]


def test_progress_then_failure_in_one_chunk_stays_pending() -> None:
    since, reason, events = update_failure_state(
        "TASK22_FLOW phase=physical_end physical_rollout_id=3\n"
        "ray.exceptions.ActorDiedError: actor exited\n",
        now=10.0,
        hard_failure_since=None,
        hard_failure_reason=None,
    )

    assert since == 10.0
    assert reason == "ActorDiedError"
    assert events == [("hard_failure_observed", "ActorDiedError")]


def test_repeated_failure_does_not_extend_grace_deadline() -> None:
    since, reason, _ = update_failure_state(
        "CUDA error: first\n",
        now=10.0,
        hard_failure_since=None,
        hard_failure_reason=None,
    )
    since, reason, _ = update_failure_state(
        "CUDA error: repeated\n",
        now=200.0,
        hard_failure_since=since,
        hard_failure_reason=reason,
    )

    assert since == 10.0
    assert reason == "CUDA error"


def test_monitor_exits_when_leader_is_gone_even_if_process_group_remains(
    monkeypatch, tmp_path
) -> None:
    process_checks: list[tuple[int, str | None]] = []

    def fake_process_exists(pid: int, expected_identity: str | None = None) -> bool:
        process_checks.append((pid, expected_identity))
        return False

    monkeypatch.setattr(
        "scripts.task22.monitor_admission_health._process_exists",
        fake_process_exists,
    )
    monkeypatch.setattr(
        "scripts.task22.monitor_admission_health._process_group_exists",
        lambda process_group_id: True,
    )

    assert (
        monitor(
            tmp_path,
            pid=123,
            process_group_id=123,
            pid_start_identity="leader-start",
            sampler_pid=None,
            sampler_start_identity=None,
            poll_interval=0.001,
            no_progress_timeout=1.0,
            hard_failure_grace=1.0,
            term_timeout=0.0,
        )
        == 0
    )
    assert process_checks == [(123, None)]
    assert '"event": "training_exited"' in (
        tmp_path / "online_monitor.jsonl"
    ).read_text()


def test_gpu_sampler_builds_one_complete_snapshot_before_write(monkeypatch) -> None:
    class Result:
        stdout = (
            "0, 100 MiB, 10 %, 20 %, 30 W\n"
            "1, 100 MiB, 10 %, 20 %, 30 W\n"
            "2, 100 MiB, 10 %, 20 %, 30 W\n"
            "3, 100 MiB, 10 %, 20 %, 30 W\n"
        )

    monkeypatch.setattr(sample_gpu_state.subprocess, "run", lambda *args, **kwargs: Result())

    block = sample_gpu_state.snapshot_block()

    lines = block.splitlines()
    assert lines[0].startswith("20")
    assert lines[1:] == Result.stdout.splitlines()
