# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.engine.rollout.admission import (
    AdmissionMode,
    AdmissionPolicy,
    DebtAwareAdmissionConfig,
    DebtAwareAdmissionController,
    PartitionTransferPlanner,
    UnrecoverableFinalBackfillError,
    config_from_namespace,
    plan_next_admission,
    previous_partition_debt_remaining,
    previous_partition_release_remaining,
    require_final_backfill_deficit,
    split_transfer_counts,
    validate_admission_namespace,
)


def _enabled_config(mode: AdmissionMode = AdmissionMode.ON) -> DebtAwareAdmissionConfig:
    return DebtAwareAdmissionConfig(
        mode=mode,
        min_inflight_groups=4,
        max_inflight_groups=8,
        slack_groups=2,
    )


def _work_conserving_config(mode: AdmissionMode = AdmissionMode.ON) -> DebtAwareAdmissionConfig:
    return DebtAwareAdmissionConfig(
        mode=mode,
        policy=AdmissionPolicy.WORK_CONSERVING,
        min_inflight_groups=12,
        max_inflight_groups=16,
        slack_groups=4,
    )


def _decide(
    controller: DebtAwareAdmissionController,
    *,
    inflight: int,
    debt: int,
    available: int = 20,
    eager: int = 20,
):
    return controller.admit_count(
        inflight_groups=inflight,
        debt_remaining=debt,
        available_groups=available,
        eager_admit_groups=eager,
    )


def test_admission_off_preserves_eager_count() -> None:
    controller = DebtAwareAdmissionController(DebtAwareAdmissionConfig())

    decision = _decide(controller, inflight=0, debt=6, available=14, eager=14)

    assert decision.actual_admit_groups == 14
    assert decision.bypass_reason == "disabled"


def test_admission_shadow_preserves_eager_but_reports_instant_bounded_count() -> None:
    controller = DebtAwareAdmissionController(_enabled_config(AdmissionMode.SHADOW))

    decision = _decide(controller, inflight=0, debt=6, available=14, eager=14)

    assert decision.desired_inflight_groups == 8
    assert decision.bounded_admit_groups == 8
    assert decision.actual_admit_groups == 14


def test_admission_on_tapers_without_refilling_while_debt_closes() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    first = _decide(controller, inflight=0, debt=6, available=14, eager=14)
    second = _decide(controller, inflight=6, debt=4, available=8, eager=0)
    third = _decide(controller, inflight=4, debt=2, available=6, eager=0)

    assert first.actual_admit_groups == 8
    assert first.desired_inflight_groups == 8
    assert second.actual_admit_groups == 0
    assert second.desired_inflight_groups == 6
    assert third.actual_admit_groups == 0
    assert third.desired_inflight_groups == 4


def test_admission_on_refills_when_work_is_dropped_without_closing_debt() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    decision = _decide(controller, inflight=7, debt=6, available=7, eager=0)

    assert decision.desired_inflight_groups == 8
    assert decision.actual_admit_groups == 1


def test_admission_on_uses_normal_max_window_after_debt_closes() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    decision = _decide(controller, inflight=2, debt=0, available=20, eager=0)

    assert decision.desired_inflight_groups == 8
    assert decision.actual_admit_groups == 6


def test_work_conserving_policy_keeps_saturation_floor_after_debt_closes() -> None:
    controller = DebtAwareAdmissionController(_work_conserving_config())

    decision = controller.admit_count(
        inflight_groups=6,
        inflight_requests=48,
        requests_per_group=8,
        debt_remaining=0,
        available_groups=8,
        eager_admit_groups=0,
    )

    assert decision.desired_inflight_groups == 12
    assert decision.actual_admit_groups == 6


def test_work_conserving_policy_refills_partial_group_request_tail() -> None:
    controller = DebtAwareAdmissionController(_work_conserving_config())

    decision = controller.admit_count(
        inflight_groups=12,
        inflight_requests=40,
        requests_per_group=8,
        debt_remaining=5,
        available_groups=4,
        eager_admit_groups=0,
    )

    # Twelve whole group tasks are still alive, but only 40 of their 96
    # request slots remain. Admit useful filler instead of treating every tail
    # group as eight active requests.
    assert decision.desired_inflight_groups == 12
    assert decision.actual_admit_groups == 4


def test_work_conserving_policy_respects_request_ceiling() -> None:
    controller = DebtAwareAdmissionController(_work_conserving_config())

    decision = controller.admit_count(
        inflight_groups=12,
        inflight_requests=124,
        requests_per_group=8,
        debt_remaining=5,
        available_groups=4,
        eager_admit_groups=0,
    )

    assert decision.actual_admit_groups == 0


def test_admission_never_fetches_more_than_useful_available_groups() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    decision = _decide(controller, inflight=2, debt=0, available=3, eager=0)

    assert decision.actual_admit_groups == 3


def test_admission_final_backfill_caps_eager_count_at_remaining_debt() -> None:
    controller = DebtAwareAdmissionController(_enabled_config(), final_backfill=True)

    decision = _decide(controller, inflight=0, debt=6, available=6, eager=14)

    assert decision.actual_admit_groups == 6
    assert decision.bypass_reason == "final_backfill"


def test_admission_final_backfill_never_submits_past_available_debt() -> None:
    controller = DebtAwareAdmissionController(_enabled_config(), final_backfill=True)

    decision = _decide(controller, inflight=1, debt=3, available=2, eager=14)

    assert decision.actual_admit_groups == 2


def test_process_restart_fails_closed_when_final_backfill_deficit_is_lost() -> None:
    # The durable completion flag says train_7 is incomplete, while a restarted
    # rollout process has lost GenerateState.last_step_current_deficit.
    with pytest.raises(
        UnrecoverableFinalBackfillError,
        match=r"durable partition train_7 is incomplete.*in-memory deficit was lost",
    ):
        require_final_backfill_deficit(rollout_id=8, deficit_groups=0)


def test_final_backfill_accepts_preserved_in_memory_deficit() -> None:
    assert require_final_backfill_deficit(rollout_id=8, deficit_groups=3) == 3


def test_admission_fail_open_preserves_eager_count_and_records_reason() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())
    controller.fail_open("no_progress")

    decision = _decide(controller, inflight=0, debt=4, available=6, eager=14)

    assert decision.actual_admit_groups == 14
    assert decision.bypass_reason == "fail_open:no_progress"
    assert controller.metrics()["rollout/admission/failed_open"] == 1.0


def test_previous_partition_debt_does_not_wait_for_transfer_batch_fill() -> None:
    assert previous_partition_debt_remaining(previous_debt_groups=6, completed_groups=0) == 6
    assert previous_partition_debt_remaining(previous_debt_groups=6, completed_groups=5) == 1
    assert previous_partition_debt_remaining(previous_debt_groups=6, completed_groups=6) == 0


def test_previous_partition_release_compatibility_wrapper_uses_logical_debt() -> None:
    assert (
        previous_partition_release_remaining(
            previous_debt_groups=6,
            completed_groups=0,
            transfer_batch_groups=8,
        )
        == 6
    )
    assert (
        previous_partition_release_remaining(
            previous_debt_groups=6,
            completed_groups=6,
            transfer_batch_groups=8,
        )
        == 0
    )


@pytest.mark.parametrize(
    ("batch_groups", "remaining_previous_debt", "expected"),
    [
        (3, 0, (0, 3)),
        (3, 2, (2, 1)),
        (3, 3, (3, 0)),
        (3, 5, (3, 0)),
    ],
)
def test_split_transfer_counts_uses_remaining_previous_debt(
    batch_groups: int,
    remaining_previous_debt: int,
    expected: tuple[int, int],
) -> None:
    assert split_transfer_counts(batch_groups, remaining_previous_debt) == expected


def test_plan_next_admission_matches_task22_fully_async_partial_false_shape() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    initial = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=0,
        transferred_groups=0,
        inflight_groups=0,
        cumulative_submitted_groups=0,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )
    near_close = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=6,
        transferred_groups=6,
        inflight_groups=2,
        cumulative_submitted_groups=8,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )

    assert initial.actual_admit_groups == 8
    assert initial.debt_remaining == 6
    assert near_close.actual_admit_groups == 6
    assert near_close.debt_remaining == 0


def test_plan_next_admission_refills_after_dynamic_filter_drop() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    decision = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=2,
        transferred_groups=2,
        inflight_groups=5,
        cumulative_submitted_groups=7,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )

    assert decision.desired_inflight_groups == 6
    assert decision.actual_admit_groups == 1


def test_plan_next_admission_counts_aborted_group_as_rollout_progress_not_transfer() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    decision = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=6,
        transferred_groups=5,
        inflight_groups=2,
        cumulative_submitted_groups=8,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )

    assert decision.available_groups == 6
    assert decision.debt_remaining == 1
    assert decision.actual_admit_groups == 2


def test_plan_next_admission_closes_debt_then_opens_normal_window_without_overfetch() -> None:
    controller = DebtAwareAdmissionController(_enabled_config())

    first = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=0,
        transferred_groups=0,
        inflight_groups=0,
        cumulative_submitted_groups=0,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )
    close_debt = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=6,
        transferred_groups=6,
        inflight_groups=2,
        cumulative_submitted_groups=8,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )
    normal = plan_next_admission(
        controller,
        target_groups=14,
        progress_groups=8,
        transferred_groups=8,
        inflight_groups=6,
        cumulative_submitted_groups=14,
        previous_debt_groups=6,
        transfer_batch_groups=8,
        eager_fetch_groups=14,
    )

    assert first.actual_admit_groups == 8
    assert close_debt.actual_admit_groups == 6
    assert normal.actual_admit_groups == 0
    assert (
        sum(
            decision.actual_admit_groups
            for decision in (
                first,
                close_debt,
                normal,
            )
        )
        == 14
    )


def test_partition_transfer_planner_closes_previous_at_logical_debt_boundary() -> None:
    planner: PartitionTransferPlanner[str] = PartitionTransferPlanner(
        previous_quota_groups=6,
        current_quota_groups=8,
        preferred_batch_groups=8,
    )

    assert planner.add_completed("p0", now=10.0) == []
    for index in range(1, 5):
        assert planner.add_completed(f"p{index}", now=10.0 + index) == []
    batches = planner.add_completed("p5", now=15.0)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.partition_kind == "previous"
    assert batch.groups == ("p0", "p1", "p2", "p3", "p4", "p5")
    assert batch.flush_reason == "logical_debt_closed"
    assert batch.is_last
    assert batch.buffer_enter_abs == 10.0
    assert batch.flush_trigger_abs == 15.0
    assert planner.previous_remaining_groups == 0


def test_partition_transfer_planner_batches_current_independently() -> None:
    planner: PartitionTransferPlanner[str] = PartitionTransferPlanner(
        previous_quota_groups=2,
        current_quota_groups=8,
        preferred_batch_groups=8,
    )
    planner.add_completed("previous-0", now=1.0)
    planner.add_completed("previous-1", now=2.0)

    for index in range(7):
        assert planner.add_completed(f"current-{index}", now=3.0 + index) == []
    batches = planner.add_completed("current-7", now=10.0)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.partition_kind == "current"
    assert batch.groups == tuple(f"current-{index}" for index in range(8))
    assert batch.flush_reason == "preferred_size"
    assert batch.is_last


def test_partition_transfer_planner_flushes_underfilled_current_tail_without_closing() -> None:
    planner: PartitionTransferPlanner[str] = PartitionTransferPlanner(
        previous_quota_groups=0,
        current_quota_groups=8,
        preferred_batch_groups=8,
    )
    for index in range(3):
        planner.add_completed(f"current-{index}", now=1.0 + index)

    batches = planner.flush_tail(now=5.0)

    assert len(batches) == 1
    assert batches[0].partition_kind == "current"
    assert batches[0].groups == ("current-0", "current-1", "current-2")
    assert batches[0].flush_reason == "physical_close"
    assert not batches[0].is_last


@pytest.mark.parametrize("mode", [AdmissionMode.OFF, AdmissionMode.SHADOW])
def test_non_on_modes_keep_eager_top_up_active_until_target_is_submitted(mode: AdmissionMode) -> None:
    controller = DebtAwareAdmissionController(
        DebtAwareAdmissionConfig(
            mode=mode,
            min_inflight_groups=4 if mode is AdmissionMode.SHADOW else 1,
            max_inflight_groups=8 if mode is AdmissionMode.SHADOW else 1,
            slack_groups=2 if mode is AdmissionMode.SHADOW else 0,
        )
    )

    actual = []
    for cumulative in (0, 3, 6, 9):
        decision = plan_next_admission(
            controller,
            target_groups=8,
            progress_groups=0,
            transferred_groups=0,
            inflight_groups=cumulative,
            cumulative_submitted_groups=cumulative,
            previous_debt_groups=0,
            transfer_batch_groups=8,
            eager_fetch_groups=3,
        )
        actual.append(decision.actual_admit_groups)

    assert actual == [3, 3, 3, 0]


def test_admission_config_defaults_to_off_when_attributes_are_absent() -> None:
    config, error = config_from_namespace(Namespace())

    assert config == DebtAwareAdmissionConfig()
    assert error is None


def test_admission_config_fails_open_when_enabled_settings_are_incomplete() -> None:
    config, error = config_from_namespace(Namespace(partition_critical_admission_mode="on"))

    assert config.mode is AdmissionMode.OFF
    assert error is not None
    assert "missing admission settings" in error


def test_admission_config_fails_open_when_enabled_settings_are_none() -> None:
    config, error = config_from_namespace(
        Namespace(
            partition_critical_admission_mode="shadow",
            partition_critical_admission_min_inflight_groups=None,
            partition_critical_admission_max_inflight_groups=None,
            partition_critical_admission_slack_groups=None,
        )
    )

    assert config.mode is AdmissionMode.OFF
    assert error is not None
    assert "missing admission settings" in error


def test_admission_config_accepts_complete_cli_namespace() -> None:
    config, error = config_from_namespace(
        Namespace(
            partition_critical_admission_mode="on",
            partition_critical_admission_min_inflight_groups=4,
            partition_critical_admission_max_inflight_groups=8,
            partition_critical_admission_slack_groups=2,
        )
    )

    assert error is None
    assert config == _enabled_config()


def test_admission_config_accepts_work_conserving_policy() -> None:
    config, error = config_from_namespace(
        Namespace(
            partition_critical_admission_mode="on",
            partition_critical_admission_policy="work_conserving",
            partition_critical_admission_min_inflight_groups=12,
            partition_critical_admission_max_inflight_groups=16,
            partition_critical_admission_slack_groups=4,
        )
    )

    assert error is None
    assert config == _work_conserving_config()


def test_admission_cli_validation_requires_fully_async_when_enabled() -> None:
    args = Namespace(
        fully_async=False,
        hybrid=False,
        partition_critical_admission_mode="on",
        partition_critical_admission_min_inflight_groups=4,
        partition_critical_admission_max_inflight_groups=8,
        partition_critical_admission_slack_groups=2,
    )

    with pytest.raises(ValueError, match="requires --fully-async"):
        validate_admission_namespace(args)


def test_admission_cli_validation_accepts_fully_async_complete_config() -> None:
    args = Namespace(
        fully_async=True,
        partition_critical_admission_mode="shadow",
        partition_critical_admission_min_inflight_groups=4,
        partition_critical_admission_max_inflight_groups=8,
        partition_critical_admission_slack_groups=2,
    )

    assert validate_admission_namespace(args).mode is AdmissionMode.SHADOW


def test_admission_cli_validation_accepts_hybrid_before_normalization() -> None:
    args = Namespace(
        fully_async=False,
        hybrid=True,
        partition_critical_admission_mode="shadow",
        partition_critical_admission_min_inflight_groups=4,
        partition_critical_admission_max_inflight_groups=8,
        partition_critical_admission_slack_groups=2,
    )

    assert validate_admission_namespace(args).mode is AdmissionMode.SHADOW


def test_work_conserving_cli_validation_rejects_dispatch_capacity_below_ceiling() -> None:
    args = Namespace(
        fully_async=True,
        hybrid=True,
        partition_critical_admission_mode="on",
        partition_critical_admission_policy="work_conserving",
        partition_critical_admission_min_inflight_groups=12,
        partition_critical_admission_max_inflight_groups=16,
        partition_critical_admission_slack_groups=4,
        sglang_server_concurrency=16,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        n_samples_per_prompt=8,
    )

    with pytest.raises(ValueError, match="client request capacity"):
        validate_admission_namespace(args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_inflight_groups": 0, "max_inflight_groups": 8}, "must be positive"),
        ({"min_inflight_groups": 8, "max_inflight_groups": 4}, "must be >="),
        ({"slack_groups": -1}, "must be non-negative"),
    ],
)
def test_admission_config_rejects_unsafe_windows(kwargs: dict, message: str) -> None:
    config_kwargs = {
        "mode": AdmissionMode.ON,
        "min_inflight_groups": 4,
        "max_inflight_groups": 8,
        "slack_groups": 2,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        DebtAwareAdmissionController(DebtAwareAdmissionConfig(**config_kwargs))


@pytest.mark.parametrize(
    "field",
    ["inflight_groups", "debt_remaining", "available_groups", "eager_admit_groups"],
)
def test_admission_rejects_negative_runtime_state(field: str) -> None:
    controller = DebtAwareAdmissionController(_enabled_config())
    values = {
        "inflight_groups": 0,
        "debt_remaining": 1,
        "available_groups": 1,
        "eager_admit_groups": 1,
    }
    values[field] = -1

    with pytest.raises(ValueError, match=field):
        controller.decide(**values)
