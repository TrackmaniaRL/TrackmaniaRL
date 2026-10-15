from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.observability.wandb_tracker_support import (
    _FailingRun,
    _FakeRun,
    _FakeWandb,
    _install_wandb,
    _metric_event,
    _offline_pretraining_payload,
    _Scenario,
    _scenario,
    _tracker,
)
from trackmaniarl.observability.trackers import WandbTracker


def _assert_update_axes(run: _FakeRun) -> None:
    update = _metric_event(run, "learner/loss_total")
    assert update["trainer/update"] == 7
    assert update["env/transitions"] == 512
    assert isinstance(update["runtime/elapsed_s"], float)
    assert update["pipeline/policy_lag_updates"] == 4.0
    assert update["health/active_actors"] == 1
    assert update["health/spool_bytes"] == 128
    assert update["health/wal_pending_rows"] == 3
    assert update["health/wal_pending_payload_bytes"] == 1_024


def test_rollout_uses_cumulative_transition_axis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _FakeRun()
    tracker, _ = _tracker(monkeypatch, tmp_path, run)
    tracker.log(
        "train/rollout",
        {"finished_episodes": 2, "transitions": 2048, "counters": {"transitions": 4096}},
        step=1,
    )
    tracker.close()
    event = _metric_event(run, "rollout/finished_episodes")
    assert event["rollout/finished_episodes"] == 2
    assert event["rollout/transitions"] == 2048
    assert event["env/transitions"] == 4096


def _assert_bounded_metric_catalog(run: _FakeRun) -> None:
    update = _metric_event(run, "learner/loss_total")
    assert update["learner/action_batch_entropy"] == 0.8
    assert update["learner/bootstrap_discount_mean"] == 0.95
    assert update["learner/n_step_return_mean"] == 1.5
    assert update["learner/q_target_mean"] == 2.25
    assert update["learner/demo_sample_fraction"] == 0.25
    assert update["learner/demo_accuracy"] == 0.8
    assert update["learner/demo_steering_switch_fraction"] == 0.2
    assert update["learner/demo_steering_switch_accuracy"] == 0.6
    assert update["learner/demo_steady_accuracy"] == 0.85
    assert update["replay/demo_sample_fraction"] == 0.25
    assert update["replay/expert_demo_active_fraction"] == 0.02
    assert update["replay/expert_demo_sample_fraction"] == 0.25
    assert update["replay/expert_demo_target_fraction"] == 0.2
    assert "learner/initialized_exact_tensors" not in update
    assert all(key.count("/") <= 1 for item in run.logged for key in item)
    assert not any("progress_bin" in key for item in run.logged for key in item)


def _assert_episode_metrics(run: _FakeRun) -> None:
    episodes = [item for item in run.logged if "episode/return" in item]
    assert [item["env/episode"] for item in episodes] == [1, 2]
    assert episodes[0]["episode/reward_time"] == -1.5
    assert episodes[0]["episode/control_gas_fraction"] == 0.75
    assert episodes[0]["episode/termination_time_limit"] == 1.0
    assert episodes[0]["episode/timing_step_race_ms_p99"] == 51.0
    assert episodes[0]["episode/telemetry_skipped_frames_total"] == 4.0
    assert episodes[0]["episode/controller_apply_ms_mean"] == 1.5


def test_wandb_tracker_maps_training_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _scenario(monkeypatch, tmp_path).run
    _assert_update_axes(run)
    _assert_bounded_metric_catalog(run)
    _assert_episode_metrics(run)


def test_wandb_tracker_maps_offline_pretraining_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _FakeRun()
    tracker, _ = _tracker(monkeypatch, tmp_path, run)
    tracker.log("train/offline_pretrain", _offline_pretraining_payload(), step=1_000)
    tracker.close()

    offline = _metric_event(run, "learner/loss_total")
    assert offline["trainer/update"] == 1_000
    assert offline["learner/loss_objectives"] == 0.2
    assert offline["learner/demo_accuracy"] == 0.8
    assert offline["replay/expert_demo_sample_fraction"] == 1.0


def _assert_evaluation_summary(run: _FakeRun) -> None:
    evaluation = _metric_event(run, "evaluation/finish_rate")
    assert evaluation["eval/batch"] == 1
    assert evaluation["evaluation/control_gas_fraction_mean"] == 0.8
    assert evaluation["evaluation/step_race_time_ms_max"] == 75.0
    assert evaluation["evaluation/step_race_time_measurement_count"] == 40.0
    assert evaluation["evaluation/step_race_time_measurements_valid"] == 1.0
    assert evaluation["evaluation/telemetry_skipped_frames_total"] == 6.0


def _assert_evaluation_suite(run: _FakeRun) -> None:
    suite = next(item for item in run.logged if item.get("eval/batch") == 2)
    _assert_suite_outcomes(suite)
    _assert_suite_timing(suite)


def _assert_suite_outcomes(suite: dict[str, object]) -> None:
    assert suite["evaluation/finish_rate"] == 0.5
    assert suite["evaluation/finish_time_mean_s"] == 35.0
    assert suite["evaluation/finish_time_median_s"] == 35.0
    assert suite["evaluation/finish_time_best_s"] == 35.0
    assert suite["evaluation/reward"] == 3.0
    assert suite["evaluation/crash_rate"] == 0.5
    assert suite["evaluation/sub_36_rate"] == 0.5
    assert suite["evaluation/sub_38_rate"] == 0.5
    assert suite["evaluation/sub_40_rate"] == 0.5


def _assert_suite_timing(suite: dict[str, object]) -> None:
    assert suite["evaluation/controller_apply_ms"] == 1.5
    assert suite["evaluation/telemetry_wait_ms"] == pytest.approx(98 / 3)
    assert suite["evaluation/control_brake_tap_fraction"] == pytest.approx(1 / 3)
    assert suite["evaluation/step_race_time_ms_p99"] == 52.0
    assert suite["evaluation/step_race_time_ms_max"] == 60.0
    assert suite["evaluation/step_race_time_measurement_count"] == 2.0
    assert suite["evaluation/step_race_time_expected_measurement_count"] == 3.0
    assert suite["evaluation/step_race_time_measurements_valid"] == 0.0
    assert suite["evaluation/telemetry_skipped_frames_total"] == 3.0


def test_wandb_tracker_maps_evaluation_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _scenario(monkeypatch, tmp_path).run
    _assert_evaluation_summary(run)
    _assert_evaluation_suite(run)


def test_wandb_tracker_maps_evaluation_leaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _scenario(monkeypatch, tmp_path).run
    best = _metric_event(run, "checkpoint_best/finish_time_best_s")
    fastest = _metric_event(run, "checkpoint_fastest/finish_time_best_s")

    assert best["trainer/update"] == 9
    assert best["checkpoint_best/finish_rate"] == 1.0
    assert best["checkpoint_best/release_qualified"] == 1.0
    assert best["checkpoint_best/step_race_time_measurements_valid"] == 1.0
    assert "checkpoint_best/path" not in best
    assert fastest["checkpoint_fastest/exact_policy"] == 1.0
    assert fastest["checkpoint_fastest/shared_with_reliable"] == 0.0


def test_wandb_tracker_maps_expert_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expert = _metric_event(_scenario(monkeypatch, tmp_path).run, "expert/exact_action_accuracy")

    assert expert["expert/transitions"] == 16_798
    assert expert["expert/demonstrations_count"] == 23
    assert expert["expert/count"] == 16_798.0
    assert expert["expert/steering_bin_accuracy"] == 0.75
    assert expert["expert/steering_switch_recall"] == 0.6
    assert expert["expert/expert_steering_switch_step_accuracy"] == 0.4
    assert expert["expert/expert_steering_steady_step_accuracy"] == 0.8
    assert "expert/progress_bins" not in expert


def _assert_configuration(scenario: _Scenario) -> None:
    definitions = dict(scenario.run.definitions)
    assert definitions["checkpoint_best/*"] == {"step_metric": "trainer/update"}
    assert definitions["checkpoint_fastest/*"] == {"step_metric": "trainer/update"}
    assert definitions["expert/*"] == {"step_metric": "expert/transitions"}
    assert definitions["learner/*"] == {"step_metric": "trainer/update"}
    assert definitions["episode/*"] == {"step_metric": "env/episode"}
    assert definitions["rollout/*"] == {"step_metric": "env/transitions"}
    assert definitions["evaluation/*"] == {"step_metric": "eval/batch"}
    assert definitions["health/*"] == {"step_metric": "runtime/elapsed_s"}
    assert scenario.wandb.init_kwargs["name"] == "run-17"
    assert scenario.wandb.init_kwargs["group"] == "run-17"
    settings = scenario.wandb.init_kwargs["settings"]
    assert isinstance(settings, _FakeWandb.Settings)
    assert settings.kwargs["console"] == "wrap"
    config = scenario.wandb.init_kwargs["config"]
    assert isinstance(config, dict)
    assert config["observability/attempt_id"] == "attempt-2"
    assert config["observability/resumed_from"] == "checkpoint-1"
    assert scenario.run.finished == [0]


def test_wandb_tracker_configuration_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _assert_configuration(_scenario(monkeypatch, tmp_path))
    with pytest.raises(ValueError, match="queue_size must be positive"):
        WandbTracker("project", queue_size=0)


def test_wandb_tracker_marks_remote_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _FailingRun()
    _install_wandb(monkeypatch, run)
    tracker = WandbTracker("project", run_dir=str(tmp_path))

    tracker.log("train/update", {"loss/total": 1.0}, step=1)
    tracker.close()

    assert run.finished == [1]
