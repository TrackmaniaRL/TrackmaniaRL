"""Race outcomes inside on-policy segments must survive collection logging."""

from trackmaniarl.core.collector import CollectionResult
from trackmaniarl.core.data import EpisodeArtifact
from trackmaniarl.core.training_support import rollout_metrics
from trackmaniarl.observability.wandb_metrics import _event_metrics


def test_rollout_counts_finishes_before_the_final_boundary() -> None:
    telemetry = [
        {"termination_reason": "finished", "telemetry_skipped_frames": 2},
        {"termination_reason": None, "step_race_time_ms": 140},
        {"termination_reason": "no_progress"},
        {"termination_reason": "finished"},
        {"termination_reason": "rollout_boundary"},
    ]
    result = CollectionResult(
        5,
        7.0,
        EpisodeArtifact("rollout", telemetry, [], [], []),
        completed_episodes=4,
    )
    metrics = rollout_metrics(result)
    assert metrics["finished_episodes"] == 2
    assert metrics["completed_episodes"] == 3
    assert metrics["rollout_boundaries"] == 1
    assert metrics["step_race_time_ms_max"] == 140
    assert metrics["telemetry_skipped_frames_total"] == 2
    remote = _event_metrics("train/rollout", {"reward": 7.0, "transitions": 5, **metrics})
    assert remote["rollout/finished_episodes"] == 2
    assert remote["rollout/completed_episodes"] == 3
    assert remote["rollout/transitions"] == 5
    assert remote["rollout/reward"] == 7.0
    assert "episode/finished" not in remote


def test_rollout_does_not_invent_an_ending_for_an_open_race() -> None:
    result = CollectionResult(
        1,
        0.0,
        EpisodeArtifact("rollout", [{"termination_reason": None}], [], [], []),
        completed_episodes=0,
    )
    metrics = rollout_metrics(result)
    assert metrics["finished_episodes"] == 0
    assert metrics["completed_episodes"] == 0
    assert metrics["rollout_boundaries"] == 0
