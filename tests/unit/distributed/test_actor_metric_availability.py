"""Missing policy measurements must survive collection, transport and reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from math import log
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.integration.runtime.test_distributed_submission import _base_payload
from tests.unit.learning._algorithm_fixtures import DiscreteSacModel
from trackmaniarl.algorithms.stable_discrete_soft_actor_critic import _DiscretePolicy
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.distributed.actor_collection import (
    CollectionContext,
    PolicyStep,
    TrainingEpisode,
    _record_training_step,
)
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics, summarize_episode
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator_evaluation import _evaluation_batch_stats
from trackmaniarl.distributed.coordinator_ingest import log_episode
from trackmaniarl.distributed.coordinator_validation import (
    _validate_episode_summary,
    _validate_submit_payload,
)
from trackmaniarl.experiments.evaluation import EvaluationResult
from trackmaniarl.trackmania.diagnostics import (
    ProgressBinDiagnostics,
    ProgressDiagnosticRecord,
    aggregate_progress_bins,
)


class _EpsilonPolicy:
    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        return 0

    def set_exploration_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon


def _episode_summary(metrics: EpisodeMetrics, steps: int = 1) -> dict[str, Any]:
    info = metrics.summary_info(0.3, 7, steps)
    return {**summarize_episode(0.0, info, steps), "episode_id": "actor/session/episode"}


def test_sd_sac_marks_unmeasured_q_and_unused_epsilon_as_null() -> None:
    policy = _DiscretePolicy(DiscreteSacModel().actor, torch.device("cpu"))
    metrics = EpisodeMetrics.from_policy(policy)
    metrics.record_policy(policy, 0)
    action, policy_info = policy.act_with_info(torch.zeros(4), PolicyMode.EVALUATION)
    metrics.record_diagnostics(action, policy, {}, policy_info=policy_info)
    summary = _episode_summary(metrics)

    assert summary["exploration_epsilon"] is None
    assert summary["exploration_epsilon_used"] is False
    assert summary["q_margin/mean"] is None
    assert summary["q_margin/min"] is None
    assert summary["q_margin/start_mean"] is None
    assert summary["q_margin/sample_count"] == 0
    assert summary["progress_bin/00_005/q_margin_mean"] is None
    assert summary["progress_bin/00_005/q_max_mean"] is None
    assert (
        summary["policy/behavior_entropy_nats_mean"]
        == policy_info["_trackmaniarl_behavior_entropy"]
    )
    assert summary["policy/behavior_entropy_sample_count"] == 1


def test_real_zero_q_margin_and_epsilon_policy_remain_measured() -> None:
    policy = _EpsilonPolicy()
    policy.set_exploration_epsilon(0.3)
    policy.last_q_margin = 0.0
    policy.last_q_max = 0.0
    metrics = EpisodeMetrics.from_policy(policy)
    metrics.record_policy(policy, 0)
    metrics.record_diagnostics(0, policy, {})
    summary = _episode_summary(metrics)

    assert summary["exploration_epsilon"] == 0.3
    assert summary["exploration_epsilon_used"] is True
    assert policy.epsilon == 0.3
    assert summary["q_margin/mean"] == 0.0
    assert summary["q_margin/min"] == 0.0
    assert summary["q_margin/sample_count"] == 1
    assert summary["progress_bin/00_005/q_margin_mean"] == 0.0
    assert summary["progress_bin/00_005/q_max_mean"] == 0.0
    assert summary["policy/behavior_entropy_nats_mean"] is None
    assert summary["policy/behavior_entropy_sample_count"] == 0


def test_margin_observed_only_late_does_not_invent_a_start_measurement() -> None:
    metrics = EpisodeMetrics.from_policy(object())
    metrics.record_policy(SimpleNamespace(last_q_margin=0.0), 50)
    summary = _episode_summary(metrics)
    assert summary["q_margin/mean"] == 0.0
    assert summary["q_margin/start_mean"] is None
    assert summary["q_margin/start_sample_count"] == 0


def test_conditional_entropy_nats_is_separate_from_normalized_action_histogram() -> None:
    metrics = EpisodeMetrics.from_policy(object())
    for action in (0, 1):
        metrics.record_diagnostics(action, object(), {"_trackmaniarl_behavior_entropy": 0.25})
    summary = _episode_summary(metrics, 2)
    prefix = "progress_bin/00_005/"
    assert summary["policy/behavior_entropy_nats_mean"] == 0.25
    assert summary["policy/behavior_entropy_sample_count"] == 2
    assert summary[prefix + "action_histogram_entropy_normalized"] == pytest.approx(
        log(2) / log(78)
    )
    assert (
        summary[prefix + "action_entropy"]
        == summary[prefix + "action_histogram_entropy_normalized"]
    )


@pytest.mark.parametrize("entropy", [None, -0.1, float("nan"), float("inf"), True, "0.1"])
def test_unavailable_or_invalid_entropy_is_not_reported_as_zero(entropy: Any) -> None:
    metrics = EpisodeMetrics.from_policy(object())
    metrics.record_diagnostics(0, object(), {"_trackmaniarl_behavior_entropy": entropy})
    summary = _episode_summary(metrics)
    assert summary["policy/behavior_entropy_nats_mean"] is None
    assert summary["policy/behavior_entropy_sample_count"] == 0


def test_collection_passes_behavior_info_without_overwriting_control_measurements() -> None:
    policy = _DiscretePolicy(DiscreteSacModel().actor, torch.device("cpu"))
    metrics = EpisodeMetrics.from_policy(policy)
    context = CollectionContext(
        SimpleNamespace(_should_flush=lambda *_: False),
        None,
        SimpleNamespace(transform_observation=lambda value: value),
    )
    state = TrainingEpisode(np.zeros(4), policy, 0.3, 7, "actor/session/episode", metrics)
    policy_info = {"_trackmaniarl_behavior_entropy": 0.0, "control_gas": 999.0}
    info = {"control_gas": 1.0, "step_race_time_ms": 50.0}
    step = PolicyStep(0, 3, policy_info, np.ones(4), 1.0, False, False, info, 0.001)

    _record_training_step(context, state, step)

    summary = _episode_summary(metrics)
    assert summary["policy/behavior_entropy_nats_mean"] == 0.0
    assert summary["policy/behavior_entropy_sample_count"] == 1
    assert summary["control/gas_fraction"] == 1.0
    assert info == {"control_gas": 1.0, "step_race_time_ms": 50.0}
    transition = context.buffers.transitions[0]
    assert transition.action == 3
    assert transition.info["_trackmaniarl_behavior_entropy"] == 0.0
    np.testing.assert_array_equal(transition.next_observation, np.ones(4))


def test_nullable_summary_survives_wire_validation_and_ingest_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary = _episode_summary(EpisodeMetrics.from_policy(object()))
    codec = WireCodec(1024 * 1024)
    payload = _base_payload(0)
    payload["episodes"] = [summary]
    decoded = codec.decode(codec.encode(payload))
    _validate_submit_payload(decoded, codec)
    records: list[tuple[str, dict[str, Any]]] = []
    coordinator = SimpleNamespace(
        counters=SimpleNamespace(episodes=1, finishes=0, best_finish_time_s=0.0, updates=7),
        run=SimpleNamespace(
            replay_store=[],
            logger=SimpleNamespace(log=lambda event, data, **_: records.append((event, data))),
        ),
    )
    with caplog.at_level(logging.INFO, logger="trackmaniarl.distributed.coordinator"):
        log_episode(coordinator, payload, decoded["episodes"][0])
    assert records[0][1]["exploration_epsilon"] is None
    assert records[0][1]["q_margin/mean"] is None
    assert "epsilon=unused" in caplog.text
    assert "q_margin(start=unknown, min=unknown)" in caplog.text


@pytest.mark.parametrize("epsilon_used", [True, False])
def test_wire_rejects_contradictory_exploration_availability(*, epsilon_used: bool) -> None:
    summary = _episode_summary(EpisodeMetrics.from_policy(object()))
    summary["exploration_epsilon_used"] = epsilon_used
    summary["exploration_epsilon"] = None if epsilon_used else 0.3
    with pytest.raises(ValueError, match="exploration_epsilon"):
        _validate_episode_summary(summary)


def test_evaluation_reducers_skip_unknown_q_but_preserve_real_zero() -> None:
    unknown = _episode_summary(EpisodeMetrics.from_policy(object()))
    zero_metrics = EpisodeMetrics.from_policy(object())
    zero_metrics.record_policy(SimpleNamespace(last_q_margin=0.0), 0)
    zero_metrics.record_diagnostics(0, SimpleNamespace(last_q_margin=0.0, last_q_max=0.0), {})
    zero = _episode_summary(zero_metrics)
    missing_stats = _evaluation_batch_stats([unknown], (150.0,))
    mixed_stats = _evaluation_batch_stats([unknown, zero], (150.0,))

    assert missing_stats["q_margin_start_mean"] is None
    assert missing_stats["q_margin_start_sample_count"] == 0
    assert missing_stats["progress_bin/00_005/q_margin_mean"] is None
    assert mixed_stats["q_margin_start_mean"] == 0.0
    assert mixed_stats["q_margin_start_sample_count"] == 1
    assert mixed_stats["progress_bin/00_005/q_margin_mean"] == 0.0
    assert mixed_stats["progress_bin/00_005/q_margin_sample_count"] == 1


def test_progress_aggregation_weights_q_by_measurements_and_reads_legacy_histograms() -> None:
    summaries = [
        {
            "00_100": {
                "action_count": 1.0,
                "action_entropy": 0.5,
                "action_coverage": 0.5,
                "q_margin_mean": 2.0,
                "q_margin_min": 2.0,
                "q_max_mean": 3.0,
            }
        },
        {
            "00_100": {
                "action_count": 100.0,
                "action_entropy": 0.0,
                "action_coverage": 0.5,
                "q_margin_mean": None,
                "q_margin_min": None,
                "q_max_mean": None,
            }
        },
        {
            "00_100": {
                "action_count": 0.0,
                "action_entropy": 0.0,
                "action_coverage": 0.0,
                "q_margin_mean": 0.0,
                "q_margin_min": 0.0,
                "q_max_mean": 0.0,
            }
        },
    ]
    summary = aggregate_progress_bins(summaries)
    assert summary["progress_bin/00_100/q_margin_mean"] == 2.0
    assert summary["progress_bin/00_100/q_margin_min"] == 2.0
    assert summary["progress_bin/00_100/q_margin_sample_count"] == 1
    assert summary["progress_bin/00_100/action_histogram_entropy_normalized"] == pytest.approx(
        0.5 / 101
    )


def test_evaluation_artifact_preserves_null_progress_measurements() -> None:
    diagnostics = ProgressBinDiagnostics(78, bin_count=1)
    diagnostics.record(ProgressDiagnosticRecord(0.0, 0, object()))
    result = EvaluationResult(
        False, None, False, 0.0, 0.0, 1.0, progress_bins=diagnostics.summary()
    )
    restored = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert restored["progress_bins"]["00_100"]["q_margin_mean"] is None
    assert restored["progress_bins"]["00_100"]["q_margin_sample_count"] == 0
