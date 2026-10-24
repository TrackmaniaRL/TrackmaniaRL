"""The incidental transition cap must not alter SDK exploration or replay."""

import sys
from types import SimpleNamespace

import pytest

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.coordinator_rpc import epsilon
from trackmaniarl.experiments.full_qr.campaign_spec import for_four_hour_budget


def specimen() -> RunSpec:
    components = {
        "learner": {"class_path": "trackmaniarl.algorithms.value_based:DiscreteValueLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"},
    }
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "fixture-no-model-or-environment",
            "components": components,
            "training": {"total_transitions": 8192000},
            "distributed": {
                "epsilon_start": 0.3,
                "epsilon_final": 0.01,
                "epsilon_decay_transitions": 500000,
            },
            "metadata": {"requested_wall_time_seconds": 14400},
        }
    )


def test_only_transition_stop_changes_and_actual_sdk_epsilon_is_identical() -> None:
    before = specimen()
    after = for_four_hour_budget(before)
    expected = before.model_dump(mode="json")
    expected["training"]["total_transitions"] = sys.maxsize
    assert after.model_dump(mode="json") == expected
    for count in (0, 10000, 250000, 500000, 8192000, 10000000):
        values = [
            epsilon(
                SimpleNamespace(
                    run=SimpleNamespace(spec=spec),
                    counters=SimpleNamespace(transitions=count, updates=count // 4),
                ),
                0,
            )
            for spec in (before, after)
        ]
        assert values[0] == values[1]
        assert before.training.replay_beta(count) == after.training.replay_beta(count)


def test_implicit_replay_annealing_cannot_change_with_the_cap() -> None:
    data = specimen().model_dump(mode="json")
    data["training"].update(beta=0.4, per_beta_final=1.0)
    with pytest.raises(ValueError, match="explicit historical schedules"):
        for_four_hour_budget(RunSpec.model_validate(data))


def test_other_exploration_schedule_or_wall_budget_is_rejected() -> None:
    for section, key, value in (
        ("distributed", "epsilon_decay_updates", 100),
        ("metadata", "requested_wall_time_seconds", 28800),
    ):
        data = specimen().model_dump(mode="json")
        data[section][key] = value
        with pytest.raises(ValueError, match="four-hour"):
            for_four_hour_budget(RunSpec.model_validate(data))
