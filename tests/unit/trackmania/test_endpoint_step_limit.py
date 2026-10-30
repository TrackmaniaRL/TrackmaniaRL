"""Collectors reject known incompatible episode limits before resetting."""

from types import SimpleNamespace

import pytest

from trackmaniarl.core.collector import EpisodeCollector
from trackmaniarl.core.observation_endpoints import validate_endpoint_step_limit
from trackmaniarl.distributed.actor_collection import collect


def test_collectors_reject_endpoint_cap_before_reset() -> None:
    environment = SimpleNamespace(
        capabilities={"trainingEpisodeStepCap": 20},
        reset=lambda: pytest.fail("reset must not run"),
    )
    pipeline = SimpleNamespace(requires_environment_endpoint=True)
    with pytest.raises(ValueError, match="at least 20 steps"):
        EpisodeCollector(None, pipeline, None).collect(environment, "episode", 19)
    runtime = SimpleNamespace(spec=SimpleNamespace(training=SimpleNamespace(max_episode_steps=19)))
    with pytest.raises(ValueError, match="at least 20 steps"):
        collect(runtime, environment, pipeline)


@pytest.mark.parametrize("cap", [None, 19, 20])
def test_compatible_or_unknown_cap_is_accepted(cap: int | None) -> None:
    environment = SimpleNamespace(capabilities={"trainingEpisodeStepCap": cap})
    pipeline = SimpleNamespace(requires_environment_endpoint=True)
    validate_endpoint_step_limit(pipeline, environment, 20)


def test_other_pipelines_may_stop_before_environment_cap() -> None:
    environment = SimpleNamespace(capabilities={"trainingEpisodeStepCap": 20})
    validate_endpoint_step_limit(object(), environment, 19)
