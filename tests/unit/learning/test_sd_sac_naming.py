"""The canonical SD-SAC name also accepts existing serialized configuration names."""

from __future__ import annotations

import pytest
import yaml

from experiments.tmrl_test_comparison.generate import ALGORITHMS, configuration
from trackmaniarl.algorithms.sac_config import DiscreteSACConfig, SDSACConfig
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.models.sensor_actor_critic import SensorActorCriticModelFactory
from trackmaniarl.project.scaffold_run_templates import (
    _trackmania_actor_critic_config,
    _trackmania_vision_config,
)
from trackmaniarl.trackmania.vision_models import VisionActorCriticModelFactory


@pytest.mark.parametrize("name", ["sd-sac", "discrete-sac", "dsac"])
def test_sd_sac_accepts_legacy_model_names(name: str) -> None:
    sensor = SensorActorCriticModelFactory(
        name,
        {"class_path": "trackmaniarl.trackmania.vision_models:VisionSensorEncoder"},
    )
    vision = VisionActorCriticModelFactory(name)
    assert sensor.algorithm == vision.algorithm == "sd-sac"
    assert sensor.model_contract == vision.model_contract == ModelContract.DISCRETE_ACTOR_CRITIC


@pytest.mark.parametrize("name", ["sd-sac", "discrete-sac", "dsac"])
def test_new_comparison_and_scaffold_configs_use_sd_sac(name: str) -> None:
    spec = RunSpec.model_validate(configuration(name, 17, "pilot"))
    assert spec.run_id == "tmrl-test-pilot-sd-sac-s17"
    assert spec.components.model_factory.kwargs["algorithm"] == "sd-sac"
    assert spec.metadata["algorithm"] == "sd-sac"
    generated = RunSpec.model_validate(yaml.safe_load(_trackmania_actor_critic_config(name)))
    assert generated.run_id == "trackmania-sd-sac-telemetry"
    assert generated.training.n_step == 1
    vision = RunSpec.model_validate(yaml.safe_load(_trackmania_vision_config(name)))
    assert vision.training.n_step == 1


def test_sd_sac_config_preserves_the_previous_import() -> None:
    assert DiscreteSACConfig is SDSACConfig
    assert "sd-sac" not in ALGORITHMS
    assert "discrete-sac" not in ALGORITHMS
