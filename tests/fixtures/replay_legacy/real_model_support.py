"""Small first-party model and synthetic run shared by fixture and restore test."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from tests.integration.runtime.distributed_runtime_support import _resolved_run
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.core.replay import SequenceSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.models.factory import CompositeValueModelFactory


def _component(path: str, **kwargs: object) -> dict:
    return {"class_path": path, "kwargs": kwargs}


def make_run(root: Path) -> ResolvedRun:
    """Create matching runtime components on both release and current source."""
    training = {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }
    run = _resolved_run(root, "real-model-fixture", training)
    factory_config = {
        "encoder": _component(
            "trackmaniarl.models.encoders:MlpSensorEncoder", input_dim=4, output_dim=6, hidden_dim=8
        ),
        "temporal": _component("trackmaniarl.models.temporal:IdentityTemporalCore", input_dim=6),
        "head": _component("trackmaniarl.models.heads:ScalarQHead", feature_dim=6, action_count=3),
        "strategy": _component("trackmaniarl.models.strategies:ScalarValueStrategy"),
    }
    config = run.spec.model_dump(mode="json")
    config["components"]["learner"] = _component(
        "trackmaniarl.algorithms.value_based:DiscreteValueLearner",
        execution={"device": "cpu", "precision": "float32"},
    )
    config["components"]["model_factory"] = _component(
        "trackmaniarl.models.factory:CompositeValueModelFactory", **factory_config
    )
    config["components"]["feature_pipeline"] = _component(
        "trackmaniarl.core.builtins:IdentityFeaturePipeline"
    )
    config["components"]["sampler"] = _component(
        "trackmaniarl.core.replay:SequenceSampler", sequence_length=2, seed=41
    )
    spec = RunSpec.model_validate(config)
    torch.manual_seed(17)
    factory = CompositeValueModelFactory(**factory_config)
    learner = DiscreteValueLearner(
        factory.build(), execution={"device": "cpu", "precision": "float32"}
    )
    learner.setup({"seed": 17})
    pipeline = IdentityFeaturePipeline()
    return replace(
        run,
        spec=spec,
        learner=learner,
        model_factory=factory,
        feature_pipeline=pipeline,
        sampler=SequenceSampler(pipeline, sequence_length=2, seed=41),
    )


def transition(step: int) -> Transition:
    return Transition(
        observation=torch.tensor([step, 1.0, -1.0, 0.5], dtype=torch.float32),
        action=step % 3,
        reward=float(step + 1),
        next_observation=torch.tensor([step + 1, 1.0, -1.0, 0.5], dtype=torch.float32),
        terminated=step == 2,
        truncated=False,
        episode_id="fixture-actor/fixture-session/episode",
        step=step,
        info={"policy_version": 0},
    )
