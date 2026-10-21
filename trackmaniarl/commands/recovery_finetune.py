"""Supervised fine-tuning of the incident-gated TrackMania recovery adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackmaniarl.commands.common import (
    _new_attempt_spec,
    _with_model_initialization_checkpoint,
)
from trackmaniarl.commands.helpers import _learner_context, _training_learner_state
from trackmaniarl.commands.recovery_finetune_checkpoint import (
    _file_sha256,
    _save_checkpoint,
)
from trackmaniarl.commands.recovery_finetune_data import (
    _build_recovery_data,
    _model_actions,
    _resolve_recovery_paths,
    _split_episodes,
)
from trackmaniarl.commands.recovery_finetune_training import (
    _fine_tune,
    _settings,
    _summary,
)
from trackmaniarl.commands.recovery_finetune_types import (
    _CheckpointRequest,
    _FineTuneContext,
    _FineTuneSettings,
    _RecoveryBuildContext,
    _RecoveryEpisode,
)
from trackmaniarl.core.runtime import prepare_run, record_run_attempt, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.experiments.graph_iqn_v6 import (
    IncidentGatedTrackGnnSimbaEncoderV6,
    IncidentRecoveryOnlyDiscreteValueLearner,
)
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry


@dataclass(frozen=True, slots=True)
class _CommandRequest:
    checkpoint: Path
    recoveries: tuple[Path, ...]
    settings: _FineTuneSettings
    output: Path | None


def _recovery_finetune(args: argparse.Namespace) -> None:
    """Fine-tune the incident adapter on human takeover labels."""

    config = args.config.resolve()
    request = _command_request(args)
    run = resolve_run(_fine_tune_spec(config, request.checkpoint), base_dir=config.parent)
    try:
        data, result, target = _execute_fine_tune(run, request)
    finally:
        run.logger.close()
    print(_summary(data, result))
    print(f"Recovery policy checkpoint: {target}")


def _command_request(args: argparse.Namespace) -> _CommandRequest:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recovery source checkpoint does not exist: {checkpoint}")
    return _CommandRequest(
        checkpoint,
        _resolve_recovery_paths(args.recovery),
        _settings(args),
        _validated_output(args.output, checkpoint),
    )


def _validated_output(output: Path | None, source: Path) -> Path | None:
    if output is None:
        return None
    target = output.resolve()
    if target == source:
        raise ValueError("recovery output checkpoint must differ from the source checkpoint")
    if target.exists():
        raise FileExistsError(f"recovery output checkpoint already exists: {target}")
    return target


def _execute_fine_tune(run: Any, request: _CommandRequest) -> tuple[Any, Any, Path]:
    prepare_run(run)
    run.learner.setup(_learner_context(run))
    record_run_attempt(run)
    learner, components = _components(run)
    _require_source_architecture(run, learner, request.checkpoint)
    context = _build_context(request, learner, components)
    data = _build_recovery_data(request.recoveries, context, request.settings)
    _print_data_gate(data)
    result = _fine_tune(data, _FineTuneContext(learner, request.settings, run.logger))
    learner.finish_supervised_recovery()
    checkpoint = _CheckpointRequest(
        request.checkpoint, data, request.settings, result, request.output
    )
    return data, result, _save_checkpoint(run, checkpoint)


def _require_source_architecture(
    run: Any, learner: IncidentRecoveryOnlyDiscreteValueLearner, checkpoint: Path
) -> None:
    """Validate and strictly restore the complete source policy."""

    outer = run.checkpoint_codec.load(checkpoint)
    source = _training_learner_state(outer)
    compatible = {
        learner.model.architecture_fingerprint(),
        learner.model.legacy_architecture_fingerprint(),
    }
    if source.get("architecture_fingerprint") not in compatible:
        raise ValueError(
            "recovery source checkpoint architecture does not match the configured model"
        )
    learner.load_policy_state_dict(source)


def _print_data_gate(data: Any) -> None:
    print(
        "Recovery data gate passed: "
        f"train={len(data.train)} samples/{data.train_episodes} episodes, "
        f"validation={len(data.validation)} samples/{data.validation_episodes} episodes."
    )


def _build_context(
    request: _CommandRequest,
    learner: IncidentRecoveryOnlyDiscreteValueLearner,
    components: tuple[TrackmaniaEnvironmentConfig, BoundaryGraphFeaturePipelineV5],
) -> _RecoveryBuildContext:
    environment, pipeline = components
    geometry = BoundaryGeometry(
        environment.geometry_path, expected_map_uid=environment.expected_map_uid
    )
    return _RecoveryBuildContext(
        environment,
        geometry,
        pipeline,
        learner,
        request.settings.minimum_gate,
        _file_sha256(request.checkpoint),
    )


def _fine_tune_spec(config: Path, checkpoint: Path) -> RunSpec:
    source = RunSpec.from_yaml(config)
    initialized = _with_model_initialization_checkpoint(source, checkpoint)
    return _new_attempt_spec(config, initialized, argparse.Namespace())


def _components(
    run: Any,
) -> tuple[
    IncidentRecoveryOnlyDiscreteValueLearner,
    tuple[TrackmaniaEnvironmentConfig, BoundaryGraphFeaturePipelineV5],
]:
    learner = _recovery_learner(run.learner)
    pipeline = _recovery_pipeline(run.feature_pipeline)
    factory = run.environment_factory
    if not isinstance(factory, OpenPlanetEnvironmentFactory):
        raise TypeError("recovery-finetune requires OpenPlanetEnvironmentFactory")
    return learner, (factory.config, pipeline)


def _recovery_learner(learner: Any) -> IncidentRecoveryOnlyDiscreteValueLearner:
    if not isinstance(learner, IncidentRecoveryOnlyDiscreteValueLearner):
        raise TypeError("recovery-finetune requires IncidentRecoveryOnlyDiscreteValueLearner")
    if not isinstance(learner.model.encoder, IncidentGatedTrackGnnSimbaEncoderV6):
        raise TypeError("recovery-finetune requires the incident-gated V6 encoder")
    return learner


def _recovery_pipeline(pipeline: Any) -> BoundaryGraphFeaturePipelineV5:
    if not isinstance(pipeline, BoundaryGraphFeaturePipelineV5):
        raise TypeError("recovery-finetune requires BoundaryGraphFeaturePipelineV5")
    return pipeline


__all__ = [
    "_RecoveryEpisode",
    "_model_actions",
    "_recovery_finetune",
    "_resolve_recovery_paths",
    "_settings",
    "_split_episodes",
]
