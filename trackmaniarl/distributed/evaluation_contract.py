"""Fail-closed map binding for scheduled distributed evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trackmaniarl.core.runtime import _resolve_evaluation_map
from trackmaniarl.core.spec import RunSpec


def validate_scheduled_evaluation(  # noqa: PLR0913
    spec: RunSpec, factory: Any, pipeline: Any, base_dir: Path
) -> None:
    """Require an attested single map until distributed map switching exists."""
    suite = spec.evaluation
    if spec.training.evaluate_every_episodes is None or suite is None:
        return
    if len(suite.maps) != 1:
        raise ValueError(
            "scheduled distributed evaluation does not support multiple maps; "
            "use the local benchmark for a multi-map suite"
        )
    validate = getattr(factory, "validate_scheduled_evaluation_map", None)
    if not callable(validate):
        raise ValueError(
            "scheduled distributed evaluation requires an environment factory "
            "that attests its evaluation map and feature pipeline"
        )
    validate(_resolve_evaluation_map(suite.maps[0], base_dir), pipeline)


def validate_pipeline_binding(pipeline: Any, asset: Any, map_spec: Any) -> None:
    """Require a known map-independent pipeline or explicit map attestation."""
    from trackmaniarl.builtins.features import TransitionFeaturePipeline
    from trackmaniarl.core.builtins import IdentityFeaturePipeline
    from trackmaniarl.trackmania.common_training.features import CommonFeaturePipeline
    from trackmaniarl.trackmania.features import LidarFeaturePipeline, TelemetryFeaturePipeline
    from trackmaniarl.trackmania.multimodal import LidarVisionFeaturePipeline
    from trackmaniarl.trackmania.native_headless.vision import RibbonVisionFeaturePipeline
    from trackmaniarl.trackmania.vision import PreparedVisionFeaturePipeline, VisionFeaturePipeline

    if type(pipeline) in (
        IdentityFeaturePipeline,
        TelemetryFeaturePipeline,
        TransitionFeaturePipeline,
        CommonFeaturePipeline,
        VisionFeaturePipeline,
        PreparedVisionFeaturePipeline,
        RibbonVisionFeaturePipeline,
    ):
        return
    if type(pipeline) is LidarVisionFeaturePipeline:
        validate_pipeline_binding(pipeline.lidar, asset, map_spec)
        return
    if type(pipeline) is LidarFeaturePipeline:
        geometry = pipeline.geometry
        if geometry.map_uid != map_spec.expected_map_uid or geometry.sha256 != asset.sha256:
            raise ValueError("scheduled evaluation pipeline geometry differs from the pinned map")
        return
    validate = getattr(pipeline, "validate_scheduled_evaluation_map", None)
    if not callable(validate):
        raise ValueError("feature pipeline must attest its scheduled evaluation map binding")
    validate(map_spec)
