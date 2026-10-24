"""Opt-in preparation of terminal bootstrap observations at collection boundaries."""

from __future__ import annotations

from typing import Any

from trackmaniarl.core.contracts import FeaturePipeline


def validate_endpoint_flags(pipeline: object, *, terminated: object, truncated: object) -> None:
    """Require unambiguous flags only for the explicit optional-endpoint contract."""
    if getattr(pipeline, "terminal_next_observation_optional", False) is not True:
        return
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("Optional terminal endpoints require bool terminated and truncated flags")
    if terminated and truncated:
        raise ValueError("An optional terminal endpoint cannot be both terminated and truncated")


def prepare_bootstrap_observation(
    pipeline: FeaturePipeline,
    observation: Any,
    *,
    flags: tuple[object, object],
) -> Any:
    """Skip terminal NEXT only on literal opt-in; CURRENT still uses the pipeline.

    This collection hook does not establish wire, sampler or learner compatibility.
    Nonterminal and truncated endpoints always follow the ordinary transform path.
    """
    terminated, truncated = flags
    validate_endpoint_flags(pipeline, terminated=terminated, truncated=truncated)
    if (
        getattr(pipeline, "terminal_next_observation_optional", False) is True
        and terminated is True
    ):
        return None
    return pipeline.transform_observation(observation)
