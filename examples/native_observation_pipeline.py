"""A small, untrained example of a custom native observation pipeline.

Copy this module into your project. Pair its ``kinematics`` output with a
feature pipeline and model configured for a six-element sensor. This is an API
example, not a qualified replacement for the historical V5 preprocessing.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from trackmaniarl.builtins.features import TransitionFeaturePipeline


class _Frame(Protocol):
    @property
    def channels(self) -> tuple[float, ...]: ...


class PositionVelocityPipeline:
    """Select measured position and velocity from each native frame."""

    def reset_episode(self) -> None:
        """This stateless example has no episode history to clear."""

    def _features(self, frame: _Frame) -> dict[str, NDArray[np.float32]]:
        return {"kinematics": np.asarray(frame.channels[1:7], dtype=np.float32)}

    def transform_observation(self, frame: _Frame) -> dict[str, NDArray[np.float32]]:
        return self._features(frame)

    def transform_terminal(self, frame: _Frame) -> dict[str, NDArray[np.float32]]:
        return self._features(frame)

    def transform_committed_terminal(self, frame: _Frame) -> dict[str, NDArray[np.float32]]:
        return self._features(frame)


class PositionVelocityFeatures(TransitionFeaturePipeline):
    """Prepare a tensor for policy inference and collate replay transitions."""

    def transform_observation(
        self, observation: dict[str, NDArray[np.float32]]
    ) -> dict[str, torch.Tensor]:
        values = np.asarray(observation["kinematics"], dtype=np.float32)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("kinematics must contain six finite values")
        return {"kinematics": torch.from_numpy(values.copy())}
