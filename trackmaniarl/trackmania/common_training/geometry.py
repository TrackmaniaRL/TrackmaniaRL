"""One immutable reference and continuous projection shared by both producers.

Coordinates retain the engine/reference unit. SI meters are not inferred. The
closest point is searched across all nonzero segments; ties select the lowest
original segment index. Repeated stationary samples add zero arc. There is no
vehicle model, history, index window or inferred lookahead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.trackmania.common_training.contract import (
    ProjectionLimits,
    RewardMeasurement,
    RewardSpec,
    require_digest,
)
from trackmaniarl.trackmania.parity import COMMON_SCALES

PROJECTION_PROFILE = "tmrl.common-continuous-global-polyline/v1"
type Position = Sequence[float] | np.ndarray[Any, Any]


@dataclass(frozen=True)
class Projection:
    segment_index: int
    segment_fraction: float
    progress_units: float
    deviation_units: float


class CommonPolylineReference:
    def __init__(
        self,
        reference_json: bytes,
        *,
        expected_sha256: str,
        limits: ProjectionLimits | None = None,
    ) -> None:
        require_digest(expected_sha256, "full reference JSON file")
        if (
            type(reference_json) is not bytes
            or hashlib.sha256(reference_json).hexdigest() != expected_sha256
        ):
            raise ValueError("full reference JSON SHA-256 does not match the pinned artifact")
        parsed = json.loads(reference_json)
        points = parsed["points"] if isinstance(parsed, dict) else parsed
        self.file_sha256 = expected_sha256
        self.limits = limits or ProjectionLimits()
        array = np.array(points, dtype=np.float64, copy=True)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 3:
            raise ValueError("reference requires at least two actual XYZ points")
        if not np.isfinite(array).all():
            raise ValueError("finite actual reference points required")
        differences = np.diff(array, axis=0)
        squared_lengths = np.einsum("ij,ij->i", differences, differences)
        nonzero = squared_lengths > 0
        if not np.isfinite(squared_lengths).all() or not nonzero.any():
            raise ValueError("finite nonzero reference segments required")
        lengths = np.sqrt(squared_lengths)
        arc = np.concatenate(([0.0], np.cumsum(lengths)))
        if not np.isfinite(arc).all():
            raise ValueError("finite reference arc length required")
        # Stationary recorded samples legitimately repeat points. Their segments
        # contribute zero arc and no new geometric locus; never divide by zero.
        self._starts, self._differences = array[:-1][nonzero], differences[nonzero]
        self._segment_indices = np.flatnonzero(nonzero)
        self._squared_lengths = squared_lengths[nonzero]
        self._lengths, self._arc = lengths[nonzero], arc[:-1][nonzero]
        for value in (
            self._starts,
            self._differences,
            self._segment_indices,
            self._squared_lengths,
            self._lengths,
            self._arc,
        ):
            value.flags.writeable = False

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        limits: ProjectionLimits | None = None,
    ) -> CommonPolylineReference:
        return cls(path.read_bytes(), expected_sha256=expected_sha256, limits=limits)

    def reward_spec(self) -> RewardSpec:
        return RewardSpec(
            reference_sha256=self.file_sha256,
            projection_profile=PROJECTION_PROFILE,
            projection_limits=self.limits,
        )

    def project_world(self, position: Position) -> Projection:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("actual finite world XYZ position required")
        relative = point - self._starts
        fractions = np.clip(
            np.einsum("ij,ij->i", relative, self._differences) / self._squared_lengths,
            0.0,
            1.0,
        )
        residuals = relative - self._differences * fractions[:, None]
        distances_squared = np.einsum("ij,ij->i", residuals, residuals)
        if not np.isfinite(distances_squared).all():
            raise ValueError("finite real projection distances required")
        index = int(np.argmin(distances_squared))
        deviation = float(np.sqrt(distances_squared[index]))
        if self.limits.max_deviation_units is not None and (
            deviation > self.limits.max_deviation_units
        ):
            raise ValueError("actual projection exceeds declared deviation rejection limit")
        return Projection(
            int(self._segment_indices[index]),
            float(fractions[index]),
            float(self._arc[index] + fractions[index] * self._lengths[index]),
            deviation,
        )

    def project_features(self, observation: torch.Tensor) -> Projection:
        if (
            not isinstance(observation, torch.Tensor)
            or tuple(observation.shape) not in ((6,), (8,))
            or observation.dtype != torch.float32
            or observation.device.type != "cpu"
            or not bool(torch.isfinite(observation).all())
        ):
            raise ValueError("actual finite CPU float32 common6/common8 observation required")
        # Invert only the declared normalization; this does not introduce SI units.
        position = observation[:3].to(torch.float64).numpy() * COMMON_SCALES[:3]
        return self.project_world(position)

    def measure_world(
        self,
        positions: tuple[Position, Position],
        *,
        finished: bool,
    ) -> RewardMeasurement:
        before, after = (self.project_world(position) for position in positions)
        return self._measurement(before, after, finished=finished)

    def measure_features(
        self,
        observations: tuple[torch.Tensor, torch.Tensor],
        *,
        finished: bool,
    ) -> RewardMeasurement:
        before, after = (self.project_features(observation) for observation in observations)
        return self._measurement(before, after, finished=finished)

    def _measurement(
        self,
        before: Projection,
        after: Projection,
        *,
        finished: bool,
    ) -> RewardMeasurement:
        measurement = RewardMeasurement(
            progress_before_units=before.progress_units,
            progress_after_units=after.progress_units,
            deviation_after_units=after.deviation_units,
            finished=finished,
        )
        # Enforce the identical explicit limits in projection and the reward contract.
        self.reward_spec().score(measurement, 1.0)
        return measurement
