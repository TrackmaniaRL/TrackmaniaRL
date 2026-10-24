"""Full V5 graph computations from a typed native input, without a fake API packet.

Only the 23 channels consumed by the historical V5 feature calculations are
carried. Unused API fields remain absent. Capture/clock qualification belongs
to the experimental backend; this module does not assert Windows parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from struct import unpack_from
from typing import Any

import numpy as np
import torch

from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import MeasuredAdherence
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import PartialGoalCapture
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCapture

GRAPH_FIELDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32)


def _speed(velocity: tuple[float, ...]) -> float:
    squares = [np.float32(np.float32(value) * np.float32(value)) for value in velocity]
    squared = np.float32(np.float32(squares[0] + squares[1]) + squares[2])
    return float(np.float32(np.sqrt(squared)))


@dataclass(frozen=True, slots=True)
class NativeGraphFrame:
    """Exactly the consumed channels in GRAPH_FIELDS order, never 33-field API telemetry."""

    channels: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.channels, dtype=np.float32)
        if values.shape != (len(GRAPH_FIELDS),) or not np.isfinite(values).all():
            raise ValueError("all 23 consumed native graph channels must be finite")
        time_ms = float(values[0])
        if not 0 <= time_ms <= 150_000 or time_ms % 50:
            raise ValueError("native graph frame requires the qualified 50ms/150s domain")

    @classmethod
    def from_capture(cls, capture: VehicleCapture) -> NativeGraphFrame:
        """Extract fresh visual/vendor-cache channels under the explicit native clock contract.

        A nonnull adherence owner is unavailable until its resolver is qualified.
        Null selects the pinned vendor getter's source-proven default1.0. This
        is explicitly a getter default, not a measured ground-grip coefficient.
        """
        if capture.global_owner != 0:
            raise ValueError("nonnull adherence owner requires a qualified resolver")
        return cls._from_channels(capture, 1.0)

    @classmethod
    def _from_channels(cls, capture: VehicleCapture, coefficient: float) -> NativeGraphFrame:
        return cls(cls._capture_channels(capture, coefficient))

    @staticmethod
    def _capture_channels(capture: VehicleCapture, coefficient: float) -> tuple[float, ...]:
        visual, cache = capture.visual_state, capture.native_cache
        velocity = unpack_from("<3f", visual, 0x5C)
        matrix = unpack_from("<9f", visual, 0x2C)
        rpm, gear = unpack_from("<fI", cache, 0x38)
        return (
            float(capture.native_t - 2200),
            *unpack_from("<3f", cache),
            *velocity,
            matrix[2],
            matrix[5],
            matrix[8],
            _speed(velocity),
            rpm,
            float(gear),
            *capture.slip_connect_order,
            float(capture.skidding_count),
            float(capture.flying_duration),
            coefficient,
            unpack_from("<f", cache, 0x2C)[0],
            unpack_from("<f", cache, 0x30)[0],
            float(unpack_from("<I", cache, 0x34)[0]),
        )

    def _feature_carrier(self) -> np.ndarray:
        # Private indexing carrier only. NaNs make an accidental read of an
        # absent field fail output validation; it is never a valid API packet.
        carrier = np.full(33, np.nan, dtype=np.float32)
        carrier[list(GRAPH_FIELDS)] = self.channels
        return carrier

    @classmethod
    def from_measured_capture(
        cls, capture: VehicleCapture, adherence: MeasuredAdherence
    ) -> NativeGraphFrame:
        """Join two complete streams at the same actual native boundary.

        The measured coefficient overrides the legacy getter-default path.
        Other channels use the same vendor-cache computations. A different
        temporary arena context is still an error, including a nonnull one.
        """
        if (capture.native_t, capture.sequence) != (adherence.native_t, adherence.sequence):
            raise ValueError("measured adherence and vehicle boundary mismatch")
        if capture.global_owner not in (0, adherence.arena):
            raise ValueError("vehicle context belongs to another adherence arena")
        return cls._from_channels(capture, adherence.coefficient)


@dataclass(frozen=True, slots=True)
class NativeGoalTerminalFrame:
    """Actual partial final state; deliberately not a regular NativeGraphFrame."""

    capture: PartialGoalCapture
    adherence: MeasuredAdherence

    def __post_init__(self) -> None:
        vehicle, adherence = self.capture.vehicle, self.adherence
        if (
            (vehicle.native_t, vehicle.sequence) != (adherence.native_t, adherence.sequence)
            or vehicle.global_owner not in (0, adherence.arena)
            or not np.isfinite(np.asarray(self.channels, dtype=np.float32)).all()
        ):
            raise ValueError("complete measured terminal vehicle/adherence binding required")

    @property
    def channels(self) -> tuple[float, ...]:
        return NativeGraphFrame._capture_channels(self.capture.vehicle, self.adherence.coefficient)

    def _feature_carrier(self) -> np.ndarray:
        carrier = np.full(33, np.nan, dtype=np.float32)
        carrier[list(GRAPH_FIELDS)] = self.channels
        return carrier


class NativeGraphFeaturePipeline(BoundaryGraphFeaturePipelineV5):
    """Preserve all V5 graph/context/recovery calculations and causal history."""

    schema_version = "native-v5-experimental-1"

    def reset_episode(self) -> None:
        super().reset_episode()
        self._native_last_time: float | None = None
        self._native_last_observation: dict[str, torch.Tensor] | None = None

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, Mapping):
            return self._validate_prepared_v5(observation)
        if not isinstance(observation, NativeGraphFrame):
            raise TypeError("native graph pipeline requires a typed NativeGraphFrame")
        values = observation._feature_carrier()
        time_ms = float(values[3])
        expected = 0.0 if self._native_last_time is None else self._native_last_time + 50.0
        if time_ms != expected:
            raise ValueError("native graph episode must start at0 and advance exactly50ms")
        return self._transform_native_values(values, time_ms)

    def transform_terminal(self, observation: NativeGoalTerminalFrame) -> dict[str, torch.Tensor]:
        if type(observation) is not NativeGoalTerminalFrame:
            raise TypeError("a separate actual partial Goal terminal frame is required")
        if self._native_last_time != observation.capture.previous_native_t - 2200:
            raise ValueError("terminal frame must follow this episode's last regular step")
        values = observation._feature_carrier()
        return self._transform_native_values(values, float(values[3]))

    def transform_committed_terminal(self, observation: Any) -> dict[str, torch.Tensor]:
        from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
            NativeCommittedGoalFrame,
        )

        if type(observation) is not NativeCommittedGoalFrame:
            raise TypeError("a separately timestamped committed Goal frame is required")
        if self._native_last_time != observation.capture.previous_native_t - 2200:
            raise ValueError("committed frame must follow this episode's last regular step")
        values = observation._feature_carrier()
        if float(values[3]) == self._native_last_time:
            if self._native_last_observation is None:
                raise ValueError("previous actual observation is unavailable")
            return self._native_last_observation
        return self._transform_native_values(values, float(values[3]))

    def _transform_native_values(
        self, values: np.ndarray, time_ms: float
    ) -> dict[str, torch.Tensor]:
        position = values[[4, 5, 6]]
        nearest = self._nearest(position)
        track, curvature = self._track_features(position, values[[10, 12]], nearest)
        result = {
            "physics": torch.from_numpy(self._physics_v2(values, nearest, curvature)),
            "track": torch.from_numpy(track),
            "context": torch.from_numpy(self._context_v3(values, nearest)),
            "recovery": torch.from_numpy(self._recovery_v5(values, nearest)),
        }
        self._validate_v5_tensors(result)
        self._native_last_time = time_ms
        self._native_last_observation = result
        return result
