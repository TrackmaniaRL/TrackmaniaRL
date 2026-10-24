"""A committed native score and its separately timestamped measured observation."""

from dataclasses import dataclass

import numpy as np

from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import MeasuredAdherence
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    GRAPH_FIELDS,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import CommittedGoalCapture


@dataclass(frozen=True, slots=True)
class NativeCommittedGoalFrame:
    capture: CommittedGoalCapture
    adherence: MeasuredAdherence

    def __post_init__(self) -> None:
        if type(self.capture) is not CommittedGoalCapture:
            raise TypeError("complete bound committed Goal evidence required")
        vehicle = self.capture.vehicle
        if (
            (vehicle.native_t, vehicle.sequence)
            != (self.adherence.native_t, self.adherence.sequence)
            or vehicle.global_owner not in (0, self.adherence.arena)
            or not np.isfinite(np.asarray(self.channels, dtype=np.float32)).all()
        ):
            raise ValueError("actual committed-terminal vehicle/adherence binding required")

    @property
    def score_race_time_ms(self) -> int:
        return self.capture.fact.event_native_t - 2200

    @property
    def channels(self) -> tuple[float, ...]:
        # Channel0 remains the physical sample clock, never the later score.
        return NativeGraphFrame._capture_channels(self.capture.vehicle, self.adherence.coefficient)

    def _feature_carrier(self) -> np.ndarray:
        values = np.full(33, np.nan, dtype=np.float32)
        values[list(GRAPH_FIELDS)] = self.channels
        return values


@dataclass(frozen=True, slots=True)
class CommittedGoalEventWindow:
    start_ms: int
    sampled_ms: int
    score_ms: int
    collision_detected: bool
    final_sample_completed: bool = False

    def __post_init__(self) -> None:
        if (
            any(type(v) is not int for v in (self.start_ms, self.sampled_ms, self.score_ms))
            or not 0 <= self.start_ms <= self.sampled_ms < self.score_ms <= 150000
            or self.start_ms % 50
            or self.sampled_ms % 10
            or not 0 < self.score_ms - self.sampled_ms < 10
            or type(self.final_sample_completed) is not bool
            or (
                self.sampled_ms - self.start_ms != 50
                if self.final_sample_completed
                else not self.score_ms - self.start_ms < 50
            )
            or type(self.collision_detected) is not bool
        ):
            raise ValueError("complete separately timestamped committed Goal interval required")
