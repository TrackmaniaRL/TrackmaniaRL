"""Measured mapping from the virtual gamepad's stick to the steer Trackmania applies.

Trackmania passes a pad's stick through its own Analog Sensitivity and Analog Dead
Zone settings before the car sees it. At a low sensitivity the curve is steep: on
one measured install (sensitivity 0.1, dead zone 0.05) every stick value up to 0.5
steered straight and 0.667 steered 6%, so 7 of the 13 default steering levels
drove identically and the policy could not tell them apart. At sensitivity 1.0
the same install was close to linear.

``measure_steering_curve`` records the curve through the telemetry's ``InputSteer``
field. The gamepad backend then inverts it, so an action's steer is the steer the
game applies rather than a raw stick position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import copysign
from pathlib import Path
from typing import Protocol

import numpy as np

from trackmaniarl.trackmania.telemetry import TelemetryFrame

INPUT_STEER_INDEX = 30
"""``api.InputSteer`` offset in the supported 33-field telemetry packet."""

_FORMAT_VERSION = 1
_PLATEAU_DECIMALS = 4


@dataclass(frozen=True, slots=True)
class SteeringCurve:
    """Game steer measured at increasing stick positions on one side, 0 to 1."""

    stick: tuple[float, ...]
    steer: tuple[float, ...]

    def __post_init__(self) -> None:
        stick, steer = np.asarray(self.stick), np.asarray(self.steer)
        if stick.shape != steer.shape or stick.size < 2:
            raise ValueError("a steering curve needs matching stick and steer samples")
        if stick[0] != 0.0 or np.any(np.diff(stick) <= 0.0) or stick[-1] > 1.0:
            raise ValueError("stick samples must rise strictly from 0 to at most 1")
        if np.any(steer < 0.0) or np.any(steer > 1.0) or np.any(np.diff(steer) < 0.0):
            raise ValueError("steer samples must be non-decreasing values in [0, 1]")
        if steer[-1] <= 0.0:
            raise ValueError("the measured steer never left zero")

    def steer_at(self, stick: float) -> float:
        """Steer the game applies for a raw stick position."""

        return copysign(float(np.interp(abs(stick), self.stick, self.steer)), stick)

    def stick_for(self, steer: float) -> float:
        """Raw stick position that makes the game apply ``steer``.

        Where several stick positions give the same steer, aim for the middle of
        that band so measurement noise cannot tip it into a neighbour. The zero band
        is the exception: aim at its far edge, so the smallest requested steer
        leaves the dead zone instead of disappearing into it.
        """

        magnitude = min(abs(steer), 1.0)
        if magnitude == 0.0:
            return 0.0
        levels, sticks = self._inverse_samples()
        return copysign(float(np.interp(magnitude, levels, sticks)), steer)

    def _inverse_samples(self) -> tuple[np.ndarray, np.ndarray]:
        steer = np.round(np.asarray(self.steer), _PLATEAU_DECIMALS)
        stick = np.asarray(self.stick)
        levels: list[float] = []
        sticks: list[float] = []
        for level in np.unique(steer):
            band = stick[steer == level]
            levels.append(float(level))
            sticks.append(float(band[-1] if level == 0.0 else (band[0] + band[-1]) / 2))
        return np.asarray(levels), np.asarray(sticks)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"version": _FORMAT_VERSION, "stick": self.stick, "steer": self.steer}
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> SteeringCurve:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported steering curve version in {path}")
        return cls(tuple(document["stick"]), tuple(document["steer"]))


def distinct_steering_levels(curve: SteeringCurve, levels: np.ndarray) -> int:
    """How many of the requested raw stick levels produce different game steer."""

    applied = {round(curve.steer_at(float(level)), _PLATEAU_DECIMALS) for level in levels}
    return len(applied)


class _SteeringInput(Protocol):
    def apply(self, action: np.ndarray) -> None: ...


class _TelemetrySource(Protocol):
    def read(self) -> TelemetryFrame: ...


@dataclass(frozen=True, slots=True)
class SteeringMeasurement:
    """Hold each stick position still and read back the steer the game applied.

    The car must sit on a loaded map with no gas: only the stick moves.
    """

    controller: _SteeringInput
    client: _TelemetrySource
    points: int = 41
    settle_frames: int = 8


def measure_steering_curve(request: SteeringMeasurement) -> SteeringCurve:
    """Sweep both stick directions and average them into one symmetric curve."""

    if request.points < 2 or request.settle_frames < 1:
        raise ValueError("measurement needs at least two points and one settle frame")
    sticks = np.linspace(0.0, 1.0, request.points)
    try:
        right = [_read_steer(request, float(stick)) for stick in sticks]
        left = [-_read_steer(request, -float(stick)) for stick in sticks]
    finally:
        request.controller.apply(np.zeros(3, dtype=np.float32))
    if max(right) < 0.5 or max(left) < 0.5:
        raise RuntimeError(
            "InputSteer did not follow the virtual stick to full lock. Sit on a loaded "
            "map with the car visible and check that the gamepad backend drives the car."
        )
    # Small readback noise must not break monotonicity, so each sample keeps the
    # largest steer seen at or below its stick position.
    steer = np.maximum.accumulate(np.clip((np.asarray(right) + np.asarray(left)) / 2, 0.0, 1.0))
    return SteeringCurve(tuple(float(s) for s in sticks), tuple(float(s) for s in steer))


def _read_steer(request: SteeringMeasurement, stick: float) -> float:
    request.controller.apply(np.asarray([0.0, 0.0, stick], dtype=np.float32))
    readings = [
        float(request.client.read().values[INPUT_STEER_INDEX]) for _ in range(request.settle_frames)
    ]
    # The first frames can still carry the previous stick position.
    return float(np.median(readings[-3:]))
