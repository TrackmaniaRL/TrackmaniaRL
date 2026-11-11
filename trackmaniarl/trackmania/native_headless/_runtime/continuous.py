"""Experimental float input records; no claim of native actuator qualification.

The record layout is distinct from the actual values consumed by game physics.
Callers must independently qualify the latter before using this for training.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class ContinuousControl:
    """SDK order [gas, brake, steer], rounded to native float32 at entry."""

    gas: float
    brake: float
    steer: float

    def __post_init__(self) -> None:
        for name, lower, upper in (("gas", 0.0, 1.0), ("brake", 0.0, 1.0), ("steer", -1.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside the finite action domain")
            object.__setattr__(self, name, struct.unpack("<f", struct.pack("<f", value))[0])

    def wire_record(self) -> bytes:
        return struct.pack("<3f", self.steer, self.gas, self.brake)

    def receipt_identity(self) -> dict[str, Any]:
        return {
            "profile": "experimental-continuous-sgb-f32/v1",
            "record_hex": self.wire_record().hex(),
        }
