"""Versioned partial observations and auditable pending-input scheduling.

Field availability is shared; equivalence of the engines' units and timing is
a separate qualification. No native clock, pending control or guessed physics.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COMMON_PROFILE = "tmrl.common-kinematics8/v1"
COMMON_FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "engine_rpm",
    "gear_raw",
)
COMMON_SCALES = (1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0, 10000.0, 10.0)


def _encode(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (8,) or not np.isfinite(raw).all():
        raise ValueError("eight actual finite fields required")
    if raw[7] < 0 or raw[7] != math.floor(raw[7]):
        raise ValueError("actual nonnegative integral gear required")
    result = (raw / COMMON_SCALES).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("common observation overflows float32")
    return result


def from_connect33(values: Sequence[float]) -> np.ndarray:
    if len(values) != 33:
        raise ValueError("Connect 2.4.0 packet of exactly 33 fields required")
    return _encode([*values[4:10], values[17], values[18]])


def from_native(state: Mapping[str, Any]) -> np.ndarray:
    from trackmaniarl.trackmania.native_headless._runtime.features import native16

    if state.get("native76Hex") is None:
        raise ValueError("actual native endpoint is unavailable")
    raw = native16(bytes.fromhex(state["native76Hex"]))
    return _encode([*raw["physicalPosition"], *raw["physicalVelocity"], raw["rpm"], raw["gearRaw"]])


def checkpoint_contract(*, reset_profile: str, input_schedule: str) -> dict[str, Any]:
    """Persist and compare this envelope before loading any model/optimizer."""
    body = {
        "profile": COMMON_PROFILE,
        "fields": COMMON_FIELDS,
        "scales": COMMON_SCALES,
        "reset_profile": reset_profile,
        "input_schedule": input_schedule,
        "clock_mapping": None,
        "action_profile": "discrete78/v1",
    }
    return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def require_checkpoint_contract(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    # Canonical JSON also makes tuple/list roundtrips unambiguous.
    if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("incompatible observation/reset/input checkpoint; train a fresh model")


@dataclass(frozen=True)
class InputSchedule:
    issued_action: int
    issued_sgb: tuple[tuple[float, float, float], ...]
    applied_sgb: tuple[tuple[float, float, float], ...]
    pending_sgb: tuple[float, float, float]


def pending_schedule(action: int, pending: Sequence[float], intervals: int) -> InputSchedule:
    from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb

    if type(intervals) is not int or not 1 <= intervals <= 5:
        raise ValueError("one through five native intervals required")
    if len(pending) != 3 or not all(math.isfinite(v) for v in pending):
        raise ValueError("actual pending SGB required")
    if not (-1 <= pending[0] <= 1 and pending[1] in (0, 1) and pending[2] in (0, 1)):
        raise ValueError("pending SGB domain")
    issued = tuple(action_sgb(action))
    pending_sgb = (pending[0], pending[1], pending[2])
    return InputSchedule(
        action, issued, (pending_sgb, *issued[: intervals - 1]), issued[intervals - 1]
    )


def verify_applied(
    before: Mapping[str, Any], action: int, result: Mapping[str, Any]
) -> InputSchedule:
    """Verify literal POST inputs; write receipts alone are insufficient."""
    schedule = pending_schedule(action, before["pendingSGB"], result["actualIterations"])
    events = result["authorization"]["nativeEventHex"]
    if len(events) != len(schedule.applied_sgb):
        raise ValueError("actual POST count differs from interval count")
    for raw_hex, want in zip(events, schedule.applied_sgb, strict=True):
        raw = bytes.fromhex(raw_hex)
        if len(raw) != 256 or raw[:8] != b"TMPOST03":
            raise ValueError("actual POST256 required")
        gas, brake, steer = struct.unpack_from("<3f", raw, 88)
        if struct.pack("<3f", steer, gas, brake) != struct.pack("<3f", *want):
            raise ValueError("applied controls differ from pending schedule")
    if (
        not result["terminated"]
        and tuple(result["observation"]["pendingSGB"]) != schedule.pending_sgb
    ):
        raise ValueError("pending endpoint differs from issued future record")
    return schedule


def require_learning_comparison(evidence: Mapping[str, Any]) -> None:
    keys = (
        "reset_qualified",
        "applied_inputs_qualified",
        "clock_mapping_qualified",
        "observation_semantics_qualified",
        "same_model_reward_budget",
    )
    missing = [key for key in keys if evidence.get(key) is not True]
    if missing:
        raise ValueError("learning comparison blocked: " + ", ".join(missing))
