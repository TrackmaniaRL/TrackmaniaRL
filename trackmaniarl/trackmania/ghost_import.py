"""Build a map's training assets from one driven lap, such as a record ghost.

TrackmaniaRL normally needs a person to drive along both walls of every new map.
A ghost of a fast lap already holds what the agent needs to start on that map:

- its line, which becomes the reward centre line inside a corridor of virtual walls,
  so the lidar reports how far the car strays from the record line;
- its timing, which becomes the pace reference, so the reward can say how far
  ahead of or behind the record the car is at every point of the track.

The input is a JSON export of a ``.Replay.Gbx`` or ``.Ghost.Gbx`` (any GBX.NET-based
exporter; the format is in readme/trackmania.md): samples every 50 ms with position,
velocity, rotation, speed and the applied inputs, plus checkpoint times.

Engine RPM, gear, surface material and adherence are not recorded exactly in a
ghost. Frames built here leave the RPM, material and adherence fields at zero and
approximate slip from the ghost's per-wheel slip flags; the fields the lidar
geometry uses (position, velocity, heading, speed, timing, inputs) are exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstration_data import Demonstration
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT

_MIN_POINT_SPACING_M = 0.5
_INNER_WIDTH_OF_RADIUS = 0.8


@dataclass(frozen=True, slots=True)
class GhostLap:
    """One finished lap sampled at a fixed rate, cut at the finish."""

    map_uid: str
    times_ms: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    speeds: np.ndarray
    controls: np.ndarray
    slipping_wheels: np.ndarray
    checkpoint_times_ms: np.ndarray
    finish_time_ms: float
    input_times_ms: np.ndarray
    """When each input change reached the car; ``input_controls`` holds the new state."""
    input_controls: np.ndarray


def load_ghost_lap(path: Path, map_uid: str | None = None) -> GhostLap:
    """Read a ghost's JSON export and keep the samples up to the finish.

    A ``.Ghost.Gbx`` carries no map UID; pass the map's UID for one.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    map_uid = map_uid or str(document.get("map_uid") or "")
    if not map_uid:
        raise ValueError(f"{path} has no map UID; pass the map's UID")
    checkpoints = document.get("checkpoints") or []
    if not checkpoints:
        raise ValueError(f"{path} has no checkpoint times, so its finish is unknown")
    # A single-lap run finishes at its last checkpoint; exports do not always flag it.
    finish_time_ms = float(checkpoints[-1]["time_ms"])
    samples = [s for s in document["samples"] if 0 <= s["time_ms"] <= finish_time_ms]
    if len(samples) < 2:
        raise ValueError(f"{path} has fewer than two samples before the finish")
    extras = [s.get("extras") or {} for s in samples]
    return GhostLap(
        map_uid=map_uid,
        times_ms=np.asarray([s["time_ms"] for s in samples], dtype=np.float64),
        positions=np.asarray([s["position"] for s in samples], dtype=np.float64),
        velocities=np.asarray([s["velocity"] for s in samples], dtype=np.float64),
        rotations=np.asarray([s["rotation"] for s in samples], dtype=np.float64),
        speeds=np.asarray([s["speed"] for s in samples], dtype=np.float64),
        controls=np.asarray([_control(e) for e in extras], dtype=np.float64),
        slipping_wheels=np.asarray(
            [e.get("wheel_slip") or [False] * 4 for e in extras], dtype=np.float64
        ),
        checkpoint_times_ms=np.asarray([c["time_ms"] for c in checkpoints[:-1]], dtype=np.float64),
        finish_time_ms=finish_time_ms,
        **_input_events(document.get("inputs") or [], samples, extras),
    )


def _input_events(
    events: list[dict[str, Any]], samples: list[dict[str, Any]], extras: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    """Tick-exact input changes, or the sampled inputs when the export has none."""

    if not events:
        times = [s["time_ms"] for s in samples]
        return {
            "input_times_ms": np.asarray(times, dtype=np.float64),
            "input_controls": np.asarray([_control(e) for e in extras], dtype=np.float64),
        }
    controls = [
        (float(e.get("throttle", 0.0)), float(e.get("brake", 0.0)), float(e.get("steer", 0.0)))
        for e in events
    ]
    return {
        "input_times_ms": np.asarray([e["time_ms"] for e in events], dtype=np.float64),
        "input_controls": np.asarray(controls, dtype=np.float64),
    }


def _control(extras: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(extras.get("gas", 0.0)),
        float(extras.get("brake", 0.0)),
        float(extras.get("steer", 0.0)),
    )


def ghost_frames(lap: GhostLap) -> np.ndarray:
    """Telemetry frames in the plugin's 33-field layout, one per sample plus the finish.

    The last frame sits exactly at the finish time, extrapolated from the final
    sample's velocity, and is the only one with the finished flag set.
    """

    finish_dt_s = (lap.finish_time_ms - lap.times_ms[-1]) / 1_000.0
    times = np.append(lap.times_ms, lap.finish_time_ms)
    positions = np.vstack([lap.positions, lap.positions[-1] + lap.velocities[-1] * finish_dt_s])
    rows = np.append(np.arange(len(lap.times_ms)), len(lap.times_ms) - 1)
    frames = np.zeros((len(times), DEFAULT_TELEMETRY_FIELD_COUNT), dtype=np.float32)
    frames[:, 0] = np.searchsorted(lap.checkpoint_times_ms, times, side="right")
    frames[-1, 2] = 1.0
    frames[:, 3] = times
    frames[:, 4:7] = positions
    frames[:, 7:10] = lap.velocities[rows]
    frames[:, 10:13] = _rotate(lap.rotations[rows], np.array([0.0, 0.0, 1.0]))
    frames[:, 13:16] = _rotate(lap.rotations[rows], np.array([0.0, 1.0, 0.0]))
    frames[:, 16] = lap.speeds[rows]
    frames[:, 19:23] = lap.slipping_wheels[rows]
    frames[:, 27] = lap.slipping_wheels[rows].sum(axis=1)
    frames[:, 30] = lap.controls[rows, 2]
    frames[:, 31] = lap.controls[rows, 0]
    frames[:, 32] = lap.controls[rows, 1] > 0.5
    return frames


def _rotate(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector by each ``[x, y, z, w]`` quaternion."""

    axis, w = quaternions[:, :3], quaternions[:, 3:4]
    twice_cross = 2.0 * np.cross(axis, vector)
    return vector + w * twice_cross + np.cross(axis, twice_cross)


def corridor_boundaries(lap: GhostLap, half_width_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Virtual left and right walls ``half_width_m`` either side of the lap's line.

    On a tight turn the inner wall comes in closer, to at most 80% of the turn
    radius, so it never folds back over itself.
    """

    if half_width_m <= 0.0:
        raise ValueError("half_width_m must be positive")
    line = _moving_points(lap.positions)
    tangent = np.gradient(line[:, [0, 2]], axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    # Trackmania is left-handed with Y up: facing +Z, +X is to the driver's left.
    left = np.stack([tangent[:, 1], np.zeros(len(line)), -tangent[:, 0]], axis=1)
    left_turn = _signed_curvature(line, tangent)
    radius = 1.0 / np.maximum(np.abs(left_turn), 1e-9)
    inner = np.minimum(half_width_m, _INNER_WIDTH_OF_RADIUS * radius)
    left_width = np.where(left_turn > 0.0, inner, half_width_m)
    right_width = np.where(left_turn < 0.0, inner, half_width_m)
    return line + left * left_width[:, None], line - left * right_width[:, None]


def _moving_points(positions: np.ndarray) -> np.ndarray:
    kept = [positions[0]]
    for point in positions[1:]:
        if np.linalg.norm(point - kept[-1]) >= _MIN_POINT_SPACING_M:
            kept.append(point)
    if len(kept) < 3:
        raise ValueError("the lap barely moves, so it has no line to follow")
    return np.asarray(kept)


def _signed_curvature(line: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Heading change per metre; positive turns toward the driver's left."""

    heading = np.unwrap(np.arctan2(tangent[:, 0], tangent[:, 1]))
    distance = np.maximum(np.gradient(np.r_[0.0, np.cumsum(_steps(line))]), 1e-9)
    return np.asarray(np.gradient(heading) / distance)


def _steps(line: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(np.diff(line, axis=0), axis=1))


@dataclass(frozen=True, slots=True)
class GhostAssetRequest:
    """Where to write the assets built from one ghost lap."""

    ghost_path: Path
    map_path: Path
    output_dir: Path
    half_width_m: float = 8.0
    spacing_m: float = 2.0
    map_uid: str | None = None


@dataclass(frozen=True, slots=True)
class GhostAssets:
    map_uid: str
    geometry_path: Path
    pace_path: Path
    finish_time_s: float


def build_ghost_assets(request: GhostAssetRequest) -> GhostAssets:
    """Write corridor walls, a geometry asset and a pace reference for the ghost's map."""

    lap = load_ghost_lap(request.ghost_path, request.map_uid)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    stem = request.output_dir / lap.map_uid
    left, right = corridor_boundaries(lap, request.half_width_m)
    left_path, right_path = Path(f"{stem}-left.npy"), Path(f"{stem}-right.npy")
    np.save(left_path, left.astype(np.float32))
    np.save(right_path, right.astype(np.float32))
    geometry_path = build_geometry_asset(
        GeometryBuildRequest(
            Path(f"{stem}.geometry.npz"),
            left_path,
            right_path,
            lap.map_uid,
            request.map_path,
            request.spacing_m,
            already_paired=True,
        )
    )
    geometry = BoundaryGeometry(geometry_path, expected_map_uid=lap.map_uid)
    pace_path = Path(f"{stem}.pace.npz")
    np.savez_compressed(
        pace_path,
        map_uid=np.asarray(lap.map_uid),
        geometry_sha256=np.asarray(geometry.sha256),
        frames=ghost_frames(lap),
        finish_time_s=np.asarray(lap.finish_time_ms / 1_000.0),
    )
    return GhostAssets(lap.map_uid, geometry_path, pace_path, lap.finish_time_ms / 1_000.0)


DEMONSTRATION_STEP_MS = 10.0
"""One physics tick: the cadence the demonstration quality gate expects of native telemetry."""


def ghost_demonstration(lap: GhostLap, geometry_sha256: str) -> Demonstration:
    """A demonstration archive for ``lap`` on the geometry with ``geometry_sha256``.

    Ghost samples are 50 ms apart, too sparse for the demonstration gate and for
    short taps. Frames are rebuilt every 10 ms tick: position by cubic Hermite
    interpolation between samples (each sample's velocity is its slope), the
    other motion fields linearly, and the controls from the ghost's tick-exact
    input changes rather than from the samples.
    """

    if not np.any(lap.input_controls):
        # A line-only export (no input events, no inputs in the samples) reads as a
        # lap driven with nothing pressed. It still gives geometry and pace.
        raise ValueError("ghost records no inputs, so it cannot be a demonstration")
    samples = ghost_frames(lap)
    times = np.append(np.arange(0.0, lap.finish_time_ms, DEMONSTRATION_STEP_MS), lap.finish_time_ms)
    frames = _interpolated_frames(samples, times)
    frames[:, 0] = np.searchsorted(lap.checkpoint_times_ms, times, side="right")
    frames[:, 2] = 0.0
    frames[-1, 2] = 1.0
    change = np.searchsorted(lap.input_times_ms, times, side="right") - 1
    controls = np.where(change[:, None] >= 0, lap.input_controls[np.maximum(change, 0)], 0.0)
    frames[:, 30] = controls[:, 2]
    frames[:, 31] = controls[:, 0]
    frames[:, 32] = controls[:, 1] > 0.5
    _, table = build_brake_tap_action_table()
    applied = controls[:-1].astype(np.float32)
    return Demonstration(
        map_uid=lap.map_uid,
        geometry_sha256=geometry_sha256,
        action_repeat_frames=1,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(applied, table),
        controls=applied,
        finish_time_s=lap.finish_time_ms / 1_000.0,
    )


def _interpolated_frames(samples: np.ndarray, times: np.ndarray) -> np.ndarray:
    sample_times = samples[:, 3].astype(np.float64)
    left = np.clip(np.searchsorted(sample_times, times, side="right") - 1, 0, len(samples) - 2)
    span_s = (sample_times[left + 1] - sample_times[left]) / 1_000.0
    u = np.clip((times - sample_times[left]) / 1_000.0 / np.maximum(span_s, 1e-9), 0.0, 1.0)
    before, after = samples[left].astype(np.float64), samples[left + 1].astype(np.float64)
    frames = before + (after - before) * u[:, None]
    u = u[:, None]
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    frames[:, 4:7] = (
        h00 * before[:, 4:7]
        + h10 * span_s[:, None] * before[:, 7:10]
        + h01 * after[:, 4:7]
        + h11 * span_s[:, None] * after[:, 7:10]
    )
    for start in (10, 13):
        vectors = frames[:, start : start + 3]
        frames[:, start : start + 3] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    held = u[:, 0] < 1.0
    frames[:, 19:28] = np.where(held[:, None], before[:, 19:28], after[:, 19:28])
    frames[:, 3] = times
    return frames.astype(np.float32)
