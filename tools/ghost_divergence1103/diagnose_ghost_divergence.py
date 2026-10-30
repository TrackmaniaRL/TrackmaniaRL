"""Compare observed ghost/headless samples without interpolation or clock fitting.

Run with ``python -m tools.ghost_divergence1103.diagnose_ghost_divergence``.
Inputs are decoded JSON; the CLI writes only an explicitly requested NEW report.
It never launches a simulator, imports training code, or changes its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Vector = tuple[float, float, float]
SCHEMA = "tmrl.ghost-divergence-series/v1"


def file_path(path: Path) -> Path:
    """Support long private Windows evidence paths without relocating anything."""
    value = os.path.abspath(path)
    if os.name == "nt" and len(value) >= 248 and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def mapping(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise ValueError("JSON array required")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric value required")
    return float(value)


def tick(value: Json) -> int:
    if type(value) is not int or value % 10:
        raise ValueError("literal integer timestamp on the ten-ms grid required")
    return value


def vector(value: Json) -> Vector | None:
    if value is None:
        return None
    values = array(value)
    if len(values) != 3:
        raise ValueError("three coordinates required")
    return number(values[0]), number(values[1]), number(values[2])


def read_json(path: Path) -> dict[str, Json]:
    raw = file_path(path).read_bytes()
    if len(raw) > 64 * 1024**2:
        raise ValueError("diagnostic JSON exceeds64MiB")
    value = cast(Json, json.loads(raw.decode("utf-8-sig")))
    return mapping(value)


@dataclass(frozen=True)
class Frame:
    time_ms: int
    position: Vector | None
    velocity: Vector | None
    control: Vector | None
    contacts: Json = None
    phase: str = "unclassified"


@dataclass(frozen=True)
class Series:
    map_uid: str
    frames: dict[int, Frame]
    provenance: dict[str, Json]


def load_series(path: Path) -> Series:
    document = read_json(path)
    if document.get("schema") != SCHEMA:
        raise ValueError("normalized decoded diagnostic series required")
    uid = document.get("map_uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("explicit map UID required")
    if document.get("coordinate_frame") != "world_xyz_y_up":
        raise ValueError("explicit world XYZ/Y-up coordinates required")
    frames: dict[int, Frame] = {}
    previous: int | None = None
    for value in array(document.get("frames")):
        row = mapping(value)
        time = tick(row.get("time_ms"))
        if previous is not None and time <= previous:
            raise ValueError("strictly increasing unique timestamps required")
        control = vector(row.get("control_sgb"))
        if control is not None and not (
            -1 <= control[0] <= 1 and control[1] in (0, 1) and control[2] in (0, 1)
        ):
            raise ValueError("decoded steer/gas/brake outside digital control domain")
        frames[time] = Frame(
            time, vector(row.get("position_m")), vector(row.get("velocity_m_s")),
            control, row.get("contacts"), str(row.get("phase", "unclassified")),
        )
        previous = time
    if not frames:
        raise ValueError("nonempty decoded series required")
    return Series(uid, frames, mapping(document.get("provenance", {})))


@dataclass(frozen=True)
class Settings:
    position_threshold_m: float = 1e-3
    velocity_threshold_m_s: float = 1e-3
    input_start_ms: int = 0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (
            self.position_threshold_m, self.velocity_threshold_m_s,
        )):
            raise ValueError("positive finite thresholds required")
        tick(self.input_start_ms)


def distance(left: Vector | None, right: Vector | None) -> float | None:
    return None if left is None or right is None else math.dist(left, right)


def frame_json(frame: Frame | None) -> Json:
    if frame is None:
        return None
    return {
        "position_m": None if frame.position is None else list(frame.position),
        "velocity_m_s": None if frame.velocity is None else list(frame.velocity),
        "control_sgb": None if frame.control is None else list(frame.control),
        "contacts": frame.contacts, "phase": frame.phase,
    }


def cadence(series: Series, field: str) -> dict[str, Json]:
    times = [t for t, frame in series.frames.items() if getattr(frame, field) is not None]
    gaps = Counter(b - a for a, b in pairwise(times))
    return {
        "observed_count": len(times), "first_ms": times[0] if times else None,
        "last_ms": times[-1] if times else None,
        "gap_histogram_ms": {str(k): v for k, v in sorted(gaps.items())},
    }


def context(series: tuple[Series, Series], at: int) -> list[Json]:
    ghost, headless = series
    result: list[Json] = []
    for time in range(at - 100, at + 101, 10):
        left, right = ghost.frames.get(time), headless.frames.get(time)
        result.append({
            "time_ms": time, "ghost": frame_json(left), "headless": frame_json(right),
            "position_error_m": None if left is None or right is None else distance(
                left.position, right.position
            ),
            "velocity_error_m_s": None if left is None or right is None else distance(
                left.velocity, right.velocity
            ),
        })
    return result


def compare(series: tuple[Series, Series], settings: Settings) -> dict[str, Json]:
    ghost, headless = series
    if ghost.map_uid != headless.map_uid:
        raise ValueError("ghost and headless map UIDs differ")
    first: dict[str, Json] = {}
    last_matching: dict[str, int] = {}
    compared = {"position": 0, "velocity": 0, "input": 0}
    differences = {"position": 0, "velocity": 0, "input": 0}
    maximum: dict[str, Json] = {"position": None, "velocity": None}
    for time in sorted(ghost.frames.keys() & headless.frames.keys()):
        left, right = ghost.frames[time], headless.frames[time]
        errors = {
            "position": (distance(left.position, right.position), settings.position_threshold_m),
            "velocity": (distance(left.velocity, right.velocity), settings.velocity_threshold_m_s),
        }
        for name, (error, threshold) in errors.items():
            if error is None:
                continue
            compared[name] += 1
            prior_maximum = maximum[name]
            if prior_maximum is None or error > number(prior_maximum):
                maximum[name] = error
            if error <= threshold:
                last_matching[name] = time
            else:
                differences[name] += 1
                if name not in first:
                    previous = last_matching.get(name)
                    first[name] = {
                        "first_observed_difference_ms": time, "error": error,
                        "last_observed_within_threshold_ms": previous,
                        "possible_first_tick_interval_ms": None if previous is None else [
                            previous + 10, time
                        ],
                        "context": context(series, time),
                    }
        if left.control is None or right.control is None:
            continue
        different = left.control != right.control
        if different and "input_including_reset" not in first:
            first["input_including_reset"] = {
                "first_observed_difference_ms": time, "context": context(series, time)
            }
        if time >= settings.input_start_ms:
            compared["input"] += 1
            differences["input"] += int(different)
            if different and "input" not in first:
                first["input"] = {
                    "first_observed_difference_ms": time, "context": context(series, time)
                }
    ghost_times = {t for t, f in ghost.frames.items() if f.position is not None}
    native_times = {t for t, f in headless.frames.items() if f.position is not None}
    overlaps = bool(ghost_times & native_times)
    state_grid_complete = overlaps and all(
        t in ghost_times and t in native_times for t in range(
            max(min(ghost_times), min(native_times)),
            min(max(ghost_times), max(native_times)) + 1, 10
        )
    )
    status = "DIFFERENCES_OBSERVED" if first else "NO_DIFFERENCES_AT_SHARED_OBSERVATIONS"
    if not all(compared.values()):
        status = "INSUFFICIENT_SHARED_OBSERVATIONS"
    return {
        "schema": "tmrl.ghost-divergence-report/v1", "map_uid": ghost.map_uid,
        "status": status,
        "thresholds": {"position_m": settings.position_threshold_m,
                       "velocity_m_s": settings.velocity_threshold_m_s},
        "input_comparison_starts_ms": settings.input_start_ms,
        "compared_samples": dict(compared), "different_samples": dict(differences),
        "maximum_observed_error": maximum, "first_differences": first,
        "complete_ten_ms_state_grid": state_grid_complete,
        "cadence": {"ghost_position": cadence(ghost, "position"),
                    "ghost_velocity": cadence(ghost, "velocity"),
                    "ghost_control": cadence(ghost, "control"),
                    "headless_position": cadence(headless, "position")},
        "countdown_and_reset_context": context(series, 0),
        "provenance": {"ghost": ghost.provenance, "headless": headless.provenance},
        "limitations": [
            "Missing states remain null. No interpolation, input holding or automatic time shift.",
            "A threshold crossing is a recorded-data difference, not proof of a physics bug.",
            "Packed ghost velocity and position precision may exceed the requested thresholds.",
            "An exact first10ms physics divergence requires observed reference states every10ms.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghost", type=Path, required=True)
    parser.add_argument("--headless", type=Path, required=True)
    parser.add_argument("--position-threshold-m", type=float, default=1e-3)
    parser.add_argument("--velocity-threshold-m-s", type=float, default=1e-3)
    parser.add_argument("--input-start-ms", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ghost_path, headless_path = Path(args.ghost), Path(args.headless)
    report = compare((load_series(ghost_path), load_series(headless_path)), Settings(
        float(args.position_threshold_m), float(args.velocity_threshold_m_s),
        int(args.input_start_ms)
    ))
    report["input_sha256"] = {
        "ghost": hashlib.sha256(file_path(ghost_path).read_bytes()).hexdigest(),
        "headless": hashlib.sha256(file_path(headless_path).read_bytes()).hexdigest(),
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        with file_path(Path(args.output)).open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
