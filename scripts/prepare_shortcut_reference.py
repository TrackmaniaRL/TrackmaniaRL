"""Prepare reference evidence only; never start a simulator or update a model.

The input is an explicit, hash-pinned RE ghost export. Recorded poses are useful
for a reference corridor, but are neither full simulator snapshots nor replay
transitions. Resampling controls below is diagnostic, not a successful replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb

MAX_EXPORT_BYTES = 16 * 1024**2


def read_export(archive: Path, member: str, expected_sha256: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member)
        if info.file_size > MAX_EXPORT_BYTES:
            raise ValueError("reference export exceeds 16 MiB")
        raw = source.read(info)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("reference export SHA-256 mismatch")
    return json.loads(raw)


def control_plan(document: dict[str, Any], map_uid: str) -> tuple[np.ndarray, np.ndarray]:
    if document["metadata"]["map_uid_literals"] != [map_uid]:
        raise ValueError("reference map UID mismatch")
    archives = document["input_archives"]
    if len(archives) != 1 or archives[0]["packet_period_ms"] != 10:
        raise ValueError("one literal 10 ms input archive required")
    packets = [p for p in archives[0]["packets"] if p["race_time_ms"] >= 0]
    recorded_times = [p["race_time_ms"] for p in packets]
    if any(type(value) is not int for value in recorded_times):
        raise ValueError("recorded control timestamps must be literal integer milliseconds")
    times = np.asarray(recorded_times, dtype=np.int64)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) != 10):
        raise ValueError("consecutive recorded controls from race time zero required")
    if any(p.get("respawn_input") is not False for p in packets):
        raise ValueError("respawn or unknown respawn controls cannot guide a clean lap")
    controls = np.asarray(
        [[p["steer_normalized"], p["accel"], p["brake"]] for p in packets],
        dtype=np.float32,
    )
    if controls.shape != (len(times), 3) or not np.isfinite(controls).all():
        raise ValueError("finite recorded steer/gas/brake triplets required")
    return times, controls


def exact_action_ids(controls: np.ndarray) -> np.ndarray:
    table: dict[bytes, int] = {}
    for action in range(78):
        table.setdefault(struct.pack("<3f", *action_sgb(action)[0]), action)
    try:
        return np.asarray([table[row.tobytes()] for row in controls], dtype=np.int64)
    except KeyError as error:
        raise ValueError("a reference command is outside the exact native action table") from error


def cadence_report(times: np.ndarray, controls: np.ndarray, step_ms: int) -> dict[str, Any]:
    if step_ms < 10 or step_ms % 10:
        raise ValueError("decision cadence must be a positive multiple of 10 ms")
    indices = np.arange(len(times))
    held = controls[(indices // (step_ms // 10)) * (step_ms // 10)]
    different = np.any(held.view(np.uint32) != controls.view(np.uint32), axis=1)
    changed = np.any(controls[1:].view(np.uint32) != controls[:-1].view(np.uint32), axis=1)
    return {
        "decision_interval_ms": step_ms,
        "source_control_changes": int(changed.sum()),
        "off_grid_changes": int(np.count_nonzero(times[1:][changed] % step_ms)),
        "changed_source_ticks": int(different.sum()),
        "changed_source_tick_fraction": float(different.mean()),
        "exact_source_controls_preserved": not bool(different.any()),
        "native_replay_qualified": False,
    }


def reference_poses(document: dict[str, Any]) -> dict[str, np.ndarray]:
    samples = document["samples"]
    result = {
        "record_time_ms": np.asarray([p["record_time_ms"] for p in samples], dtype=np.int64),
        "position_m": np.asarray([p["position_m"] for p in samples], dtype=np.float32),
        "linear_velocity_m_s": np.asarray(
            [p["linear_velocity_m_s"] for p in samples], dtype=np.float32
        ),
        "orientation_xyzw": np.asarray([p["orientation_xyzw"] for p in samples], dtype=np.float32),
    }
    count = len(samples)
    expected = {
        "record_time_ms": (count,),
        "position_m": (count, 3),
        "linear_velocity_m_s": (count, 3),
        "orientation_xyzw": (count, 4),
    }
    if count < 2 or any(
        a.shape != expected[k] or not np.isfinite(a).all() for k, a in result.items()
    ):
        raise ValueError("complete finite recorded pose arrays required")
    if np.any(np.diff(result["record_time_ms"]) <= 0):
        raise ValueError("reference sample timestamps must increase strictly")
    return result


def corridor_distances(positions: np.ndarray, center: np.ndarray) -> np.ndarray:
    if center.ndim != 2 or center.shape[1] != 3 or len(center) < 2:
        raise ValueError("centerline must have shape (points >= 2, 3)")
    if not np.isfinite(center).all():
        raise ValueError("centerline must be finite")
    starts, directions = center[:-1], np.diff(center, axis=0)
    squared = np.sum(directions.astype(np.float64) ** 2, axis=1)
    if np.any(squared == 0):
        raise ValueError("centerline contains a zero-length segment")
    distances = []
    for position in positions:
        offset = position - starts
        projection = np.clip(np.sum(offset * directions, axis=1) / squared, 0, 1)
        delta = offset - projection[:, None] * directions
        distances.append(math.sqrt(float(np.min(np.sum(delta**2, axis=1)))))
    return np.asarray(distances)


def prepare(args: argparse.Namespace) -> None:
    document = read_export(args.archive, args.member, args.sha256)
    times, controls = control_plan(document, args.map_uid)
    ids = exact_action_ids(controls)
    poses = reference_poses(document)
    with np.load(args.geometry, allow_pickle=False) as geometry:
        if str(geometry["map_uid"].item()) != args.map_uid:
            raise ValueError("geometry map UID mismatch")
        recorded_count = int(geometry["recorded_count"].item())
        distances = corridor_distances(poses["position_m"], geometry["center"][:recorded_count])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "source-controls.npz",
        race_time_ms=times,
        steer_gas_brake=controls,
        native_action_ids=ids,
    )
    np.savez_compressed(output / "reference-poses.npz", **poses)
    summary = {
        "schema": "tmrl.shortcut-reference-preparation/v1",
        "status": "PREPARED_NOT_TRAINING_QUALIFIED",
        "map_uid": args.map_uid,
        "source_export_sha256": args.sha256,
        "geometry_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        "declared_reference_finish_ms": document["metadata"]["result_race_time_ms"],
        "recorded_pose_end_ms": int(poses["record_time_ms"][-1]),
        "source_parser_pin": document.get("parser_pin"),
        "source_capabilities": document["capabilities"],
        "poses_are_full_simulator_states": False,
        "generated_training_transitions": 0,
        "learner_updates": 0,
        "native_sessions_started": 0,
        "cadences": [cadence_report(times, controls, step) for step in (10, 20, 50)],
        "distance_from_current_centerline_m": {
            "max": float(distances.max()),
            "recorded_poses_beyond_15m": int(np.count_nonzero(distances > 15)),
            "warning": "Geometric diagnostic only; not a reward or native replay qualification",
        },
    }
    for name in ("source-controls.npz", "reference-poses.npz"):
        summary.setdefault("artifacts", {})[name] = hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()
    (output / "reference-audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
