"""Read recorded native visual fields; never simulate, control, or invent fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tarfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantized_pose_agrees(physical: float, visual: float) -> bool:
    """Cross-check millimetre quantization, including f32 projection boundaries.

    Both archived projections are f32. At a half-millimetre boundary the
    original internal coordinate can lie on either side of the rounded POST.
    Accept only the exact grid values reachable from its f32 rounding cell.
    This is a projection check, not a claim of bitwise simulation parity.
    """
    if not math.isfinite(physical) or not math.isfinite(visual):
        return False
    bits = struct.unpack("<I", struct.pack("<f", physical))[0]
    if physical == 0:
        previous, following = -(2.0**-149), 2.0**-149
    else:
        direction = 1 if physical > 0 else -1
        previous = struct.unpack("<f", struct.pack("<I", bits - direction))[0]
        following = struct.unpack("<f", struct.pack("<I", bits + direction))[0]
    low, high = (previous + physical) / 2, (following + physical) / 2
    if not math.isfinite(low) or not math.isfinite(high):
        return False
    first, last = math.floor(low * 1000 + 0.5), math.floor(high * 1000 + 0.5)
    if last - first > 2:
        return False
    return any(float32(i * float32(0.001)) == visual for i in range(first, last + 1))


def visual_records(raw: bytes, pid: int) -> dict[int, bytes]:
    if not raw or len(raw) % 1280:
        raise ValueError("Incomplete native vehicle capture")
    records: dict[int, bytes] = {}
    for at in range(0, len(raw), 1280):
        row = raw[at : at + 1280]
        if row[:8] != b"TMVH0001" or struct.unpack_from("<4I", row, 8) != (
            1280, 1, pid, pid
        ):
            raise ValueError("Unexpected capture format or process owner")
        clock = struct.unpack_from("<I", row, 32)[0]
        state = row[208:1072]
        if struct.unpack_from("<I", state, 12)[0] != clock:
            raise ValueError("Native visual timestamp differs from capture timestamp")
        rpm_cache = struct.unpack_from("<f", row, 64 + 0x38)[0]
        rpm_vis = struct.unpack_from("<f", state, 0x198)[0]
        if not math.isfinite(rpm_vis) or rpm_cache != rpm_vis:
            raise ValueError("Independent native RPM projections disagree")
        if clock in records and records[clock] != state:
            raise ValueError("Ambiguous visual states for one native tick")
        records[clock] = state
    return records


def prepare(physics_path: Path, schema_path: Path) -> dict[str, Any]:
    physics: dict[str, Any] = json.loads(physics_path.read_text(encoding="utf-8-sig"))
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    if physics["goal_ms"] != 17712 or schema["size"] != 864:
        raise ValueError("Wrong bound source or visual-state schema")
    offsets = {str(m["name"]): int(m["raw_offset_u16"]) for m in schema["members"]}
    anchors = {"Position": 80, "WorldVel": 92, "CurGear": 420,
               "InputSteer": 16, "FLSlipCoef": 188, "RLSlipCoef": 320}
    if any(offsets[name] != expected for name, expected in anchors.items()):
        raise ValueError("Windows reflection disagrees with native capture ABI anchors")
    provenance = physics["provenance"]
    archive = Path(provenance["archive"])
    if digest(archive) != provenance["archive_sha256"]:
        raise ValueError("Archived autonomous source hash changed")
    pid = int(provenance["native_pid"])
    with tarfile.open(archive, "r:gz") as source:
        stream = source.extractfile(f"trace/posts/vehicle-{pid}.bin")
        if stream is None:
            raise ValueError("Actual native visual capture missing")
        records = visual_records(stream.read(), pid)
    scalars = ["InputSteer", "InputGasPedal", "InputBrakePedal", "InputVertical",
               "FrontSpeed", "TurboTime", "SimulationTimeCoef", "BulletTimeNormed",
               "AirBrakeNormed", "SpoilerOpenNormed", "WingsOpenNormed", "WetnessValue01"]
    for wheel in ("FL", "FR", "RR", "RL"):
        scalars.extend(wheel + field for field in (
            "SteerAngle", "WheelRot", "WheelRotSpeed", "DamperLen", "SlipCoef",
            "Icing01", "TireWear01", "BreakNormedCoef"
        ))
    frames: list[dict[str, Any]] = []
    for frame in physics["frames"]:
        # Source resetNativeT=2200 is already proven by extract.py and1116 audit.
        state = records[int(frame["t"]) + 2200]
        for field, source_key in (("Position", "p"), ("WorldVel", "v")):
            actual = struct.unpack_from("<3f", state, offsets[field])
            if not all(quantized_pose_agrees(float(p), v)
                       for p, v in zip(frame[source_key], actual, strict=True)):
                raise ValueError(f"Visual {field} differs from independent physical POST")
        fields = {name: struct.unpack_from("<f", state, offsets[name])[0] for name in scalars}
        fields["RPM"] = struct.unpack_from("<f", state, 0x198)[0]
        if not all(math.isfinite(value) for value in fields.values()):
            raise ValueError("Nonfinite recorded visual field")
        frames.append({"t": frame["t"], "fields": fields,
                       "CurGear": struct.unpack_from("<I", state, offsets["CurGear"])[0]})
    ranges = {name: [min(f["fields"][name] for f in frames),
                     max(f["fields"][name] for f in frames)] for name in scalars + ["RPM"]}
    return {
        "status": "VERIFIED_RECORDED_VISUAL_FIELDS_NOT_ENCODED_REPLAY",
        "goal_ms": 17712, "frames": frames, "ranges": ranges,
        "provenance": {**provenance, "physics_json_sha256": digest(physics_path),
                       "reflection_schema_sha256": digest(schema_path),
                       "actual_visual_records": len(records),
                       "independent_position_velocity_comparisons": len(frames) * 2,
                       "pose_comparison": "Exact native0.001 grid within each independent f32 POST rounding cell; not bitwise parity",
                       "fields_with_unresolved_custom_getters": [
                           name for name, offset in offsets.items() if offset == 65535
                       ],
                       "new_simulations": 0, "state_or_input_changes": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse visual evidence overwrite")
    result = prepare(args.physics, args.schema)
    args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "frames": len(result["frames"]),
                      "ranges": result["ranges"], "provenance": result["provenance"]}))


if __name__ == "__main__":
    main()
