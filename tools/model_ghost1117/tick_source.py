"""Export this audited AI trace to TICK text without running game controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

POLICY_SHA = "bb22eba32ca1da83da6f3346e7ad74e67cc55fdaaa455d486f0792e94221b0d4"
GHOST_SHA = "b370c16b01f60c7cc993b56a98b22cb7a8f14069abd970209d605e70d74860c0"
ARCHIVE_SHA = "f6ee8407107c298ee75995a3889899faf706b90ac08727f295ce45274208f2a5"


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected JSON object with string keys")
    return cast(dict[str, object], value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected JSON array")
    return cast(list[object], value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(trace: Path, metadata: Path, validated_ghost: Path, output: Path) -> None:
    """Write only a fresh output directory; validate all input clocks and provenance."""
    if _sha(validated_ghost) != GHOST_SHA:
        raise ValueError("Wrong independently validated model ghost")
    data = _object(json.loads(trace.read_text(encoding="utf-8-sig")))
    provenance = _object(data["provenance"])
    if (provenance.get("policy_sha256") != POLICY_SHA
            or provenance.get("archive_sha256") != ARCHIVE_SHA
            or provenance.get("teacher_files_used") is not False
            or provenance.get("actual_model_commands_verified") != 1775):
        raise ValueError("Wrong model trace provenance")
    if data.get("goal_ms") != 17712 or data.get("physical_end_ms") != 17750:
        raise ValueError("Wrong native Goal or physical clock")
    validation = _object(json.loads(metadata.read_text(encoding="utf-8-sig")))
    seed = validation.get("validation_seed")
    if (validation.get("status") != "READ_ONLY_VALIDATION_METADATA"
            or type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF):
        raise ValueError("Missing validated uint32 seed")
    frames = _array(data["frames"])
    if len(frames) != 1776:
        raise ValueError("Expected all 1776 actual native frames")
    states: list[tuple[int, int, int]] = []
    for index, value in enumerate(frames):
        frame = _object(value)
        if type(frame.get("t")) is not int or frame["t"] != index * 10:
            raise ValueError("Noncanonical or missing actual native tick")
        sgb = _array(frame["sgb"])
        if (len(sgb) != 3 or any(type(v) not in (int, float) for v in sgb)
                or sgb[0] not in (-1, 0, 1)
                or sgb[1] not in (0, 1) or sgb[2] not in (0, 1)):
            raise ValueError("Expected exact recorded digital controls")
        steer = int(cast(float, sgb[0])) * 127
        gas = int(cast(float, sgb[1]))
        brake = int(cast(float, sgb[2]))
        states.append((steer, gas, brake))
    if states[0] != (0, 0, 0):
        raise ValueError("Initial observed native state is not neutral")
    lines = [f"0 seed {seed}"]
    names = ("steer", "accel", "brake")
    previous: tuple[int, int, int] | None = None
    for index, state in enumerate(states):
        for channel, name in enumerate(names):
            if previous is None or state[channel] != previous[channel]:
                lines.append(f"{index * 10} {name} {state[channel]}")
        previous = state
    source = "\n".join(lines) + "\n"
    # Expand the actual text back onto every tick to catch any timing shift.
    events: dict[int, list[tuple[str, int]]] = {}
    for line in source.splitlines():
        time, name, raw = line.split()
        events.setdefault(int(time), []).append((name, int(raw)))
    current = {"steer": 0, "accel": 0, "brake": 0}
    for index, expected in enumerate(states):
        for name, value in events.get(index * 10, []):
            if name == "seed":
                if index != 0 or value != seed:
                    raise ValueError("Seed clock changed")
            else:
                current[name] = value
        if tuple(current[name] for name in names) != expected:
            raise ValueError("Text round trip changed an actual model command")
    output.mkdir(parents=True, exist_ok=False)
    (output / "TrackmaniaRL-AI-17712.txt").write_text(source, encoding="utf-8")
    receipt = {
        "status": "PASS_ALL_1776_RECORDED_TICKS_EXACT_TEXT_ROUNDTRIP",
        "purpose": "native client replay of recorded AI controls, not live inference",
        "source_policy_sha256": POLICY_SHA,
        "source_trace_sha256": _sha(trace),
        "validated_ghost_sha256": GHOST_SHA,
        "validation_metadata_sha256": _sha(metadata),
        "source_archive_sha256": ARCHIVE_SHA,
        "validation_seed": seed,
        "native_goal_ms": 17712,
        "physical_end_ms": 17750,
        "ticks": len(states),
        "text_actions": len(lines),
        "text_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "synthetic_trailing_release": False,
        "teacher_racing_controls_imported": False,
        "new_simulations": 0,
        "public_submission": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(receipt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("validated_ghost", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.trace, args.metadata, args.validated_ghost, args.output)


if __name__ == "__main__":
    main()
