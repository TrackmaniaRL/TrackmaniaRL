"""Artificial protocol records built from zero-filled buffers for parser tests.

These data are not a game capture or evidence of physical behavior.
"""

import json
from pathlib import Path
from struct import pack_into
from typing import Any

from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import PRODUCER_RVA

PID = 123
META = {
    "module_slide": 0x10000000,
    "participant": 0x40000000,
    "sim": 0x50000000,
    "controller": 0x60000000,
    "vehicle": 0x30000000,
}


def goal_facts(callback: int = 28490, event: int = 28505, dispatch: int = 28500) -> bytes:
    manager, table, table_owner, map_pointer = 0x70000000, 0x73000000, 0x74000000, 0x75000000
    writer = [bytearray(512), bytearray(512)]
    for i, row in enumerate(writer):
        row[:8] = b"TMWP0001"
        pack_into("<4I", row, 8, 512, 1, PID, PID)
        pack_into("<Q", row, 24, i + 1)
        pack_into(
            "<IIQQII",
            row,
            32,
            i,
            (0x1216411, 0x12167D6)[i],
            META["sim"],
            META["controller"],
            callback,
            (1231, 799)[i],
        )
        pack_into("<QQ4I", row, 248, META["participant"], META["participant"], 10, 11, 2200, 2200)
    a, b = writer
    regs = [0] * 23
    for i, value in {
        8: manager,
        9: META["participant"],
        12: event,
        14: dispatch,
        10: manager,
        1: 1,
    }.items():
        regs[i] = value
    pack_into("<23Q", a, 64, *regs)
    pack_into("<QQ", a, 360, manager, META["module_slide"] + 0x11FBA27)
    pack_into("<QQ", a, 392, 0x71000000, META["module_slide"] + 0x120AC95)
    pack_into("<II", a, 416, 0, dispatch)
    pack_into("<I", a, 440, 123456)
    pack_into("<I", a, 456, event)
    pack_into("<4Q", a, 464, 0x72000000, manager, 0x120AC95, 0x71000000)
    pack_into("<III", b, 288, 0, event, 1)
    for offset, selector in ((416, 10), (448, 11)):
        pack_into("<6I", b, offset, 12, selector, 0, 1, event, 0)
    rows = []
    for i, wr in enumerate(writer):
        row = bytearray(768)
        row[:8] = b"TMGF0001"
        pack_into("<4I", row, 8, 768, 1, PID, PID)
        pack_into("<QII", row, 24, i + 1, i, callback)
        pack_into("<4Q", row, 40, manager, table_owner, table, map_pointer)
        pack_into("<4I", row, 72, 0, 1, event, dispatch)
        pack_into("<3Q", row, 88, META["participant"], META["sim"], META["controller"])
        pack_into("<Q", row, 112, table)
        row[120:124] = b"Goal"
        row[120 + 0x51] = 1
        row[240:752] = wr
        pack_into("<Q", row, 752, META["module_slide"])
        rows.append(row)
    return bytes(rows[0] + rows[1])


def goal_receipt(clock: int, event: int | None = None) -> bytes:
    row = bytearray(96)
    row[:8] = b"TMGR0001"
    pack_into("<4I", row, 8, 96, 1, PID, PID)
    pack_into("<QII", row, 24, (clock - 2190) // 10 + 1, clock, 0)
    pack_into("<QII", row, 40, int(event is not None), event or 0, int(event is not None))
    pack_into(
        "<4Q", row, 56, META["participant"], META["vehicle"], META["sim"], META["module_slide"]
    )
    return bytes(row)


def vehicle(clock: int) -> bytes:
    raw = bytearray(1280)
    raw[:8] = b"TMVH0001"
    pack_into("<4I", raw, 8, 1280, 1, PID, PID)
    seq = (clock - 2190) // 10 + 1
    pack_into("<QII", raw, 24, seq, clock, 0 if seq == 1 else 10)
    pack_into("<3Q", raw, 40, META["participant"], META["vehicle"], 0)
    pack_into("<I", raw, 204, 1)
    pack_into("<I", raw, 220, clock)
    pack_into("<9f", raw, 208 + 0x2C, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    for wheel in range(4):
        raw[208 + 0xB8 + wheel * 0x2C] = 16
        pack_into("<I", raw, 208 + 0xD0 + wheel * 0x2C, 4)
    pack_into("<I", raw, 1240, PRODUCER_RVA)
    pack_into("<4I", raw, 1248, *([clock] * 4))
    pack_into("<2Q", raw, 1264, META["vehicle"] + 0x848, META["module_slide"] + PRODUCER_RVA)
    return bytes(raw)


def adherence(clock: int) -> bytes:
    raw = bytearray(128)
    raw[:8] = b"TMAD0001"
    pack_into("<4I", raw, 8, 128, 1, PID, PID)
    pack_into("<QII", raw, 24, (clock - 2190) // 10 + 1, clock, 10)
    pack_into("<4Q", raw, 40, META["participant"], META["vehicle"], 0x76000000, 0x77000000)
    pack_into(
        "<IIIIQfIII",
        raw,
        72,
        0x02000001,
        1,
        0x0A020000,
        1,
        META["vehicle"],
        0.75,
        0x121C7A0,
        0x1195D70,
        0xA48730,
    )
    pack_into("<II", raw, 120, clock, 0)
    return bytes(raw)


def collector(clock: int = 55000, event: int = 55007) -> bytes:
    raw = bytearray(896)
    raw[:8] = b"TMFN0001"
    pack_into("<4I", raw, 8, 896, 1, PID, PID)
    pack_into("<QII", raw, 24, 1, clock, clock - 10)
    pack_into("<4Q", raw, 40, META["sim"], META["controller"], META["participant"], META["vehicle"])
    pack_into("<IIQ", raw, 72, 1, event, 1)
    pack_into("<2Q", raw, 88, META["module_slide"], META["module_slide"] + 0x11834F2)
    pack_into("<II", raw, 104, 0x47FF, 1)
    data = memoryview(raw)[128:]
    data[:8] = b"TMRESULT"
    pack_into("<4I", data, 8, 768, 1, PID, PID)
    pack_into("<Q", data, 24, 1)
    pack_into("<2Q", data, 40, META["controller"], META["sim"])
    pack_into("<Q", data, 80, META["participant"])
    pack_into("<I", data, 128, clock)
    pack_into("<I", data, 148, 0x47FF)
    pack_into("<Q", data, 232, META["module_slide"] + 0x11834F3)
    pack_into("<I", data, 240, 13)
    pack_into("<I", data, 248, 1)
    return bytes(raw)


def material(clock: int = 55000) -> bytes:
    raw = bytearray(1536)
    raw[:8] = b"TMFM0001"
    pack_into("<4I", raw, 8, 1536, 1, PID, PID)
    pack_into("<QII", raw, 24, 1, clock, 0)
    pack_into(
        "<7Q",
        raw,
        40,
        META["sim"],
        META["controller"],
        META["participant"],
        META["vehicle"],
        0x78000000,
        0x79000000,
        META["module_slide"],
    )
    pack_into("<III", raw, 96, 1, 4, 16)
    pack_into("<I", raw, 108, 0x11834F2)
    pack_into("<4Q", raw, 176, 0x71000000, 0x72000000, 0x73000000, 0x74000000)
    for i in range(4):
        pack_into("<f", raw, 208 + i * 32 + 24, 1.0 if i < 2 else -1.0)
    pack_into("<Q", raw, 336, 0x7A000000)
    raw[344:488] = bytes([7]) * 144
    raw[640:1504] = vehicle(clock)[208:1072]
    return bytes(raw)


def goal_context() -> NativeGoalContext:
    return NativeGoalContext(PID, META.copy())


def goal_fixture() -> tuple[NativeGoalContext, dict[str, bytes]]:
    event = 28505
    receipts = b"".join(
        goal_receipt(clock, event if clock >= 28500 else None) for clock in range(2190, 28551, 10)
    )
    return goal_context(), {
        "facts.bin": goal_facts(),
        "initial.bin": goal_receipt(2200),
        "pending.bin": goal_receipt(28500, event),
        "active.bin": goal_receipt(28550, event),
        "receipts.bin": receipts,
    }


def paired_evidence() -> tuple[dict[str, bytes], dict[str, str]]:
    uid = "synthetic-map"
    score = {"NbCheckpoints": 1, "NbRespawns": 0, "Time": 28731, "Score": 0}

    def record(name: str) -> dict[str, Any]:
        return {
            "MapUid": uid,
            "IsValid": True,
            "FileName": name,
            "ValidatedResult": score,
            "DeclaredResult": score,
            "Inputs": "synthetic controls",
        }

    # The status string is a protocol token required by qualify_reference;
    # it does not describe an observed run in these artificial test inputs.
    client = {
        "status": "PASS_ACTUAL_FULL_V6_REFERENCE_UI_FINISH",
        "training_started": False,
        "learner_updates": 0,
        "owned_controller_closed_under_mutex": True,
        "actual_reset_start_time": 100,
        "actual_finish_snapshot": {"resolved_map_identifier": uid, "ui_sequence": 11, "player": 1},
        "ui_timing_observations": [
            {"start_time_ms": 100, "race_ms": 0, "ui_sequence": 0, "player": 1},
            {"start_time_ms": 100, "race_ms": 28731, "ui_sequence": 11, "player": 1},
        ],
    }
    observer = {
        "cleanup": {"remainingOwned": [], "cleanupErrors": [], "forced": []},
        "exits": {"plain-validator": 0},
        "identity": {"worker": {"pid": PID}},
    }
    # These names are strict qualify_reference API constants. The records are
    # constructed here; no actual replay, ghost or gameplay data is included.
    plain = [record("actual-window-pb.Replay.Gbx"), record("actual-window-pb.Ghost.Gbx")]
    observed = [record("actual-window-pb.Ghost.Gbx")]
    return {
        **{
            k: json.dumps(v).encode()
            for k, v in {
                "client": client,
                "observer": observer,
                "plain": plain,
                "observed": observed,
            }.items()
        },
        "goal": goal_facts(callback=30920, event=30931, dispatch=30930),
    }, {"server": "1" * 64, "map": "2" * 64, "shim": "3" * 64}


def write_fixtures(root: Path) -> None:
    """Replace the named historical capture files with explicitly artificial records."""
    import hashlib

    def emit(name: str, files: dict[str, bytes], context: dict[str, Any]) -> None:
        folder = root / name
        folder.mkdir(exist_ok=True)
        for key, data in files.items():
            (folder / key).write_bytes(data)
        context["origin"] = (
            "Synthetic records generated from zero-filled buffers; not engine captures."
        )
        context["files"] = {key: hashlib.sha256(data).hexdigest() for key, data in files.items()}
        (folder / "context.json").write_bytes(
            (json.dumps(context, indent=2) + "\n").encode("utf-8")
        )

    _, goal = goal_fixture()
    emit("goal_delivery380", goal, {"pid": PID, "meta": META.copy()})
    partial = {
        "vehicle.bin": vehicle(71990),
        "goals.bin": goal_facts(71950, 71951, 71950),
        "goal-receipt.bin": goal_receipt(71990, 71951),
    }
    emit(
        "terminal_partial537",
        partial,
        {"pid": PID, "meta": META.copy(), "previous_native_t": 71950, "held_native_t": 71990},
    )
    committed = {
        "vehicle.bin": vehicle(54990),
        "goals.bin": goal_facts(54980, 54992, 54990),
        "goal-receipt.bin": goal_receipt(54990, 54992),
        "adherence.bin": adherence(54990),
    }
    emit(
        "terminal_committed554",
        committed,
        {"pid": PID, "meta": META.copy(), "previous_native_t": 54950, "held_native_t": 54990},
    )
    final = {
        "collector.bin": collector(),
        "goals.bin": goal_facts(54990, 55007, 55000),
        "goal-receipt.bin": goal_receipt(55000, 55007),
        "vehicle.bin": vehicle(55000),
        "adherence.bin": adherence(55000),
    }
    emit("final_collector580", final, {"pid": PID, "meta": META.copy()})
    emit(
        "final_material584",
        {**final, "material.bin": material()},
        {"pid": PID, "meta": META.copy()},
    )
    evidence, pins = paired_evidence()
    files = {key + (".bin" if key == "goal" else ".json"): value for key, value in evidence.items()}
    emit("paired_finish403", files, {"asset_pins": pins})


if __name__ == "__main__":
    write_fixtures(Path(__file__).parents[2] / "fixtures/full_qr")
