"""Synthetic V7 wire fixtures; no bytes or measurements originate from the game."""

from __future__ import annotations

import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v7, features

PID = 1234
META = {"controller": 0x100000, "sim": 0x200000, "input_base": 0x300000, "input_count": 4152}


def reader(index: int, anchor: int) -> bytes:
    raw = bytearray(256)
    raw[:8] = b"TMREAD01"
    struct.pack_into("<4I", raw, 8, 256, 1, PID, PID)
    struct.pack_into("<2Q", raw, 40, META["controller"], META["sim"])
    struct.pack_into("<Q", raw, 88, META["input_base"])
    struct.pack_into("<Q", raw, 104, META["input_count"])
    struct.pack_into("<4I", raw, 112, index, META["input_count"], anchor + 10, anchor)
    struct.pack_into("<I", raw, 148, 447)
    struct.pack_into("<I3f4I", raw, 152, 2, 0.0, 0.0, 0.0, 11, 12, 13, 14)
    raw[184:188] = raw[156:160]
    struct.pack_into("<I", raw, 188, META["input_count"])
    struct.pack_into("<2I", raw, 208, 1, 11)
    return bytes(raw)


def post(native_t: int, hit: int, channels: tuple[float, float, float]) -> bytes:
    steer, gas, brake = channels
    raw = bytearray(256)
    raw[:8] = b"TMPOST03"
    struct.pack_into("<4I", raw, 8, 256, 3, PID, PID)
    struct.pack_into("<Q", raw, 24, hit)
    struct.pack_into("<2Q", raw, 40, META["controller"], META["sim"])
    struct.pack_into("<4f", raw, 88, gas, brake, steer, steer)
    struct.pack_into("<II", raw, 112, native_t, native_t)
    struct.pack_into("<II", raw, 136, 10, 10)
    struct.pack_into("<II", raw, 144, native_t, 75)
    struct.pack_into("<10f", raw, 152, 1, 0, 0, 0, 10, 20, 30, 1, 2, 3)
    struct.pack_into("<fIfII", raw, 192, 100, 1, 100, 1, 0)
    struct.pack_into("<I", raw, 212, 10)
    return bytes(raw)


def frame(  # noqa: PLR0913 -- fixed fields in the simulated wire header
    kind: int, seq: int, count: int, epoch: int, event: bytes, receipt: bytes = b""
) -> bytes:
    raw = bytearray(448)
    raw[:8] = b"TMR7\x07" + bytes((kind, 0, 0))
    struct.pack_into("<IIQII", raw, 8, seq, count, epoch, 416, 4 if kind == 6 else 0)
    raw[32:288] = event
    if receipt:
        assert len(receipt) == 160
        raw[288:] = receipt
    return bytes(raw)


def capture(index: int = 0) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build independently authored input events and the corresponding protocol frames."""
    if index not in (0, 1):
        raise ValueError("synthetic fixture index")
    action, count = (39, 1) if index == 0 else (2, 5)
    anchor, selected, epoch, seq = 2200, 100, 10, 15
    original_reader = reader(selected, anchor)
    old = original_reader[152:184]
    before_channels = old[4:16]
    afters = [struct.pack("<3f", *channels) for channels in features.action_sgb(action)[:count]]
    receipt = (
        struct.pack("<4I", selected, selected, count, 2)
        + b"".join(before_channels + after for after in afters)
        + bytes(144 - count * 24)
    )
    audits = []
    posts = []
    for number in range(count):
        new = old[:4] + afters[number] + old[16:]
        audit = bytearray(256)
        audit[:8] = b"TMAUD003"
        struct.pack_into("<4I", audit, 8, 256, 2, PID, PID)
        struct.pack_into("<Q", audit, 24, 100 + number)
        audit[40:72], audit[72:104] = old, new
        audit[104:136], audit[136:140] = old, old[:4]
        audit[140:152], audit[152:168] = afters[0], old[16:]
        audit[168:172], audit[172:176] = before_channels[:4], afters[number][:4]
        struct.pack_into("<4I", audit, 176, selected, selected + number, count, anchor)
        struct.pack_into("<Q", audit, 192, epoch)
        struct.pack_into("<4I", audit, 200, seq, 1, selected, 12)
        struct.pack_into("<Q", audit, 216, META["input_base"])
        struct.pack_into("<Q", audit, 232, META["input_base"] - 4 + (selected + number) * 32)
        struct.pack_into("<Q", audit, 240, META["input_base"] - 4 + selected * 32)
        struct.pack_into("<2I", audit, 248, count, number)
        audits.append(bytes(audit))
        posts.append(
            post(anchor + (number + 1) * 10, 200 + number, features.action_sgb(action)[number])
        )
    next_reader = reader(selected + count, anchor + count * 10)
    frames = [
        frame(6, seq, count, epoch, original_reader, receipt),
        frame(1, seq, count, epoch + 1, posts[-1]),
        frame(3, seq + 1, 0, epoch + 2, next_reader),
    ]
    before = {"nativeT": anchor, "readerHex": original_reader.hex(), "epoch": epoch}
    authorization = {
        "anchor": anchor,
        "ownP": selected,
        "first": selected,
        "n": count,
        "actionId": action,
        "epoch": epoch,
        "sequence": seq,
        "originalRecordHex": [old.hex()] * count,
        "receiptHex": receipt.hex(),
        "auditEventHex": [event.hex() for event in audits],
        "nativeEventHex": [event.hex() for event in posts],
    }
    packet = {
        "profile": codec_v7.PROFILE_ID,
        "pid": PID,
        "meta": META.copy(),
        "epoch": epoch + 2,
        "boundaryHex": posts[-1].hex(),
        "readerHex": next_reader.hex(),
        "eof": False,
        "eofStage": None,
        "responseFramesHex": [event.hex() for event in frames],
        "authorization": authorization,
    }
    return before, {}, packet
