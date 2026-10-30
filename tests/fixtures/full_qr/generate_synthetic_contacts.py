"""Create wholly artificial protocol samples for contact decoder unit tests.

These records describe invented identifiers and values, not game observations.
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
P = struct.pack_into


def dump(name: str, files: dict[str, bytes], info: dict[str, Any]) -> None:
    root = ROOT / name
    for path, data in files.items():
        (root / path).write_bytes(data)
    (root / "context.json").write_bytes((json.dumps(info, indent=2) + "\n").encode("utf-8"))


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


base = {
    "pid": 127,
    "birth": 550,
    "module_base": 0x10000000,
    "vehicle": 0x20000,
    "world": 0x30000,
    "physical_state": 0x40000,
    "tracer_tid": 129,
    "wheel_count": 4,
}
wheels = (0x50004, 0x50030, 0x5005C, 0x50088)
owner = 0x60000
vehicle_model = 0x70000
header = bytes(range(32))
thresholds = struct.pack("<4f", 40.0, 10.0, 20.0, 100.0)
categories = (0, 1, 0, 1)
wheel_headers = tuple(
    struct.pack("<4I", 100 + i, categories[i], 200 + i, 300 + i) for i in range(4)
)


def wheel_record(i: int) -> bytes:
    raw = bytearray(2048)
    raw[:8] = b"TMPI0001"
    P("<4I", raw, 8, 2048, 1, base["pid"], base["tracer_tid"])
    P("<2Q", raw, 24, i + 1, base["module_base"] + 0xAAD9F0)
    P("<QII", raw, 48, 1, 4, 5)
    P("<2I", raw, 72, 2200 + i * 10, 0)
    P("<Q", raw, 80 + 11 * 8, 0x80000)
    P("<Q", raw, 80 + 12 * 8, i % 4)
    P("<Q", raw, 80 + 13 * 8, base["vehicle"])
    P("<Q", raw, 80 + 14 * 8, base["world"])
    P("<Q", raw, 80 + 16 * 8, base["module_base"] + 0xAAD9F0)
    P("<Q", raw, 304, base["module_base"] + 0xAAD38E + 5)
    P("<Q", raw, 848, base["birth"])
    P("<I", raw, 928, 122)
    raw[936 + 0x18] = 9 if i % 2 == 0 else 16
    P("<3f", raw, 936 + 0x1C, *((1.0, 0.0, 0.0) if i % 2 == 0 else (0.0, 1.0, 0.0)))
    P("<3f", raw, 936 + 0x28, 1.0, 2.0, 3.0)
    P("<3f", raw, 936 + 0x40, 30.0, 20.0, 0.0)
    P("<Q", raw, 1288, base["physical_state"])
    raw[1440:1448] = b"TMWHEEL1"
    P("<Q", raw, 1448, base["vehicle"] + 0x388 + (i % 4) * 8)
    P("<Q", raw, 1456, wheels[i % 4])
    raw[1464:1480] = wheel_headers[i % 4]
    P("<I", raw, 1480, 4)
    P("<Q", raw, 1520, vehicle_model)
    P("<Q", raw, 1600, owner)
    raw[1608:1640] = header
    raw[1640:1656] = thresholds
    return bytes(raw)


def wheel_expected(i: int) -> dict[str, Any]:
    return {
        "sequence": i + 1,
        "native_time": 2200 + i * 10,
        "wheel_index": i % 4,
        "material": 9 if i % 2 == 0 else 16,
        "normal": [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0],
        "point": [1.0, 2.0, 3.0],
        "contact_velocity": [30.0, 20.0, 0.0],
    }


prepared_context = {
    "pid": 127,
    "birth": 550,
    "module_base": 0x10000000,
    "vehicle": 0x20000,
    "physical_state": 0x40000,
    "tracer_tid": 129,
    "normalization_owner": owner,
    "normalization_header": header.hex(),
    "normalization_thresholds": thresholds.hex(),
    "wheel_models": wheels,
}


def prepared_record(i: int) -> bytes:
    raw = bytearray(2048)
    wheel = i - 1
    raw[:8] = b"TMIA0001"
    P("<4I", raw, 8, 2048, 1, 127, 129)
    P("<2Q", raw, 24, i + 1, base["module_base"] + 0x9CE390)
    P("<QII", raw, 48, 1, 4, 5)
    P("<2I", raw, 72, 26800, 0)
    P("<Q", raw, 80 + 11 * 8, wheel & 0xFFFFFFFF)
    P("<Q", raw, 80 + 12 * 8, 0x80000)
    P("<Q", raw, 80 + 13 * 8, base["vehicle"])
    P("<Q", raw, 80 + 14 * 8, 26805)
    P("<Q", raw, 80 + 16 * 8, base["module_base"] + 0x9CE390)
    P("<Q", raw, 304, base["module_base"] + 0xAACC92)
    P("<Q", raw, 848, 550)
    P("<I", raw, 928, 212)
    P("<Q", raw, 1288, base["physical_state"])
    if wheel >= 0:
        raw[1440:1448] = b"TMWHEEL1"
        P("<Q", raw, 1448, base["vehicle"] + 0x388 + wheel * 8)
        P("<Q", raw, 1456, wheels[wheel])
        raw[1464:1480] = wheel_headers[wheel]
        P("<I", raw, 1480, 4)
    P("<Q", raw, 1600, owner)
    raw[1608:1640] = header
    raw[1640:1656] = thresholds
    raw[1680] = 9
    P("<3f", raw, 1684, *((1.0, 0.0, 0.0) if wheel < 0 else (0.0, 1.0, 0.0)))
    P("<3f", raw, 1696, 1.0, 2.0, 3.0)
    P("<3f", raw, 1708, *((80.0, 0.0, 0.0) if wheel < 0 else (0.0, 0.5, 0.0)))
    raw[768:816] = raw[1680:1728]
    P("<i", raw, 1732, wheel)
    P("<I", raw, 1740, 26805)
    return bytes(raw)


def expected_prepared(i: int, clock: int = 26800, event: int = 26805) -> dict[str, Any]:
    wheel = i - 1
    return {
        "sequence": i + 1,
        "native_time": clock,
        "event_time": event,
        "wheel_index": wheel,
        "wheel_category": None if wheel < 0 else categories[wheel],
        "material": 9,
        "normal": [1.0, 0.0, 0.0] if wheel < 0 else [0.0, 1.0, 0.0],
        "point": [1.0, 2.0, 3.0],
        "contact_velocity": [80.0, 0.0, 0.0] if wheel < 0 else [0.0, 0.5, 0.0],
        "normalization_thresholds": [40.0, 10.0, 20.0, 100.0],
    }


impact_context = {
    "pid": 127,
    "tid": 129,
    "module_base": 0x10000000,
    "vehicle": 0x20000,
    "participant": 0x90000,
    "sim": 0xA0000,
    "controller": 0xB0000,
    "normalization_owner": owner,
    "normalization_header": header.hex(),
    "normalization_thresholds": thresholds.hex(),
    "wheel_models": wheels,
    "wheel_headers": [x.hex() for x in wheel_headers],
}


def impact_record(  # noqa: PLR0913
    i: int, clock: int = 26800, event: int = 26805, wheel: int | None = None
) -> bytes:
    wheel = i - 1 if wheel is None else wheel
    raw = bytearray(512)
    raw[:8] = b"TMIH0001"
    P("<4I", raw, 8, 512, 1, 127, 129)
    P("<QIIIi", raw, 24, i + 1, 0, clock, event, wheel)
    P("<5Q", raw, 48, 0x20000, 0x10000000 + 0x9CE395, 0x10000000 + 0xAACC92, 0x80000, event)
    raw[96] = 9
    P("<3f", raw, 100, *((1.0, 0.0, 0.0) if wheel < 0 else (0.0, 1.0, 0.0)))
    P("<3f", raw, 112, 1.0, 2.0, 3.0)
    P("<3f", raw, 124, *((80.0, 0.0, 0.0) if wheel < 0 else (0.0, 0.5, 0.0)))
    P("<Q", raw, 144, owner)
    raw[152:184] = header
    raw[184:200] = thresholds
    if wheel >= 0:
        P("<Q", raw, 200, wheels[wheel])
        raw[208:224] = wheel_headers[wheel]
    P("<3QIIQQ", raw, 224, 0x90000, 0xA0000, 0xB0000, 31, 4, 0x10000000, 0xA0000 + 0x48)
    return bytes(raw)


def contact_receipt(i: int, count: int, callback: int, event: int) -> bytes:  # noqa: PLR0913
    raw = bytearray(96)
    raw[:8] = b"TMCR0001"
    P("<4I", raw, 8, 96, 1, 127, 129)
    P("<QII", raw, 24, i + 1, 2190 + i * 10, 0)
    P("<QII", raw, 40, count, callback, event)
    P("<4Q", raw, 56, 0x90000, 0x20000, 0xA0000, 0x10000000)
    P("<Q", raw, 88, count)
    return bytes(raw)


def main() -> None:
    raw = b"".join(wheel_record(i) for i in range(8))
    dump(
        "wheel_contacts_v1",
        {"contacts.bin": raw},
        {
            "origin": "synthetic",
            "context": base,
            "sha256": digest(raw),
            "expected": [wheel_expected(i) for i in range(8)],
        },
    )
    raw = b"".join(wheel_record(i) for i in range(23))
    dump(
        "wheel_contact_sources_v1",
        {"contacts.bin": raw},
        {
            "origin": "synthetic",
            "context": base,
            "sha256": digest(raw),
            "wheel_models": wheels,
            "vehicle_model": vehicle_model,
            "normalization_owner": owner,
            "normalization_owner_header_hex": header.hex(),
            "records": 23,
            "expected_categories": categories,
            "expected_thresholds_hex": thresholds.hex(),
        },
    )
    raw = b"".join(prepared_record(i) for i in range(5))
    dump(
        "prepared_contacts_v1",
        {"contacts.bin": raw},
        {
            "origin": "synthetic",
            "context": prepared_context,
            "sha256": digest(raw),
            "expected": [expected_prepared(i) for i in range(5)],
        },
    )
    raw = b"".join(impact_record(i) for i in range(5))
    dump(
        "prepared_impact_hook_v1",
        {"contacts.bin": raw},
        {
            "origin": "synthetic",
            "context": impact_context,
            "fixture_sha256": digest(raw),
            "expected": [expected_prepared(i) for i in range(5)],
        },
    )
    contacts = b"".join(
        impact_record(i, clock=2190 if i < 4 else 2200, event=2200 if i < 4 else 2210, wheel=i % 4)
        for i in range(5)
    )
    receipts = b"".join(
        contact_receipt(*args)
        for args in [(0, 0, 0, 0), (1, 4, 2190, 2200), (2, 5, 2200, 2210), (3, 5, 2200, 2210)]
    )
    files = {"receipts.bin": receipts, "contacts.bin": contacts}
    dump(
        "contact_delivery361",
        files,
        {
            "origin": "synthetic",
            "context": impact_context,
            "final_native_t": 2220,
            "contacts": 5,
            "files": [{"name": k, "sha256": digest(v)} for k, v in files.items()],
        },
    )


if __name__ == "__main__":
    main()
