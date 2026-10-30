"""Read Rust exports and closed V7 archives; never run physics or extract archives.

This deliberately supports record-v10 digital ghosts and the documented V7 POST
layout only. Other encodings require an explicit adapter, not guessed fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .diagnose_ghost_divergence import SCHEMA, Json, array, file_path, mapping, number, read_json


def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- assertion helper
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        require(0 <= size <= 16 * 1024**2 and self.offset + size <= len(self.data), "record bounds")
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def integer(self) -> int:
        return int(struct.unpack("<i", self.take(4))[0])

    def count(self) -> int:
        value = self.integer()
        require(0 <= value <= 100_000, "bounded nonnegative record count")
        return value

    def flag(self) -> bool:
        value = self.take(1)[0]
        require(value in (0, 1), "record flag must be zero or one")
        return bool(value)

    def buffer(self) -> bytes:
        return self.take(self.integer())


@dataclass(frozen=True)
class Entity:
    class_id: int
    times: list[int]
    samples: list[bytes]


def record_entities(data: bytes) -> list[Entity]:
    """Consume the complete v10 grammar, including notices and custom modules."""
    reader = Reader(data)
    reader.take(8)
    descriptors: list[int] = []
    for _ in range(reader.count()):
        descriptors.append(reader.integer())
        reader.take(12)
        reader.buffer()
        reader.integer()
    reader.take(reader.count() * 12)
    entities: list[Entity] = []
    more = reader.flag()
    while more:
        kind = reader.integer()
        require(0 <= kind < len(descriptors), "entity descriptor index")
        reader.take(16)
        times: list[int] = []
        samples: list[bytes] = []
        while reader.flag():
            times.append(reader.integer())
            samples.append(reader.buffer())
        more = reader.flag()
        while reader.flag():
            reader.take(8)
            reader.buffer()
        entities.append(Entity(descriptors[kind], times, samples))
    while reader.flag():
        reader.take(8)
        reader.buffer()
    for _ in range(reader.count()):
        while reader.flag():
            reader.integer()
            reader.buffer()
            reader.buffer()
        reader.integer()
    require(reader.offset == len(data), "complete record grammar must be consumed")
    return entities


def decode_record(dump: Path, manifest: dict[str, Json], map_uid: str) -> Entity:
    """Use the existing Rust container dump and independently verify its manifest."""
    body = bytearray()
    for line in file_path(dump).read_text(encoding="utf-8-sig").splitlines():
        require(int(line[:8]) == len(body), "Rust dump byte offsets must be contiguous")
        body.extend(bytes.fromhex(line[10:58]))
    require(len(body) == mapping(manifest["header"])["body_bytes"], "full Rust body dump required")
    require(bool(map_uid) and body.count(map_uid.encode("ascii")) == 1,
            "one explicit expected map UID literal in Rust-decoded body required")
    chunks = [mapping(v) for v in array(manifest["chunks"])
              if mapping(v)["id"] == "0x0911F000"]
    require(len(chunks) == 1, "one unambiguous record chunk required")
    offset = int(number(chunks[0]["offset"]))
    marker = struct.pack("<I", 0x0911F000)
    require(body[offset:offset + 4] == marker, "Rust record chunk offset")
    while body[offset + 4:offset + 8] == marker:
        offset += 4
    version, size, compressed = struct.unpack_from("<III", body, offset + 4)
    require(version == 10 and size < 16 * 1024**2, "only bounded v10 records supported")
    decoder = zlib.decompressobj()
    blob = decoder.decompress(bytes(body[offset + 16:offset + 16 + compressed]), size + 1)
    require(decoder.eof and not decoder.unused_data and len(blob) == size, "complete zlib record")
    declared = mapping(manifest["record"])
    require(size == declared["bytes_total"] == declared["bytes_consumed"], "Rust record size")
    require(struct.unpack_from("<ii", blob) == (declared["start_ms"], declared["end_ms"]),
            "Rust literal record clock bounds")
    entities = record_entities(blob)
    expected = array(declared["entities"])
    require(len(entities) == len(expected), "Rust entity count")
    for actual, value in zip(entities, expected, strict=True):
        row = mapping(value)
        require(actual.class_id == int(str(row["class_id"]), 16), "Rust entity class")
        require(len(actual.samples) == row["samples"], "Rust entity sample count")
        if actual.times:
            require((actual.times[0], actual.times[-1]) == (row["first_ms"], row["last_ms"]),
                    "Rust entity timestamps")
    cars = [e for e in entities if e.class_id == 0x0A018000]
    require(len(cars) == 1 and bool(cars[0].samples), "one unambiguous controlled car required")
    require(cars[0].samples[0].hex() == mapping(declared["first_vehicle_sample"])["raw_hex"],
            "independent Rust first raw sample")
    return cars[0]


def stored_state(raw: bytes) -> tuple[list[Json], list[Json]]:
    require(len(raw) >= 69, "complete recorded transform required")
    position: list[Json] = [float(v) for v in struct.unpack_from("<3f", raw, 47)]
    speed_log, heading, pitch = struct.unpack_from("<hbb", raw, 65)
    speed = math.exp(speed_log / 1000)
    horizontal = heading / 127 * math.pi
    vertical = pitch / 127 * math.pi / 2
    velocity: list[Json] = [speed * math.cos(vertical) * math.cos(horizontal),
                            speed * math.cos(vertical) * math.sin(horizontal),
                            speed * math.sin(vertical)]
    return position, velocity


@dataclass(frozen=True)
class RustInputs:
    dump: Path
    manifest: Path
    controls: Path
    map_uid: str


def ghost_document(inputs: RustInputs) -> dict[str, Json]:
    manifest = read_json(inputs.manifest)
    car = decode_record(inputs.dump, manifest, inputs.map_uid)
    rows: dict[int, dict[str, Json]] = {}
    previous: int | None = None
    with file_path(inputs.controls).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            time = int(row["race_ms"])
            steer, gas, brake = int(row["steer"]), int(row["accel"]), int(row["brake"])
            require(time not in rows and time % 10 == 0, "unique input ticks")
            require(previous is None or time == previous + 10, "complete ordered input grid")
            previous = time
            require(steer in (-127, 0, 127) and gas in (0, 1) and brake in (0, 1),
                    "digital tape required; no analog approximation")
            rows[time] = {"time_ms": time, "control_sgb": [steer / 127, gas, brake],
                          "phase": "countdown" if time < 0 else "recorded_race"}
    archives = array(mapping(manifest["tape"])["archives"])
    require(len(archives) == 1, "one unambiguous input archive required")
    tape = mapping(archives[0])
    require(len(rows) == tape["packets"] and min(rows) == tape["start_offset_ms"],
            "full Rust tape including countdown required")
    for time, raw in zip(car.times, car.samples, strict=True):
        position, velocity = stored_state(raw)
        row = rows.setdefault(time, {"time_ms": time})
        row.update(position_m=position, velocity_m_s=velocity)
        require(len(raw) >= 92, "full wheel rotation fields required")
        row["contacts"] = {
            "stored_wheel_rotation_rad_FL_FR_RR_RL": [
                (raw[i] / 255 + raw[i + 1]) * math.tau for i in (6, 8, 10, 12)
            ],
            "ground_mode_raw_unverified": raw[89],
        }
    first_position, first_velocity = stored_state(car.samples[0])
    first = mapping(mapping(manifest["record"])["first_vehicle_sample"])
    for actual, field in ((first_position, "position"), (first_velocity, "velocity")):
        expected = array(first[field])
        require(len(expected) == 3 and all(abs(number(a) - number(b)) < 1e-8
                for a, b in zip(actual, expected, strict=True)), "Rust transform control")
    return {
        "schema": SCHEMA, "coordinate_frame": "world_xyz_y_up",
        "map_uid": inputs.map_uid,
        "frames": [rows[t] for t in sorted(rows)],
        "provenance": {
            "decoder": "Rust container/tape exports; Python v10 record grammar and Rust transform",
            "manifest_sha256": hashlib.sha256(file_path(inputs.manifest).read_bytes()).hexdigest(),
            "dump_sha256": hashlib.sha256(file_path(inputs.dump).read_bytes()).hexdigest(),
            "controls_sha256": hashlib.sha256(file_path(inputs.controls).read_bytes()).hexdigest(),
            "rust_manifest_entity_and_first_transform_checks": True,
            "velocity": "stored log-speed and signed8-bit direction; not finite differences",
            "input_clock": "literal Rust race_ms, including negative countdown",
        },
    }


def archive_member(archive: tarfile.TarFile, name: str) -> bytes:
    entry = archive.getmember(name)
    require(entry.isfile() and entry.size <= 32 * 1024**2, "bounded regular trace member")
    stream = archive.extractfile(entry)
    if stream is None:
        raise ValueError("trace member must be readable")
    with stream:
        return stream.read()


def uint(raw: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", raw, offset)[0])


def archived_auxiliary(archive: tarfile.TarFile, metadata: dict[str, Json]) -> dict[int, Json]:
    """Summarize archived wheel getters and delivered-hook counts, not full contacts."""
    pid = int(number(metadata["server_pid"]))
    result: dict[int, Json] = {}
    vehicle_name = f"trace/posts/vehicle-{pid}.bin"
    contacts_name = f"trace/posts/contact-receipt-{pid}.bin"
    if vehicle_name in archive.getnames():
        data = archive_member(archive, vehicle_name)
        require(len(data) % 1280 == 0, "whole archived vehicle records")
        for at in range(0, len(data), 1280):
            raw = data[at:at + 1280]
            require(raw[:8] == b"TMVH0001" and struct.unpack_from("<4I", raw, 8)
                    == (1280, 1, pid, pid), "archived vehicle header/owner")
            clock = uint(raw, 32)
            visual = raw[208:1072]
            require(uint(visual, 12) == clock, "archived wheel/vehicle clock join")
            flags = [uint(visual, 0xD0 + i * 0x2C) for i in range(4)]
            result[clock] = {
                "wheel_flags_raw_native_index_order": [cast(Json, flag) for flag in flags],
                "wheel_material_getter_native_index_order": [
                    13 if flag & 2 else visual[0xB8 + i * 0x2C] if flag & 4 else 80
                    for i, flag in enumerate(flags)
                ],
                "wheel_slip_connect_native_index_order": [
                    float(struct.unpack_from("<f", visual, 0xBC + i * 0x2C)[0])
                    for i in range(4)
                ],
            }
    if contacts_name in archive.getnames():
        data = archive_member(archive, contacts_name)
        require(len(data) % 96 == 0, "whole delivered contact receipts")
        for at in range(0, len(data), 96):
            raw = data[at:at + 96]
            require(raw[:8] == b"TMCR0001" and struct.unpack_from("<4I", raw, 8)
                    == (96, 1, pid, pid), "contact receipt header/owner")
            clock = uint(raw, 32)
            counter = int(struct.unpack_from("<Q", raw, 40)[0])
            require(counter == struct.unpack_from("<Q", raw, 88)[0], "contact count echo")
            values = mapping(result.setdefault(clock, {}))
            values.update(delivered_hook_contacts_cumulative=counter,
                          last_callback_native_ms=uint(raw, 48),
                          last_contact_event_native_ms=uint(raw, 52))
    return result


def audited_controls(post: bytes, pid: int) -> dict[int, bytes]:
    commits: dict[int, bytes] = {}
    for at in range(0, len(post), 256):
        raw = post[at:at + 256]
        if raw[:8] != b"TMAUD003":
            continue
        require(struct.unpack_from("<4I", raw, 8) == (256, 2, pid, pid), "input audit owner")
        count, ordinal = uint(raw, 184), uint(raw, 252)
        require(1 <= count <= 5 and ordinal < count, "input audit interval ordinal")
        end = uint(raw, 188) + (ordinal + 1) * 10
        require(end not in commits, "unique audited input interval")
        commits[end] = raw[76:88]
    return commits


def headless_document(path: Path, receipt: Path, factory_config: Path) -> dict[str, Json]:
    raw = file_path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    declared = array(read_json(receipt)["archives"])
    config = read_json(factory_config)
    require(any(mapping(v)["archive_sha256"] == digest for v in declared), "recorded archive SHA")
    with tarfile.open(file_path(path), "r:gz") as archive:
        metadata = mapping(cast(Json, json.loads(archive_member(
            archive, "trace/episode-0000/metadata.json"
        ))))
        reset = mapping(cast(Json, json.loads(archive_member(
            archive, "trace/episode-0000/uncredited-neutral-reset-v7.json"
        ))))
        origin = int(number(reset["resetNativeT"]))
        pid = int(number(metadata["server_pid"]))
        post = archive_member(archive, f"trace/posts/post-{pid}.bin")
        auxiliary = archived_auxiliary(archive, metadata)
    require(len(post) % 256 == 0 and origin == 2200, "closed V7 record layout/reset")
    commits = audited_controls(post, pid)
    frames: list[Json] = []
    previous: int | None = None
    for at in range(0, len(post), 256):
        row = post[at:at + 256]
        if row[:8] != b"TMPOST03" or uint(row, 148) != 75:
            continue
        native_t = uint(row, 112)
        require((uint(row, 8), uint(row, 12), uint(row, 16), uint(row, 20), uint(row, 212))
                == (256, 3, pid, pid, 10), "actual POST header/owner/role")
        require(native_t == uint(row, 116) == uint(row, 144) and native_t % 10 == 0,
                "actual POST clock grid")
        require(previous is None or native_t == previous + 10, "contiguous actual POST ticks")
        previous = native_t
        gas, brake, steer = struct.unpack_from("<3f", row, 88)
        if native_t >= origin:
            require(commits.get(native_t) == struct.pack("<3f", steer, gas, brake),
                    "actual POST controls must match pre-load write audit")
        physical = struct.unpack_from("<10f", row, 152)
        time = native_t - origin
        frames.append({
            "time_ms": time, "position_m": [float(v) for v in physical[4:7]],
            "velocity_m_s": [float(v) for v in physical[7:10]],
            "control_sgb": [float(steer), float(gas), float(brake)],
            "phase": "driving" if time > 0 else "neutral_reset" if time == 0 else "bootstrap",
            "contacts": auxiliary.get(native_t),
        })
    return {
        "schema": SCHEMA, "coordinate_frame": "world_xyz_y_up",
        "map_uid": config["expected_map_uid"], "frames": frames,
        "provenance": {
            "archive_sha256": digest, "pid": pid, "native_race_origin_ms": origin,
            "factory_config_sha256": hashlib.sha256(
                file_path(factory_config).read_bytes()
            ).hexdigest(),
            "map_binding": "expected_map_uid in saved factory configuration",
            "clock_mapping": "race_ms = observed NativeSimClock - archived resetNativeT",
            "control_semantics": "actual POST applied SGB; interval ending at this timestamp",
            "reset": "uncredited neutral interval ending at race0, not a policy action",
            "contacts": "archived wheel getters and delivered-hook cumulative counters only",
            "not_a_contact_qualification": True,
        },
    }


def write_new(path: Path, document: dict[str, Json]) -> None:
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with file_path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dump", "manifest", "controls", "archive", "receipt", "factory-config",
                 "ghost-output",
                 "headless-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--map-uid", required=True)
    args = parser.parse_args()
    ghost = ghost_document(RustInputs(args.dump, args.manifest, args.controls, args.map_uid))
    native = headless_document(args.archive, args.receipt, args.factory_config)
    require(ghost["map_uid"] == native["map_uid"], "ghost/config map UID match")
    write_new(args.ghost_output, ghost)
    write_new(args.headless_output, native)


if __name__ == "__main__":
    main()
