"""Replay saved Windows decisions at observed clocks through the V7 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
from collections.abc import Callable
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from scripts.headless_immediate_probe import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime import env, features
from trackmaniarl.trackmania.native_headless._runtime.host import u32, u64
from trackmaniarl.trackmania.native_headless._runtime.shapes import RequestSpec, ResponseSpec


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(value: dict[str, Any]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def signature(values: list[float]) -> bytes:
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("six finite measured physics fields required")
    return struct.pack("<6f", *values)


def measured_time(raw: list[float]) -> int:
    value = raw[3]
    if not math.isfinite(value) or int(value) != value:
        raise ValueError("Windows observed race time must be an exact integer")
    return int(value)


def state(session: ImmediateSession) -> dict[str, Any]:
    raw, _ = env.qualify_post(session.boundary, session.pid, session.meta)
    return cast(dict[str, Any], features.native16(raw))


def compare(native: dict[str, Any], windows: list[float]) -> dict[str, Any]:
    expected = windows[4:10]
    actual = native["physicalPosition"] + native["physicalVelocity"]
    return {
        "exact_physics6": signature(actual) == signature(expected),
        "native_physics6_hex": signature(actual).hex(),
        "windows_physics6_hex": signature(expected).hex(),
        "position_distance_readout_units": math.dist(actual[:3], expected[:3]),
        "position_difference": [a - b for a, b in zip(actual[:3], expected[:3], strict=True)],
        "velocity_difference": [a - b for a, b in zip(actual[3:], expected[3:], strict=True)],
    }


class ObservedSession(ImmediateSession):
    """Explicit observed waveform transport, without manufacturing action IDs."""

    def step_observed(self, source: dict[str, Any]) -> dict[str, Any]:
        reader, anchor = self.reader, u32(self.boundary, 112)
        p = u32(reader, 112)
        original = self.read_mem(u64(reader, 88) - 4 + p * 32, 32)
        before, after = original[4:16], bytes.fromhex(source["sgbHex"])
        steer, gas, brake = struct.unpack("<3f", after)
        if (
            not math.isfinite(steer)
            or not -1 <= steer <= 1
            or gas not in (0, 1)
            or brake not in (0, 1)
        ):
            raise ValueError("observed waveform outside finite native SGB domain")
        epoch, sequence = self.epoch, self.seq + 1
        self.request(RequestSpec(op=6, n=1, p=p, k=p, befores=[before], afters=[after]))
        committed, receipt = self.response(ResponseSpec(kind=6, seq=sequence, n=1, epoch=epoch))
        if committed != reader or receipt != struct.pack(
            "<4I", p, p, 1, 2
        ) + before + after + bytes(120):
            raise ValueError("observed waveform exact current-record receipt mismatch")
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=sequence, n=1, epoch=epoch + 1))
        self.arm()
        if u32(self.boundary, 112) != anchor + 10:
            raise ValueError("observed waveform advanced beyond exactly one native interval")
        all_posts = (self.local_posts / f"post-{self.pid}.bin").read_bytes()
        if len(all_posts) % 256:
            raise ValueError("observed waveform partial raw POST log")
        records = [all_posts[i : i + 256] for i in range(0, len(all_posts), 256)]
        posts = [
            raw
            for raw in records
            if raw[:8] == b"TMPOST03" and u32(raw, 148) == 75 and u32(raw, 112) == anchor + 10
        ]
        audits = [raw for raw in records if raw[:8] == b"TMAUD003" and u32(raw, 200) == sequence]
        if len(posts) != 1 or len(audits) != 1:
            raise ValueError("observed waveform requires exactly one POST and write audit")
        post, audit = posts[0], audits[0]
        env.qualify_post(post, self.pid, self.meta)
        actual_gas, actual_brake, actual_steer = struct.unpack_from("<3f", post, 88)
        if struct.pack("<3f", actual_steer, actual_gas, actual_brake) != after:
            raise ValueError("actual POST does not match observed desired waveform bytes")
        if (
            u32(audit, 176),
            u32(audit, 180),
            u32(audit, 184),
            u32(audit, 188),
            u32(audit, 204),
            u32(audit, 248),
            u32(audit, 252),
        ) != (p, p, 1, anchor, 1, 1, 0):
            raise ValueError("observed waveform native write authorization mismatch")
        expected = original[:4] + after + original[16:]
        if (
            audit[40:72] != original
            or audit[72:104] != expected
            or audit[104:136] != original
            or audit[136:168] != expected
            or audit[168:172] != before[:4]
            or audit[172:176] != after[:4]
        ):
            raise ValueError("observed waveform exact channel-only current/shadow write mismatch")
        return {
            "waveformSource": source,
            "beforeNativeT": anchor,
            "afterNativeT": anchor + 10,
            "p": p,
            "postHex": post.hex(),
            "auditHex": audit.hex(),
            "receiptHex": receipt.hex(),
        }


def observed_tape(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """Keep exact acquisition order and refuse ambiguous or missing control ticks."""
    measured = [("initial", -1, -1, source["initial_raw33"])]
    for decision, row in enumerate(source["rows"]):
        measured.append(("before", decision, -1, row["before_raw33"]))
        measured.extend(
            ("packet", decision, index, packet["raw33"])
            for index, packet in enumerate(row.get("packets", []))
        )
        measured.append(("after", decision, -1, row["after_raw33"]))
    by_time: dict[int, dict[str, Any]] = {}
    previous = None
    for kind, decision, index, raw in measured:
        race = measured_time(raw)
        if previous is not None and race < previous:
            raise ValueError("observed packet clock regressed in acquisition order")
        previous = race
        physical, sgb = signature(raw[4:10]), struct.pack("<3f", *raw[30:33])
        entry = {
            "raceMs": race,
            "kind": kind,
            "decisionIndex": decision,
            "packetIndex": index,
            "packetSha256": hashlib.sha256(struct.pack("<33f", *raw)).hexdigest(),
            "physics6Hex": physical.hex(),
            "sgbHex": sgb.hex(),
            "raw33": raw,
        }
        if race in by_time:
            prior = by_time[race]
            if prior["physics6Hex"] != entry["physics6Hex"] or prior["sgbHex"] != entry["sgbHex"]:
                raise ValueError(
                    f"different physical/input measurements share observed race clock {race}"
                )
            if kind == "packet" and prior["kind"] != "packet":
                by_time[race] = entry
        else:
            by_time[race] = entry
    times = list(by_time)
    missing = [t for t in range(times[0], times[-1] + 1, 10) if t not in by_time]
    return list(by_time.values()), missing


def replay_observed(  # noqa: PLR0913 -- explicit independent source/build/clock evidence
    config: dict[str, Any],
    receipt: dict[str, Any],
    windows: dict[str, Any],
    output: Path,
    native_time: Callable[[int], int],
    driver_sha: str,
    report: dict[str, Any],
) -> None:
    report.update(
        schema="tmrl.observed-windows-input-replay/v1",
        observed_input_phase="Windows SGB at raceT drives native interval raceT→raceT+10",
        missing_tick_policy="stop before first missing control/physical sample; never interpolate",
        exact_physics_parity=False,
    )
    try:
        for name, source in windows["cases"].items():
            if not source["rows"]:
                continue
            report["current_case"] = name
            tape, missing = observed_tape(source)
            # Preserve a strict complete prefix. The first missing physical or
            # control tick ends it, even when surrounding held values agree.
            if missing:
                tape = [entry for entry in tape if entry["raceMs"] < missing[0]]
            case: dict[str, Any] = {
                "rows": [],
                "missing_race_ticks": missing,
                "complete_source": not missing,
                "completed": False,
                "administrative_neutral_seek": [],
            }
            report["cases"][name] = case
            session = ObservedSession(
                Path(config["runtime_root"]),
                receipt,
                output / f"{output.name}-{name}",
                native_time(0),
                driver_sha,
            )
            try:
                session.reset()
                remaining = (native_time(tape[0]["raceMs"]) - native_time(0)) // 10
                while remaining:
                    n = min(5, remaining)
                    case["administrative_neutral_seek"].append(session.step(36, n))
                    remaining -= n
                case["initial_comparison"] = compare(state(session), tape[0]["raw33"])
                if not case["initial_comparison"]["exact_physics6"]:
                    raise ValueError(
                        "observed waveform experiment requires identical initial physics"
                    )
                for before, after in pairwise(tape):
                    if (output / "STOP").exists():
                        raise InterruptedError("observed waveform cooperative STOP")
                    if after["raceMs"] != before["raceMs"] + 10:
                        raise ValueError("strict observed waveform prefix contains a missing tick")
                    evidence = session.step_observed(before)
                    if evidence["afterNativeT"] != native_time(after["raceMs"]):
                        raise ValueError("observed waveform sample clock mismatch")
                    case["rows"].append(
                        {
                            "beforeRaceMs": before["raceMs"],
                            "afterRaceMs": after["raceMs"],
                            "evidence": evidence,
                            **compare(state(session), after["raw33"]),
                        }
                    )
                case["exact_matches_including_initial"] = 1 + sum(
                    row["exact_physics6"] for row in case["rows"]
                )
                case["compared_samples_including_initial"] = 1 + len(case["rows"])
                case["exact_matching_fraction"] = (
                    case["exact_matches_including_initial"]
                    / case["compared_samples_including_initial"]
                )
                case["first_mismatch"] = next(
                    (row for row in case["rows"] if not row["exact_physics6"]), None
                )
                endpoint = case["rows"][-1] if case["rows"] else case["initial_comparison"]
                case["endpoint_position_distance_readout_units"] = endpoint[
                    "position_distance_readout_units"
                ]
                case["completed"] = True
            finally:
                session.close(error=not case["completed"])
                case["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("observed waveform owned processes remain")
        report["passed_collection"] = bool(report["cases"]) and all(
            case["completed"] for case in report["cases"].values()
        )
        report["exact_physics_parity"] = bool(report["cases"]) and all(
            case["complete_source"] and case["rows"] and case["first_mismatch"] is None
            for case in report["cases"].values()
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main() -> None:
    config_path, receipt_path, clock_path, windows_path, output = map(Path, sys.argv[1:6])
    config = json.loads(config_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    clock = json.loads(clock_path.read_bytes())
    windows = json.loads(windows_path.read_bytes())
    if clock.get("relative_clock_alignment_qualified") is not True:
        raise ValueError("independent observed clock qualification is required")
    slope, intercept = fraction(clock["slope"]), fraction(clock["intercept"])

    def native_time(race_ms: int) -> int:
        value = (race_ms - intercept) / slope
        if value.denominator != 1 or value % 10:
            raise ValueError("Windows sample does not map to an observed native boundary")
        return int(value)

    deployment_raw = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(deployment_raw).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest differs")
    driver_sha = json.loads(deployment_raw)["driverBuildApproval"]["driverSha256"]
    if output.exists():
        raise FileExistsError("fresh replay output required")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": "tmrl.saved-windows-tape-replay/v1",
        "passed_collection": False,
        "exact_physics_parity": False,
        "learning_executed": False,
        "sources": {
            "windows": {"path": str(windows_path), "sha256": digest(windows_path)},
            "clock_alignment": {"path": str(clock_path), "sha256": digest(clock_path)},
            "v7_build": {"path": str(receipt_path), "sha256": digest(receipt_path)},
        },
        "comparison": "exact float32 positionXYZ and velocityXYZ at observed equal sample times",
        "windows_input_fields": "Connect33[30:33] = steer,gas,braking; sampled endpoint only",
        "cases": {},
    }
    begin = time.monotonic()
    if "--observed" in sys.argv[6:]:
        replay_observed(config, receipt, windows, output, native_time, driver_sha, report)
        return
    try:
        for name, source in windows["cases"].items():
            if not source["rows"]:
                continue  # A dense clock-reference trace contains no issued tape.
            case: dict[str, Any] = {
                "rows": [],
                "completed": False,
                "administrative_neutral_seek": [],
            }
            report["cases"][name] = case
            first_race = measured_time(source["initial_raw33"])
            first_native = native_time(first_race)
            # Start at the independently aligned race-zero state and reproduce
            # the neutral reset interval explicitly before the first decision.
            start = native_time(0)
            session = ImmediateSession(
                Path(config["runtime_root"]),
                receipt,
                output / f"{output.name}-{name}",
                start,
                driver_sha,
            )
            try:
                session.reset()
                if first_native < start:
                    raise ValueError("requested initial sample precedes the retained start")
                remaining = (first_native - start) // 10
                while remaining:
                    n = min(5, remaining)
                    case["administrative_neutral_seek"].append(session.step(36, n))
                    remaining -= n
                case["initial_native_t"] = first_native
                case["initial_windows_race_ms"] = first_race
                case["initial_native_state"] = state(session)
                native_by_time = {first_native: state(session)}
                case["initial_comparison"] = compare(state(session), source["initial_raw33"])
                case["first_mismatch"] = (
                    None
                    if case["initial_comparison"]["exact_physics6"]
                    else {"step": -1, **case["initial_comparison"]}
                )
                previous_race = first_race
                for ordinal, row in enumerate(source["rows"]):
                    if (output / "STOP").exists() or time.monotonic() - begin > 170:
                        raise InterruptedError("finite replay STOP/deadline")
                    before_race, after_race = (
                        measured_time(row["before_raw33"]),
                        measured_time(row["after_raw33"]),
                    )
                    delta = after_race - before_race
                    if before_race != previous_race or delta <= 0 or delta % 10:
                        raise ValueError(
                            "saved Windows clock continuity/positive native grid required"
                        )
                    action = row["action"]
                    intervals = (native_time(after_race) - native_time(before_race)) // 10
                    if intervals < 1 or intervals > 1000:
                        raise ValueError("bounded real observed intervals required")
                    chunks, offset = [], 0
                    while offset < intervals:
                        n = min(5, intervals - offset)
                        # A tap has one braking interval for the original
                        # decision. Splitting transport never retriggers it.
                        transport_action = action - 2 if action % 3 == 2 and offset else action
                        chunks.append(session.step(transport_action, n))
                        for post_hex in chunks[-1]["appliedPostHex"]:
                            raw_state, post_time = env.qualify_post(
                                bytes.fromhex(post_hex), session.pid, session.meta
                            )
                            native_by_time[post_time] = features.native16(raw_state)
                        offset += n
                    native = state(session)
                    comparison = compare(native, row["after_raw33"])
                    nominal = features.action_sgb(action)[min(intervals - 1, 4)]
                    observed_windows = row["after_raw33"][30:33]
                    observed_native = [native["steer"], native["gas"], native["brake"]]
                    case["rows"].append(
                        {
                            "step": ordinal,
                            "issued_action": action,
                            "before_windows_race_ms": before_race,
                            "after_windows_race_ms": after_race,
                            "native_t": native_time(after_race),
                            "actual_delta_race_ms": delta,
                            "native_intervals": intervals,
                            "transport_chunks": chunks,
                            "native_state": native,
                            "windows_raw33": row["after_raw33"],
                            "nominal_endpoint_sgb": nominal,
                            "native_observed_endpoint_sgb": observed_native,
                            "windows_observed_endpoint_sgb": observed_windows,
                            "windows_sgb_equals_nominal": struct.pack("<3f", *observed_windows)
                            == struct.pack("<3f", *nominal),
                            "native_sgb_equals_nominal": struct.pack("<3f", *observed_native)
                            == struct.pack("<3f", *nominal),
                            **comparison,
                        }
                    )
                    if case["first_mismatch"] is None and not comparison["exact_physics6"]:
                        case["first_mismatch"] = {
                            "step": ordinal,
                            "action": action,
                            "race_ms": after_race,
                            **comparison,
                        }
                    previous_race = after_race
                total = 1 + len(case["rows"])
                exact = int(case["initial_comparison"]["exact_physics6"]) + sum(
                    row["exact_physics6"] for row in case["rows"]
                )
                case["exact_matches_including_initial"] = exact
                case["compared_samples_including_initial"] = total
                case["exact_matching_fraction"] = exact / total
                case["endpoint_position_distance_readout_units"] = case["rows"][-1][
                    "position_distance_readout_units"
                ]
                case["windows_nominal_input_mismatches"] = sum(
                    not row["windows_sgb_equals_nominal"] for row in case["rows"]
                )
                dense_packets = source.get("packets") or [
                    packet
                    for source_row in source["rows"]
                    for packet in source_row.get("packets", [])
                ]
                if dense_packets:
                    case["dense_packet_comparisons"] = []
                    physical_at_time: dict[int, bytes] = {}
                    inputs_at_time: dict[int, set[bytes]] = {}
                    for packet in dense_packets:
                        raw = packet["raw33"]
                        race_time = measured_time(raw)
                        sample_time = native_time(race_time)
                        if sample_time not in native_by_time:
                            continue
                        physical = signature(raw[4:10])
                        if (
                            race_time in physical_at_time
                            and physical_at_time[race_time] != physical
                        ):
                            raise ValueError(
                                "different dense Windows physical states share one clock"
                            )
                        physical_at_time[race_time] = physical
                        inputs_at_time.setdefault(race_time, set()).add(
                            struct.pack("<3f", *raw[30:33])
                        )
                        native = native_by_time[sample_time]
                        case["dense_packet_comparisons"].append(
                            {
                                "race_ms": race_time,
                                "native_t": sample_time,
                                "windows_sgb": raw[30:33],
                                "native_sgb": [native["steer"], native["gas"], native["brake"]],
                                **compare(native, raw),
                            }
                        )
                    case["dense_unique_race_times"] = len(physical_at_time)
                    case["dense_input_ambiguous_times"] = [
                        t for t, values in inputs_at_time.items() if len(values) > 1
                    ]
                    case["dense_exact_packet_matches"] = sum(
                        packet["exact_physics6"] for packet in case["dense_packet_comparisons"]
                    )
                case["completed"] = True
            finally:
                session.close(error=not case["completed"])
                case["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("owned process remained after saved tape replay")
        report["passed_collection"] = bool(report["cases"]) and all(
            case["completed"] for case in report["cases"].values()
        )
        report["exact_physics_parity"] = bool(report["cases"]) and all(
            case["rows"] and case["first_mismatch"] is None for case in report["cases"].values()
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["host_seconds"] = time.monotonic() - begin
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
