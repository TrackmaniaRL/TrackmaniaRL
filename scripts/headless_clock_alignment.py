"""Infer relative sample clocks only from unique, exact held-out physical states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any


class AlignmentError(ValueError):
    """The supplied traces do not establish a unique physical sample clock."""


@dataclass(frozen=True)
class Sample:
    time: int
    physics: tuple[float, ...]

    @property
    def signature(self) -> bytes:
        if len(self.physics) != 6 or not all(math.isfinite(value) for value in self.physics):
            raise AlignmentError("six finite measured position/velocity fields required")
        return struct.pack("<6f", *self.physics)


def _validated(samples: list[Sample]) -> dict[bytes, list[Sample]]:
    signatures: dict[bytes, list[Sample]] = {}
    prior = None
    for sample in samples:
        if type(sample.time) is not int or (prior is not None and sample.time <= prior):
            raise AlignmentError("strictly increasing integer clock samples required")
        prior = sample.time
        signatures.setdefault(sample.signature, []).append(sample)
    return signatures


def align_clocks(
    native: list[Sample], windows: list[Sample], *, minimum_matches: int = 6
) -> dict[str, Any]:
    """Fit on alternating exact matches and validate the unused matches.

    Repeated signatures do not establish correspondence: they are excluded before
    fitting. No configured offset, tolerated distance, or expected native unit is
    used. The result describes only the observed physical sampling phases.
    """
    if minimum_matches < 6:
        raise ValueError("at least six unique exact measured matches required")
    ns, ws = _validated(native), _validated(windows)
    common = ns.keys() & ws.keys()
    ambiguous = [
        signature for signature in common if len(ns[signature]) != 1 or len(ws[signature]) != 1
    ]
    matches = sorted(
        ((ns[signature][0], ws[signature][0]) for signature in common - set(ambiguous)),
        key=lambda pair: pair[0].time,
    )
    if len(matches) < minimum_matches:
        raise AlignmentError(
            f"only {len(matches)} unique exact physical matches; "
            f"{len(ambiguous)} repeated signatures excluded"
        )
    if any(right.time <= previous.time for (_, previous), (_, right) in pairwise(matches)):
        raise AlignmentError("physical matches disagree about temporal ordering")
    training, heldout = matches[::2], matches[1::2]
    if len(training) < 3 or len(heldout) < 3:
        raise AlignmentError("three fit and three held-out exact matches required")
    first, last = training[0], training[-1]
    slope = Fraction(last[1].time - first[1].time, last[0].time - first[0].time)
    intercept = Fraction(first[1].time) - slope * first[0].time
    if slope <= 0:
        raise AlignmentError("positive monotonic relative clock slope required")
    for split, pairs in (("fit", training), ("heldout", heldout)):
        if any(
            slope * native_sample.time + intercept != windows_sample.time
            for native_sample, windows_sample in pairs
        ):
            raise AlignmentError(
                f"{split} physical matches reject a single affine clock relationship"
            )
    if matches[-1][1].time - matches[0][1].time < 50:
        raise AlignmentError("at least 50 observed race milliseconds of validation span required")
    heldout_times = {sample.time for sample, _ in heldout}
    return {
        "schema": "tmrl.physical-sample-clock-alignment/v1",
        "relative_clock_alignment_qualified": True,
        "fields": ["positionX", "positionY", "positionZ", "velocityX", "velocityY", "velocityZ"],
        "comparison": "exact float32 bytes; no tolerance or rounding",
        "relationship": "windows_race_ms = slope * native_counter + intercept",
        "slope": {"numerator": slope.numerator, "denominator": slope.denominator},
        "intercept": {"numerator": intercept.numerator, "denominator": intercept.denominator},
        "native_physical_sample_race_zero": {
            "numerator": (-intercept / slope).numerator,
            "denominator": (-intercept / slope).denominator,
        },
        "fit_matches": len(training),
        "heldout_matches": len(heldout),
        "ambiguous_repeated_signatures_excluded": len(ambiguous),
        "sampled_race_span_ms": matches[-1][1].time - matches[0][1].time,
        "input_archive_clock_qualified": False,
        "scope": (
            "observed physical sampling phases on supplied map/reset/input trace; "
            "no wall-clock or arbitrary-map claim"
        ),
        "matches": [
            {
                "native_counter": n.time,
                "windows_race_ms": w.time,
                "split": "heldout" if n.time in heldout_times else "fit",
                "physics6Hex": n.signature.hex(),
                "position": list(n.physics[:3]),
                "velocity": list(n.physics[3:]),
            }
            for n, w in matches
        ],
    }


def native_samples(data: dict[str, Any]) -> list[Sample]:
    rows = data["rows"]
    if any(type(row["native_t"]) is not int for row in rows):
        raise AlignmentError("literal integer native counters required")
    return [
        Sample(
            row["native_t"],
            tuple(row["state"]["physicalPosition"] + row["state"]["physicalVelocity"]),
        )
        for row in rows
    ]


def reader_phase(raw: bytes) -> dict[str, int | str]:
    """Decode literal reader operands; source arithmetic is separate from clock fit."""
    if len(raw) != 256 or raw[:8] != b"TMREAD01":
        raise AlignmentError("exact measured native reader required for input phase")
    p, count, next_native, anchor = struct.unpack_from("<4I", raw, 112)
    start_offset = struct.unpack_from("<i", raw, 248)[0]
    effective_r8 = struct.unpack_from("<i", raw, 144)[0]
    query = struct.unpack_from("<I", raw, 252)[0]
    if p >= count or next_native != anchor + 10 or effective_r8 != start_offset + 10:
        raise AlignmentError("measured reader source clock/grid operands disagree")
    if query != p * 10:
        raise AlignmentError("reader query does not lie on the observed input archive grid")
    getter_start = next_native - query - start_offset
    return {
        "held_native_boundary": anchor,
        "next_native_boundary": next_native,
        "input_record_index": p,
        "archive_start_offset_ms": start_offset,
        "archive_input_label_ms": p * 10 + start_offset,
        "reader_query_after_subtractions": query,
        "source_start_getter_result": getter_start,
        "source": "reader0x119f0f0; call0x1219f89; start getter0x121cf50",
    }


def windows_samples(data: dict[str, Any]) -> list[Sample]:
    if "cases" in data and "rows" not in data:
        case = data["cases"]["neutral"]
        raw_rows = [case["initial_raw33"], *(row["after_raw33"] for row in case["rows"])]
    else:
        raw_rows = [row["raw33"] if isinstance(row, dict) else row for row in data["rows"]]
    by_time = {}
    previous_time = None
    for raw in raw_rows:
        if len(raw) < 10 or not math.isfinite(raw[3]) or int(raw[3]) != raw[3]:
            raise AlignmentError("measured Connect33 with exact integer race milliseconds required")
        sample = Sample(int(raw[3]), tuple(raw[4:10]))
        if previous_time is not None and sample.time < previous_time:
            raise AlignmentError("Windows race timestamp regressed in acquisition order")
        previous_time = sample.time
        prior = by_time.get(sample.time)
        if prior is not None and prior.signature != sample.signature:
            raise AlignmentError("different physical states share one Windows race timestamp")
        by_time[sample.time] = sample
    return list(by_time.values())


def capture_native(  # noqa: PLR0913 -- explicit independent evidence/configuration inputs
    config_path: Path, receipt_path: Path, output: Path, count: int, start: int
) -> None:
    # Import only for an explicit capture; normal matching is completely offline.
    from headless_immediate_probe import ImmediateSession

    from trackmaniarl.trackmania.native_headless._runtime import env, features

    config, receipt = json.loads(config_path.read_bytes()), json.loads(receipt_path.read_bytes())
    raw = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("pinned original deployment manifest differs")
    driver_sha = json.loads(raw)["driverBuildApproval"]["driverSha256"]
    if not 20 <= count <= 200:
        raise ValueError("finite neutral capture requires 20..200 intervals")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": "tmrl.neutral-physical-clock-capture/v1",
        "passed": False,
        "rows": [],
        "action": 36,
        "clock_mapping_qualified": False,
    }
    session = ImmediateSession(
        Path(config["runtime_root"]),
        receipt,
        output / (output.name + "-session"),
        start,
        driver_sha,
    )
    try:
        initial = session.reset()

        def append(post: bytes) -> None:
            state, clock = env.qualify_post(post, session.pid, session.meta)
            report["rows"].append(
                {
                    "native_t": clock,
                    "state": features.native16(state),
                    "postHex": post.hex(),
                    "readerHex": session.reader.hex(),
                }
            )

        append(initial)
        for _ in range(count):
            if (output / "STOP").exists():
                raise InterruptedError("neutral clock capture cooperative STOP")
            step = session.step(36, 1)
            append(bytes.fromhex(step["appliedPostHex"][0]))
        report["passed"] = True
    finally:
        session.close(error=not report["passed"])
        report["cleanup"] = session.cleanup
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match")
    match.add_argument("native", type=Path)
    match.add_argument("windows", type=Path)
    match.add_argument("output", type=Path)
    capture = commands.add_parser("capture-native")
    capture.add_argument("config", type=Path)
    capture.add_argument("receipt", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--intervals", type=int, default=65)
    capture.add_argument("--start", type=int, default=2190)
    args = parser.parse_args()
    if args.command == "capture-native":
        capture_native(args.config, args.receipt, args.output, args.intervals, args.start)
        return
    paths = {"native": args.native, "windows": args.windows}
    report = {
        "relative_clock_alignment_qualified": False,
        "sources": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()
        },
    }
    try:
        native_data = json.loads(args.native.read_bytes())
        report.update(
            align_clocks(
                native_samples(native_data),
                windows_samples(json.loads(args.windows.read_bytes())),
            )
        )
        rows = {row["native_t"]: row for row in native_data["rows"]}
        if all("readerHex" in rows[row["native_counter"]] for row in report["matches"]):
            for match in report["matches"]:
                phase = reader_phase(bytes.fromhex(rows[match["native_counter"]]["readerHex"]))
                match["input_phase"] = phase
                if phase["archive_input_label_ms"] != match["windows_race_ms"]:
                    raise AlignmentError(
                        "input archive label differs from matched current sample race"
                    )
            report["observed_input_phase"] = (
                "next consumed input archive label equals current Windows physical sample race; "
                "next POST physical sample advances by 10 race milliseconds"
            )
    except AlignmentError as exc:
        report["relative_clock_alignment_qualified"] = False
        report["reason"] = str(exc)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not report["relative_clock_alignment_qualified"]:
        raise SystemExit(report["reason"])


if __name__ == "__main__":
    main()
