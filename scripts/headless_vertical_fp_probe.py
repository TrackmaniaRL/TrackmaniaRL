"""Read saved interrupted FP controls through a separately pinned V8 shim."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from scripts.headless_replay_windows_tapes import ObservedSession
from scripts.headless_vertical_fp_decode import decode_saved_fp
from trackmaniarl.trackmania.native_headless._runtime import codec_v7
from trackmaniarl.trackmania.native_headless._runtime.host import u32
from trackmaniarl.trackmania.native_headless._runtime.shapes import RequestSpec, ResponseSpec


class FPObservedSession(ObservedSession):
    """V8 transport is explicit; V7 production decoder and hashes are unchanged."""

    wire_version = 8

    def request(self, spec: RequestSpec) -> None:
        self.seq += 1
        raw = bytearray(
            codec_v7.encode_request(
                op=spec.op,
                sequence=self.seq,
                n=spec.n,
                epoch=self.epoch,
                p=spec.p,
                k=spec.k,
                befores=tuple(spec.befores),
                afters=tuple(spec.afters),
            )
        )
        raw[:5] = f"TMA{self.wire_version}".encode() + bytes([self.wire_version])
        self.requests.append({"experimentalWireVersion": self.wire_version, "hex": raw.hex()})
        self.connection.sendall(raw)

    def response(self, spec: ResponseSpec) -> tuple[bytes, bytes]:
        raw = self.exact(448)
        if raw[:5] != f"TMR{self.wire_version}".encode() + bytes([self.wire_version]):
            raise ValueError("independent experimental response version required")
        view = bytearray(raw)
        view[:5] = b"TMR7\x07"
        decoded = codec_v7.decode_v7_response(bytes(view))
        if (decoded["kind"], decoded["sequence"], decoded["iterations"], decoded["epoch"]) != (
            spec.kind,
            spec.seq,
            spec.n,
            spec.epoch,
        ):
            raise ValueError("V8 response control identity mismatch")
        if decoded["event"]["pid"] != self.pid:
            raise ValueError("foreign V8 response")
        self.responses.append(raw)
        self.epoch = spec.epoch
        return raw[32:288], raw[288:]

    def saved_fp_at_current_boundary(self) -> dict[str, Any]:
        raw = (self.local_posts / f"post-{self.pid}.bin").read_bytes()
        if len(raw) % 256:
            raise ValueError("partial V8 FP/POST stream")
        rows = [raw[offset : offset + 256] for offset in range(0, len(raw), 256)]
        matches = [
            row
            for row in rows
            if row[:8] == b"TMFPU001" and u32(row, 40) == u32(self.boundary, 112)
        ]
        if len(matches) != 1:
            raise ValueError("one saved interrupted FP snapshot per boundary required")
        result = decode_saved_fp(matches[0])
        if result["pid"] != self.pid or result["tid"] != self.pid:
            raise ValueError("saved FP controls must belong to owned validator thread")
        return result


def main() -> None:
    config_path, receipt_path, baseline_path, layout_path, output = map(Path, sys.argv[1:6])
    config = json.loads(config_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    baseline = json.loads(baseline_path.read_bytes())["cases"]["mild_left"]["rows"]
    layout = json.loads(layout_path.read_bytes())
    if layout != {
        "ucontext_fpregs_offset": 224,
        "mxcsr_offset": 24,
        "cwd_offset": 0,
        "swd_offset": 2,
        "mask_offset": 28,
    }:
        raise ValueError("host ABI proof disagrees with experimental FP read offsets")
    if (
        len(baseline) != 200
        or receipt["wire"] != "TMA8/TMR8/version8"
        or receipt["physical_fp_state_writes"] is not False
    ):
        raise ValueError("only the approved finite read-only FP experiment is allowed")
    deployment_raw = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(deployment_raw).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest differs")
    driver_sha = json.loads(deployment_raw)["driverBuildApproval"]["driverSha256"]
    if output.exists():
        raise FileExistsError("fresh FP capture output required")
    output.mkdir(parents=True)
    report = {
        "schema": "tmrl.interrupted-fp-controls/v1",
        "passed": False,
        "experimental_wire": 8,
        "physical_fp_state_writes": False,
        "source_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "build_receipt": receipt,
        "host_abi_layout": layout,
        "rows": [],
    }
    session = FPObservedSession(
        Path(config["runtime_root"]), receipt, output / output.name, 2190, driver_sha
    )
    try:
        session.reset()
        report["initial_saved_fp"] = session.saved_fp_at_current_boundary()
        report["administrative_neutral_seek"] = session.step(36, 1)
        for source in baseline:
            if (output / "STOP").exists():
                raise InterruptedError("FP capture cooperative STOP")
            evidence = session.step_observed(source["evidence"]["waveformSource"])
            post = bytes.fromhex(evidence["postHex"])
            old = bytes.fromhex(source["evidence"]["postHex"])
            report["rows"].append(
                {
                    "race_ms": source["afterRaceMs"],
                    "same_native76_as_v7": post[152:212] + post[88:104]
                    == old[152:212] + old[88:104],
                    "saved_interrupted_fp": session.saved_fp_at_current_boundary(),
                    "evidence": evidence,
                }
            )
        report["first_native76_difference_from_v7"] = next(
            (row["race_ms"] for row in report["rows"] if not row["same_native76_as_v7"]), None
        )
        report["passed"] = report["first_native76_difference_from_v7"] is None
    finally:
        session.close(error=not report["passed"])
        report["cleanup"] = session.cleanup
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
