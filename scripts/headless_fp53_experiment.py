"""Isolated V9 x87 precision experiment. Never provisions production or edits V7.

Only the saved x87 precision bits are written: 64->53, retaining rounding,
exception masks and all other saved FXSAVE bytes. No physical state is written.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.diagnostic_inputs import configured_path
from scripts.headless_vertical_fp_decode import decode_saved_fp
from scripts.headless_vertical_fp_probe import FPObservedSession


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Precision53Session(FPObservedSession):
    wire_version = 9

    def saved_fp_at_current_boundary(self) -> dict[str, Any]:
        native_t = struct.unpack_from("<I", self.boundary, 112)[0]
        raw = (self.local_posts / f"post-{self.pid}.bin").read_bytes()
        if len(raw) % 256:
            raise ValueError("partial precision experiment sidecar")
        matches = [
            raw[o : o + 256]
            for o in range(0, len(raw), 256)
            if raw[o : o + 8] == b"TMFPU009"
            and struct.unpack_from("<I", raw, o + 40)[0] == native_t
        ]
        if len(matches) != 1:
            raise ValueError("exactly one V9 FP sidecar required")
        record = matches[0]
        if struct.unpack_from("<II", record, 8) != (256, 9):
            raise ValueError("literal V9 sidecar size/version required")
        view = bytearray(record)
        view[:8] = b"TMFPU001"
        struct.pack_into("<I", view, 12, 1)
        before = decode_saved_fp(bytes(view))
        after = record[128:160]
        old_cw, new_cw = struct.unpack_from("<H", record, 96)[0], struct.unpack_from("<H", after)[0]
        if (
            before["pid"] != self.pid
            or before["tid"] != self.pid
            or new_cw != (old_cw & ~0x300) | 0x200
            or after[2:] != record[98:128]
            or struct.unpack_from("<II", record, 72) != (new_cw, 1)
        ):
            raise ValueError("precision-only saved-context mutation proof failed")
        return {"before": before, "after_cw": new_cw, "after_fxsave32_hex": after.hex()}


def build(root: Path, v8: Path, runtime: Path) -> dict[str, Any]:
    imported = json.loads((v8 / "v7-import-manifest.json").read_bytes())
    modified = json.loads((v8 / "modified-source-manifest.json").read_bytes())
    overrides = {entry["path"]: entry["sha256"] for entry in modified["files"]}
    source, manifest = root / "source/tools", []
    source.mkdir(parents=True)
    for entry in imported["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest escapes fresh experiment")
        original, target = v8 / "source/tools" / relative, source / relative
        if sha(original) != overrides.get(entry["path"], entry["sha256"]):
            raise ValueError("existing V8 source differs: " + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        manifest.append({"path": str(relative), "sha256": sha(original)})
    patches: list[str] = []
    for relative_name in (
        "search/forkshim/src/post_observe.rs",
        "search/forkshim/src/post_observe/native_ipc.rs",
    ):
        path = source / relative_name
        old = path.read_text()
        if relative_name.endswith("native_ipc.rs"):
            new = old.replace('b"TMR8"', 'b"TMR9"').replace("h[4]=8;", "h[4]=9;")
            new = new.replace('b"TMA8"', 'b"TMA9"').replace("request[4]!=8", "request[4]!=9")
        else:
            new = old.replace('b"TMFPU001"', 'b"TMFPU009"').replace(
                "w32(&mut d,12,1);", "w32(&mut d,12,9);"
            )
            needle = " d[96..128].copy_from_slice(&saved);"
            if old.count(needle) != 1:
                raise ValueError("exact saved-FP capture site required")
            replacement = (
                needle
                + """
 // V9 only: explicit x87 precision compatibility experiment, no other FP writes.
 let old_cw=u16::from_le_bytes(saved[..2].try_into().unwrap());
 let new_cw=(old_cw & !0x0300u16)|0x0200u16;
 core::ptr::write_volatile(fp as *mut u16,new_cw);
 let mut after=[0u8;32];if !read(fp,&mut after){fail(365)}
 if u16::from_le_bytes(after[..2].try_into().unwrap())!=new_cw || after[2..]!=saved[2..]{fail(366)}
 w32(&mut d,72,new_cw as u32);w32(&mut d,76,1);d[128..160].copy_from_slice(&after);"""
            )
            new = new.replace(needle, replacement)
        if new == old:
            raise ValueError("expected explicit experimental modification absent")
        patches.extend(
            difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), relative_name, relative_name
            )
        )
        path.write_text(new)
    (root / "source-import.json").write_text(json.dumps(manifest, indent=2))
    (root / "v8-to-v9.patch").write_text("".join(patches))
    tool = Path(str(configured_path("TRACKMANIARL_RUST_TOOLS")))
    build_env = dict(
        os.environ,
        CARGO_HOME=str(tool / "cargo"),
        RUSTUP_HOME=str(tool / "rustup"),
        CARGO_TARGET_DIR=str(root / "target"),
        PATH=str(tool / "cargo/bin") + os.pathsep + os.environ.get("PATH", ""),
    )
    with (root / "build.log").open("xb") as log:
        subprocess.run(
            [
                str(tool / "cargo/bin/cargo"),
                "build",
                "--offline",
                "--locked",
                "--release",
                "-p",
                "forkshim",
            ],
            cwd=source / "search",
            env=build_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=True,
        )
    candidates = list((root / "target").rglob("libforkshim.so"))
    if not candidates:
        raise ValueError("experimental shim build absent")
    binary = min(candidates, key=lambda path: len(path.parts))
    directory = runtime / "build/forkoracle" / root.name
    directory.mkdir(exist_ok=False)
    artifact = directory / "libforkshim.so"
    shutil.copyfile(binary, artifact)
    receipt = {
        "schema": "tmrl.experimental-fp53-build/v1",
        "artifact": str(artifact),
        "sha256": sha(artifact),
        "wire": "TMA9/TMR9/version9",
        "x87_precision_written": 53,
        "physical_state_written": False,
        "patch_sha256": sha(root / "v8-to-v9.patch"),
        "build_workspace": str(source / "search"),
    }
    (root / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    root = Path(sys.argv[1])
    if sys.platform != "linux" or not root.is_absolute() or str(root).startswith("/mnt/"):
        raise ValueError("fresh Linux/ext4 experiment required")
    root.mkdir(parents=True, exist_ok=False)
    v8 = Path(str(configured_path("TRACKMANIARL_NATIVE_SOURCE")))
    config_path = Path(str(configured_path("TRACKMANIARL_NATIVE_CONFIG")))
    config = json.loads(config_path.read_bytes())
    runtime = Path(config["runtime_root"])
    receipt = build(root, v8, runtime)
    deployment = Path(config["deployment_manifest"])
    if sha(deployment) != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest differs")
    driver = json.loads(deployment.read_bytes())["driverBuildApproval"]["driverSha256"]
    baseline_path = Path(str(configured_path("TRACKMANIARL_NATIVE_REFERENCE")))
    baseline = json.loads(baseline_path.read_bytes())["cases"]["mild_left"]["rows"]
    if len(baseline) != 200:
        raise ValueError("only the saved finite 200-interval waveform is permitted")
    report: dict[str, Any] = {
        "schema": "tmrl.fp53-experiment/v1",
        "production_modified": False,
        "x87_precision_written": 53,
        "physical_state_written": False,
        "baseline_sha256": sha(baseline_path),
        "rows": [],
    }
    session = Precision53Session(runtime, receipt, root / root.name, 2190, driver)
    completed = False
    try:
        session.reset()
        report["initial_fp"] = session.saved_fp_at_current_boundary()
        report["administrative_neutral"] = session.step(36, 1)
        for row in baseline:
            if (root / "STOP").exists():
                raise InterruptedError("FP53 experiment STOP")
            evidence = session.step_observed(row["evidence"]["waveformSource"])
            raw = bytes.fromhex(evidence["postHex"])
            report["rows"].append(
                {
                    "race_ms": row["afterRaceMs"],
                    "native_physics6_hex": raw[168:192].hex(),
                    "windows_physics6_hex": row["windows_physics6_hex"],
                    "exact_windows_physics6": raw[168:192].hex() == row["windows_physics6_hex"],
                    "saved_fp": session.saved_fp_at_current_boundary(),
                    "evidence": evidence,
                }
            )
        completed = True
    finally:
        session.close(error=not completed)
        report["completed"], report["cleanup"] = completed, session.cleanup
        report["exact_matches"] = sum(row["exact_windows_physics6"] for row in report["rows"])
        (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "completed": completed,
                "exact_matches": report["exact_matches"],
                "samples": len(report["rows"]),
                "remainingOwned": session.cleanup["remainingOwned"],
            }
        )
    )


if __name__ == "__main__":
    main()
