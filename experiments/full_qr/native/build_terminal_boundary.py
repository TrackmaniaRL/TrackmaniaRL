"""Build a separate diagnostic shim with actual collector source witnesses.

Never changes the frozen379 shim, the training source557 or a running cohort.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from scripts.diagnostic_inputs import configured_path


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"independently pinned patch site changed: {before[:100]}")
    return text.replace(before, after)


def main() -> None:
    root = Path(str(configured_path("TRACKMANIARL_NATIVE_EVIDENCE")))
    checkout = Path(str(Path(__file__).resolve().parents[3]))
    assert shutil.disk_usage("/mnt/c").free >= 10 * 1024**3
    assert shutil.disk_usage(root).free >= 4 * 1024**3
    build = root / "terminal-boundary-build583"
    if build.exists():
        raise ValueError("a new build directory is required")
    shutil.copytree(root / "goal-build379/source", build / "source")
    modules = build / "source/tools/search/forkshim/src/post_observe"
    source = checkout / "experiments/full_qr/native/terminal_boundary_observer.rs"
    shutil.copyfile(source, modules / "terminal_boundary_observe.rs")
    for module in (
        "vehicle_observe",
        "adherence_observe",
        "contact_receipt_observe",
        "goal_observe",
    ):
        path = modules / (module + ".rs")
        text = path.read_text()
        # The original reader entry point keeps its provenance. A different
        # entry point accepts only the complete actual collector witness.
        pattern = r"pub unsafe fn at_reader\((?:event|raw):\s*&\[u8;256\]\)\s*\{"
        assert len(re.findall(pattern, text)) == 1
        wrapper = """pub unsafe fn at_reader(event:&[u8;256]) {
    capture(super::terminal_boundary_observe::ClockWitness::Reader(event));
}
pub(super) unsafe fn at_collector(event:&[u8;768]) {
    capture(super::terminal_boundary_observe::ClockWitness::Collector(event));
}
unsafe fn capture(witness:super::terminal_boundary_observe::ClockWitness<'_>) {"""
        text = re.sub(pattern, wrapper, text)
        if module == "vehicle_observe":
            start = text.index("    if u32::from_le_bytes(event[148..152]")
            end = text.index("    if VEHICLE_PID", start)
            text = text[:start] + "    if !witness.valid(clock) {fail(705)}\n" + text[end:]
            text += "\npub(super) fn latest_clock()->usize {VEHICLE_LAST.load(Ordering::SeqCst)}\n"
        else:
            pattern = (
                r"\|\|\s*u32::from_le_bytes\(event\[124\.\.128\]"
                r"\.try_into\(\)\.unwrap\(\)\) as usize\s*!=\s*clock\s*"
                r"\|\|\s*u32::from_le_bytes\(event\[148\.\.152\]"
                r"\.try_into\(\)\.unwrap\(\)\)\s*&\s*0x1bf\s*!=\s*0x1bf"
            )
            if module == "goal_observe":
                before = "|| u(raw,124) as usize!=clock || u(raw,148)&0x1bf!=0x1bf"
                text = replace_once(text, before, "|| !witness.valid(clock)")
                text = replace_once(
                    text, "let participant=q(raw,192);", "let participant=witness.participant();"
                )
            else:
                assert len(re.findall(pattern, text)) == 1, module
                text = re.sub(pattern, "|| !witness.valid(clock)", text)
            if module == "goal_observe":
                # Goal's original function called its reader argument raw.
                # No data substitution: capture now consumes the typed witness.
                text += (
                    "\npub(super) fn delivery_snapshot()->(usize,usize,usize,usize) {"
                    "(FACT_SEQ.load(Ordering::SeqCst)/2,GOALS.load(Ordering::SeqCst),"
                    "LAST_GOAL.load(Ordering::SeqCst),PENDING.load(Ordering::SeqCst))}\n"
                )
        path.write_text(text)
    parent = modules.parent / "post_observe.rs"
    parent.write_text(
        replace_once(
            parent.read_text(),
            "mod goal_observe;",
            "mod goal_observe;\nmod terminal_boundary_observe;",
        )
    )
    result = modules / "result_observe.rs"
    result.write_text(
        replace_once(
            result.read_text(),
            " // Exact 41 8b 85 dc",
            " super::terminal_boundary_observe::at_collector(d);\n // Exact 41 8b 85 dc",
        )
    )
    rust = str(configured_path("TRACKMANIARL_RUST_TOOLS"))
    env = dict(
        os.environ,
        PATH=rust + "/rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:" + os.environ["PATH"],
        CARGO_HOME=rust + "/cargo",
        RUSTUP_HOME=rust + "/rustup",
        CARGO_TARGET_DIR=str(build / "target"),
    )
    with (build / "build.log").open("wb") as log:
        run = subprocess.run(
            ["cargo", "build", "--locked", "--offline", "--release", "-p", "forkshim"],
            cwd=build / "source/tools/search",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if run.returncode:
        raise RuntimeError((build / "build.log").read_text()[-8000:])
    destination = root / "waypoint-runtime002/build/forkoracle/terminal583"
    destination.mkdir(exist_ok=False)
    binary = destination / "libforkshim-impact.so"
    shutil.copyfile(build / "target/release/libforkshim.so", binary)
    receipt = {
        "status": "PASS_DIAGNOSTIC_BUILD_ONLY",
        "training_qualified": False,
        "shim": str(binary),
        "sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
        "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
    }
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
