"""Inspect the exact shipped reader's bounded code window with REA/Ghidra.

The slice provenance is explicit; external CFG and calls stay unknown.
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1])
    target = Path(sys.argv[2])
    output = Path(sys.argv[3])
    output.mkdir(mode=0o700)
    tools = json.loads((root / "toolchain.json").read_bytes())
    original = target.read_bytes()
    if (
        hashlib.sha256(original).hexdigest()
        != "3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779"
    ):
        raise ValueError("the exact approved shipped ELF is required")
    virtual_address, length = 0x119F0F0, 0x400
    if original[:6] != b"\x7fELF\x02\x01":
        raise ValueError("little-endian 64-bit ELF required")
    phoff = struct.unpack_from("<Q", original, 32)[0]
    phsize, phnum = struct.unpack_from("<HH", original, 54)
    mappings = [struct.unpack_from("<IIQQQQQQ", original, phoff + i * phsize) for i in range(phnum)]
    load = next(
        p
        for p in mappings
        if p[0] == 1 and p[3] <= virtual_address and virtual_address + length <= p[3] + p[5]
    )
    offset = load[2] + virtual_address - load[3]
    content = original[offset : offset + length]
    if content[:4] != bytes.fromhex("554889e5"):
        raise ValueError("reader function prologue changed")
    binary = output / "reader-code-window.bin"
    binary.write_bytes(content)
    (output / "slice-provenance.json").write_text(
        json.dumps(
            {
                "original": str(target),
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "file_offset": offset,
                "original_virtual_address": virtual_address,
                "length": length,
                "slice_sha256": hashlib.sha256(content).hexdigest(),
                "limitation": (
                    "bounded static code window; external targets and whole-program CFG unknown"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(
        os.environ,
        JAVA_HOME=tools["java"],
        GHIDRA_INSTALL_DIR=tools["ghidra"],
        PATH=tools["node"] + "/bin:" + os.environ["PATH"],
        REA_GHIDRA_STARTUP_TIMEOUT_MS="60000",
    )
    command = [
        tools["node"] + "/bin/node",
        tools["rea"] + "/scripts/rea.mjs",
        "decompile",
        str(binary),
        "--target-format",
        "raw-binary",
        "--architecture",
        "x86_64",
        "--entry-point",
        hex(virtual_address),
        "--procedure",
        hex(virtual_address),
        "--provider",
        "ghidra",
        "--json",
    ]
    with (
        (output / "rea-evidence.json").open("x") as stdout,
        (output / "rea.stderr").open("x") as stderr,
    ):
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
