"""Build a separate V8 source deployment without changing the R7/R10 runtime.

This is preparation, not a source qualification or a training launcher. Each
explicit substitution is checked against the current implementation. In
particular the EOF/collector checks are copied intact, never reverted to the
older diagnostic snapshot's behavior. The native driver/shim/assets must still
be independently pinned by the caller's deployment manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from trackmaniarl.trackmania.native_headless.full_qr.profile import (
    ASSET_SCHEMA,
    CALIBRATION_I8,
    PROFILE,
    PROTOCOL,
    WIRE_PROFILE,
)

_ROOT = "trackmaniarl/trackmania/native_headless/"
_REPLACEMENTS = {
    _ROOT + "config.py": (
        (
            'IMMEDIATE_ASSET_SCHEMA = "tmrl.native-immediate.assets/v1"',
            f'IMMEDIATE_ASSET_SCHEMA = "{ASSET_SCHEMA}"',
        ),
        (
            'IMMEDIATE_PROFILE = "native-immediate-kinematics6/v7"',
            f'IMMEDIATE_PROFILE = "{PROFILE}"',
        ),
        ('IMMEDIATE_PROTOCOL = "immediate-current/v7"', f'IMMEDIATE_PROTOCOL = "{PROTOCOL}"'),
        ("0 < duration <= 47770", "0 < duration <= 150000"),
        ("10..47770 in steps", "10..150000 in steps"),
    ),
    _ROOT + "_runtime/codec_v6.py": (
        (
            'PROFILE_ID = "tmre.long4152-declared40000/v6"',
            'PROFILE_ID = "tmre.full-qr16152-raw/v8"',
        ),
        ("INPUT_COUNT = 4152", "INPUT_COUNT = 16152"),
        ("MAX_NATIVE = 50000", "MAX_NATIVE = 162200"),
    ),
    _ROOT + "_runtime/codec_v7.py": (
        ('PROFILE_ID = "tmre.long4152-immediate/v7"', f'PROFILE_ID = "{WIRE_PROFILE}"'),
        ('b"TMR7\\x07"', 'b"TMR8\\x08"'),
        ('b"TMA7\\x07"', 'b"TMA8\\x08"'),
        ("version=7,", "version=8,"),
        (
            'wireProfile="TMR7/v7; op6 pre-load current input"',
            'wireProfile="TMR8/v8; op6 pre-load current input; full QR 150s"',
        ),
    ),
    _ROOT + "_runtime/env.py": (
        (
            'RUNTIME_PROFILE = "tmre.long4152-declared40000/v6"',
            'RUNTIME_PROFILE = "tmre.full-qr16152-raw/v8"',
        ),
        ('meta["input_count"] == 4152', 'meta["input_count"] == 16152'),
        ("u(raw, 112) < 4152", "u(raw, 112) < 16152"),
        ("p + 1 + length <= 4152", "p + 1 + length <= 16152"),
    ),
    _ROOT + "_runtime/immediate.py": (
        (
            'self.ready["metadata"]["input_count"] == 4152',
            'self.ready["metadata"]["input_count"] == 16152',
        ),
    ),
    _ROOT + "_runtime/features.py": (
        (
            'steer = struct.unpack("<f", struct.pack("<f", (steerbin - 6) / 6))[0]',
            'steer = struct.unpack("<f", struct.pack("<f", '
            f"{CALIBRATION_I8!r}[steerbin] / 127.0))[0]",
        ),
    ),
}


def transform_source(relative: str, text: str) -> str:
    """Apply only the independently versioned profile's reviewed differences."""
    for before, after in _REPLACEMENTS.get(relative, ()):
        if text.count(before) != 1:
            raise ValueError(f"V8 source prerequisite changed: {relative}: {before!r}")
        text = text.replace(before, after)
    return text


def build_source_deployment(source: Path, destination: Path) -> dict[str, object]:
    """Write a fresh ext4 deployment, retaining original source hashes for review.

    No imports from the new deployment occur here. No process, environment,
    frozen deployment, existing output, STOP or checkpoint is modified.
    """
    source = source.resolve(strict=True)
    if (
        not destination.is_absolute()
        or destination.exists()
        or destination.is_symlink()
        or str(destination).startswith("/mnt/")
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise ValueError("a fresh canonical absolute ext4 deployment directory is required")
    if shutil.disk_usage(source).free < 10 * 1024**3:
        raise OSError("source volume requires at least10GiB reserve")
    if shutil.disk_usage(destination.parent).free < 4 * 1024**3:
        raise OSError("deployment volume requires at least4GiB reserve")
    # Validate every source prerequisite before creating an output directory.
    changed = {}
    for relative in _REPLACEMENTS:
        raw = (source / relative).read_bytes()
        changed[relative] = (raw, transform_source(relative, raw.decode("utf-8")))
    destination.mkdir()
    for name in ("trackmaniarl",):
        shutil.copytree(
            source / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    changes = []
    for relative, (before, after) in changed.items():
        target = destination / relative
        target.write_text(after, encoding="utf-8", newline="\n")
        changes.append(
            {
                "path": relative,
                "source_sha256": hashlib.sha256(before).hexdigest(),
                "deployment_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    receipt: dict[str, object] = {
        "schema": "tmrl.full-qr-source-deployment/v1",
        "profile": PROFILE,
        "protocol": PROTOCOL,
        "wire_profile": WIRE_PROFILE,
        "source": str(source),
        "destination": str(destination),
        "changes": changes,
        "steering_i8": CALIBRATION_I8,
        "qualification": False,
        "training_started": False,
    }
    (destination / "source-deployment.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
