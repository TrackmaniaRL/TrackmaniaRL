"""Export a trusted value-policy checkpoint without altering its source or identity.

Usage: python -m trackmaniarl.core.policy_export SOURCE.pt DESTINATION.pt
The output is for evaluation/compatible weight initialization, never exact resume.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.checkpoints import validate_policy_checkpoint_v2


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_policy_checkpoint(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy validated policy fields verbatim; omit replay, targets and optimizer.

    Load only trusted Torch artifacts. Preserve the original architecture
    fingerprint. The consumer must still check observation/action semantics;
    this export is not a compatibility migration.
    """
    source_path, target = Path(source).resolve(strict=True), Path(destination).resolve()
    if target == source_path or target.exists():
        raise FileExistsError("Policy export requires a new destination distinct from the source")
    before = _sha256(source_path)
    codec = TorchCheckpointCodec()
    checkpoint = codec.load(source_path)
    if checkpoint.get("schema_version") != "2.0":
        raise ValueError("Policy export requires checkpoint schema 2.0")
    learner = checkpoint.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError("Checkpoint must contain a learner mapping")
    validate_policy_checkpoint_v2(learner)
    policy = {key: learner[key] for key in ("schema_version", "architecture_fingerprint", "online")}
    report = {
        "schema_version": "trackmaniarl-policy-export-v1",
        "checkpoint_kind": "policy_only",
        "resume_supported": False,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": before,
        "source_run_fingerprint": checkpoint.get("run_fingerprint"),
        "architecture_fingerprint": learner["architecture_fingerprint"],
        "transformation": "Verbatim online modules; no tensor, fingerprint or action conversion",
    }
    if _sha256(source_path) != before:
        raise RuntimeError("Source checkpoint changed during export")
    codec.save({"schema_version": "2.0", "learner": policy, "policy_export": report}, target)
    return {**report, "destination": str(target), "destination_sha256": _sha256(target)}


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report_path = args.destination.with_suffix(".export.json")
    if report_path.exists():
        raise FileExistsError(f"Export report already exists: {report_path}")
    report = export_policy_checkpoint(args.source, args.destination)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
