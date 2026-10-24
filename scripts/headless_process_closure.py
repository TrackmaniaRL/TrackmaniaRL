"""Recheck recorded process identities without signaling any process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.ownership import (
    process_identity,
    same_identity,
)


def closure(roots: list[Path]) -> dict[str, Any]:
    if sys.platform != "linux":
        raise ValueError("Linux process identity records require a Linux recheck")
    sources, identities, remaining = [], {}, []
    for root in roots:
        for path in sorted(root.rglob("*supervisor.json")):
            raw = path.read_bytes()
            report = json.loads(raw)
            if report.get("schema") != "tmrl.native-headless.supervisor/v1":
                raise ValueError("unknown supervisor evidence schema: " + str(path))
            sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            recorded = [
                report["driverIdentity"],
                report["supervisorIdentity"],
                *report.get("capturedOwned", []),
            ]
            for row in recorded:
                if row is None:
                    # Startup fault evidence can precede driver identity capture;
                    # cleanup still records its subreaper-owned descendants.
                    continue
                identities[row["pid"], row["birth"], row["uid"]] = row
    if not sources:
        raise ValueError("actual supervisor evidence required")
    for row in identities.values():
        if same_identity(row, process_identity(row["pid"])):
            remaining.append(row)
    return {
        "schema": "tmrl.recorded-process-closure/v1",
        "passed": not remaining,
        "sources": sources,
        "identitiesChecked": len(identities),
        "remainingOwned": remaining,
        "scope": "recorded driver/supervisor/descendant identities only; read-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = closure(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    keys = ("passed", "identitiesChecked", "remainingOwned")
    print(json.dumps({key: report[key] for key in keys}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
