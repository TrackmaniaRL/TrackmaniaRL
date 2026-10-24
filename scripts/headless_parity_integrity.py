"""Read-only verification of frozen inputs and final owned-process closure."""

import json
import sys
from pathlib import Path

from scripts.diagnostic_inputs import configured_path
from trackmaniarl.trackmania.native_headless._runtime.ownership import (
    process_identity,
    same_identity,
)
from trackmaniarl.trackmania.native_headless.assets import verify_assets
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


def main() -> None:
    deployment, report_path, destination = map(Path, sys.argv[1:4])
    baseline = Path(str(configured_path("TRACKMANIARL_NATIVE_BASELINE")))
    sys.path.append(str(baseline / "package"))
    from preview200.common import verify_source

    source = verify_source(baseline / "package")
    assets = verify_assets(
        NativeRuntimeConfig.from_mapping(
            json.loads((baseline / "runtime/config.json").read_bytes())["runtime"]
        )
    )
    current = verify_assets(
        NativeRuntimeConfig.from_mapping(json.loads((deployment / "config.json").read_bytes()))
    )
    probe = json.loads(report_path.read_bytes())
    remaining = [
        identity
        for case in probe["cases"]
        for identity in case.get("owned_identities", [])
        if same_identity(identity, process_identity(identity["pid"]))
    ]
    assert not remaining
    value = {
        "passed": True,
        "frozen_source": source,
        "frozen_assets_sha256": assets.manifest_sha256,
        "development_assets_sha256": current.manifest_sha256,
        "remaining_owned_processes": remaining,
        "baseline_modified": False,
        "scope": "manifest-bound frozen source/assets; no engine execution",
    }
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
