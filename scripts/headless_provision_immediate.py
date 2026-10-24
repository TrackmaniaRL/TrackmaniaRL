"""Provision a fresh V7 deployment from private, explicitly supplied local assets.

No engine runs and no files are downloaded. The source-only patch and manifest
under native_input_v7 document rebuilding the supplied shim from its private
source archive; this command verifies the tested artifact before provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import portable
from trackmaniarl.trackmania.native_headless.assets import verify_assets
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_ASSET_SCHEMA,
    IMMEDIATE_PROFILE,
    IMMEDIATE_PROTOCOL,
    IMMEDIATE_RESET_PROFILE,
    NativeRuntimeConfig,
)


def save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_selected(source: Path, target: Path, selected: set[str]) -> None:
    def ignored(directory: str, names: list[str]) -> list[str]:
        relative = Path(directory).relative_to(source)
        return [
            name
            for name in names
            if not any(
                item == (relative / name).as_posix()
                or item.startswith((relative / name).as_posix() + "/")
                for item in selected
            )
        ]

    shutil.copytree(source, target, ignore=ignored)


def provision(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination
    if (
        sys.platform != "linux"
        or not destination.is_absolute()
        or str(destination).startswith("/mnt/")
        or destination.exists()
        or destination.resolve() != destination
        or any(parent.is_symlink() for parent in destination.parents)
    ):
        raise ValueError("a fresh canonical absolute Linux/ext4 destination is required")
    raw_config = json.loads(args.v6_config.read_bytes())
    original = NativeRuntimeConfig.from_mapping(raw_config.get("runtime", raw_config))
    if original.protocol_profile != "pending-future/v6":
        raise ValueError("an explicitly approved V6 deployment is required as source")
    verified = verify_assets(original)
    old_deployment = json.loads(original.deployment_manifest.read_bytes())
    source_manifest = json.loads(original.asset_manifest.read_bytes())
    package = Path(portable.__file__).parent
    pins = json.loads((package / "immediate_profile.json").read_bytes())
    required = {
        "original-source-manifest.json": pins["sourceManifestSha256"],
        "v6-to-v7.patch": pins["patchSha256"],
        "build-receipt.json": pins["buildReceiptSha256"],
    }
    for name, expected in required.items():
        if portable.file_sha(args.source_evidence_dir / name) != expected:
            raise ValueError("exact tested V7 source/build evidence required: " + name)
    if portable.file_sha(args.shim) != pins["shimSha256"]:
        raise ValueError("supplied V7 shim differs from the tested source/build receipt")
    roles = {role: source_manifest["roles"][role] for role in ("server", "driver", "map", "tape")}
    selected = set(roles.values()) | {row["path"] for row in old_deployment["supportAssets"]}
    # Verify every source path before creating the target. No historical work or
    # results are part of this whitelist.
    for role, path in roles.items():
        if verified.paths[role] != original.runtime_root / path:
            raise ValueError("the tested fixed role layout is required")
    destination.mkdir(mode=0o700, parents=True)
    runtime = destination / "runtime"
    copy_selected(original.runtime_root, runtime, selected)
    shim_directory = runtime / "build/forkoracle/native-immediate-v7-001"
    copy_selected(args.shim.parent, shim_directory, {args.shim.name})
    roles["shim"] = (shim_directory / args.shim.name).relative_to(runtime).as_posix()
    evidence = shim_directory / "evidence"
    copy_selected(args.source_evidence_dir, evidence, set(required))
    for role, name in {
        "shim_source_manifest": "original-source-manifest.json",
        "shim_patch": "v6-to-v7.patch",
        "shim_build_receipt": "build-receipt.json",
    }.items():
        roles[role] = (evidence / name).relative_to(runtime).as_posix()
    assets = {
        name: {
            "sha256": portable.file_sha(runtime / name),
            "bytes": (runtime / name).stat().st_size,
        }
        for name in roles.values()
    }
    save(
        runtime / "assets.json",
        {
            "schema": IMMEDIATE_ASSET_SCHEMA,
            "profile": IMMEDIATE_PROFILE,
            "runtime_root": str(runtime),
            "roles": roles,
            "assets": assets,
        },
    )
    deployment = {
        **old_deployment,
        "schema": "tmrl.native-immediate-deployment/v1",
        "runtime_root": str(runtime),
        "immediateBuild": pins,
    }
    save(runtime / "deployment.json", deployment)
    settings = {
        "runtime_root": str(runtime),
        "trace_dir": str(destination / "results"),
        "work_dir": str(runtime / "build/server-probe/immediate-work"),
        "asset_manifest": str(runtime / "assets.json"),
        "deployment_manifest": str(runtime / "deployment.json"),
        "deployment_manifest_sha256": portable.file_sha(runtime / "deployment.json"),
        "protocol_profile": IMMEDIATE_PROTOCOL,
        "reset_profile": IMMEDIATE_RESET_PROFILE,
        "timeout_s": original.timeout_s,
        "duration_limit": args.duration_limit,
        "max_trace_bytes": args.max_trace_bytes,
        "allow_native": True,
    }
    config = NativeRuntimeConfig.from_mapping(settings)
    checked = verify_assets(config)
    save(destination / "config.json", settings)
    receipt = {
        "schema": "tmrl.native-immediate-provision/v1",
        "profile": checked.profile,
        "runtime": str(runtime),
        "config": str(destination / "config.json"),
        "assetManifestSha256": checked.manifest_sha256,
        "deploymentManifestSha256": settings["deployment_manifest_sha256"],
        "sourceAssetManifestSha256": verified.manifest_sha256,
        "pinnedAssetAndSupportFiles": len(roles) + len(deployment["supportAssets"]),
        "engineExecuted": False,
        "runtimeQualifiedByCopy": False,
    }
    receipt["sourceEvidenceSha256"] = hashlib.sha256(
        json.dumps(required, sort_keys=True).encode()
    ).hexdigest()
    save(destination / "provision.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument(
        "--source-evidence-dir",
        type=Path,
        default=Path(portable.__file__).with_name("native_input_v7"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--duration-limit", type=int, default=39950)
    parser.add_argument("--max-trace-bytes", type=int, default=64 * 1024 * 1024)
    print(json.dumps(provision(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
