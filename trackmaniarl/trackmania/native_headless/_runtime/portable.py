"""Preview200 deployment identity checks. Reads only; no engine at import."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(path: Path, *, directory: bool) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("explicit canonical absolute deployment path required")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("deployment path symlink rejected")
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError("wrong deployment path type")
    return path


def expected_role_hashes(settings: Any, baseline: Mapping[str, str]) -> dict[str, str]:
    """Only a hash-bound211 build approval can replace the old fixed driver.

    The SHA in settings is an explicit trusted caller input, not a signature or
    redistribution permission. The six other original role hashes stay fixed.
    """
    root = canonical(settings.runtime_root, directory=True)
    path = settings.deployment_manifest
    expected = settings.deployment_manifest_sha256
    if path is None or expected is None:
        raise ValueError("preview200 requires explicit deployment manifest and SHA")
    canonical(path, directory=False)
    if file_sha(path) != expected:
        raise ValueError("deployment manifest SHA differs")
    data = json.loads(path.read_bytes())
    profile = json.loads(Path(__file__).with_name("portable_profile.json").read_bytes())
    immediate = getattr(settings, "protocol_profile", "pending-future/v6") == "immediate-current/v7"
    schema = "tmrl.native-immediate-deployment/v1" if immediate else "tmre.preview200-deployment/v1"
    if data.get("schema") != schema or data.get("runtime_root") != str(root):
        raise ValueError("deployment root/schema mismatch")
    if data.get("driverBuildApprovalFileSha256") != profile["driverBuildReceiptSha256"]:
        raise ValueError("exact215 driver build receipt identity required")
    approval = data.get("driverBuildApproval")
    if not isinstance(approval, dict) or any(
        approval.get(key) != value
        for key, value in {
            "schema": "tmre.preview200-driver211-build-approval/v1",
            "passed": True,
            "sourceManifestSha256": profile["driverSourceManifestSha256"],
            "runtimeRootConfigurable": True,
            "wireABI": "V6-request192-response448",
        }.items()
    ):
        raise ValueError("explicit source211 build approval required")
    driver_hash = approval.get("driverSha256")
    if (
        not isinstance(driver_hash, str)
        or len(driver_hash) != 64
        or any(character not in "0123456789abcdef" for character in driver_hash)
        or driver_hash == baseline["driver"]
        or driver_hash != profile["portableDriverSha256"]
    ):
        raise ValueError("new portable211 driver hash required; old fixed driver unsupported")
    if data.get("supportAssets") != profile["supportAssets"]:
        raise ValueError("complete tested four-file Packs identity required")
    for record in profile["supportAssets"]:
        support = canonical(root / record["path"], directory=False)
        if (
            not support.is_relative_to(root)
            or support.stat().st_size != record["bytes"]
            or file_sha(support) != record["sha256"]
        ):
            raise ValueError("server Packs bytes differ")
    result = {**baseline, "driver": driver_hash}
    if immediate:
        pins = json.loads(Path(__file__).with_name("immediate_profile.json").read_bytes())
        if data.get("immediateBuild") != pins:
            raise ValueError("exact source/patch/build receipt identities required for V7")
        result.update(
            shim=pins["shimSha256"],
            shim_source_manifest=pins["sourceManifestSha256"],
            shim_patch=pins["patchSha256"],
            shim_build_receipt=pins["buildReceiptSha256"],
        )
        result.pop("reference", None)
        result.pop("cold_post", None)
    return result
