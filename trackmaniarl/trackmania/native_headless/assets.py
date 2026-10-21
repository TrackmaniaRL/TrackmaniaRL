"""Hash-verified external assets; no bundled game files or import-time I/O."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from trackmaniarl.trackmania.native_headless.config import (
    ASSET_SCHEMA,
    NATIVE_PROFILE,
    NativeRuntimeConfig,
)

# These identities describe exactly one tested external profile, not a discovery API.
PINNED_ROLE_SHA256 = {
    "server": "3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779",
    "driver": "669486302fd66cdd8833128e06879089d1c4073ceed0dc11bcc0963fea06e1c4",
    "shim": "a14837a1c64595a697ddea89637e4f12a904df7ddaa84d814b4fc2cd47e97781",
    "map": "ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc",
    "tape": "1cb2dfe4749c9743c8aa7cc710c5442f82c1193157e2c15b8e973f5bb5d6979e",
    "reference": "49b7663cc35f4fefedb68309a54e37c627634c990d51aec9de3590a217efdeea",
    "cold_post": "b30d31c5d0758319237c3251f7d79c4b42d5105bda8a66e457847d3de9a7d324",
}


class NativeAssetError(ValueError):
    """An external asset manifest or pinned file failed verification."""


@dataclass(frozen=True, slots=True)
class VerifiedAssets:
    profile: str
    runtime_root: Path
    paths: Mapping[str, Path]
    manifest_sha256: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeAssetError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise NativeAssetError("asset paths must be explicit relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise NativeAssetError(f"asset path escapes or aliases its root: {value}")
    if str(path) != value or "\x00" in value:
        raise NativeAssetError(f"noncanonical asset path: {value}")
    return value


def verify_assets(settings: NativeRuntimeConfig) -> VerifiedAssets:
    """Verify every declared path and byte hash; no verification cache is trusted.

    This performs reads only. It never creates directories, loads a DLL, calls an
    engine or interprets reference JSON as executable code.
    """
    try:
        from trackmaniarl.trackmania.native_headless._runtime.portable import expected_role_hashes

        role_hashes = expected_role_hashes(settings, PINNED_ROLE_SHA256)
        root = settings.runtime_root.resolve(strict=True)
        raw = settings.asset_manifest.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(manifest, dict):
            raise NativeAssetError("asset manifest must be a JSON object")
        if manifest.get("schema") != ASSET_SCHEMA or manifest.get("profile") != NATIVE_PROFILE:
            raise NativeAssetError("unsupported native asset schema or profile")
        if manifest.get("runtime_root") != str(settings.runtime_root):
            raise NativeAssetError(
                "manifest runtime_root does not match the explicit configuration"
            )
        roles, assets = manifest.get("roles"), manifest.get("assets")
        if not isinstance(roles, dict) or set(roles) != set(PINNED_ROLE_SHA256):
            raise NativeAssetError(
                "manifest requires exactly server/driver/shim/map/tape/reference/cold_post"
            )
        if not isinstance(assets, dict):
            raise NativeAssetError("manifest assets must be a path-to-hash mapping")
        role_paths = {role: _relative_path(path) for role, path in roles.items()}
        if len(set(role_paths.values())) != len(role_paths):
            raise NativeAssetError("asset roles must name separate files")
        if set(assets) != set(role_paths.values()):
            raise NativeAssetError("assets must contain exactly the explicit role paths")
        verified: dict[str, Path] = {}
        for name, record in assets.items():
            relative = _relative_path(name)
            if not isinstance(record, dict):
                raise NativeAssetError(f"invalid asset record: {relative}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise NativeAssetError(f"invalid SHA-256: {relative}")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise NativeAssetError(
                    f"asset is not a regular file within runtime_root: {relative}"
                )
            data = path.read_bytes()
            if "bytes" in record and (
                type(record["bytes"]) is not int or record["bytes"] != len(data)
            ):
                raise NativeAssetError(f"asset size mismatch: {relative}")
            if sha256(data).hexdigest() != digest:
                raise NativeAssetError(f"asset SHA-256 mismatch: {relative}")
            verified[relative] = path
        for role, name in role_paths.items():
            if assets[name]["sha256"] != role_hashes[role]:
                raise NativeAssetError(f"unsupported {role} build for this native profile")
        return VerifiedAssets(
            profile=NATIVE_PROFILE,
            runtime_root=root,
            paths=MappingProxyType({role: verified[name] for role, name in role_paths.items()}),
            manifest_sha256=sha256(raw).hexdigest(),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeAssetError(f"cannot verify native assets: {error}") from error
