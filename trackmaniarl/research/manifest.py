"""Map identities and split boundaries; this module never contacts the game."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class MapAsset(BaseModel):
    """An actual map and the environment-only data used to measure its progress."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    uid: str = Field(min_length=1, pattern=r"^\S+$")
    family: str = Field(min_length=1)
    split: Literal["development", "train", "validation", "test"]
    reward_kind: Literal["reward_points", "geometry"]
    reward_path: str
    reward_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_path: str | None = None
    binding: Literal["uid", "uid+sha256"]
    car: str
    physics: str
    camera: str
    features: list[str]
    rights: str
    preparation: str


class MapManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"]
    maps: list[MapAsset] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_identity_and_split(self) -> MapManifest:
        ids: set[str] = set()
        uids: set[str] = set()
        families: dict[str, str] = {}
        for entry in self.maps:
            if entry.id in ids or entry.uid in uids:
                raise ValueError("map IDs and UIDs must be unique")
            if entry.family in families and families[entry.family] != entry.split:
                raise ValueError(f"map family {entry.family!r} leaks across splits")
            ids.add(entry.id)
            uids.add(entry.uid)
            families[entry.family] = entry.split
        return self


def load_manifest(path: Path) -> MapManifest:
    """Read a versioned manifest; relative asset paths are relative to its directory."""
    return MapManifest.model_validate_json(path.read_text(encoding="utf-8"))


def preflight(
    path: Path, map_id: str | None = None, observed_uid: str | None = None
) -> dict[str, Any]:
    """Validate local assets and an optional operator-reported UID, without game I/O."""
    from trackmaniarl.trackmania.reward_points import load_reward_asset

    manifest = load_manifest(path)
    entries = [item for item in manifest.maps if map_id is None or item.id == map_id]
    if not entries:
        raise ValueError(f"map {map_id!r} is absent from the manifest")
    if observed_uid is not None and len(entries) != 1:
        raise ValueError("an observed UID requires exactly one selected map")
    checked = []
    for entry in entries:
        asset_path = (path.parent / entry.reward_path).resolve()
        if not asset_path.is_file():
            raise ValueError(f"missing reward asset for {entry.id}: {asset_path}")
        if sha256(asset_path) != entry.reward_sha256:
            raise ValueError(f"reward asset checksum mismatch for {entry.id}")
        map_path = (path.parent / entry.map_path).resolve() if entry.map_path else None
        if map_path is not None and not map_path.is_file():
            raise ValueError(f"missing map binary for {entry.id}: {map_path}")
        geometry = asset_path if entry.reward_kind == "geometry" else None
        points = asset_path if entry.reward_kind == "reward_points" else None
        asset = load_reward_asset(geometry, points, entry.uid)
        actual_binding = "uid+sha256" if asset.map_sha256 else "uid"
        if actual_binding != entry.binding:
            raise ValueError(f"declared map binding differs from reward asset for {entry.id}")
        asset.validate_map(map_path)
        if observed_uid is not None and observed_uid != entry.uid:
            raise ValueError(f"loaded map UID mismatch for {entry.id}")
        checked.append(
            {
                "id": entry.id,
                "uid": entry.uid,
                "split": entry.split,
                "reward_sha256": entry.reward_sha256,
                "binding": entry.binding,
                "map_sha256": asset.map_sha256 or None,
            }
        )
    return {
        "schema_version": "1",
        "status": "assets_validated",
        "maps": checked,
        "live_game_verified": False,
        "next_step": "Manually load the exact map; runtime verifies UID and readiness.",
    }


def write_json(path: Path, payload: object) -> None:
    """Write a new artifact, refusing to overwrite an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
