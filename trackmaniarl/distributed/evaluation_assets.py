"""Stable content identity for a multi-map evaluation assignment.

This binds configured files, not the bytes loaded by a game process. A backend
must independently attest the active map and its isolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def evaluation_assets_digest(suite: Any, base_dir: Path) -> str:
    maps = []
    for item in suite.maps:
        if item.map_path is None:
            raise ValueError("multi-map evaluation requires an explicit map file")
        assets: dict[str, str | None] = {}
        for name in ("map_path", "geometry_path", "reward_points_path"):
            path = getattr(item, name)
            if path is None:
                assets[name] = None
                continue
            with (base_dir / path).open("rb") as stream:
                assets[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        maps.append({"id": item.id, "uid": item.expected_map_uid, "assets": assets})
    data = {
        "name": suite.name,
        "version": suite.version,
        "maps": maps,
        "trials_per_map": suite.trials_per_map,
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
