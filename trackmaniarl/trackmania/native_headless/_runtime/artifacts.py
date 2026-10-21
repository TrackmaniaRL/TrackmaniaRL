from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import features

ROLE_PATHS = {
    "server": "build/server-probe/linux-validate-map2-001/TrackmaniaServer",
    "driver": "build/tmrl-shortcut/long-horizon-001/driver/target/release/tmre-reset-driver",
    "shim": "build/forkoracle/native-long-v6-001/libforkshim.so",
    "map": "analysis/tmrl-shortcut/assets-001/tmrl-test.Map.Gbx",
    "tape": "analysis/tmrl-shortcut/long-horizon-controls-001/n4152-decl40000.Ghost.Gbx",
    "reference": "experiments/headless-native16-training-001/reference.json",
    "cold_post": "analysis/headless-normalstart/run-001/posts/post-828.bin",
}
PINS = {
    ROLE_PATHS["server"]: "3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779",
    ROLE_PATHS["driver"]: "669486302fd66cdd8833128e06879089d1c4073ceed0dc11bcc0963fea06e1c4",
    ROLE_PATHS["shim"]: "a14837a1c64595a697ddea89637e4f12a904df7ddaa84d814b4fc2cd47e97781",
    ROLE_PATHS["map"]: "ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc",
    ROLE_PATHS["tape"]: "1cb2dfe4749c9743c8aa7cc710c5442f82c1193157e2c15b8e973f5bb5d6979e",
    ROLE_PATHS["reference"]: "49b7663cc35f4fefedb68309a54e37c627634c990d51aec9de3590a217efdeea",
    ROLE_PATHS["cold_post"]: "b30d31c5d0758319237c3251f7d79c4b42d5105bda8a66e457847d3de9a7d324",
}


def dependencies(root: Path) -> tuple[Any, features.Reference, dict[str, Any]]:
    path = root / ROLE_PATHS["reference"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[ROLE_PATHS["reference"]]:
        raise ValueError("external reference artifact changed")
    data: dict[str, Any] = json.loads(raw)
    if (
        data["source"]["file"] != ROLE_PATHS["cold_post"]
        or data["source"]["sha256"] != PINS[ROLE_PATHS["cold_post"]]
    ):
        raise ValueError("reference cold trajectory identity differs")
    reference = features.Reference(data["points"], data["source"])
    if reference.fingerprint != data["referenceFingerprint"]:
        raise ValueError("reference fingerprint differs")
    return (features, reference, data)
