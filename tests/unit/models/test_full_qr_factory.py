"""Asset/source refusals; synthetic files never qualify a native deployment."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.experiments.full_qr.runtime_profile import ASSET_SCHEMA, PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.full_qr.factory import (
    FullQrEnvironmentFactory,
    verify_bound_assets,
)


def assets(tmp_path: Path) -> tuple[SimpleNamespace, dict, Path]:
    root = tmp_path / "runtime"
    root.mkdir()
    records = {}
    roles = {name: name + ".bin" for name in ("server", "driver", "shim", "map", "tape")}
    for name in roles.values():
        path = root / name
        path.write_bytes(b"synthetic fixture only: " + name.encode())
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": ASSET_SCHEMA,
        "profile": PROFILE,
        "runtime_root": str(root),
        "roles": roles,
        "assets": records,
    }
    path = root / "assets.json"
    path.write_text(json.dumps(manifest))
    return (
        SimpleNamespace(protocol_profile=PROTOCOL, runtime_root=root, asset_manifest=path),
        manifest,
        path,
    )


def test_missing_event_source_is_rejected_before_any_runtime_or_path_access() -> None:
    with pytest.raises(ValueError, match="event-source"):
        FullQrEnvironmentFactory({})


def test_manifest_pin_and_actual_asset_bytes_both_remain_required(tmp_path: Path) -> None:
    settings, _, path = assets(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = verify_bound_assets(settings, expected)
    with pytest.raises(TypeError):
        verified.paths["server"] = tmp_path
    with pytest.raises(ValueError, match="manifest changed"):
        verify_bound_assets(settings, "0" * 64)
    verified.paths["map"].write_bytes(b"changed actual bytes")
    with pytest.raises(ValueError, match="asset differs"):
        verify_bound_assets(settings, expected)


def test_correctly_hashed_old_diagnostic_manifest_is_not_a_full_v8_manifest(tmp_path: Path) -> None:
    settings, manifest, path = assets(tmp_path)
    manifest["schema"] = "tmrl.experimental-long150-diagnostic.assets/v1"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema/root/roles"):
        verify_bound_assets(settings, hashlib.sha256(path.read_bytes()).hexdigest())


def test_duplicate_json_keys_cannot_override_the_pinned_deployment_schema(tmp_path: Path) -> None:
    settings, _, path = assets(tmp_path)
    raw = path.read_text()
    path.write_text('{"schema": "wrong",' + raw[1:])
    with pytest.raises(ValueError, match="duplicate manifest key"):
        verify_bound_assets(settings, hashlib.sha256(path.read_bytes()).hexdigest())
