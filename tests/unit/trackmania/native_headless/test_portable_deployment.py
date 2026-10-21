"""Deployment identity checks with synthetic files; no server or model execution."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless._runtime import portable
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


@pytest.fixture
def deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[NativeRuntimeConfig, dict[str, str], dict[str, object]]:
    support = tmp_path / "support.fixture"
    support.write_bytes(b"synthetic support; not a game asset")
    profile = {
        "driverBuildReceiptSha256": "a" * 64,
        "driverSourceManifestSha256": "b" * 64,
        "portableDriverSha256": "c" * 64,
        "supportAssets": [
            {
                "path": support.name,
                "bytes": support.stat().st_size,
                "sha256": sha256(support.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "portable_profile.json").write_text(json.dumps(profile))
    monkeypatch.setattr(portable, "__file__", str(tmp_path / "portable.py"))
    document: dict[str, object] = {
        "schema": "tmre.preview200-deployment/v1",
        "runtime_root": str(tmp_path),
        "driverBuildApprovalFileSha256": profile["driverBuildReceiptSha256"],
        "driverBuildApproval": {
            "schema": "tmre.preview200-driver211-build-approval/v1",
            "passed": True,
            "sourceManifestSha256": profile["driverSourceManifestSha256"],
            "runtimeRootConfigurable": True,
            "wireABI": "V6-request192-response448",
            "driverSha256": profile["portableDriverSha256"],
        },
        "supportAssets": profile["supportAssets"],
    }
    manifest = tmp_path / "deployment.json"
    manifest.write_text(json.dumps(document))
    settings = NativeRuntimeConfig(
        runtime_root=tmp_path,
        trace_dir=tmp_path / "trace",
        work_dir=tmp_path / "work",
        asset_manifest=tmp_path / "assets.json",
        deployment_manifest=manifest,
        deployment_manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
    )
    return settings, {"driver": "d" * 64, "server": "e" * 64}, document


def test_portable_deployment_preserves_other_role_pins(
    deployment: tuple[NativeRuntimeConfig, dict[str, str], dict[str, object]],
) -> None:
    settings, baseline, _ = deployment
    result = portable.expected_role_hashes(settings, baseline)
    assert result == {"driver": "c" * 64, "server": baseline["server"]}
    assert baseline["driver"] == "d" * 64


def test_deployment_rejects_missing_approval_and_modified_support(
    deployment: tuple[NativeRuntimeConfig, dict[str, str], dict[str, object]],
) -> None:
    settings, baseline, _ = deployment
    with pytest.raises(ValueError, match="explicit deployment manifest"):
        portable.expected_role_hashes(
            replace(settings, deployment_manifest=None, deployment_manifest_sha256=None), baseline
        )
    (settings.runtime_root / "support.fixture").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Packs bytes differ"):
        portable.expected_role_hashes(settings, baseline)


@pytest.mark.parametrize("field", ["driverBuildApprovalFileSha256", "driverBuildApproval"])
def test_rehashing_changed_deployment_does_not_approve_another_driver(
    deployment: tuple[NativeRuntimeConfig, dict[str, str], dict[str, object]], field: str
) -> None:
    settings, baseline, document = deployment
    document[field] = "unexpected"
    manifest = settings.deployment_manifest
    assert manifest is not None
    manifest.write_text(json.dumps(document))
    settings = replace(
        settings, deployment_manifest_sha256=sha256(manifest.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match=r"receipt identity|build approval"):
        portable.expected_role_hashes(settings, baseline)


def test_changed_manifest_bytes_require_the_explicit_digest(
    deployment: tuple[NativeRuntimeConfig, dict[str, str], dict[str, object]],
) -> None:
    settings, baseline, _ = deployment
    manifest = settings.deployment_manifest
    assert manifest is not None
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest SHA differs"):
        portable.expected_role_hashes(settings, baseline)
