"""Provisioning is available as an installed module and its old wrapper still works."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import headless_provision_immediate as legacy
from trackmaniarl.trackmania.native_headless import provision_immediate as installed

ROOT = Path(__file__).resolve().parents[3]


def test_legacy_api_stays_bound_to_the_packaged_implementation() -> None:
    assert legacy.provision is installed.provision
    assert legacy.copy_selected is installed.copy_selected
    assert legacy.save is installed.save
    assert legacy.main is installed.main


def test_provision_module_help_works_from_outside_the_checkout(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trackmaniarl.trackmania.native_headless.provision_immediate",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--v6-config" in completed.stdout
    assert "--source-evidence-dir" in completed.stdout
    assert "--destination" in completed.stdout
    assert "private build-receipt.json is not included" in " ".join(completed.stdout.split())


def test_missing_private_evidence_explains_how_to_supply_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SimpleNamespace(
        protocol_profile="pending-future/v6",
        deployment_manifest=tmp_path / "deployment.json",
        asset_manifest=tmp_path / "assets.json",
    )
    for path in (original.deployment_manifest, original.asset_manifest):
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "immediate_profile.json").write_text(
        json.dumps(
            {
                "sourceManifestSha256": "synthetic",
                "patchSha256": "synthetic",
                "buildReceiptSha256": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "native_input_v7"
    evidence.mkdir()
    (evidence / "original-source-manifest.json").write_text("", encoding="utf-8")
    (evidence / "v6-to-v7.patch").write_text("", encoding="utf-8")
    monkeypatch.setattr(installed.sys, "platform", "linux")
    monkeypatch.setattr(installed.portable, "__file__", str(tmp_path / "portable.py"))
    monkeypatch.setattr(installed.portable, "file_sha", lambda _: "synthetic")
    monkeypatch.setattr(installed.NativeRuntimeConfig, "from_mapping", lambda _: original)
    monkeypatch.setattr(installed, "verify_assets", lambda _: object())
    config = tmp_path / "v6.json"
    config.write_text("{}", encoding="utf-8")
    manifest = evidence / "original-source-manifest.json"
    manifest.unlink()
    with pytest.raises(ValueError, match=r"original-source-manifest\.json.*--source-evidence-dir"):
        installed.provision(
            argparse.Namespace(
                destination=tmp_path / "fresh",
                v6_config=config,
                source_evidence_dir=evidence,
            )
        )
    manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(installed.portable, "file_sha", lambda _: "wrong hash")
    with pytest.raises(ValueError, match="exact tested V7 source/build evidence required"):
        installed.provision(
            argparse.Namespace(
                destination=tmp_path / "fresh",
                v6_config=config,
                source_evidence_dir=evidence,
            )
        )
    monkeypatch.setattr(installed.portable, "file_sha", lambda _: "synthetic")
    with pytest.raises(ValueError, match=r"build-receipt\.json.*--source-evidence-dir"):
        installed.provision(
            argparse.Namespace(
                destination=tmp_path / "fresh",
                v6_config=config,
                source_evidence_dir=evidence,
            )
        )

    # A caller may place the private file in the default package directory.
    # Once all three files exist and match, provisioning reaches the next check.
    (evidence / "build-receipt.json").write_text("", encoding="utf-8")
    shim = tmp_path / "shim.so"

    def stop_at_shim(path: Path) -> str:
        if path == shim:
            raise RuntimeError("reached shim verification")
        return "synthetic"

    monkeypatch.setattr(installed.portable, "file_sha", stop_at_shim)
    with pytest.raises(RuntimeError, match="reached shim verification"):
        installed.provision(
            argparse.Namespace(
                destination=tmp_path / "fresh",
                v6_config=config,
                source_evidence_dir=evidence,
                shim=shim,
            )
        )


def test_missing_v6_deployment_manifest_is_reported_before_asset_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "v6.json"
    config.write_text("{}", encoding="utf-8")
    original = SimpleNamespace(protocol_profile="pending-future/v6", deployment_manifest=None)
    monkeypatch.setattr(installed.sys, "platform", "linux")
    monkeypatch.setattr(installed.NativeRuntimeConfig, "from_mapping", lambda _: original)

    def unexpected_asset_access(_: object) -> None:
        pytest.fail("asset verification must not be attempted without the deployment manifest")

    monkeypatch.setattr(installed, "verify_assets", unexpected_asset_access)
    with pytest.raises(ValueError, match="requires a deployment_manifest"):
        installed.provision(argparse.Namespace(destination=tmp_path / "fresh", v6_config=config))
