"""Portable external-file fixtures, not game assets or runtime qualification."""

import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from trackmaniarl.trackmania.native_headless import assets
from trackmaniarl.trackmania.native_headless.config import (
    ASSET_SCHEMA,
    NATIVE_PROFILE,
    NativeConfigurationError,
    NativeRuntimeConfig,
)


@contextmanager
def raises(kind: type[Exception]) -> Iterator[None]:
    try:
        yield
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


class TestAssets(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.settings = NativeRuntimeConfig(
            runtime_root=self.root,
            trace_dir=self.root / "traces",
            work_dir=self.root / "work",
            asset_manifest=self.root / "assets.json",
        )
        self.roles = {role: role + ".fixture" for role in assets.PINNED_ROLE_SHA256}
        records = {}
        hashes = {}
        for role, name in self.roles.items():
            data = ("unit fixture, never executed: " + role).encode()
            (self.root / name).write_bytes(data)
            hashes[role] = sha256(data).hexdigest()
            records[name] = {"sha256": hashes[role], "bytes": len(data)}
        self.manifest = {
            "schema": ASSET_SCHEMA,
            "profile": NATIVE_PROFILE,
            "runtime_root": str(self.root),
            "roles": self.roles,
            "assets": records,
        }
        # Only unit fixtures replace the approved build identities; no production override exists.
        pins = patch.object(assets, "PINNED_ROLE_SHA256", hashes)
        pins.start()
        self.addCleanup(pins.stop)
        # These tests isolate asset bytes and paths. The deployment verifier
        # has its own unmocked rejection tests using explicit synthetic inputs.
        deployment = patch(
            "trackmaniarl.trackmania.native_headless._runtime.portable.expected_role_hashes",
            return_value=hashes,
        )
        deployment.start()
        self.addCleanup(deployment.stop)
        self.write_manifest()

    def write_manifest(self) -> None:
        self.settings.asset_manifest.write_text(json.dumps(self.manifest))

    def test_explicit_config_constructs_without_file_reads(self) -> None:
        with patch.object(Path, "read_bytes", side_effect=AssertionError("construction I/O")):
            parsed = NativeRuntimeConfig.from_mapping(
                {
                    "runtime_root": str(self.root),
                    "trace_dir": str(self.settings.trace_dir),
                    "work_dir": str(self.settings.work_dir),
                    "asset_manifest": str(self.settings.asset_manifest),
                }
            )
        assert parsed == self.settings
        assert not self.settings.trace_dir.exists()

    def test_all_roles_verified_and_mutation_not_hidden_by_cache(self) -> None:
        verified = assets.verify_assets(self.settings)
        assert set(verified.paths) == set(self.roles)
        assert (
            verified.manifest_sha256
            == sha256(self.settings.asset_manifest.read_bytes()).hexdigest()
        )
        (self.root / self.roles["server"]).write_bytes(b"changed fixture")
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)

    def test_size_pin_and_known_profile_identity_rejected(self) -> None:
        record = self.manifest["assets"][self.roles["server"]]
        record["bytes"] += 1
        self.write_manifest()
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)
        record["bytes"] -= 1
        record["sha256"] = "0" * 64
        self.write_manifest()
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)

    def test_duplicate_json_keys_and_missing_roles_rejected(self) -> None:
        self.settings.asset_manifest.write_text('{"schema":"a","schema":"b"}')
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)
        self.roles.pop("shim")
        self.write_manifest()
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)

    def test_root_and_relative_path_boundaries(self) -> None:
        self.manifest["runtime_root"] = "unrelated-root"
        self.write_manifest()
        with raises(assets.NativeAssetError):
            assets.verify_assets(self.settings)
        self.manifest["runtime_root"] = str(self.root)
        for invalid in ("../server", "/server", "a/../server", "a//server", "C:\\server"):
            self.roles["server"] = invalid
            self.write_manifest()
            with raises(assets.NativeAssetError):
                assets.verify_assets(self.settings)

    def test_configuration_caps_are_value_only(self) -> None:
        for timeout in (0, 241, float("nan"), True):
            with raises(NativeConfigurationError):
                replace(self.settings, timeout_s=timeout)
        for duration in (0, 1, 47780, True):
            with raises(NativeConfigurationError):
                replace(self.settings, duration_limit=duration)
        with raises(NativeConfigurationError):
            replace(self.settings, work_dir=self.settings.trace_dir)
