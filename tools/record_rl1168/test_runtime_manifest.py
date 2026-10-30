"""An asset manifest cannot be replaced after native qualification."""

import hashlib
import json
from collections import namedtuple

import pytest

from tools.record_rl1168 import runtime


def test_manifest_replacement_between_qualification_and_training_fails(tmp_path, monkeypatch):
    disk_usage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: disk_usage(10**12, 0, 10**12))
    source = tmp_path / "source"
    server = source / "server" / "server-bin"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"native engine")
    manifest = tmp_path / "asset-manifest.json"
    original = {
        "runtime_root": str(source),
        "roles": {"server": "server/server-bin"},
        "assets": {
            "server/server-bin": {"sha256": hashlib.sha256(server.read_bytes()).hexdigest()}
        },
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")
    config = {
        "runtime_root": str(source),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    qualified = runtime.isolate_runtime(config, qualification)
    assert (
        qualified["manifest_sha256"]
        == hashlib.sha256((qualification / "assets.json").read_bytes()).hexdigest()
    )
    # Even a manifest with otherwise correct asset hashes must not be re-pinned.
    manifest.write_text(json.dumps({**original, "new_metadata": True}), encoding="utf-8")
    training = tmp_path / "training"
    training.mkdir()
    with pytest.raises(ValueError, match="source asset manifest SHA mismatch"):
        runtime.isolate_runtime(config, training)
    assert not (training / "runtime").exists()
    assert not (training / "assets.json").exists()


def test_missing_original_manifest_sha_fails_closed(tmp_path):
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source asset manifest SHA mismatch"):
        runtime.isolate_runtime(
            {"runtime_root": str(tmp_path), "manifest": str(manifest)}, tmp_path
        )
