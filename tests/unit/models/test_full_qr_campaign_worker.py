"""Reject altered worker requests before resolving a model or opening the SDK."""

import hashlib
from pathlib import Path

import pytest

from trackmaniarl.experiments.full_qr.campaign_worker import WorkerSettings, publish_once


def test_changed_run_spec_is_refused_before_yaml_or_model_setup(tmp_path: Path) -> None:
    config, token, stop = (tmp_path / name for name in ("run.yaml", "token", "stop"))
    config.write_bytes(b"original pinned bytes")
    pin = hashlib.sha256(config.read_bytes()).hexdigest()
    token.write_text("fixture-token-only-" * 4)
    stop.write_bytes(b"\0")
    settings = WorkerSettings(
        tmp_path,
        config,
        pin,
        token,
        stop,
        "127.0.0.1:12345",
        "learner",
        Path("/mnt/f/AITrackmania/reports/fixture/artifacts"),
    )
    config.write_bytes(b"altered invalid YAML: [")
    with pytest.raises(ValueError, match="immutable campaign"):
        settings.validate()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.yaml", "stop", "token"]


def test_worker_ready_is_published_once_and_never_replaces_prior_identity(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    publish_once(path, {"pid": 1, "scope": "fixture"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_once(path, {"pid": 2, "scope": "fixture"})
    assert path.read_bytes() == original
