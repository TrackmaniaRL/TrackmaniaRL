"""Pinned process requests fail before SDK/model setup."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from trackmaniarl.experiments.full_qr.campaign_entry import load_worker_request
from trackmaniarl.experiments.full_qr.campaign_worker import WorkerSettings

pytestmark = pytest.mark.skipif(os.name != "posix", reason="private Linux process requests")


def request(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    values = asdict(
        WorkerSettings(
            tmp_path,
            tmp_path / "run.yaml",
            "a" * 64,
            tmp_path / "token",
            tmp_path / "stop",
            "127.0.0.1:12345",
            "learner",
            Path("/mnt/f/AITrackmania/reports/unit-fixture/artifacts"),
        )
    )
    data = {key: str(value) for key, value in values.items()}
    path = tmp_path / "learner.json"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), data


def test_pinned_worker_request_preserves_exact_paths_and_identity(tmp_path: Path) -> None:
    path, pin, data = request(tmp_path)
    settings = load_worker_request(path, pin)
    assert settings.directory == tmp_path
    assert settings.worker_name == data["worker_name"]
    assert settings.config_sha256 == data["config_sha256"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["learner.json"]


def test_changed_request_and_extra_unreviewed_options_are_refused(tmp_path: Path) -> None:
    path, pin, data = request(tmp_path)
    data["allow_unqualified_events"] = "true"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="immutable worker request"):
        load_worker_request(path, pin)
    new_pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exact explicitly typed"):
        load_worker_request(path, new_pin)


def test_public_or_foreign_campaign_request_is_refused(tmp_path: Path) -> None:
    path, pin, data = request(tmp_path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private worker request"):
        load_worker_request(path, pin)
    path.chmod(0o600)
    data["directory"] = str(tmp_path / "other")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="belong to its campaign"):
        load_worker_request(path, hashlib.sha256(path.read_bytes()).hexdigest())
