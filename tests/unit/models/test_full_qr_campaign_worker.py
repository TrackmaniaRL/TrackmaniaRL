"""Reject altered worker requests before resolving a model or opening the SDK."""

import hashlib
import json
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from trackmaniarl.experiments.full_qr.campaign_worker import (
    WorkerProgress,
    WorkerSettings,
    publish_once,
)


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


def test_real_counter_progress_does_not_depend_on_metric_log_cadence(tmp_path: Path) -> None:
    counters = SimpleNamespace(transitions=10000, updates=0, episodes=3, update_credit=0.0)
    signal = Event()
    path = tmp_path / "progress.json"
    progress = WorkerProgress(SimpleNamespace(counters=counters), path, signal, interval=0.01)
    progress.start()
    try:
        deadline = monotonic() + 3
        while not path.exists() and monotonic() < deadline:
            sleep(0.01)
        assert json.loads(path.read_text())["updates"] == 0
        counters.updates = 7
        while json.loads(path.read_text())["updates"] != 7 and monotonic() < deadline:
            sleep(0.01)
        assert json.loads(path.read_text())["updates"] == 7
        assert not signal.is_set()
    finally:
        progress.close()
    assert not progress.thread.is_alive()
    assert not path.with_suffix(".stage").exists()
