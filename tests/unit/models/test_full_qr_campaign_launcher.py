"""Completion requires all workers, a full budget and the unchanged checkpoint."""

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from trackmaniarl.experiments.full_qr.campaign_launcher import (
    LaunchSettings,
    _read_ready,
    launch,
    verify_campaign_result,
)
from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal
from trackmaniarl.experiments.full_qr.campaign_worker import WorkerSettings


def evidence(tmp_path: Path) -> tuple[dict, dict, dict]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-only-final-checkpoint")
    counters = {"transitions": 10001, "updates": 1, "episodes": 1}
    result = {
        "schema": "tmrl.full-qr-learner-result/v1",
        "budget": {
            "stop_reason": "wall_clock_limit",
            "duration_seconds": 14400,
            "elapsed_seconds": 14400.01,
        },
        **counters,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **counters,
        },
    }
    return (
        result,
        {"remainingOwned": [], "cleanupErrors": [], "forced": []},
        {
            "learner": 0,
            "actor-000": 0,
            "actor-001": 0,
        },
    )


@pytest.mark.parametrize("actual_clock", [None, "monotonic", "monotonic_raw"])
def test_raw_completion_requires_the_declared_clock_domain(
    tmp_path: Path, actual_clock: str | None
) -> None:
    result, closure, exits = evidence(tmp_path)
    if actual_clock is not None:
        result["budget"]["clock_source"] = actual_clock
    if actual_clock == "monotonic_raw":
        verify_campaign_result(result, closure, exits, 2, clock_source="monotonic_raw")
    else:
        with pytest.raises(ValueError, match="worker completion"):
            verify_campaign_result(result, closure, exits, 2, clock_source="monotonic_raw")


def test_diagnostic_result_cannot_claim_qualification(tmp_path: Path) -> None:
    result, closure, exits = evidence(tmp_path)
    result["budget"]["duration_seconds"] = 1800
    result["budget"]["elapsed_seconds"] = 1800
    with pytest.raises(ValueError, match="worker completion"):
        verify_campaign_result(result, closure, exits, 2, duration_seconds=1800)
    result["campaign_mode"] = "diagnostic"
    verify_campaign_result(result, closure, exits, 2, duration_seconds=1800)


def test_diagnostic_requires_full_budget_unchanged_checkpoint_and_clean_closure(
    tmp_path: Path,
) -> None:
    result, closure, exits = evidence(tmp_path)
    result["campaign_mode"] = "diagnostic"
    result["budget"].update(duration_seconds=1800, elapsed_seconds=1800)
    for field, bad_value in (("elapsed_seconds", 1799.99), ("stop_reason", "storage_reserve")):
        altered = copy.deepcopy(result)
        altered["budget"][field] = bad_value
        with pytest.raises(ValueError, match="worker completion"):
            verify_campaign_result(altered, closure, exits, 2, duration_seconds=1800)
    with pytest.raises(ValueError, match="clean owned closure"):
        verify_campaign_result(
            result, {**closure, "remainingOwned": [{"pid": 1}]}, exits, 2, duration_seconds=1800
        )
    Path(result["checkpoint"]["path"]).write_bytes(b"fixture-only-FINAL-checkpoint")
    with pytest.raises(ValueError, match="changed after learner"):
        verify_campaign_result(result, closure, exits, 2, duration_seconds=1800)


@pytest.mark.parametrize("elapsed", [0, 14399.99, float("nan"), float("inf"), True])
def test_timer_reason_without_full_elapsed_budget_is_not_completion(
    tmp_path: Path, elapsed: float
) -> None:
    result, closure, exits = evidence(tmp_path)
    result["budget"]["elapsed_seconds"] = elapsed
    with pytest.raises(ValueError, match="four-hour"):
        verify_campaign_result(result, closure, exits, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("remainingOwned", [{"pid": 1}]),
        ("cleanupErrors", ["lost lease"]),
        ("forced", ["actor-001"]),
    ],
)
def test_survivor_cleanup_error_or_forced_actor_cannot_complete(
    tmp_path: Path, field: str, value: list
) -> None:
    result, closure, exits = evidence(tmp_path)
    closure[field] = value
    with pytest.raises(ValueError, match="clean owned closure"):
        verify_campaign_result(result, closure, exits, 2)


def test_missing_failed_or_foreign_actor_cannot_complete(tmp_path: Path) -> None:
    result, closure, exits = evidence(tmp_path)
    for change in (
        {"learner": 0, "actor-000": 0},
        {**exits, "actor-001": 1},
        {**exits, "foreign": 0},
    ):
        with pytest.raises(ValueError, match="worker completion"):
            verify_campaign_result(result, closure, change, 2)


def test_actual_checkpoint_change_after_learner_verification_is_detected(tmp_path: Path) -> None:
    result, closure, exits = evidence(tmp_path)
    verify_campaign_result(result, closure, exits, 2)
    path = Path(result["checkpoint"]["path"])
    path.write_bytes(b"fixture-only-FINAL-checkpoint")  # Same size, different actual bytes.
    with pytest.raises(ValueError, match="changed after learner"):
        verify_campaign_result(result, closure, exits, 2)


def test_checkpoint_counters_must_include_actual_updates_and_episodes(tmp_path: Path) -> None:
    result, closure, exits = evidence(tmp_path)
    for name in ("updates", "episodes", "transitions"):
        altered = copy.deepcopy(result)
        altered["checkpoint"][name] -= 1
        with pytest.raises(ValueError, match="checkpoint counters"):
            verify_campaign_result(altered, closure, exits, 2)


def test_ready_must_belong_to_current_learner_with_real_assigned_port(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    path.write_text(json.dumps({"pid": 2, "port": 32123, "duration_seconds": 14400}))
    assert _read_ready(path, {"worker": {"pid": 2}})["port"] == 32123
    with pytest.raises(ValueError, match="identity"):
        _read_ready(path, {"worker": {"pid": 3}})
    path.write_text(json.dumps({"pid": 2, "port": 0, "duration_seconds": 14400}))
    with pytest.raises(ValueError, match="bound port"):
        _read_ready(path, {"worker": {"pid": 2}})


@pytest.mark.skipif(os.name != "posix", reason="actual Linux pidfd/subreaper ownership")
def test_actual_invalid_learner_leaves_no_budget_actors_or_owned_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deliberately bypass ONLY parent preflight to exercise its actual cleanup
    # after the unchanged worker entry rejects invalid YAML before SDK setup.
    config = tmp_path / "run.yaml"
    config.write_text("invalid-yaml: [")
    token = tmp_path / "token"
    token.write_text("fixture-private-token-" * 3)
    stop = tmp_path / "stop"
    signal = MappedStopSignal(stop, create=True)
    signal.close()
    worker = WorkerSettings(
        tmp_path,
        config,
        hashlib.sha256(config.read_bytes()).hexdigest(),
        token,
        stop,
        "127.0.0.1:0",
        "learner",
        Path("/mnt/f/test-fixture/artifacts"),
    )
    settings = LaunchSettings(worker, Path(sys.executable), Path(sys.executable), 2, 10)
    monkeypatch.setattr(LaunchSettings, "preflight", lambda self: None)
    # The checkout may be mounted from Windows. Exercise the unchanged real
    # worker entry from a private Linux deployment, as the launcher requires.
    source_root = tmp_path / "source"
    shutil.copytree(
        Path(__file__).parents[3] / "trackmaniarl",
        source_root / "trackmaniarl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    with pytest.raises(RuntimeError, match="owned closure recorded"):
        launch(settings, source_root=source_root)
    report = json.loads((tmp_path / "parent-result.json").read_text())
    assert report["phase"] == "failed"
    assert report["closure"]["remainingOwned"] == report["closure"]["cleanupErrors"] == []
    assert report["closure"]["forced"] == []
    assert report["exits"] == {"learner": 1}
    assert not (tmp_path / "budget").exists()
    assert not (tmp_path / "actor-000.json").exists()
