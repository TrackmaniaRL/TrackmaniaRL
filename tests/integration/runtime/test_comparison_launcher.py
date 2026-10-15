"""Launcher guards must stop before opening a live game controller."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.tmrl_test_comparison import launch_checks, preflight
from experiments.tmrl_test_comparison.generate import configuration
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.spec import RunSpec


def test_preflight_refuses_existing_run_before_creating_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "artifacts" / "existing"
    existing.mkdir(parents=True)
    spec = SimpleNamespace(artifacts_dir=tmp_path / "artifacts", run_id="existing")
    monkeypatch.setattr(RunSpec, "from_yaml", lambda _: spec)

    def forbidden_controller(**kwargs: object) -> None:
        raise AssertionError("Controller created before existing-run guard")

    monkeypatch.setattr(preflight, "GamepadController", forbidden_controller)
    with pytest.raises(RuntimeError, match="Run already exists"):
        preflight.run_preflight(tmp_path / "config.yaml", require_new_run=True)


@pytest.fixture
def launch_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> launch_checks.LaunchRun:
    spec = RunSpec.model_validate(configuration("iqn", 17, "full"))
    directory = tmp_path / spec.run_id
    directory.mkdir()
    monkeypatch.setattr(launch_checks, "run_fingerprint", lambda *_: "test-fingerprint")
    return launch_checks.LaunchRun(tmp_path / "run.yaml", spec, directory)


def _saved_checkpoint(run: launch_checks.LaunchRun, family: str) -> tuple[Path, dict]:
    distributed = family == "distributed-update"
    updates = 509500 if distributed else 2048000 // run.spec.training.sequence_length
    checkpoint = run.directory / "checkpoints" / f"{family}-{updates:08d}.pt"
    state = {
        "run_fingerprint": "test-fingerprint",
        "learner": {"model": torch.tensor([1.0]), "processed_transitions": 2048000},
        "distributed" if distributed else "counters": {
            "transitions": 2048000,
            "updates": updates,
            "update_credit": 0.0,
            "fractional_updates": 0.0,
        },
    }
    TorchCheckpointCodec().save(state, checkpoint)
    events = [
        {"event": event, "payload": {"path": str(checkpoint)}}
        for event in ("train/checkpoint", "train/checkpoint_completed")
    ]
    (run.directory / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    return checkpoint, state


@pytest.mark.parametrize("fault", ["fractional_credit", "missing_update", "unprocessed"])
def test_final_checkpoint_refuses_collected_but_unlearned_ppo_rollout(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    ppo = RunSpec.model_validate(configuration("ppo", 17, "full"))
    run = launch_checks.LaunchRun(launch_run.config, ppo, launch_run.directory)
    checkpoint, state = _saved_checkpoint(run, "update")
    if fault == "fractional_credit":
        state["counters"]["fractional_updates"] = 1.0
    elif fault == "missing_update":
        # Use a matching saved filename so this exercises the rollout budget check.
        state["counters"]["updates"] = 999
        new_path = checkpoint.with_name("update-00000999.pt")
        events = (run.directory / "events.jsonl").read_text(encoding="utf-8")
        (run.directory / "events.jsonl").write_text(
            events.replace(checkpoint.name, new_path.name), encoding="utf-8"
        )
        checkpoint = new_path
    else:
        state["learner"]["processed_transitions"] = 2045952
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match="incomplete rollout"):
        launch_checks.final_checkpoint(run)


@pytest.mark.parametrize("family", ["distributed-update", "update"])
def test_final_checkpoint_accepts_completed_async_and_ppo(
    launch_run: launch_checks.LaunchRun, family: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, family)
    assert launch_checks.final_checkpoint(launch_run) == checkpoint


@pytest.mark.parametrize(
    "change",
    [
        ("transitions", 145408, "completed training budget"),
        ("updates", 500000, "completed training budget"),
        ("update_credit", 1.0, "undrained"),
        ("update_credit", float("nan"), "undrained"),
    ],
)
def test_final_checkpoint_refuses_incomplete_training(
    launch_run: launch_checks.LaunchRun, change: tuple[str, float, str]
) -> None:
    field, value, error = change
    checkpoint, state = _saved_checkpoint(launch_run, "distributed-update")
    state["distributed"][field] = value
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match=error):
        launch_checks.final_checkpoint(launch_run)


def test_final_checkpoint_does_not_fall_back_to_older_success(
    launch_run: launch_checks.LaunchRun,
) -> None:
    _saved_checkpoint(launch_run, "distributed-update")
    pending = launch_run.directory / "checkpoints/distributed-update-00509501.pt"
    with (launch_run.directory / "events.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            "\n" + json.dumps({"event": "train/checkpoint", "payload": {"path": str(pending)}})
        )
    with pytest.raises(RuntimeError, match="not successfully saved"):
        launch_checks.final_checkpoint(launch_run)


@pytest.mark.parametrize("fault", ["fingerprint", "nonfinite"])
def test_final_checkpoint_refuses_changed_code_or_broken_weights(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    checkpoint, state = _saved_checkpoint(launch_run, "update")
    if fault == "fingerprint":
        state["run_fingerprint"] = "different"
    else:
        state["learner"]["model"] = torch.tensor([float("nan")])
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match=r"fingerprint|non-finite"):
        launch_checks.final_checkpoint(launch_run)


def _measurement(run: launch_checks.LaunchRun, checkpoint: Path) -> tuple[Path, dict]:
    assert run.spec.evaluation is not None
    map_spec = run.spec.evaluation.maps[0]
    payload = {
        "status": "complete",
        "expected_trials": 30,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metrics": {"eval/finish_time_s": 55.0, "eval/median_finish_time_s": 55.0},
        "trials": [
            {
                "map_id": map_spec.id,
                "map_uid": map_spec.expected_map_uid,
                "trial_index": index,
                "finished": True,
                "steps": 100,
                "step_race_time_ms_max": 50,
                "step_race_time_measurement_count": 100,
                "step_race_time_measurements_valid": True,
                "telemetry_error": None,
                "controller_error": None,
                "termination_reason": "finished",
            }
            for index in range(30)
        ],
    }
    path = run.directory / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_complete_slow_measurement_is_reported_as_failed_gate_without_repeating(
    launch_run: launch_checks.LaunchRun, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "update")
    path, _ = _measurement(launch_run, checkpoint)
    launch_checks.report_measurement(launch_run, checkpoint, path)
    assert "configured benchmark gate failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault", ["cancelled", "missing_trial", "duplicate", "wrong_hash", "controller", "timing"]
)
def test_measurement_refuses_incomplete_or_invalid_results(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "update")
    path, payload = _measurement(launch_run, checkpoint)
    if fault == "cancelled":
        payload["status"] = "cancelled"
    elif fault == "missing_trial":
        payload["trials"].pop()
    elif fault == "duplicate":
        payload["trials"][1]["trial_index"] = 0
    elif fault == "wrong_hash":
        payload["checkpoint_sha256"] = "wrong"
    elif fault == "controller":
        payload["trials"][0]["controller_error"] = "failed"
    else:
        payload["trials"][0]["step_race_time_ms_max"] = 140
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        launch_checks.measurement_gate(launch_run, checkpoint, path)


@pytest.mark.parametrize("failure", ["performance", "unrelated"])
def test_benchmark_only_continues_after_exact_performance_failure(
    launch_run: launch_checks.LaunchRun, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "distributed-update")
    source, _ = _measurement(launch_run, checkpoint)
    target = launch_run.directory.with_name(launch_run.spec.run_id + "-benchmark-test")

    def fake_benchmark(_args: object) -> None:
        target.mkdir()
        artifact = target / "evaluation.json"
        shutil.copyfile(source, artifact)
        gate = launch_checks.measurement_gate(launch_run, checkpoint, artifact)
        error = launch_checks._benchmark_gate_failure(gate)
        raise RuntimeError(error if failure == "performance" else "Unrelated failure")

    monkeypatch.setattr(launch_checks, "_benchmark", fake_benchmark)
    stop = launch_run.directory / "STOP"
    if failure == "performance":
        launch_checks.benchmark_final(launch_run, checkpoint, stop)
    else:
        with pytest.raises(RuntimeError, match="Unrelated failure"):
            launch_checks.benchmark_final(launch_run, checkpoint, stop)
