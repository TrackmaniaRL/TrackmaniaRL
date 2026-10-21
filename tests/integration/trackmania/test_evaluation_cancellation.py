"""Cancellation preserves observations without inventing unstarted benchmark trials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.trackmania.test_evaluation_provenance import _ScenarioEnvironment
from tests.integration.trackmania.test_scaffold_evaluation import (
    _evaluation_suite,
    _EvaluationPolicy,
    _IdentityPipeline,
    _patch_geometry,
)
from trackmaniarl.commands.evaluation import _benchmark, _configure_evaluation_stop
from trackmaniarl.commands.parser import build_parser
from trackmaniarl.core.contracts import EvaluatorRuntimeRequest
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from trackmaniarl.trackmania.evaluation import EvaluationCancelledError, TrackmaniaEvaluator


class _StoppedEnvironment(_ScenarioEnvironment):
    def __init__(self, stop_file: Path, scenario: str) -> None:
        super().__init__("finished" if scenario == "between_trials" else "max_steps")
        self.stop_file, self.stop_scenario, self.closed = stop_file, scenario, False

    def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, Any]]:
        if self.stop_scenario == "paused_reset":
            self.stop_file.touch()
            raise EnvironmentPausedError("background")
        return super().reset(seed=seed)

    def step(self, action: Any) -> tuple[float, float, bool, bool, dict[str, Any]]:
        result = super().step(action)
        self.stop_file.touch()
        return result

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("scenario", ["paused_reset", "mid_trial", "between_trials"])
def test_stop_file_cancels_without_waiting_for_focus_and_preserves_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    _patch_geometry(monkeypatch)
    monkeypatch.setattr("trackmaniarl.trackmania.evaluation.sleep", lambda _: None)
    stop_file = tmp_path / "STOP"
    environment = _StoppedEnvironment(stop_file, scenario)
    evaluator = TrackmaniaEvaluator(
        EvaluatorRuntimeRequest(
            _evaluation_suite(tmp_path, trials_per_map=3),
            SimpleNamespace(create=lambda **kwargs: environment),
            _IdentityPipeline(),
            max_episode_steps=10,
            run_dir=tmp_path,
        )
    )
    _configure_evaluation_stop(SimpleNamespace(evaluator=evaluator), stop_file)

    with pytest.raises(EvaluationCancelledError, match="incomplete"):
        evaluator.evaluate(_EvaluationPolicy())

    artifact = json.loads((tmp_path / "evaluation.json").read_text())
    assert environment.closed
    assert artifact["status"] == "cancelled"
    assert artifact["expected_trials"] == 3
    if scenario == "paused_reset":
        assert artifact["trials"] == []
        assert artifact["metrics"] == {}
    else:
        assert len(artifact["trials"]) == 1
        trial = artifact["trials"][0]
        assert trial["steps"] == 1
        assert trial["finished"] == (scenario == "between_trials")
        assert trial["termination_reason"] == (
            "finished" if scenario == "between_trials" else "operator_interruption"
        )


def test_benchmark_stop_file_parser_and_existing_flag_rejection(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP"
    args = build_parser().parse_args(
        ["benchmark", "missing.yaml", "missing.pt", "--stop-file", str(stop_file)]
    )
    assert args.stop_file == stop_file
    stop_file.touch()
    with pytest.raises(ValueError, match="Stop file already exists"):
        _benchmark(args)


def test_unsupported_evaluator_does_not_silently_ignore_stop_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support"):
        _configure_evaluation_stop(SimpleNamespace(evaluator=object()), tmp_path / "STOP")


@pytest.mark.parametrize("replace_checkpoint", [False, True])
def test_evaluation_binds_checkpoint_bytes_and_rejects_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, replace_checkpoint: bool
) -> None:
    _patch_geometry(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"original checkpoint")
    environment = _ScenarioEnvironment("finished")
    evaluator = TrackmaniaEvaluator(
        EvaluatorRuntimeRequest(
            _evaluation_suite(tmp_path),
            SimpleNamespace(create=lambda **kwargs: environment),
            _IdentityPipeline(),
            run_dir=tmp_path,
        )
    )
    evaluator.set_checkpoint(checkpoint)
    if replace_checkpoint:
        checkpoint.write_bytes(b"different checkpoint")
        with pytest.raises(RuntimeError, match="Checkpoint changed"):
            evaluator.evaluate(_EvaluationPolicy())
        assert not (tmp_path / "evaluation.json").exists()
    else:
        evaluator.evaluate(_EvaluationPolicy())
        artifact = json.loads((tmp_path / "evaluation.json").read_text())
        assert artifact["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert artifact["status"] == "complete"
