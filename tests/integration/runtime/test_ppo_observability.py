"""Diagnostic adapters must not change PPO updates, RNG or evaluation actions."""

from __future__ import annotations

import copy
import gzip
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pilot_tools.ppo_diagnostics import InstrumentedPrefixPPO, PpoWandbTracker
from pilot_tools.ppo_prefix import CampaignPrefixPPO
from pilot_tools.ppo_trace import TraceEvaluator
from pilot_tools.train_trend import trend_evaluation
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import tree_map
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.models.sensor_actor_critic import SensorActorCriticModelFactory
from trackmaniarl.trackmania.baseline import TelemetryPpoModel
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator


@pytest.mark.parametrize("sensor_model", [False, True])
def test_diagnostic_update_preserves_weights_optimizer_metrics_and_rng(
    *, sensor_model: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = SensorActorCriticModelFactory(
        algorithm="ppo", config={"feature_dim": 192},
        encoder={"class_path": "experiments.tmrl_test_comparison.components:BatchedIncidentEncoder",
                 "kwargs": {"context_residual_scale": 0.02, "spatial_residual_scale": 0.02,
                            "recovery_residual_scale": 0.01}},
    ).build() if sensor_model else TelemetryPpoModel(hidden_dim=8)
    options = {"schedule_transitions": 8192000, "execution": {"device": "cpu"},
               "normalize_observations": False, "normalize_rewards": False,
               "update_epochs": 3, "minibatch_size": 4, "target_kl": 0.02}
    standard = CampaignPrefixPPO(copy.deepcopy(model), **options)
    diagnostic = InstrumentedPrefixPPO(copy.deepcopy(model), **options)
    for learner in (standard, diagnostic):
        learner.setup({"total_transitions": 204800, "seed": 17})
    observations = {
        key: torch.randn(1, 8, *space.shape) * 0.1
        for key, space in BoundaryGraphFeaturePipelineV5._build_observation_space().spaces.items()
    } if sensor_model else torch.randn(1, 8, 33)
    with torch.no_grad():
        actions, logs, latent = model.actor.sample_with_latent(observations)
        values = model.value(observations)
    terminated = torch.zeros(1, 8, dtype=torch.bool)
    terminated[0, -1] = True
    batch = TrainingBatch(
        data={}, observations=observations,
        next_observations=tree_map(lambda value: value + 0.01, observations),
        actions=actions, rewards=torch.linspace(-1, 1, 8).reshape(1, 8),
        terminated=terminated, truncated=torch.zeros_like(terminated),
        bootstrap_discounts=(~terminated).float() * 0.995, transition_ids=list(range(8)),
        metadata={"behavior_latent_actions": latent, "behavior_log_probabilities": logs,
                  "behavior_values": values},
    )
    rng = torch.get_rng_state()
    reference = standard.update(batch)
    next_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    measured = diagnostic.update(batch)
    assert torch.equal(next_rng, torch.get_rng_state())
    assert all(measured[key] == value for key, value in reference.items())
    assert all(torch.equal(value, diagnostic.model.state_dict()[key])
               for key, value in standard.model.state_dict().items())
    for parameter, state in standard.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            other = diagnostic.optimizer.state_dict()["state"][parameter][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert measured["state/epochs"] > 0
    assert measured["state/minibatches"] == measured["state/epochs"] * 2
    assert all(np.isfinite(value) for value in measured.values())


def test_wandb_adds_only_explicit_ppo_state_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trackmaniarl.observability.trackers.WandbTracker._project",
                        lambda *_: {"trainer/update": 3})
    tracker = PpoWandbTracker.__new__(PpoWandbTracker)
    result = tracker._project("train/update", {"state/approx_kl": 0.03,
                                               "state/private": "do not project"}, 3)
    assert result == {"trainer/update": 3, "learner/ppo_approx_kl": 0.03}


def test_trace_keeps_causal_order_and_writes_only_after_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = TraceEvaluator.__new__(TraceEvaluator)
    evaluator.run_dir = tmp_path
    evaluator.max_episode_steps = 3000
    evaluator.rows = [{"kind": "reset", "raw_observation": [1]}]
    evaluator.checkpoint = "checkpoint.pt"
    evaluator.checkpoint_sha256 = "sha"
    step = SimpleNamespace(observation=np.array([2]), action=np.array([0.8, 0.1, -0.2]),
                           action_duration_ms=1.0, terminated=True, truncated=False,
                           info={"position": [1, 2, 3], "race_time_ms": 50})
    monkeypatch.setattr(TrackmaniaEvaluator, "_take_step", lambda *_: step)
    assert evaluator._take_step(SimpleNamespace(observation=np.array([1]))) is step
    assert not list(tmp_path.iterdir())
    row = evaluator.rows[-1]
    assert row["raw_current"] == [1]
    assert row["raw_next"] == [2]
    assert row["after_action"]["race_time_ms"] == 50
    monkeypatch.setattr(TrackmaniaEvaluator, "_run_episode", lambda *_: None)
    request = SimpleNamespace(trial_index=0, map_spec=SimpleNamespace(expected_map_uid="uid"))
    evaluator._run_episode(request, None, None)
    payload = json.loads(gzip.decompress((tmp_path / "trial-00.json.gz").read_bytes()))
    assert len(payload["rows"]) == 2
    assert payload["checkpoint_sha256"] == "sha"
    assert evaluator.rows == []


def test_midpoint_hook_preserves_all_rng_and_closes_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    def evaluate(_policy: object) -> dict:
        random.random()
        np.random.rand()
        torch.rand(1)
        events.append("evaluate")
        return {"finish_rate": 0.3}

    monkeypatch.setattr("pilot_tools.train_trend.training_loop._close_rollout_environment",
                        lambda _: events.append("close"))
    monkeypatch.setattr("pilot_tools.train_trend.validate_midpoint",
                        lambda _: events.append("validate"))
    trainer = SimpleNamespace(
        run=SimpleNamespace(evaluator=SimpleNamespace(evaluate=evaluate),
                            learner=SimpleNamespace(policy=lambda: object())),
        _checkpoint_for_evaluation=lambda *_: events.append("checkpoint"),
        _log=lambda *_: events.append("log"),
    )
    session = SimpleNamespace(counters=SimpleNamespace(transitions=102400), checkpoints=[])
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    trend_evaluation(trainer, session, 0)
    assert events == ["checkpoint", "close", "evaluate", "validate", "log"]
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_midpoint_archive_survives_training_retention_and_final_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = TraceEvaluator.__new__(TraceEvaluator)
    evaluator.root = tmp_path
    evaluator.run_dir = tmp_path

    def bind(self: TraceEvaluator, path: Path) -> None:
        self.checkpoint = str(path)

    monkeypatch.setattr(TrackmaniaEvaluator, "set_checkpoint", bind)
    midpoint = tmp_path / "checkpoints/update-00000050.pt"
    midpoint.parent.mkdir()
    midpoint.write_bytes(b"midpoint")
    evaluator.set_checkpoint(midpoint)
    midpoint.unlink()
    assert Path(evaluator.checkpoint).read_bytes() == b"midpoint"
    assert not (tmp_path / "evaluation.json").exists()
    final = tmp_path / "checkpoints/update-00000100.pt"
    final.write_bytes(b"final")
    evaluator.set_checkpoint(final)

    def write(self: TraceEvaluator, *_args: object, **_kwargs: object) -> None:
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "evaluation.json").write_text('{"status":"complete"}')

    monkeypatch.setattr(TrackmaniaEvaluator, "_write_artifact", write)
    evaluator._write_artifact([], {})
    assert json.loads((tmp_path / "evaluation.json").read_text())["status"] == "complete"
    assert evaluator.run_dir.name == "update-00000100"
