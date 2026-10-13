"""Offline guard transactions and recovery semantics; no optimizer or game steps."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pilot_tools.ppo_guarded import TransactionalPPOUpdater
from pilot_tools.ppo_recovery import StalledIncidentMemory
from pilot_tools.ppo_regression_gate import regression_review
from trackmaniarl.algorithms.ppo_update import _EpochResult


class _State:
    def __init__(self) -> None:
        self.value = {"step": torch.tensor(7.0)}

    def state_dict(self) -> dict:
        return self.value

    def load_state_dict(self, state: dict) -> None:
        self.value = copy.deepcopy(state)

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none


def _owner() -> SimpleNamespace:
    return SimpleNamespace(model=torch.nn.Linear(1, 1), optimizer=_State(), scaler=_State(),
                           device=torch.device("cpu"), target_kl=.02, update_epochs=3,
                           minibatch_size=2, autocast=nullcontext)


class _SimulatedEpochs(TransactionalPPOUpdater):
    def __init__(self, owner: SimpleNamespace, kls: list[float], *, fail: bool = False) -> None:
        super().__init__(owner)
        self.kls = iter(kls)
        self.fail = fail

    def _full_kl(self, batch: object) -> float:
        return next(self.kls)

    def _optimize_epoch(self, batch: object) -> _EpochResult:
        # Manual sentinel mutation tests rollback without taking optimizer steps.
        with torch.no_grad():
            self.owner.model.weight.add_(1)
        self.owner.optimizer.value["step"] += 1
        self.owner.scaler.value["step"] += 1
        torch.rand(1)
        self.epochs += 1
        self.norms.append((1., 2., 3.))
        if self.fail:
            raise RuntimeError("simulated failure")
        return _EpochResult(torch.ones(5), 2, .001)


def test_post_epoch_guard_reverts_only_rejected_epoch_and_all_training_state() -> None:
    owner = _owner()
    weight = owner.model.weight.detach().clone()
    guard = _SimulatedEpochs(owner, [0, .01, .03])
    rng = torch.get_rng_state()
    torch.rand(1)
    accepted_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    outcome = guard._optimize_epochs(None)
    assert torch.equal(owner.model.weight, weight+1)
    assert owner.optimizer.value["step"] == 8
    assert owner.scaler.value["step"] == 8
    assert torch.equal(torch.get_rng_state(), accepted_rng)
    assert outcome.updates == 2
    assert outcome.stopped_early
    assert guard.epochs == 1
    assert guard.attempted_epochs == 2
    assert guard.rejected_epochs == 1
    assert len(guard.norms) == 1


@pytest.mark.parametrize("kl", [.021, float("inf"), float("nan")])
def test_reject_first_epoch_restores_state_and_produces_finite_zero_step_metrics(kl: float) -> None:
    owner = _owner()
    before = copy.deepcopy(owner.model.state_dict())
    guard = _SimulatedEpochs(owner, [0, kl])
    outcome = guard._optimize_epochs(None)
    assert all(torch.equal(before[k], v) for k, v in owner.model.state_dict().items())
    assert owner.optimizer.value["step"] == 7
    assert owner.scaler.value["step"] == 7
    assert outcome.updates == 0
    assert guard.epochs == 0
    assert all(np.isfinite(v) for v in guard._metrics(outcome).values())


def test_exception_rolls_back_before_propagating() -> None:
    owner = _owner()
    before = copy.deepcopy(owner.model.state_dict())
    rng = torch.get_rng_state()
    guard = _SimulatedEpochs(owner, [0], fail=True)
    with pytest.raises(RuntimeError, match="simulated"):
        guard._optimize_epochs(None)
    assert all(torch.equal(before[k], v) for k, v in owner.model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert owner.optimizer.value["step"] == 7


def test_invalid_behavior_batch_is_rejected_before_any_epoch() -> None:
    guard = _SimulatedEpochs(_owner(), [.03])
    with pytest.raises(ValueError, match="stale/invalid"):
        guard._optimize_epochs(None)
    assert guard.epochs == 0


def test_full_batch_kl_matches_weighted_empirical_formula_without_sampling() -> None:
    owner = _owner()

    class Actor:
        @staticmethod
        def _distribution(observations: torch.Tensor) -> torch.distributions.Normal:
            return torch.distributions.Normal(observations, torch.ones_like(observations))

        @staticmethod
        def _log_probability(distribution: object, actions: torch.Tensor) -> torch.Tensor:
            return distribution.log_prob(actions).sum(-1)

    owner.model.actor = Actor()
    observations = torch.tensor([[.1], [.2], [.3], [.4], [.5]])
    actions = torch.zeros_like(observations)
    old = torch.distributions.Normal(torch.zeros_like(observations),
                                     torch.ones_like(observations)).log_prob(actions).sum(-1)
    batch = SimpleNamespace(observations=observations, latent_actions=actions,
                            old_log_probabilities=old, returns=torch.zeros(5))
    rng = torch.get_rng_state()
    measured = TransactionalPPOUpdater(owner)._full_kl(batch)
    log_ratio = -.5*observations[:, 0].square()
    expected = (log_ratio.exp()-1-log_ratio).mean().item()
    assert measured == pytest.approx(expected, abs=1e-8)
    assert torch.equal(rng, torch.get_rng_state())


def test_stationary_edge_without_prior_incident_is_not_a_recovery_trigger() -> None:
    memory = StalledIncidentMemory()
    raw = np.zeros(8, dtype=np.float32)
    assert memory.apply(raw, 0, .05) is raw


def test_incident_persists_only_while_stalled_at_edge() -> None:
    memory = StalledIncidentMemory()
    incident = np.zeros(8, dtype=np.float32)
    incident[6:] = [1.0, .23]
    memory.apply(incident, .5, .06)
    faded = np.zeros(8, dtype=np.float32)
    retained = memory.apply(faded, .2, .06)
    assert np.array_equal(retained[6:], incident[6:])
    assert np.array_equal(faded, np.zeros(8))  # Never mutate stored features.
    assert memory.apply(faded, 3, .06) is faded
    assert memory.apply(faded, .2, .06) is faded


def test_leaving_edge_clears_memory_even_without_speed() -> None:
    memory = StalledIncidentMemory(memory=.23, deficit=1.)
    raw = np.zeros(8, dtype=np.float32)
    assert memory.apply(raw, 0, .5) is raw
    assert memory.memory == 0


def _suite(*, finishes: int = 10, time: float = 52., maximum: float = 50.) -> dict:
    return {"status": "complete", "expected_trials": 10, "trials": [
        {"trial_index": i, "map_uid": "oqIJ5rQDRrNwLPTh9H2p_W4tLof",
         "finished": i < finishes, "finish_time_s": time if i < finishes else None,
         "steps": 100, "step_race_time_ms_max": maximum,
         "telemetry_error": None, "controller_error": None,
         "step_race_time_measurement_count": 100, "step_race_time_measurements_valid": True,
         "termination_reason": "finished" if i < finishes else "no_progress"}
        for i in range(10)]}


def test_regression_review_keeps_quality_separate_from_timing() -> None:
    result = regression_review(_suite(), _suite(maximum=120))
    assert result["candidate"]["quality_gate_passed"] is True
    assert result["candidate"]["timing_gate_passed"] is False
    assert result["strict_gate_passed"] is False
    assert result["review_required"] is False
    assert result["automatic_checkpoint_selection"] is False


def test_finish_and_speed_regression_are_flagged_without_replacing_final_result() -> None:
    result = regression_review(_suite(), _suite(finishes=7, time=63))
    assert result["review_required"]
    assert result["reasons"] == ["fewer finishes", "median slowdown >10%"]
    assert result["candidate"]["finishes"] == 7
    assert result["automatic_full_training"] is False


def test_speed_regression_is_detected_even_when_all_trials_finish() -> None:
    result = regression_review(_suite(), _suite(time=63))
    assert result["candidate"]["quality_gate_passed"]
    assert result["reasons"] == ["median slowdown >10%"]


def test_partial_or_corrupt_suite_is_not_a_valid_quality_comparison() -> None:
    suite = _suite()
    suite["trials"].pop()
    result = regression_review(_suite(), suite)
    assert result["reasons"] == ["invalid measurement"]
    assert result["strict_gate_passed"] is False
