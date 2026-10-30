"""Meaningful policy action/RNG equivalence without claiming identical GEMM bits."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from tests.unit.trackmania.test_common_training import contract
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.contracts import RiskDistortion, RiskSpec
from trackmaniarl.trackmania.common_training import COMMON6_PROFILE, make_learner
from trackmaniarl.trackmania.common_training.batched_policy import BatchedCommonPolicy


def _policy() -> DiscreteValuePolicy:
    return make_learner(contract(observation_profile=COMMON6_PROFILE)).policy()


def _observations(size: int = 8) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(717)
    # Deliberately reverse insertion order: RNG scheduling belongs to actor IDs.
    return {
        actor: torch.randn(6, generator=generator) * (actor + 1) for actor in reversed(range(size))
    }


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("mode", [PolicyMode.ONLINE, PolicyMode.EVALUATION])
def test_actions_and_rng_preserve_serial_actor_order(epsilon: float, mode: PolicyMode) -> None:
    serial = _policy()
    serial.set_exploration_epsilon(epsilon)
    batched = deepcopy(serial)
    adapter = BatchedCommonPolicy(batched)
    observations = _observations()
    generator = torch.Generator().manual_seed(109)
    # Different starting RNG streams cover exploring and greedy mixed rounds.
    for _ in range(20):
        initial = generator.get_state()
        torch.set_rng_state(initial)
        expected = {actor: serial.act(observations[actor], mode) for actor in sorted(observations)}
        after_serial = torch.get_rng_state()
        torch.set_rng_state(initial)
        actual = adapter.act_many(observations, mode)
        assert actual == expected
        assert torch.equal(torch.get_rng_state(), after_serial)
        assert batched._state == serial._state == ()
        if mode is PolicyMode.EVALUATION or epsilon == 0:
            assert torch.equal(torch.get_rng_state(), initial)
        generator.set_state(after_serial)
        torch.rand((), generator=generator)


def test_single_actor_keeps_exact_q_diagnostics_and_reset_semantics() -> None:
    serial, observations = _policy(), _observations(1)
    batched = deepcopy(serial)
    adapter = BatchedCommonPolicy(batched)
    initial = torch.get_rng_state()
    expected = serial.act(observations[0], PolicyMode.EVALUATION)
    actual = adapter.act_many(observations, PolicyMode.EVALUATION)
    assert actual == {0: expected}
    assert serial.last_q_max == batched.last_q_max
    assert serial.last_q_margin == batched.last_q_margin
    assert torch.equal(initial, torch.get_rng_state())
    batched.reset_episode()
    assert adapter.act_many(observations, PolicyMode.EVALUATION) == actual
    assert adapter.serial_recheck_count == 0


@pytest.mark.parametrize("choices", [(5,), (1, 0, 77)])
def test_action_mask_preserves_rng_and_valid_actions(choices: tuple[int, ...]) -> None:
    serial = _policy()
    serial.policy_action_ids = choices
    serial.set_exploration_epsilon(1.0)
    batched = deepcopy(serial)
    adapter = BatchedCommonPolicy(batched)
    observations = _observations()
    initial = torch.get_rng_state()
    expected = {actor: serial.act(observations[actor]) for actor in sorted(observations)}
    final = torch.get_rng_state()
    torch.set_rng_state(initial)
    assert adapter.act_many(observations) == expected
    assert torch.equal(final, torch.get_rng_state())
    assert all(action in choices for action in expected.values())


def test_tie_recheck_uses_serial_shape_and_keeps_first_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    with torch.no_grad():
        policy.model.head.advantage.weight.zero_()
        bias = policy.model.head.advantage.bias.reshape(32, 78)
        bias.fill_(-1)
        bias[:, :2] = 1
    policy.set_exploration_epsilon(0)
    adapter = BatchedCommonPolicy(policy)
    original = adapter._q_values
    shapes: list[int] = []

    def rounded(values: torch.Tensor, mode: PolicyMode) -> torch.Tensor:
        shapes.append(len(values))
        q = original(values, mode)
        if len(values) > 1:
            # Simulate a batch kernel reversing a near tie; serial chooses zero.
            q[:, 1] += 1e-6
        return q

    monkeypatch.setattr(adapter, "_q_values", rounded)
    observations = _observations(4)
    initial = torch.get_rng_state()
    assert adapter.act_many(observations) == dict.fromkeys(range(4), 0)
    assert shapes == [4, 1, 1, 1, 1]
    assert adapter.last_serial_rechecks == (0, 1, 2, 3)
    assert adapter.serial_recheck_count == 4
    assert torch.equal(initial, torch.get_rng_state())


def test_shared_weight_refresh_changes_actions_without_rebuilding_adapter() -> None:
    policy = _policy()
    adapter = BatchedCommonPolicy(policy)
    observations = _observations(4)
    with torch.no_grad():
        policy.model.head.advantage.weight.zero_()
        policy.model.head.advantage.bias.zero_()
    assert adapter.act_many(observations, PolicyMode.EVALUATION) == dict.fromkeys(range(4), 0)
    state = deepcopy(policy.export_state())
    state["head.advantage.bias"].reshape(32, 78)[:, 31] = 4
    policy.load_state(state)
    assert adapter.act_many(observations, PolicyMode.EVALUATION) == dict.fromkeys(range(4), 31)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_round_rejected_before_any_action_or_rng(bad: float) -> None:
    adapter = BatchedCommonPolicy(_policy())
    observations = _observations()
    observations[7][0] = bad
    initial = torch.get_rng_state()
    with pytest.raises(ValueError, match="finite common6"):
        adapter.act_many(observations)
    assert torch.equal(initial, torch.get_rng_state())
    assert adapter.batch_count == 0


def test_nonfinite_model_output_rejected_before_rng() -> None:
    policy = _policy()
    adapter = BatchedCommonPolicy(policy)
    with torch.no_grad():
        policy.model.head.advantage.bias[0] = float("nan")
    initial = torch.get_rng_state()
    with pytest.raises(ValueError, match="finite batched"):
        adapter.act_many(_observations())
    assert torch.equal(initial, torch.get_rng_state())


@pytest.mark.parametrize("case", ["empty", "bool-id", "negative-id", "large-id", "rank", "dtype"])
def test_malformed_round_rejected_without_rng(case: str) -> None:
    adapter = BatchedCommonPolicy(_policy())
    row = torch.zeros(6)
    values: Any = {
        "empty": {},
        "bool-id": {True: row},
        "negative-id": {-1: row},
        "large-id": {8: row},
        "rank": {0: row[None]},
        "dtype": {0: row.double()},
    }[case]
    initial = torch.get_rng_state()
    with pytest.raises(ValueError, match=r"actor IDs|observation tensors"):
        adapter.act_many(values)
    assert torch.equal(initial, torch.get_rng_state())


@pytest.mark.parametrize(
    "case", ["selector", "epsilon", "risk", "mask", "state", "profile", "training", "dtype"]
)
def test_unsupported_profile_or_controls_rejected(case: str) -> None:
    policy = _policy()
    if case == "selector":
        policy.action_selector = object()
    elif case == "epsilon":
        policy.exploration_epsilon = float("nan")
    elif case == "risk":
        policy.online_risk = RiskSpec(RiskDistortion.UPPER_CVAR, 0.5)
    elif case == "mask":
        policy.policy_action_ids = (1, 1)
    elif case == "state":
        policy._state = (torch.zeros(1),)
    elif case == "profile":
        policy.model.encoder.input_dim = 8
    elif case == "training":
        policy.model.train()
    elif case == "dtype":
        policy.model.double()
    with pytest.raises(
        ValueError, match=r"pinned common6|stateless identity|finite exploration|unique"
    ):
        BatchedCommonPolicy(policy)


def test_controls_revalidated_after_construction() -> None:
    policy = _policy()
    adapter = BatchedCommonPolicy(policy)
    policy.action_selector = object()
    with pytest.raises(ValueError, match="without selector"):
        adapter.act_many(_observations())
