from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.actors import CategoricalActor


def _learner() -> StableDiscreteSoftActorCritic:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), entropy_penalty_reference="behavior"
    )
    learner.setup({"seed": 17})
    return learner


def test_policy_records_exact_categorical_entropy_without_sampling_in_evaluation() -> None:
    policy = _learner().policy()
    observation = torch.zeros(4)
    before = torch.get_rng_state().clone()
    action, info = policy.act_with_info(observation, PolicyMode.EVALUATION)
    p = policy.actor.probabilities(observation.unsqueeze(0)).detach()
    expected = -(p * p.log()).sum()

    assert action == int(p.argmax())
    assert info["_trackmaniarl_behavior_entropy"] == pytest.approx(float(expected))
    assert torch.equal(before, torch.get_rng_state())


def test_entropy_penalty_uses_stored_behavior_and_has_the_expected_gradient() -> None:
    learner = _learner()
    entropy = torch.tensor([0.25, 0.75], requires_grad=True)
    old = torch.tensor([0.5, 1.0], requires_grad=True)
    batch = replace(_batch(BatchKind.DISCRETE), metadata={"behavior_entropies": old})
    prepared = discrete_batch(learner._batch(batch))
    with patch.object(learner.target_model.actor, "probabilities", side_effect=AssertionError):
        penalty = learner._entropy_penalty(prepared, entropy)
        penalty.backward()

    assert float(penalty.detach()) == pytest.approx(0.0625)
    assert torch.allclose(entropy.grad, torch.tensor([-0.25, -0.25]))
    assert old.grad is None


def test_behavior_entropy_survives_replay_checkpoint_and_matches_sample_ids() -> None:
    from trackmaniarl.core.builtins import IdentityFeaturePipeline
    from trackmaniarl.core.data import BatchRequest, Transition
    from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
    from trackmaniarl.distributed.coordinator_ingest import replay_info_for_transition

    store = InMemoryReplayStore(capacity=8)
    for i in range(8):
        store.append(
            Transition(
                i,
                0,
                1.0,
                i + 1,
                True,
                False,
                info=replay_info_for_transition(
                    {"_trackmaniarl_behavior_entropy": i / 8, "unneeded_diagnostic": 123}
                ),
                episode_id=str(i),
                step=0,
            )
        )
    restored = InMemoryReplayStore(capacity=8)
    restored.load_state_dict(store.state_dict())
    batch = UniformSampler(IdentityFeaturePipeline(), seed=17).sample(restored, BatchRequest(4))

    assert torch.equal(batch.metadata["behavior_entropies"], torch.tensor(batch.transition_ids) / 8)


@pytest.mark.parametrize("actor_objective", ["sac", "soft_q_forward_kl"])
def test_discrete_sac_learns_the_known_optimal_terminal_action(actor_objective: str) -> None:
    torch.manual_seed(17)
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        entropy_penalty_reference="behavior",
        target_entropy=0.8,
        learning_rate=0.01,
        target_tau=0.2,
        actor_objective=actor_objective,
    )
    learner.setup({"seed": 17})
    observation = torch.zeros(12, 4)
    actions = torch.arange(12) % 3
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=observation,
        next_observations=observation,
        actions=actions,
        rewards=(actions == 1).float(),
        bootstrap_discounts=torch.zeros(12),
        terminated=torch.ones(12, dtype=torch.bool),
        truncated=torch.zeros(12, dtype=torch.bool),
        transition_ids=list(range(12)),
    )
    before = float(learner.model.actor.probabilities(observation)[0, 1].detach())
    for _ in range(80):
        p = learner.model.actor.probabilities(observation).detach()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(1)
        learner.update(replace(batch, metadata={"behavior_entropies": entropy}))
    probabilities = learner.model.actor.probabilities(observation).detach()

    assert int(probabilities[0].argmax()) == 1
    assert float(probabilities[0, 1]) > max(0.6, before)
    restored = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        entropy_penalty_reference="behavior",
        target_entropy=0.8,
        learning_rate=0.01,
        target_tau=0.2,
        actor_objective=actor_objective,
    )
    restored.setup({"seed": 17})
    restored.load_state_dict(learner.state_dict_for_policy(learner.policy().export_state()))
    assert torch.equal(restored.model.actor.probabilities(observation), probabilities)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "1", None])
def test_actor_transport_rejects_invalid_behavior_entropy(value: object) -> None:
    from trackmaniarl.core.data import Transition
    from trackmaniarl.distributed.coordinator_validation import _validate_wire_transition
    from trackmaniarl.distributed.protocol import transition_to_wire

    transition = Transition(
        0.0,
        1,
        1.0,
        1.0,
        True,
        False,
        info={"_trackmaniarl_behavior_entropy": value},
        episode_id="actor/session/0",
        step=0,
    )
    with pytest.raises((ValueError, TypeError), match="behavior entropy"):
        _validate_wire_transition(transition_to_wire(transition))


def _soft_values() -> torch.Tensor:
    # Two states: wait costs 0.1; advance reaches the next state then the goal.
    # Solve the optimal entropy-regularized Bellman equations independently.
    values = torch.zeros(2, dtype=torch.float64)
    for _ in range(300):
        q = torch.stack(
            [
                torch.tensor([-0.1, -0.1]) + 0.9 * values,
                torch.stack((-0.01 + 0.9 * values[1], values.new_tensor(1.0))),
            ],
            dim=1,
        )
        values = 0.05 * torch.logsumexp(q / 0.05, dim=1)
    return q.float()


@pytest.mark.parametrize("objective", ["sac", "soft_q_forward_kl"])
def test_sd_sac_learns_to_advance_to_a_delayed_goal(objective: str) -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 2, 2)
    model.q1, model.q2 = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    learner = StableDiscreteSoftActorCritic(
        model,
        execution=TorchExecutionConfig(device="cpu", torch_threads=1),
        learning_rate=0.03,
        target_tau=0.1,
        entropy_coefficient=0.05,
        learn_entropy_coefficient=False,
        entropy_penalty_coefficient=0,
        actor_objective=objective,
    )
    learner.setup({"seed": 17})
    observations = torch.eye(2).repeat_interleave(2, dim=0)
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=observations,
        actions=torch.tensor([0, 1, 0, 1]),
        rewards=torch.tensor([-0.1, -0.01, -0.1, 1.0]),
        next_observations=torch.eye(2)[torch.tensor([0, 1, 1, 0])],
        terminated=torch.tensor([False, False, False, True]),
        # A collection time limit must still bootstrap from the final observation.
        truncated=torch.tensor([True, False, False, False]),
        bootstrap_discounts=torch.tensor([0.9, 0.9, 0.9, 0.0]),
        transition_ids=list(range(4)),
    )
    for _ in range(500):
        learner.update(batch)

    oracle = _soft_values()
    for critic in (model.q1, model.q2):
        assert torch.allclose(critic(torch.eye(2)), oracle, atol=0.015)
    policy = learner.policy()
    assert [policy.act(state, PolicyMode.EVALUATION) for state in torch.eye(2)] == [1, 1]
    # Wait has a bootstrapped value; the goal action has only its known reward.
    assert float(model.q1(torch.eye(2))[0, 0].detach()) > 0.6
    assert float(model.q1(torch.eye(2))[1, 1].detach()) == pytest.approx(1.0, abs=0.01)
