"""SD-SAC entropy regularization must use the policy that collected replay."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.core.contracts import PolicyMode


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


@pytest.mark.parametrize(
    "stored",
    [None, torch.tensor([1.0]), torch.tensor([float("nan"), 1.0]), torch.tensor([-1.0, 1.0])],
)
def test_behavior_reference_rejects_missing_invalid_or_misaligned_metadata(stored: object) -> None:
    learner = _learner()
    batch = replace(_batch(BatchKind.DISCRETE), metadata={"behavior_entropies": stored})
    with pytest.raises(ValueError, match=r"entropy|entropies"):
        learner._entropy_penalty(discrete_batch(learner._batch(batch)), torch.ones(2))


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


def test_behavior_reference_supports_a_complete_finite_update() -> None:
    learner = _learner()
    batch = _batch(BatchKind.DISCRETE)
    batch = replace(batch, metadata={"behavior_entropies": torch.ones(len(batch.transition_ids))})
    metrics, priorities = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert priorities.transition_ids == batch.transition_ids


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
