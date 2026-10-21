"""Replay episode starts must represent the same history as live control."""

import pytest
import torch

from tests.unit._composite_value_fixtures import _recurrent_value_model
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler
from trackmaniarl.models.composite import BatchLayout


@pytest.mark.parametrize("burn_in", [0, 2, 5])
def test_missing_history_matches_live_steps_and_preserves_gradients(burn_in: int) -> None:
    torch.manual_seed(7)
    model = _recurrent_value_model("gru")
    padding = [0, 1, 4, 7]
    observations = torch.randn(4, 8, 4, requires_grad=True)
    masks = torch.arange(8)[None, :] >= torch.tensor(padding)[:, None]
    actual = model.encode_masked_sequence(observations, BatchLayout.SEQUENCE, burn_in, masks=masks)
    for row, start in enumerate(padding):
        state = model.initial_policy_state(1, torch.device("cpu"))
        for position in range(start, 8):
            expected, state = model.policy_step(observations[row : row + 1, position], state)
            if position >= burn_in:
                torch.testing.assert_close(actual[row, position - burn_in], expected[0])
    actual[:, -1, 0].sum().backward()
    assert observations.grad is not None
    for row, start in enumerate(padding):
        assert observations.grad[row, : max(start, burn_in)].count_nonzero() == 0
        assert observations.grad[row, -1].abs().sum() > 0


def test_recurrent_history_rejects_internal_holes() -> None:
    model = _recurrent_value_model("gru")
    with pytest.raises(ValueError, match="missing prefix"):
        model.encode_masked_sequence(
            torch.randn(1, 4, 4),
            BatchLayout.SEQUENCE,
            0,
            masks=torch.tensor([[False, True, False, True]]),
        )


def test_episode_start_replay_can_update_a_recurrent_learner() -> None:
    learner = DiscreteValueLearner(_recurrent_value_model("gru"), burn_in=5)
    learner.setup({"seed": 7})
    store = InMemoryReplayStore()
    store.append(
        Transition(
            observation=torch.randn(4),
            next_observation=torch.randn(4),
            action=1,
            reward=1.0,
            terminated=False,
            truncated=True,
            episode_id="start",
            step=0,
        )
    )
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), allow_padded_history=True)
    batch = sampler.sample(store, BatchRequest(batch_size=1, sequence_length=8, n_step=3))
    before = learner.model.temporal.recurrent.weight_ih_l0.detach().clone()
    _, priorities = learner.update(batch)
    assert list(priorities.transition_ids) == [0]
    assert not torch.equal(before, learner.model.temporal.recurrent.weight_ih_l0)
