"""The actor step must not alter which critic parameters are trainable."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.unit.learning._algorithm_fixtures import BatchKind, ContinuousModel, RedqModel, _batch
from trackmaniarl.algorithms import RandomizedEnsembleSAC, SoftActorCritic, TruncatedQuantileCritic
from trackmaniarl.algorithms.sac_support import continuous_batch, frozen_modules


def _learner(kind: str) -> Any:
    if kind == "sac":
        return SoftActorCritic(ContinuousModel())
    if kind == "redq":
        return RandomizedEnsembleSAC(RedqModel(), policy_update_interval=1)
    return TruncatedQuantileCritic(ContinuousModel(quantiles=5))


@pytest.mark.parametrize("kind", ["sac", "redq", "tqc"])
@pytest.mark.parametrize("fail_actor", [False, True])
def test_actor_step_restores_critic_parameter_flags(
    kind: str, monkeypatch: pytest.MonkeyPatch, *, fail_actor: bool
) -> None:
    learner = _learner(kind)
    learner.setup({"seed": 0})
    critics = learner.model.critics if kind == "redq" else (learner.model.q1, learner.model.q2)
    parameters = [parameter for critic in critics for parameter in critic.parameters()]
    parameters[0].requires_grad_(False)
    expected = [parameter.requires_grad for parameter in parameters]
    frozen_before = parameters[0].detach().clone()

    if fail_actor:

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("actor failure")

        monkeypatch.setattr(learner.model.actor, "forward", fail)
        prepared = continuous_batch(learner._batch(_batch(BatchKind.CONTINUOUS)))
        args = (prepared,) if kind == "redq" else (prepared, torch.tensor(0.2))
        with pytest.raises(RuntimeError, match="actor failure"):
            learner._actor_step(*args)
    else:
        learner.update(_batch(BatchKind.CONTINUOUS))
        learner.update(_batch(BatchKind.CONTINUOUS))
        assert torch.equal(parameters[0], frozen_before)

    assert [parameter.requires_grad for parameter in parameters] == expected


def test_frozen_modules_restores_shared_parameters_on_failure() -> None:
    module = torch.nn.Linear(2, 2)
    module.bias.requires_grad_(False)

    def fail_with_frozen_critics() -> None:
        with frozen_modules([module, module]):
            assert not module.weight.requires_grad
            raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        fail_with_frozen_critics()
    assert module.weight.requires_grad
    assert not module.bias.requires_grad
