import pytest

from tests.unit.learning._algorithm_fixtures import DiscreteSacModel
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic


@pytest.mark.parametrize("actor_rate", [None, 9e-4])
def test_actor_rate_does_not_change_critic_or_temperature_rate(actor_rate: float | None) -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), learning_rate=3e-4, actor_learning_rate=actor_rate
    )
    learner.setup({"seed": 17})
    assert learner.actor_optimizer.param_groups[0]["lr"] == (
        3e-4 if actor_rate is None else actor_rate
    )
    assert learner.critic_optimizer.param_groups[0]["lr"] == 3e-4
    assert learner.alpha_optimizer is not None
    assert learner.alpha_optimizer.param_groups[0]["lr"] == 3e-4


@pytest.mark.parametrize("actor_rate", [0, -1, float("nan"), float("inf")])
def test_invalid_actor_learning_rate_is_rejected(actor_rate: float) -> None:
    with pytest.raises(ValueError, match="actor_learning_rate"):
        StableDiscreteSoftActorCritic(DiscreteSacModel(), actor_learning_rate=actor_rate)
