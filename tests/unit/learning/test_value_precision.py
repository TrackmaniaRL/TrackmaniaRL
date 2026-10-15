from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import pytest
import torch

import trackmaniarl.algorithms.value_based.updater as updater
from tests.unit.experiments.test_graph_iqn import _graph_batch
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.targets import BootstrapInputs, bootstrap_target
from trackmaniarl.experiments.graph_iqn import TrackGnnSimbaEncoder
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.heads import ImplicitQuantileHead, ImplicitQuantileHeadConfig
from trackmaniarl.models.strategies import RandomQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore


@contextmanager
def _bfloat16_autocast() -> Iterator[None]:
    # CPU only allows float32 through the execution policy; this forces what a
    # GPU's "auto" precision picks, to check where reduced precision reaches.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        yield


def _bootstrap_values(
    monkeypatch: pytest.MonkeyPatch, autocast: AbstractContextManager[None] | None
) -> torch.Tensor:
    torch.manual_seed(0)
    # The generated Trackmania configuration's dueling IQN head, with values near
    # 80 as after long training at gamma 0.995 (a bfloat16 step there is 0.5).
    model = CompositeValueModel(
        CompositeModules(
            TrackGnnSimbaEncoder(),
            IdentityTemporalCore(192),
            ImplicitQuantileHead(ImplicitQuantileHeadConfig(192, 78, 64, dueling=True)),
            RandomQuantileStrategy(64, 64, 32),
        )
    )
    with torch.no_grad():
        model.head.value.bias.fill_(80.0)
    learner = DiscreteValueLearner(model, execution={"device": "cpu"})
    learner.setup({"seed": 5})
    if autocast is not None:
        monkeypatch.setattr(learner, "autocast", lambda: autocast)
    recorded: list[torch.Tensor] = []

    def recording_bootstrap_target(inputs: BootstrapInputs) -> torch.Tensor:
        recorded.append(inputs.target_values.detach().clone())
        return bootstrap_target(inputs)

    monkeypatch.setattr(updater, "bootstrap_target", recording_bootstrap_target)
    learner.update(_graph_batch())
    return recorded[0]


def test_reduced_precision_autocast_does_not_round_value_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _bootstrap_values(monkeypatch, nullcontext())
    reduced = _bootstrap_values(monkeypatch, _bfloat16_autocast())

    # Only the encoder runs in bfloat16 now. With the value head inside autocast
    # the targets were off by 0.013 on average (0.045 at most), about two steps
    # of reward; the head in float32 leaves under 0.001.
    assert (reduced - full).abs().max() < 1e-3
