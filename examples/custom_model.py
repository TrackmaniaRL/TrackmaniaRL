"""Compose a custom sensor encoder with bundled temporal, head and strategy modules."""

from __future__ import annotations

import torch
from torch import nn

from trackmaniarl.models.contracts import RiskSpec, ValuePhase
from trackmaniarl.models.factory import CompositeValueModelFactory


class MyEncoder(nn.Module):
    """Encode individual four-value observations into sixteen features."""

    output_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, self.output_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(observation))


def main() -> None:
    factory = CompositeValueModelFactory(
        encoder={"class_path": f"{__name__}:MyEncoder"},
        temporal={
            "class_path": "trackmaniarl.models.temporal:IdentityTemporalCore",
            "kwargs": {"input_dim": 16},
        },
        head={
            "class_path": "trackmaniarl.models.heads:ScalarQHead",
            "kwargs": {"feature_dim": 16, "action_count": 3},
        },
        strategy={"class_path": "trackmaniarl.models.strategies:ScalarValueStrategy"},
    )
    model = factory.build().eval()
    with torch.no_grad():
        features = model.encode_frames(torch.zeros(2, 4))
        support = model.support(features, ValuePhase.EVALUATE)
        values = model.expected_all_actions(features, support, RiskSpec())
    print(f"Action values for two observations: {tuple(values.shape)}")


if __name__ == "__main__":
    main()
