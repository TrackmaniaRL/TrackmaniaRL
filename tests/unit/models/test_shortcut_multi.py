"""Multi-lap admission and imitation release preserve explicit data contracts."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from trackmaniarl.algorithms.value_based.objectives import ValueObjectiveContext
from trackmaniarl.experiments.shortcut_demonstrations import RetainedExpertObjective
from trackmaniarl.experiments.shortcut_multi import (
    MultiDemonstrationShortcutFactory,
    PaceAwareMixedSampler,
    ReleasingExpertObjective,
)


def test_release_keeps_initial_hold_and_nonzero_floor() -> None:
    objective = ReleasingExpertObjective([1, 1])
    assert [objective.weight_at(n) for n in (0, 100_000, 300_000, 500_000, 10**9)] == [
        128,
        128,
        72,
        16,
        16,
    ]
    for invalid in (None, -1, True):
        with pytest.raises(ValueError, match="actual nonnegative"):
            objective.weight_at(invalid)


def test_slow_lap_imitation_has_less_influence_without_touching_online_rows() -> None:
    values = torch.zeros(3, 2, requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.tensor([0, 0, 0]),
        torch.ones(3, dtype=torch.bool),
        {"demo_flags": (True, True, False), "demonstration_weights": (1.0, 0.2, 0.0)},
    )
    RetainedExpertObjective([1, 1], weight=1).loss(context).backward()
    assert values.grad[1].abs().sum() == pytest.approx(values.grad[0].abs().sum() * 0.2)
    assert torch.count_nonzero(values.grad[2]) == 0
    context.metadata["demonstration_weights"] = (1.0, float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite demonstration weight"):
        RetainedExpertObjective([1, 1]).loss(context)


def test_manifest_refuses_hash_mismatch_and_duplicate_laps(tmp_path: Path) -> None:
    path = (tmp_path / "manifest.json").resolve()
    path.write_text(
        json.dumps(
            {
                "schema": "tmrl.native-multi-demonstration/v1",
                "episodes": [{"sha256": "same"}, {"sha256": "same"}],
            }
        )
    )
    factory = object.__new__(MultiDemonstrationShortcutFactory)
    factory.demonstration_binding = {"path": str(path), "sha256": "wrong"}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        factory.load_demonstration(path, None)
    factory.demonstration_binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        factory.load_demonstration(path, None)


def test_pace_weights_survive_compact_metadata_and_protected_demo_relocation() -> None:
    from trackmaniarl.core.builtins import IdentityFeaturePipeline
    from trackmaniarl.core.data import BatchRequest, Transition
    from trackmaniarl.core.replay import InMemoryReplayStore

    store = InMemoryReplayStore(capacity=8)
    for index in range(24):
        demo = index < 2
        store.append(
            Transition(
                float(index),
                0,
                1.0,
                float(index + 1),
                True,
                False,
                info={
                    "is_demo": demo,
                    "sampling/projected_lap_time_s": 17.715 if index == 0 else 18.995,
                },
                episode_id=str(index),
                step=0,
            )
        )
    sampler = PaceAwareMixedSampler(IdentityFeaturePipeline())
    batch = sampler.sample(store, BatchRequest(2, transition_count=0))
    assert all(batch.metadata["demo_flags"])
    assert min(batch.transition_ids) >= 8
    # Replay stores the lap times as float32.
    assert sorted(batch.metadata["demonstration_weights"]) == pytest.approx(
        [0.1814695, 1.0], abs=1e-6
    )
