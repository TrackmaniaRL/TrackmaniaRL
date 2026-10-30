"""Independent actions, fail-closed data admission and released expert guidance."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms.value_based.objectives import ValueObjectiveContext
from trackmaniarl.experiments.shortcut_demonstrations import (
    AnnealedExpertObjective,
    OfflineThenMixedSampler,
    load_native_demonstration,
)
from trackmaniarl.experiments.shortcut_macro import (
    CONTROL_SCHEMA,
    AutonomousMacroProvider,
    AutonomousMacroShortcutFactory,
    MacroCatalog,
    actual_goal_reward,
)
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import RewardResult


def catalog(tmp_path: Path, *, timeline: bool = False) -> dict[str, str]:
    path = tmp_path / "catalog.npz"
    fields = {
        "schema": np.asarray(CONTROL_SCHEMA),
        "programs": np.repeat(np.arange(78, dtype=np.int64)[:, None], 5, axis=1),
    }
    if timeline:
        fields["teacher_actions"] = np.asarray([39, 0, 75])
    np.savez(path, **fields)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_policy_choice_alone_selects_motor_program_regardless_of_clock(tmp_path: Path) -> None:
    vocabulary = MacroCatalog(catalog(tmp_path))
    provider = object.__new__(AutonomousMacroProvider)
    for clock in (0, 5000, 17000, 149950):
        provider._before = {"nativeT": clock + 2200}
        for action in (0, 39, 75):
            provider.program = vocabulary.program(action)
            assert (
                tuple(provider._control_for_tick(action, tick) for tick in range(5))
                == (action,) * 5
            )
    with pytest.raises(ValueError, match="outside"):
        vocabulary.program(78)


def test_inference_catalog_and_factory_reject_teacher_playback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="static macro vocabulary"):
        MacroCatalog(catalog(tmp_path, timeline=True))
    with pytest.raises(ValueError, match="playback forbidden"):
        AutonomousMacroShortcutFactory({"control_schema": CONTROL_SCHEMA, "reference_controls": {}})


def test_demonstration_rejects_unsigned_clock_rollover(monkeypatch: pytest.MonkeyPatch) -> None:
    count = 257
    times = ((np.arange(count + 1) * 50) % 256).astype(np.uint8)
    assert np.all(np.diff(times) == 50)  # Unsigned subtraction alone misses rollover.
    data = {
        "schema": np.asarray("tmrl.native-macro-demonstration/v1"),
        "catalog_sha256": np.asarray("catalog"),
        "actions": np.zeros(count, dtype=np.int64),
        "rewards": np.zeros(count),
        "terminated": np.r_[np.zeros(count - 1, dtype=bool), True],
        "observed_physics_ms": times,
    }
    monkeypatch.setattr(
        "trackmaniarl.experiments.shortcut_demonstrations.pinned_arrays", lambda _: data
    )
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="consecutive fifty-ms"):
        load_native_demonstration({}, SimpleNamespace(sha256="catalog", programs=range(78)), {})


def test_guidance_is_strong_offline_then_disappears_without_constraining_actions() -> None:
    objective = AnnealedExpertObjective(release_transitions=250_000)
    values = torch.zeros((2, 127))
    context = ValueObjectiveContext(
        values,
        torch.tensor([0, 126]),
        torch.tensor([True, True]),
        {"demo_flags": (True, True), "collected_transition_count": 0},
    )
    initial = objective.loss(context)
    assert initial is not None
    assert torch.isfinite(initial)
    assert initial > 0
    context.metadata["collected_transition_count"] = 125_000
    online = objective.loss(context)
    assert online is not None
    torch.testing.assert_close(online, initial * 0.05)
    context.metadata["collected_transition_count"] = 250_000
    assert objective.loss(context) is None
    del context.metadata["collected_transition_count"]
    with pytest.raises(ValueError, match="counter"):
        objective.loss(context)


def test_terminal_reward_distinguishes_a_one_millisecond_improvement() -> None:
    result = RewardResult(100, True, "finished")
    config = RewardConfig(time_penalty_per_second=1, time_attack_linear_scale=10)
    reference = actual_goal_reward(result, config, (17750, 17715))
    improved = actual_goal_reward(result, config, (17750, 17714))
    assert improved.reward - reference.reward == pytest.approx(0.011)
    assert improved.reason == "finished"


def test_expert_sampling_compares_pace_at_replay_storage_precision() -> None:
    sampler = OfflineThenMixedSampler(object())
    assert sampler.online.expert_demo_time_s == float(np.float32(17.715))
