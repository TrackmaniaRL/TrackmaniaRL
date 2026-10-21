"""Synthetic representation/math fixtures, never native experience evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch

from examples.native_headless.learning import (
    RaggedUpdater,
    create_learner,
    lazy_targets,
    make_policy,
    validate_batch,
)
from examples.native_headless.replay import PROFILE, REFERENCE, SCHEMA, Replay
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch


def _fixture(
    *, terminated: bool = False, truncated: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invented wire-shaped test tokens only; no rollout/qualification claim."""
    before: dict[str, Any] = {
        "features": [float(i) / 100 for i in range(44)],
        "featureProfile": PROFILE,
        "schema": SCHEMA,
        "nativeT": 19980 if terminated else 2230,
        "historyActionIds": [],
        "fixtureOnly": True,
    }
    n = 3 if terminated else 5
    after: dict[str, Any] = {
        **deepcopy(before),
        "features": None if terminated else [float(i + 1) / 100 for i in range(44)],
        "nativeT": before["nativeT"] + n * 10,
        "historyActionIds": [36],
        "native76Hex": "00" * 76,
        "projection": {"referenceFingerprint": REFERENCE},
        "pendingSGB": None if terminated else [0.0, 0.0, 0.0],
        "readerHex": None if terminated else "fixture",
        "pendingOwnP": None if terminated else 160,
    }
    result = {
        "trainingEligible": True,
        "status": "terminated" if terminated else "truncated" if truncated else "continue",
        "terminated": terminated,
        "truncated": truncated,
        "reward": 0.25 if terminated else 0.5,
        "actualIterations": n,
        "effectiveRequestedIterations": 5,
        "requestedIterations": 5,
        "deltaNative": 10 * n,
        "credit": {"intervals": n, "nativeDelta": 10 * n, "pendingNotCredited": True},
        "observation": after,
        "bootstrapFeatures": after["features"],
        "outcome": {
            "status": "terminated",
            "terminated": True,
            "truncated": False,
            "qualifiedBy": "env002.full-original-validator+TMRESULT",
            "computedResult": {"Time": 17715, "NbCheckpoints": 1},
        }
        if terminated
        else None,
        "fixtureOnly": True,
    }
    return before, result


def _batch(*, terminal_only: bool = False) -> TrainingBatch:
    replay = Replay(4, 17)
    before, result = _fixture(terminated=True)
    replay.append(before, 36, result)
    if not terminal_only:
        before, result = _fixture(truncated=True)
        replay.append(before, 36, result)
    return replay.sample(len(replay))


def test_replay_owns_rows_capacity_and_separate_rng() -> None:
    replay, same = Replay(2, 17), Replay(2, 17)
    for _ in range(3):
        before, result = _fixture()
        assert replay.append(before, 36, result) == same.append(before, 36, result)
        before["features"][0] = 999
        result["observation"]["features"][0] = 999
    assert len(replay) == 2
    assert replay.sample(2).transition_ids == same.sample(2).transition_ids
    assert float(replay.sample(1).observations[0, 0]) == 0.0


def test_mixed_targets_only_evaluate_actual_nonterminal_subset() -> None:
    batch = _batch()
    calls: list[tuple[int, ...]] = []

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(values.shape))
        return torch.full((values.shape[0], 32), 2.0)

    targets = lazy_targets(batch, evaluate)
    assert calls == [(1, 44)]
    assert tuple(targets.shape) == (2, 32)
    for i, terminal in enumerate(batch.terminated):
        expected = batch.rewards[i] if bool(terminal) else batch.rewards[i] + 0.995 * 2
        assert torch.equal(targets[i], expected.expand(32))
    assert batch.metadata["delta_native"].count(30) == 1


def test_terminal_only_targets_never_evaluate_or_encode_absent_observations() -> None:
    batch = _batch(terminal_only=True)

    def forbidden(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("terminal next observation must never be encoded")

    targets = lazy_targets(batch, forbidden)
    assert tuple(batch.next_observations.shape) == (0, 44)
    assert torch.equal(targets[0], batch.rewards[0].expand(32))
    assert float(batch.bootstrap_discounts[0]) == 0.0


def test_invalid_credit_or_terminal_evidence_never_enters_replay() -> None:
    replay = Replay(4, 17)
    for mutate in ("flags", "credit", "duration", "outcome", "missing_features"):
        before, result = _fixture(terminated=True)
        if mutate == "flags":
            result["terminated"] = None
        elif mutate == "credit":
            result["credit"]["pendingNotCredited"] = False
        elif mutate == "duration":
            result["deltaNative"] = 50
        elif mutate == "outcome":
            result["outcome"]["computedResult"]["Time"] = None
        else:
            result["observation"].pop("features")
        with pytest.raises(ValueError, match=r"flags|credit|duration|finish|terminal"):
            replay.append(before, 36, result)
    assert len(replay) == 0


def test_tensor_validation_rejects_wrong_gather_action_and_duration() -> None:
    batch = _batch()
    invalid_indices = {"bootstrap_row_indices": torch.tensor([0, 1])}
    with pytest.raises(ValueError, match="bootstrap indices"):
        validate_batch(replace(batch, data=invalid_indices))
    with pytest.raises(ValueError, match="out-of-range action"):
        validate_batch(replace(batch, actions=torch.full_like(batch.actions, 78)))
    with pytest.raises(ValueError, match="discount mismatch"):
        validate_batch(replace(batch, bootstrap_discounts=torch.ones_like(batch.rewards)))


@pytest.mark.parametrize("terminal_only", [False, True])
def test_model_main_update_uses_only_actual_current_and_ragged_next(*, terminal_only: bool) -> None:
    """CPU synthetic NN test; root authorizes execution separately from source review."""
    learner = create_learner(1729)
    batch = _batch(terminal_only=terminal_only)
    calls: dict[str, list[tuple[int, ...]]] = {"online": [], "target": []}

    def record_online(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
        calls["online"].append(tuple(args[0].shape))

    def record_target(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
        calls["target"].append(tuple(args[0].shape))

    handles = [
        learner.model.encoder.register_forward_pre_hook(record_online),
        learner.target_model.encoder.register_forward_pre_hook(record_target),
    ]
    before = {name: value.detach().clone() for name, value in learner.model.named_parameters()}
    try:
        metrics, priorities = learner.update(batch)
    finally:
        for handle in handles:
            handle.remove()
    size, subset = len(batch.transition_ids), batch.next_observations.shape[0]
    assert calls["online"] == [(size, 44)] + ([(subset, 44)] if subset else [])
    assert calls["target"] == ([(subset, 44)] if subset else [])
    assert learner.update_count == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert list(priorities.transition_ids) == list(batch.transition_ids)
    assert any(
        not torch.equal(before[name], value) for name, value in learner.model.named_parameters()
    )
    forward = RaggedUpdater(learner)._forward_step(RaggedUpdater(learner)._batch_step(batch))
    for index in batch.terminated.nonzero(as_tuple=True)[0].tolist():
        assert torch.equal(forward.targets[index, 0], batch.rewards[index].expand(32))


def test_model_policy_and_checkpoint_contract() -> None:
    learner = create_learner(1729)
    policy = make_policy(learner)
    policy.set_exploration_epsilon(0.0)
    action = policy.act(torch.zeros(44), PolicyMode.EVALUATION)
    assert type(action) is int
    assert 0 <= action < 78
    with pytest.raises(ValueError, match="no implicit old checkpoint"):
        learner.load_state_dict({})
    state = learner.state_dict()
    learner.load_state_dict(state)
    assert learner.update_count == 0
