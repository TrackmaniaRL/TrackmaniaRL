"""Real learner with explicitly synthetic split transport; never a game engine."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import headless_native_train as single
from scripts import headless_native_train_multi as multi
from tests.unit.trackmania.test_common_training_performance import _assert_exact
from tests.unit.trackmania.test_headless_native_train import components, options
from tests.unit.trackmania.test_headless_native_train_multi import (
    SyntheticAdapter,
    SyntheticPool,
    SyntheticRoundError,
    journal,
)
from trackmaniarl.trackmania import common_training
from trackmaniarl.trackmania.common_training import native_only

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux owned native training storage"
)


class SplitPool(SyntheticPool):
    def __init__(self, config: Any, **fields: Any) -> None:
        super().__init__(config, **fields)
        self.pending: Any = None
        self.cancelled = False
        self.cancel_cleanup: Any = None
        self.stop_round: int | None = None
        self.current_actors = 0

    def begin_step_many(self, requests: Any) -> None:
        assert not self.in_flight
        self.current_actors = len(requests)
        self.pending = super().step_many(requests)
        self.in_flight = True
        if self.actions == self.stop_round:
            self.stop_target.touch()

    def finish_step_many(self) -> Any:
        assert self.in_flight
        self.in_flight = False
        result, self.pending = self.pending, None
        return result

    def close_all(self) -> Any:
        if self.cancel_cleanup is not None:
            return self.cancel_cleanup
        if self.in_flight:
            self.cancelled = True
            self.in_flight = False
            self.cancel_cleanup = super().close_all()
            raise SyntheticRoundError(self.pending)
        return super().close_all()


class Factory:
    def __init__(self, **settings: Any) -> None:
        self.settings, self.instances = settings, []

    def __call__(self, config: Any, **fields: Any) -> Any:
        pool = SplitPool(config, **fields)
        for key, value in self.settings.items():
            setattr(pool, key, value)
        self.instances.append(pool)
        return pool


@pytest.fixture
def parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    result = components(tmp_path, monkeypatch)
    monkeypatch.setattr(native_only, "NativeOnlyAdapter", SyntheticAdapter)
    SyntheticAdapter.validations, SyntheticAdapter.reject_actor = [], None
    return result


def opts(tmp_path: Path, name: str, **changes: Any) -> Any:
    return replace(
        options(tmp_path, name),
        batch_size=32,
        replay_capacity=256,
        transitions=80,
        checkpoint_every=256,
        episode_native_units=4000,
        max_run_bytes=2 * 1024**3,
        **changes,
    )


@pytest.mark.parametrize("batch_actions", [False, True])
def test_updates_overlap_only_previous_valid_rows_and_warmup_is_not_lost(  # noqa: PLR0913
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch, *, batch_actions: bool
) -> None:
    factory, updates = Factory(), []
    original = common_training.make_learner

    def make(*args: Any, **kwargs: Any) -> Any:
        learner = original(*args, **kwargs)
        update = learner.update

        def checked(batch: Any) -> Any:
            pool = factory.instances[0]
            allowed = pool.raw_rows - (pool.current_actors if pool.in_flight else 0)
            assert max(batch.transition_ids) < allowed
            updates.append((pool.in_flight, pool.actions, allowed))
            return update(batch)

        learner.update = checked
        return learner

    monkeypatch.setattr(common_training, "make_learner", make)
    selected = opts(tmp_path, "run")
    report = multi.run_training(
        parts,
        selected,
        actors=8,
        pool_factory=factory,
        overlap_learning=True,
        batch_actions=batch_actions,
    )
    assert report["passed"], report
    assert report["optimizer_steps"] == len(updates) == 20
    assert updates[:8] == [(True, 5, 32)] * 8
    assert 8 <= report["maximum_prepared_updates"] <= 16
    assert report["remaining_update_credit"] == 0
    assert report["scheduler"]["policy_lag_bound_rounds"] == 2
    assert {item["update_lag"] for item in report["rounds"][4]["admissions"]} == {8}
    assert factory.instances[0].closed


def test_stop_with_prepared_rng_finishes_bounded_updates_without_fake_terminal(
    tmp_path: Path, parts: Any
) -> None:
    selected = replace(opts(tmp_path, "stopped"), transitions=160)
    factory = Factory(stop_round=5, stop_target=selected.stop_file)
    report = multi.run_training(
        parts, selected, actors=8, pool_factory=factory, overlap_learning=True
    )
    assert report["passed"], report
    assert report["stopped"]
    assert report["transitions"] == 40
    assert report["optimizer_steps"] == 10
    assert report["maximum_prepared_updates"] == 10
    assert not any(row["terminated"] or row["truncated"] for row in journal(selected.output))
    saved = torch.load(selected.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 40
    assert saved["budget"]["updates"] == 10


@pytest.mark.parametrize("batch_actions", [False, True])
def test_resume_at_physical_and_checkpoint_barrier_is_full_state_exact(
    tmp_path: Path, parts: Any, *, batch_actions: bool
) -> None:
    base = replace(
        opts(tmp_path, "full"), transitions=160, episode_native_units=2000, checkpoint_every=80
    )
    modes = {"overlap_learning": True, "batch_actions": batch_actions}
    full = multi.run_training(parts, base, actors=2, pool_factory=Factory(), **modes)
    first_options = replace(
        opts(tmp_path, "first"), transitions=160, episode_native_units=2000, checkpoint_every=80
    )
    first = multi.run_training(
        parts,
        first_options,
        actors=2,
        pool_factory=Factory(stop_round=40, stop_target=first_options.stop_file),
        **modes,
    )
    assert full["passed"], full
    assert first["passed"], first
    assert first["transitions"] == 80
    checkpoint = first_options.output / "latest.pt"
    resumed_options = replace(
        opts(tmp_path, "resumed"),
        transitions=80,
        episode_native_units=2000,
        checkpoint_every=80,
        resume=checkpoint,
        resume_sha256=single._file_sha(checkpoint),
    )
    resumed = multi.run_training(parts, resumed_options, actors=2, pool_factory=Factory(), **modes)
    assert resumed["passed"], resumed
    assert resumed["resume_loaded"]
    left = torch.load(base.output / "latest.pt", weights_only=False)
    right = torch.load(resumed_options.output / "latest.pt", weights_only=False)
    for key in ("learner", "sampler", "budget"):
        _assert_exact(left[key], right[key])


def test_learner_failure_cancels_inflight_and_preserves_last_durable_checkpoint(
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Factory()
    original = common_training.make_learner

    def make(*args: Any, **kwargs: Any) -> Any:
        learner = original(*args, **kwargs)
        update = learner.update

        def broken(batch: Any) -> Any:
            if factory.instances[0].in_flight:
                raise RuntimeError("synthetic learner failure during native work")
            return update(batch)

        learner.update = broken
        return learner

    monkeypatch.setattr(common_training, "make_learner", make)
    selected = replace(opts(tmp_path, "failed"), checkpoint_every=32)
    result = multi.run_training(
        parts, selected, actors=8, pool_factory=factory, overlap_learning=True
    )
    assert not result["passed"]
    assert "synthetic learner failure" in result["error"]
    assert result["transitions"] == 40
    assert factory.instances[0].cancelled
    assert factory.instances[0].closed
    assert result["owned_cleanup_verified"] is True
    saved = torch.load(selected.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 32
    assert saved["budget"]["updates"] == 8
    assert result["cleanup_abort_evidence"]["accepted_rows"] == 0


@pytest.mark.parametrize("changed", ["batch", "overlap", "checkpoint_interval"])
def test_mode_and_overlap_checkpoint_schedule_rejected_before_any_pool(
    tmp_path: Path, parts: Any, changed: str
) -> None:
    first_options = opts(tmp_path, "source")
    first = multi.run_training(
        parts, first_options, actors=2, pool_factory=Factory(), overlap_learning=True
    )
    assert first["passed"], first
    checkpoint = first_options.output / "latest.pt"
    second_options = replace(
        opts(tmp_path, "rejected"),
        resume=checkpoint,
        resume_sha256=single._file_sha(checkpoint),
        checkpoint_every=128 if changed == "checkpoint_interval" else 256,
    )
    factory = Factory()
    report = multi.run_training(
        parts,
        second_options,
        actors=2,
        pool_factory=factory,
        batch_actions=changed == "batch",
        overlap_learning=changed != "overlap",
    )
    assert not report["passed"]
    assert not factory.instances
    assert not (second_options.output / "latest.pt").exists()
