from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.integration.runtime.distributed_evaluation_support import _finished_evaluation
from tests.integration.runtime.test_distributed_submission import (
    _base_payload,
    _close,
    _coordinator,
)
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.journal import JournalSchemaError


def _new_coordinator(tmp_path: Path) -> Coordinator:
    coordinator = _coordinator(tmp_path, "leader-recovery")
    coordinator.run.learner.state_dict_for_policy = lambda policy: {"value": policy["value"]}
    return coordinator


def test_pending_promotion_rejects_corrupted_snapshot_evidence(tmp_path: Path) -> None:
    coordinator = _new_coordinator(tmp_path)
    try:
        coordinator.journal.stage_evaluation_promotion(1, b"original")
        with sqlite3.connect(coordinator.journal.path) as connection:
            connection.execute(
                "UPDATE metadata SET value = substr(value, 1, 65) || ? WHERE key = ?",
                ("dGFtcGVyZWQ=", "evaluation_promotion:00000000000000000001"),
            )
        with pytest.raises(JournalSchemaError, match="invalid evaluation promotion evidence"):
            coordinator.journal.pending_evaluation_promotions()
    finally:
        _close(coordinator)


@pytest.mark.parametrize("failure", ["stage", "first", "second", "complete", "after_complete"])
def test_promotions_survive_interrupted_recovery_and_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    coordinator = _new_coordinator(tmp_path)
    baseline = coordinator._checkpoint()
    coordinator._checkpoint_writer.wait()
    value = _base_payload(0)
    value["evaluations"] = [_finished_evaluation(36.0)]
    value["evaluation_snapshot"] = coordinator.codec.encode({"value": 123})
    row_id, _ = coordinator.journal.append("session", 0, coordinator.codec.encode(value))
    assert row_id == 1
    codec = coordinator._checkpoint_writer.codec
    original_save = codec.save
    original_complete = coordinator.journal.complete_evaluation_promotion

    def fail_save(state: Any, path: Path) -> None:
        if (failure == "first" and path.name.startswith("best-eval")) or (
            failure == "second" and path.name.startswith("fastest-eval")
        ):
            raise OSError("injected promotion failure")
        original_save(state, path)

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("injected promotion failure")

    def fail_after_complete(row_id: int) -> None:
        original_complete(row_id)
        raise OSError("injected promotion failure")

    with monkeypatch.context() as context:
        if failure == "stage":
            context.setattr(coordinator.journal, "stage_evaluation_promotion", fail)
        elif failure == "complete":
            context.setattr(coordinator.journal, "complete_evaluation_promotion", fail)
        elif failure == "after_complete":
            context.setattr(
                coordinator.journal, "complete_evaluation_promotion", fail_after_complete
            )
        else:
            context.setattr(codec, "save", fail_save)
        with pytest.raises(OSError, match="injected promotion failure"):
            coordinator.restore_checkpoint(baseline)
    assert coordinator.journal.pruned_through == 0
    _close(coordinator)

    resumed = _new_coordinator(tmp_path)
    try:
        resumed.restore_checkpoint(baseline)
        directory = resumed.run.run_dir / "checkpoints"
        best = list(directory.glob("best-eval-*.pt"))
        fastest = list(directory.glob("fastest-eval-*.pt"))
        assert len(best) == len(fastest) == 1
        assert resumed.run.checkpoint_codec.load(best[0])["learner"]["value"] == 123
        assert resumed.run.checkpoint_codec.load(fastest[0])["learner"]["value"] == 123
        assert resumed.journal.pending_evaluation_promotions() == []
        assert resumed.journal.pruned_through == 0
        full = resumed._checkpoint()
        resumed._checkpoint_writer.wait()
        assert resumed.journal.pruned_through == 1
    finally:
        _close(resumed)

    final = _new_coordinator(tmp_path)
    try:
        final.restore_checkpoint(full)
        assert final.counters.evaluations == 1
        assert final.journal.pending_evaluation_promotions() == []
        assert len(list((final.run.run_dir / "checkpoints").glob("*-eval-*.pt"))) == 2
    finally:
        _close(final)


def test_restore_first_leader_finishes_pending_second_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _new_coordinator(tmp_path)
    baseline = coordinator._checkpoint()
    coordinator._checkpoint_writer.wait()
    value = _base_payload(0)
    value["evaluations"] = [_finished_evaluation(36.0)]
    value["evaluation_snapshot"] = coordinator.codec.encode({"value": 123})
    coordinator.journal.append("session", 0, coordinator.codec.encode(value))
    codec = coordinator._checkpoint_writer.codec
    original_save = codec.save

    def fail_second(state: Any, path: Path) -> None:
        if path.name.startswith("fastest-eval"):
            raise OSError("interrupted second promotion")
        original_save(state, path)

    with monkeypatch.context() as context:
        context.setattr(codec, "save", fail_second)
        with pytest.raises(OSError, match="interrupted second promotion"):
            coordinator.restore_checkpoint(baseline)
    directory = coordinator.run.run_dir / "checkpoints"
    best = next(directory.glob("best-eval-*.pt"))
    assert coordinator.journal.pruned_through == 0
    _close(coordinator)

    resumed = _new_coordinator(tmp_path)
    try:
        resumed.restore_checkpoint(best)
        fastest = list(directory.glob("fastest-eval-*.pt"))
        assert len(fastest) == 1
        assert resumed.run.checkpoint_codec.load(fastest[0])["learner"]["value"] == 123
        assert resumed.journal.pending_evaluation_promotions() == []
        assert resumed.counters.evaluations == 1
    finally:
        _close(resumed)


def test_first_leader_restores_other_rank_after_pending_was_cleared(tmp_path: Path) -> None:
    coordinator = _new_coordinator(tmp_path)
    coordinator.run.learner.value = 7
    baseline = coordinator._checkpoint()
    coordinator._checkpoint_writer.wait()
    value = _base_payload(0)
    value["evaluations"] = [_finished_evaluation(36.0)]
    value["evaluation_snapshot"] = coordinator.codec.encode({"value": 123})
    coordinator.journal.append("session", 0, coordinator.codec.encode(value))
    coordinator.restore_checkpoint(baseline)
    directory = coordinator.run.run_dir / "checkpoints"
    best = next(directory.glob("best-eval-*.pt"))
    artifact = coordinator.run.checkpoint_codec.load(best)
    assert artifact["learner"]["value"] == 123
    assert artifact["training_learner"]["value"] == 7
    assert coordinator.journal.pending_evaluation_promotions() == []
    assert coordinator._fastest_evaluation is not None
    _close(coordinator)

    restored = _new_coordinator(tmp_path)
    try:
        restored.restore_checkpoint(best)
        assert restored.run.learner.value == 7
        assert restored._best_evaluation is not None
        assert restored._fastest_evaluation is not None
        assert restored.journal.pending_evaluation_promotions() == []
        assert len(list(directory.glob("fastest-eval-*.pt"))) == 1
    finally:
        _close(restored)
