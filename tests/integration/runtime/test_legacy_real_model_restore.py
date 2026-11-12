"""Read a synthetic real-learner checkpoint produced by v1.3.0."""

from __future__ import annotations

import sqlite3
import zlib
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures.replay_legacy.real_model_support import make_run
from tests.integration.runtime.distributed_runtime_support import _DISTRIBUTED_TOKEN
from tests.unit._composite_value_fixtures import _assert_nested_state_equal
from trackmaniarl.core.data import BatchRequest
from trackmaniarl.core.fingerprint import run_fingerprint
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "replay_legacy"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # The learner snapshots RNG state. Avoid even querying or seeding CUDA on
    # hosts which happen to have a GPU; these fixtures are CPU-only.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)


def _case(root: Path, *, use_saved_identity: bool) -> tuple[Coordinator, Path, dict]:
    run = make_run(root)
    journal = run.run_dir / "distributed" / "rollouts.sqlite3"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(
        zlib.decompress((_FIXTURES / "journal-real-v1-v130-eb803a21.bin").read_bytes())
    )
    with closing(sqlite3.connect(journal)) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone() == ("1",)
    checkpoint = root / "legacy-real-checkpoint.pt"
    checkpoint.write_bytes((_FIXTURES / "checkpoint-real-v130-eb803a21.bin").read_bytes())
    saved = dict(run.checkpoint_codec.load(checkpoint))
    actual_identity = run_fingerprint(run.spec, root)
    assert saved["run_fingerprint"] != actual_identity
    fingerprint = saved["run_fingerprint"] if use_saved_identity else actual_identity
    coordinator = Coordinator(
        run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, fingerprint)
    )
    return coordinator, checkpoint, saved


def test_real_learner_fixture_restores_format_and_can_update_from_recovered_replay(
    tmp_path: Path,
) -> None:
    # Inject the recorded identity at the low-level coordinator boundary only
    # to test serialized state; normal cross-revision resume rejects it below.
    coordinator, checkpoint, saved = _case(tmp_path, use_saved_identity=True)
    try:
        coordinator.restore_checkpoint(checkpoint)
        learner = coordinator.run.learner
        assert learner.update_count == 1
        assert learner.optimizer.state
        assert "cuda" not in saved["learner"]["training"]["rng"]
        _assert_nested_state_equal(learner.state_dict(), saved["learner"])
        assert coordinator.counters.transitions == 3
        assert coordinator.counters.journal_applied_frontier == 2
        assert [item.step for item in coordinator.run.replay_store.get([0, 1, 2])] == [0, 1, 2]
        assert coordinator.run.sampler.state_dict() == saved["sampler"]
        with coordinator.journal._lock:
            receipts = coordinator.journal._connection.execute(
                "SELECT session_id, sequence, row_id FROM receipts ORDER BY sequence"
            ).fetchall()
        assert receipts == [("fixture-session", 0, 1), ("fixture-session", 1, 2)]
        coordinator._recover_journal(coordinator.counters.journal_applied_frontier)
        assert coordinator.counters.transitions == 3
        assert len(coordinator.run.replay_store) == 3
        batch = coordinator.run.sampler.sample(
            coordinator.run.replay_store, BatchRequest(batch_size=1, sequence_length=2)
        )
        before_weights = [parameter.detach().clone() for parameter in learner.model.parameters()]
        before_steps = [state["step"].item() for state in learner.optimizer.state.values()]
        metrics, _ = learner.update(batch)
        assert learner.update_count == 2
        assert any(
            not torch.equal(before, after)
            for before, after in zip(before_weights, learner.model.parameters(), strict=True)
        )
        after_steps = [state["step"].item() for state in learner.optimizer.state.values()]
        assert all(
            after == before + 1 for before, after in zip(before_steps, after_steps, strict=True)
        )
        assert all(np.isfinite(float(value)) for value in metrics.values())
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_real_learner_fixture_rejects_current_source_identity_before_mutation(
    tmp_path: Path,
) -> None:
    coordinator, checkpoint, _ = _case(tmp_path, use_saved_identity=False)
    try:
        before = deepcopy(coordinator.run.learner.state_dict())
        with pytest.raises(ValueError, match="run fingerprint mismatch"):
            coordinator.restore_checkpoint(checkpoint)
        _assert_nested_state_equal(coordinator.run.learner.state_dict(), before)
        assert len(coordinator.run.replay_store) == 0
        assert coordinator.counters.transitions == 0
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()
