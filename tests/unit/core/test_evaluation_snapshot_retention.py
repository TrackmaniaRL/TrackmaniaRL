from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.runtime.core_runtime_support import runtime_spec
from trackmaniarl.core.evaluation_checkpoint import (
    create_evaluation_snapshot,
    prune_evaluation_snapshots,
)
from trackmaniarl.core.fingerprint import _fingerprint_config
from trackmaniarl.core.spec import TrainingSpec
from trackmaniarl.observability.artifacts import _resume_equivalent_manifest


def _snapshot(run_dir: Path, checkpoint: Path, keep: int | None) -> Path:
    snapshot = create_evaluation_snapshot(checkpoint, run_dir, keep)
    prune_evaluation_snapshots(snapshot, keep)
    return snapshot


def test_snapshot_retention_configuration_preserves_checkpoint_limit() -> None:
    assert TrainingSpec().evaluation_snapshot_retention == 2
    assert TrainingSpec(checkpoint_keep_last=1).evaluation_snapshot_retention == 1
    assert TrainingSpec(checkpoint_keep_last=5).evaluation_snapshot_retention == 2
    assert TrainingSpec(evaluation_snapshot_keep_last=None).evaluation_snapshot_retention is None
    assert (
        TrainingSpec(
            checkpoint_keep_last=3, evaluation_snapshot_keep_last=None
        ).evaluation_snapshot_retention
        == 3
    )


def test_snapshot_retention_does_not_change_configuration_fingerprint(tmp_path: Path) -> None:
    spec = runtime_spec(tmp_path)
    unlimited = spec.model_copy(
        update={
            "training": spec.training.model_copy(update={"evaluation_snapshot_keep_last": None})
        }
    )
    bounded = spec.model_copy(
        update={"training": spec.training.model_copy(update={"evaluation_snapshot_keep_last": 5})}
    )
    original_config, _ = _fingerprint_config(spec)
    assert _fingerprint_config(unlimited)[0] == original_config
    assert _fingerprint_config(bounded)[0] == original_config
    assert "evaluation_snapshot_keep_last" not in original_config["training"]


def test_resume_manifest_accepts_old_and_changed_retention_only() -> None:
    config = {"components": {"learner": {"kwargs": {}}}, "training": {"batch_size": 4}}
    old = json.dumps({"config": config})
    config["training"]["evaluation_snapshot_keep_last"] = 2
    bounded = json.dumps({"config": config})
    config["training"]["evaluation_snapshot_keep_last"] = None
    unlimited = json.dumps({"config": config})
    assert _resume_equivalent_manifest(old) == _resume_equivalent_manifest(bounded)
    assert _resume_equivalent_manifest(old) == _resume_equivalent_manifest(unlimited)
    config["training"]["batch_size"] = 8
    assert _resume_equivalent_manifest(old) != _resume_equivalent_manifest(
        json.dumps({"config": config})
    )


def test_default_retention_bounds_new_snapshots_and_preserves_current(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    keep = TrainingSpec().evaluation_snapshot_retention
    snapshots = [_snapshot(tmp_path, checkpoint, keep) for _ in range(4)]
    assert not snapshots[0].exists()
    assert not snapshots[1].exists()
    assert snapshots[2].is_file()
    assert snapshots[3].is_file()
    assert snapshots[3].read_bytes() == b"checkpoint"


def test_enabling_retention_preserves_existing_snapshots(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"old")
    legacy = [_snapshot(tmp_path, checkpoint, None) for _ in range(3)]
    checkpoint.write_bytes(b"new")
    new = [_snapshot(tmp_path, checkpoint, 1) for _ in range(3)]
    assert all(path.read_bytes() == b"old" for path in legacy)
    assert not new[0].exists()
    assert not new[1].exists()
    assert new[2].read_bytes() == b"new"


def test_existing_exempt_snapshots_do_not_consume_new_retention_limit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"old")
    legacy = [_snapshot(tmp_path, checkpoint, None) for _ in range(3)]
    checkpoint.write_bytes(b"new")
    new = [_snapshot(tmp_path, checkpoint, 2) for _ in range(3)]
    assert all(path.is_file() for path in legacy)
    assert not new[0].exists()
    assert new[1].is_file()
    assert new[2].is_file()


def test_explicit_unlimited_retention_keeps_all_snapshots(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    keep = TrainingSpec(evaluation_snapshot_keep_last=None).evaluation_snapshot_retention
    snapshots = [_snapshot(tmp_path, checkpoint, keep) for _ in range(5)]
    assert all(path.is_file() for path in snapshots)


def test_corrupt_exemption_record_does_not_delete_snapshots(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    saved = _snapshot(tmp_path, checkpoint, 2)
    (saved.parent / "retention-exempt.json").write_text(json.dumps({"snapshots": "bad"}))
    with pytest.raises(ValueError, match="exemption"):
        create_evaluation_snapshot(checkpoint, tmp_path, 2)
    assert saved.is_file()
