from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trackmaniarl.core.checkpoint_milestones import CheckpointProgress, preserve_milestones
from trackmaniarl.core.checkpoints import prune_checkpoint_family
from trackmaniarl.core.spec import TrainingSpec


def test_milestone_copies_survive_rotation_and_recovery_file_rewrites(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    first = directory / "distributed-update-00000001.pt"
    first.write_bytes(b"first complete snapshot")
    milestones = preserve_milestones(first, (10, 20), CheckpointProgress(0, 12, 1))
    assert len(milestones) == 1
    metadata = json.loads(milestones[0].with_suffix(".json").read_text())
    assert metadata["requested_transitions"] == 10
    assert metadata["transitions"] == 12
    first.write_bytes(b"replacement recovery state")
    second = directory / "distributed-update-00000002.pt"
    second.write_bytes(b"next complete snapshot")
    prune_checkpoint_family(second, directory, 1)
    assert not first.exists()
    assert milestones[0].read_bytes() == b"first complete snapshot"
    assert preserve_milestones(second, (10,), CheckpointProgress(0, 22, 2)) == ()


def test_resume_does_not_fabricate_earlier_milestones(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    checkpoint = directory / "update-00000009.pt"
    checkpoint.write_bytes(b"resumed")
    saved = preserve_milestones(checkpoint, (10, 20, 30), CheckpointProgress(20, 31, 9))
    assert len(saved) == 1
    assert saved[0].name.startswith("milestone-000000000030-")


def test_partial_milestone_is_not_overwritten_or_treated_as_complete(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    checkpoint = directory / "update-00000009.pt"
    checkpoint.write_bytes(b"complete")
    partial = directory / "milestone-000000000010-transitions-000000000010-update-00000008.pt"
    partial.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        preserve_milestones(checkpoint, (10,), CheckpointProgress(0, 20, 9))
    assert partial.read_bytes() == b"partial"


@pytest.mark.parametrize("milestones", [(20, 10), (10, 10), (0,), (101,)])
def test_training_rejects_invalid_checkpoint_milestones(milestones: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        TrainingSpec(total_transitions=100, checkpoint_milestones=milestones)
