"""Kill isolated writers before their transaction/atomic replacement commits."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest
import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.distributed.journal import RolloutJournal


def _pause(directory: Path) -> None:
    temporary = directory / "ready.tmp"
    temporary.write_text("ready", encoding="ascii")
    temporary.replace(directory / "ready")
    Event().wait()


def _child(directory: Path, kind: str) -> None:
    if kind == "journal":
        journal = RolloutJournal(directory / "rollouts.sqlite3")
        claim = journal._claim_canonical_evaluation

        def claim_and_pause(actor: str, assignment: str, row: int) -> bool:
            result = claim(actor, assignment, row)
            # INSERTs for chunk, receipt and canonical result have all run,
            # but the surrounding SQLite transaction has not committed.
            _pause(directory)
            return result

        journal._claim_canonical_evaluation = claim_and_pause
        journal.append("session", 1, b"payload", evaluation=("actor", "assignment", True, 0))
    else:
        import trackmaniarl.core.builtins as builtins

        path = directory / "checkpoint.pt"
        codec = TorchCheckpointCodec()
        codec.save({"value": torch.tensor([7])}, path)
        write = builtins._write_zstd_checkpoint

        def interrupted_write(state: object, temporary: Path) -> None:
            # Produce the actual compressed checkpoint bytes using the codec's
            # production writer, then stop halfway through writing the file
            # that save() intends to replace the previous checkpoint with.
            source = directory / "complete-source.tmp"
            write(state, source)
            raw = source.read_bytes()
            assert len(raw) > 2
            with temporary.open("wb") as destination:
                destination.write(raw[: len(raw) // 2])
                destination.flush()
                os.fsync(destination.fileno())
                _pause(directory)

        builtins._write_zstd_checkpoint = interrupted_write
        codec.save({"value": torch.arange(200_000)}, path)
    raise AssertionError("child unexpectedly completed")


@pytest.mark.parametrize("kind", ["journal", "checkpoint"])
def test_killed_writer_leaves_previously_committed_state_intact(tmp_path: Path, kind: str) -> None:
    project = Path(__file__).resolve().parents[3]
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from tests.integration.runtime.test_atomic_write_process_crash import _child; "
        "_child(Path(sys.argv[2]), sys.argv[3])"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(project), str(tmp_path), kind],
        cwd=project,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60.0
        while not (tmp_path / "ready").exists():
            if child.poll() is not None:
                error = child.stderr.read().decode(errors="replace") if child.stderr else ""
                pytest.fail(f"writer exited before {kind} crash point: {error}")
            if time.monotonic() >= deadline:
                pytest.fail(f"writer did not reach {kind} crash point")
            time.sleep(0.05)
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)

    if kind == "checkpoint":
        path = tmp_path / "checkpoint.pt"
        assert TorchCheckpointCodec().load(path)["value"].item() == 7
        assert list(tmp_path.glob(".checkpoint.pt.*.tmp"))
        return

    journal = RolloutJournal(tmp_path / "rollouts.sqlite3")
    try:
        assert list(journal.rows_after(0)) == []
        assert not journal.has_receipt("session", 1, b"payload")
        row, inserted = journal.append(
            "session", 1, b"payload", evaluation=("actor", "assignment", True, 0)
        )
        assert inserted
        assert journal.evaluation_row_accepted(row) is True
        assert list(journal.rows_after(0)) == [(row, b"payload")]
    finally:
        journal.close()
