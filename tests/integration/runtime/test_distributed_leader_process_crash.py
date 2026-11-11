"""Exercise promotion recovery after killing an isolated coordinator process."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tests.integration.runtime.distributed_evaluation_support import _finished_evaluation
from tests.integration.runtime.test_distributed_leader_recovery import _new_coordinator
from tests.integration.runtime.test_distributed_submission import _base_payload, _close


def _crash_child(directory: Path, boundary: str) -> None:
    """Run real writes and pause at a named boundary until the parent kills us."""
    coordinator = _new_coordinator(directory)
    baseline = coordinator._checkpoint()
    coordinator._checkpoint_writer.wait()
    payload = _base_payload(0)
    payload["evaluations"] = [_finished_evaluation(36.0)]
    payload["evaluation_snapshot"] = coordinator.codec.encode({"value": 123})
    coordinator.journal.append("session", 0, coordinator.codec.encode(payload))

    def pause() -> None:
        temporary = directory / "ready.json.tmp"
        temporary.write_text(json.dumps({"baseline": str(baseline)}), encoding="utf-8")
        temporary.replace(directory / "ready.json")
        Event().wait()

    stage = coordinator.journal.stage_evaluation_promotion
    complete = coordinator.journal.complete_evaluation_promotion
    codec = coordinator._checkpoint_writer.codec
    save = codec.save

    def stage_and_pause(*args: Any, **kwargs: Any) -> Any:
        if boundary == "before_stage":
            pause()
        result = stage(*args, **kwargs)
        if boundary == "after_stage":
            pause()
        return result

    def save_and_pause(state: Any, path: Path) -> None:
        save(state, path)
        if (boundary == "after_first" and path.name.startswith("best-eval")) or (
            boundary == "after_second" and path.name.startswith("fastest-eval")
        ):
            pause()

    def complete_and_pause(row_id: int) -> None:
        if boundary == "before_complete":
            pause()
        complete(row_id)
        if boundary == "after_complete":
            pause()

    coordinator.journal.stage_evaluation_promotion = stage_and_pause
    coordinator.journal.complete_evaluation_promotion = complete_and_pause
    codec.save = save_and_pause
    coordinator.restore_checkpoint(baseline)
    raise AssertionError(f"child unexpectedly finished before boundary {boundary}")


@pytest.mark.parametrize(
    "boundary",
    [
        "before_stage",
        "after_stage",
        "after_first",
        "after_second",
        "before_complete",
        "after_complete",
    ],
)
def test_killed_coordinator_recovers_leaders_and_preserves_wal(
    tmp_path: Path, boundary: str
) -> None:
    project = Path(__file__).resolve().parents[3]
    code = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from tests.integration.runtime.test_distributed_leader_process_crash import _crash_child; "
        "_crash_child(Path(sys.argv[2]), sys.argv[3])"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(project), str(tmp_path), boundary],
        cwd=project,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    marker = tmp_path / "ready.json"
    try:
        deadline = time.monotonic() + 60.0
        while not marker.is_file():
            if child.poll() is not None:
                error = child.stderr.read().decode(errors="replace") if child.stderr else ""
                pytest.fail(f"coordinator exited before {boundary}: {error}")
            if time.monotonic() >= deadline:
                pytest.fail(f"coordinator did not reach {boundary} within 60 seconds")
            time.sleep(0.05)
        baseline = Path(json.loads(marker.read_text(encoding="utf-8"))["baseline"])
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)

    resumed = _new_coordinator(tmp_path)
    try:
        resumed.restore_checkpoint(baseline)
        directory = resumed.run.run_dir / "checkpoints"
        best = list(directory.glob("best-eval-*.pt"))
        fastest = list(directory.glob("fastest-eval-*.pt"))
        assert len(best) == len(fastest) == 1
        assert resumed.run.checkpoint_codec.load(best[0])["learner"]["value"] == 123
        assert resumed.run.checkpoint_codec.load(fastest[0])["learner"]["value"] == 123
        assert resumed.counters.evaluations == 1
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
