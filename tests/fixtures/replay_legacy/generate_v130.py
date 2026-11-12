"""Generate synthetic checkpoint and v1 journal with the v1.3.0 release.

Run in an isolated process with v1.3.0 eb803a21 on PYTHONPATH. The output
directory must not exist; all intermediate run data is temporary.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import tempfile
import zlib
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from google.protobuf.wrappers_pb2 import BytesValue

import trackmaniarl
from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
    _resolved_run,
    _transition,
    _TransitionSpec,
    _TransitionState,
)
from trackmaniarl.core.replay.sequence_sampler import SequenceSampler
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import PROTOCOL_VERSION, transition_to_wire

_BASELINE = "eb803a2153d472ebb16cd7150201c82058dd804a"
_IDENTITY = "synthetic-fixture-fingerprint"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package_root = Path(trackmaniarl.__file__).resolve().parent.parent
    head = subprocess.run(
        ["git", "-C", str(package_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != _BASELINE:
        raise RuntimeError(f"fixture generator requires baseline {_BASELINE}; got {head}")
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="trackmaniarl-legacy-fixture-") as temporary:
        _generate(Path(temporary), args.output)


def _generate(root: Path, output: Path) -> None:
    run = _resolved_run(
        root,
        "fixture",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    run = replace(run, sampler=SequenceSampler(run.feature_pipeline, sequence_length=2, seed=41))
    run.learner.value = 11
    coordinator = Coordinator(run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, _IDENTITY))

    def submit(sequence: int, steps: list[_TransitionSpec]) -> None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": _IDENTITY,
            "actor_id": "fixture-actor",
            "session_id": "fixture-session",
            "sequence": sequence,
            "policy_version": 0,
            "transitions": [transition_to_wire(_transition(step)) for step in steps],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        coordinator._submit(
            BytesValue(value=coordinator.codec.encode(payload)),
            _Context(f"Bearer {_DISTRIBUTED_TOKEN}"),
        )

    try:
        submit(
            0,
            [
                _TransitionSpec("fixture-actor", i, float(i + 1), session="fixture-session")
                for i in range(2)
            ],
        )
        coordinator._recover_journal(0)
        checkpoint = coordinator._checkpoint()
        coordinator._checkpoint_writer.wait()
        submit(
            1,
            [
                _TransitionSpec(
                    "fixture-actor",
                    2,
                    3.0,
                    state=_TransitionState.TERMINATES,
                    session="fixture-session",
                )
            ],
        )
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()

    journal = run.run_dir / "distributed" / "rollouts.sqlite3"
    # Inspect only: preserve the schema and data produced by the release.
    with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        version = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not row[0].startswith("sqlite_")
        }
    if version != ("1",) or tables != {"chunks", "actor_profiles", "receipts", "metadata"}:
        raise RuntimeError(f"release did not produce a v1 journal: {version!r}, {tables!r}")
    (output / "checkpoint-v130-eb803a21.bin").write_bytes(checkpoint.read_bytes())
    (output / "journal-v1-v130-eb803a21.bin").write_bytes(
        zlib.compress(journal.read_bytes(), level=9)
    )


if __name__ == "__main__":
    main()
