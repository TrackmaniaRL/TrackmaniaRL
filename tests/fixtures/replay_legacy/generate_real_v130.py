"""Generate a real-learner checkpoint and v1 journal with the v1.3.0 release.

Run in an isolated process with v1.3.0 eb803a21 on PYTHONPATH. The output
directory must not exist; all intermediate run data is temporary.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import tempfile
import zlib
from contextlib import closing
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before running this CPU-only generator")

import torch
from google.protobuf.wrappers_pb2 import BytesValue
from real_model_support import make_run, transition

import trackmaniarl
from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
)
from tests.unit._composite_value_fixtures import _batch
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import PROTOCOL_VERSION, run_fingerprint, transition_to_wire

_BASELINE = "eb803a2153d472ebb16cd7150201c82058dd804a"


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable to the fixture generator")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
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
    run = make_run(root)
    run.learner.update(_batch())
    if run.learner.update_count != 1 or not run.learner.optimizer.state:
        raise RuntimeError("synthetic update did not initialize the optimizer")
    fingerprint = run_fingerprint(run.spec, root)
    coordinator = Coordinator(
        run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, fingerprint)
    )

    def submit(sequence: int, steps: list[int]) -> None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": fingerprint,
            "actor_id": "fixture-actor",
            "session_id": "fixture-session",
            "sequence": sequence,
            "policy_version": 0,
            "transitions": [transition_to_wire(transition(step)) for step in steps],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        coordinator._submit(
            BytesValue(value=coordinator.codec.encode(payload)),
            _Context(f"Bearer {_DISTRIBUTED_TOKEN}"),
        )

    try:
        submit(0, [0, 1])
        coordinator._recover_journal(0)
        checkpoint = coordinator._checkpoint()
        coordinator._checkpoint_writer.wait()
        submit(1, [2])
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
    (output / "checkpoint-real-v130-eb803a21.bin").write_bytes(checkpoint.read_bytes())
    (output / "journal-real-v1-v130-eb803a21.bin").write_bytes(
        zlib.compress(journal.read_bytes(), level=9)
    )


if __name__ == "__main__":
    main()
