"""New human-authorized seed29 confirmation after restoring the finish menu state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from pilot_tools import ppo_overnight as trial
from pilot_tools import ppo_pilot as pilot

ORIGINAL_PREPARE = pilot.prepare
ORIGINAL_VERIFY = trial.verify
FAILED = pilot.BASE / "ppo-overnight-recovery-s29-20261013"


def closed_attempt() -> dict[str, Any]:
    proof: dict[str, Any] = json.loads((FAILED / "post-closure-verification.json").read_text())
    if not proof["all_owned_closed"] or not proof["mutex_free"]:
        raise RuntimeError("Previous preflight attempt is not closed")
    for identity in proof["owned_identities"]:
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time()-identity["create_time"]) < .001 and process.is_running():
                raise RuntimeError("Previous exact process remains alive")
        except psutil.NoSuchProcess:
            pass
    if (pilot.sha(FAILED / "STOP") != proof["stop_sha256"]
            or (FAILED / "STOP").stat().st_mtime_ns != proof["stop_mtime_ns"]):
        raise RuntimeError("Previous STOP changed; preserve and stop")
    return proof


def prepare(engine: Path) -> dict[str, Any]:
    proof = closed_attempt()
    plan = ORIGINAL_PREPARE(engine)
    plan.update(
        authorization="Human: Dobra to ogarniaj dalej PPO,2026-10-14; fresh4h confirmation",
        readiness_restored="Observed Race Validation finish menu; normal Improve to active race",
        previous_preflight_queue=str(FAILED),
        prior_stop_pins=[{"predecessor_stop": str(FAILED / "STOP"),
                          "predecessor_stop_present": True,
                          "predecessor_stop_sha256": proof["stop_sha256"],
                          "predecessor_stop_mtime_ns": proof["stop_mtime_ns"]}],
    )
    return plan


def verify() -> dict[str, Any]:
    closed_attempt()
    plan = ORIGINAL_VERIFY()
    if plan.get("previous_preflight_queue") != str(FAILED):
        raise RuntimeError("New confirmation provenance differs")
    return plan


def main() -> None:
    # A new bounded trial after a new direct human instruction, never an
    # extension or restart of the completed overnight queues.
    trial.WINDOW_END = datetime(2026, 10, 14, 12, tzinfo=UTC)
    pilot.prepare = prepare
    trial.verify = verify
    trial.main()


if __name__ == "__main__":
    main()
