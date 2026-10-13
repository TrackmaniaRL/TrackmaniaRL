"""Independent fail-closed deadline supervisor; exact owned processes only."""

from __future__ import annotations

import json
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def exact(identity: dict[str, Any]) -> psutil.Process | None:
    try:
        process = psutil.Process(identity["pid"])
        if abs(process.create_time() - identity["create_time"]) < .001 and process.is_running():
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def main(queue: Path) -> None:
    launch = json.loads((queue / "launch.json").read_text())
    plan = json.loads((queue / "plan.json").read_text())
    start = datetime.fromisoformat(launch["started_at"]).timestamp()
    launcher = launch["launcher"]
    # Windows venv Python has a forwarding process between launcher and guard.
    # Neither the guard nor that forwarding ancestor is a controlled workload.
    guard_chain = {psutil.Process().pid}
    for ancestor in psutil.Process().parents():
        if ancestor.pid == launcher["pid"]:
            break
        guard_chain.add(ancestor.pid)
    observed = {(launcher["pid"], launcher["create_time"]): launcher}
    last_state = launch
    stop_paths = [queue / "STOP", queue.parent / "STOP", queue.parent.parent / "STOP",
                  queue.parent.parent / "STOP-ppo", Path(plan["source_root"]) / "artifacts/STOP",
                  Path(plan["source_root"]) / "artifacts/STOP-ppo"]
    reason = None
    stop_started = None
    with (queue / "deadline-guard-ready.json").open("x", encoding="utf-8") as target:
        json.dump({"pid": psutil.Process().pid, "create_time": psutil.Process().create_time(),
                   "hard_at": start+plan["maximum_seconds"]}, target)
    while True:
        with suppress(OSError, ValueError):
            last_state = json.loads((queue / "status.json").read_text())
        for key in ("owned_identities", "all_observed_identities"):
            for identity in last_state.get(key, []):
                if identity["pid"] not in guard_chain:
                    observed[(identity["pid"], identity["create_time"])] = identity
        # Independently capture descendants even if status publication stops.
        root = exact(launcher)
        if root is not None:
            try:
                for child in root.children(recursive=True):
                    if child.pid not in guard_chain:
                        identity = {"pid": child.pid, "create_time": child.create_time()}
                        observed[(child.pid, child.create_time())] = identity
            except psutil.NoSuchProcess:
                pass
        elapsed = time.time() - start
        if (last_state.get("status") == "COMPLETE" and root is None
                and all(exact(item) is None for item in observed.values())):
            break
        if elapsed >= plan["maximum_seconds"]:
            reason = "hard_deadline"
            break
        if root is None:
            reason = "launcher_exited_without_closed_complete"
            break
        if any(path.exists() for path in stop_paths):
            reason = "STOP"
        if elapsed >= plan["maximum_seconds"] - plan["save_margin_seconds"]:
            reason = "save_deadline"
        if reason:
            if stop_started is None:
                stop_started = time.monotonic()
            try:
                with (queue / "STOP").open("x", encoding="utf-8") as target:
                    target.write(reason+"\n")
            except FileExistsError:
                pass
            if time.monotonic() - stop_started >= 60:
                break
        time.sleep(.5)
    if reason:
        try:
            with (queue / "STOP").open("x", encoding="utf-8") as target:
                target.write(reason+"\n")
        except FileExistsError:
            pass
        # At HARD or after launcher loss, never leave an orphan controller.
        for identity in reversed(list(observed.values())):
            process = exact(identity)
            if process is not None:
                with suppress(psutil.NoSuchProcess):
                    process.kill()
    proof = {"checked_at": datetime.now(UTC).isoformat(), "reason": reason,
             "observed_identities": list(observed.values()),
             "survivors": [item for item in observed.values() if exact(item) is not None]}
    with (queue / "deadline-guard-exit.json").open("x", encoding="utf-8") as target:
        json.dump(proof, target, indent=2)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
