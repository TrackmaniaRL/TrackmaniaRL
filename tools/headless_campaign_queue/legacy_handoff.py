"""Identity-checked observation and handoff from a legacy queue supervisor.

Never removes or modifies an existing campaign lock. A live legacy owner remains
responsible for its child; this module can observe it and claim a *fresh* lock
only after the original lock disappeared and all relevant processes exited.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tools.headless_campaign_queue.recovery import (
    reject_live_competitors,
    reject_surviving_group,
)


def _proc_info(pid: int) -> tuple[dict, int, str]:
    """Read identity, parent and state from one proc stat snapshot."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    identity = {
        "pid": pid,
        "start_ticks": int(stat[19]),
        "process_group": int(stat[2]),
        "boot_id": boot,
    }
    return identity, int(stat[1]), stat[0]


def capture_running_legacy(state: dict, lock: Path) -> dict:
    """Capture live identities; never infer a supervisor birth time from a PID.

    Invoke while both recorded processes are live. Calling after a restart or
    after either process vanished cannot provide the required identity proof.
    """
    if (
        not lock.is_dir()
        or lock.is_symlink()
        or state.get("status") != "running"
        or type(state.get("job_pid")) is not int
        or type(state.get("supervisor_pid")) is not int
        or type(state.get("birth_ticks")) is not int
    ):
        raise ValueError("legacy state or campaign lock is invalid")
    try:
        supervisor, _, supervisor_status = _proc_info(state["supervisor_pid"])
        job, ppid, job_status = _proc_info(state["job_pid"])
        supervisor_again, _, supervisor_status_again = _proc_info(state["supervisor_pid"])
        job_again, ppid_again, job_status_again = _proc_info(state["job_pid"])
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ValueError("cannot establish live process identities") from exc
    if (
        supervisor != supervisor_again
        or job != job_again
        or ppid != ppid_again
        or supervisor_status in ("Z", "X")
        or supervisor_status_again in ("Z", "X")
        or job_status in ("Z", "X")
        or job_status_again in ("Z", "X")
        or job["boot_id"] != state.get("boot_id")
        or supervisor["boot_id"] != job["boot_id"]
        or job["start_ticks"] != state["birth_ticks"]
        or ppid != supervisor["pid"]
        or job["process_group"] != job["pid"]
    ):
        raise ValueError("live process identities do not match the legacy owner")
    return {"schema": "legacy-owner-observation-v1", "job": job, "supervisor": supervisor}


def _assert_identity_gone(identity: dict) -> None:
    if not isinstance(identity, dict) or any(
        k not in identity for k in ("pid", "start_ticks", "process_group", "boot_id")
    ):
        raise ValueError("missing process identity")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if current_boot != identity["boot_id"]:
        return  # identities on previous boot cannot still exist
    try:
        current, _, status = _proc_info(identity["pid"])
    except FileNotFoundError:
        return
    if current == identity and status not in ("Z", "X"):
        raise ValueError(f"owned process {identity['pid']} is still running")


def audit_owner_exited(
    observation: dict, *, reject_competitors: Callable[[], None] = reject_live_competitors
) -> None:
    if observation.get("schema") != "legacy-owner-observation-v1":
        raise ValueError("invalid owner observation")
    job = observation["job"]
    supervisor = observation["supervisor"]
    if job["boot_id"] != supervisor["boot_id"]:
        raise ValueError("inconsistent observed boot identities")
    _assert_identity_gone(job)
    _assert_identity_gone(supervisor)
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() == job["boot_id"]:
        reject_surviving_group(job["process_group"])
    reject_competitors()


def claim_released_lock(
    observation: dict,
    lock: Path,
    *,
    reject_competitors: Callable[[], None] = reject_live_competitors,
) -> tuple[int, int]:
    """Acquire a new lock after owner exit; fail closed if *any* lock exists.

    Return device/inode of our new lock for a later identity-checked release.
    Caller must preserve lock on verification/recovery failure.
    """
    audit_owner_exited(observation, reject_competitors=reject_competitors)
    lock.mkdir()  # atomic and exclusive, never remove a pre-existing lock
    info = lock.stat()
    try:
        audit_owner_exited(observation, reject_competitors=reject_competitors)
    except BaseException:
        # Keep our lock if an audit changes: another process may have started.
        raise
    return info.st_dev, info.st_ino


def release_claimed_lock(lock: Path, identity: tuple[int, int]) -> None:
    """Release only the same empty directory created by claim_released_lock."""
    if lock.is_symlink():
        raise ValueError("campaign lock is a symlink")
    info = lock.stat()
    if (info.st_dev, info.st_ino) != identity:
        raise ValueError("campaign lock identity changed")
    lock.rmdir()
