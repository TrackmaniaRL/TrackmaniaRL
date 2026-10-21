from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any


def process_identity(pid: Any) -> Any:
    """Linux identity for cleanup, never authorize by historical numeric PID."""
    try:
        directory = Path("/proc") / str(pid)
        raw = (directory / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        return {
            "pid": pid,
            "state": fields[0],
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "birth": int(fields[19]),
            "uid": directory.stat().st_uid,
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_identity(a: Any, b: Any) -> Any:
    return (
        a is not None
        and b is not None
        and all(a[k] == b[k] for k in ("pid", "birth", "uid", "sid", "pgid"))
    )


def signal_identity(owned: Any, sig: Any) -> Any:
    """Bind signal to pidfd; recheck the complete captured identity after open."""
    try:
        fd = os.pidfd_open(owned["pid"])
    except ProcessLookupError:
        return False
    try:
        current = process_identity(owned["pid"])
        if not same_identity(current, owned) or current["state"] in ("Z", "X"):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def owned_session_members(owner: Any) -> Any:
    """A live unchanged session leader authorizes discovery, not a historical SID."""
    current = process_identity(owner["pid"]) if owner is not None else None
    if current is None or not same_identity(owner, current) or current["state"] in ("Z", "X"):
        return []
    members = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdecimal():
            row = process_identity(int(directory.name))
            if (
                row is not None
                and row["state"] not in ("Z", "X")
                and (row["sid"] == owner["sid"])
                and (row["uid"] == owner["uid"])
                and (row["birth"] >= owner["birth"])
            ):
                members.append(row)
    return members
