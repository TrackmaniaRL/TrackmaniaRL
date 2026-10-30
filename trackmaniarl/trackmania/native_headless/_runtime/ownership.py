from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os, linux_signal
else:
    # The lifetime supervisor loads this sibling directly, without the SDK.
    from posix_api import linux_os, linux_signal


def process_identity(pid: Any) -> Any:
    """Linux identity for cleanup, never authorize by historical numeric PID."""
    try:
        directory = Path("/proc") / str(pid)
        raw = (directory / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
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
        fd = linux_os.pidfd_open(owned["pid"])
    except ProcessLookupError:
        return False
    except OSError as error:
        # WSL may return EINVAL for a PID reaped between discovery and open.
        # Only absence/a different birth authorizes treating that as gone;
        # an error for the same live identity must remain an explicit failure.
        if error.errno == errno.EINVAL and not same_identity(process_identity(owned["pid"]), owned):
            return False
        raise
    try:
        current = process_identity(owned["pid"])
        if not same_identity(current, owned) or current["state"] in ("Z", "X"):
            return False
        linux_signal.pidfd_send_signal(fd, sig)
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


def owned_descendants(owner: Any) -> list[dict[str, Any]]:
    """Discover only children of a still-identical owner, including new sessions.

    The supervisor is a Linux child subreaper. When the driver exits, surviving
    descendants become its direct children, so discovery never relies on a dead
    leader's historical session ID.
    """
    current = process_identity(owner["pid"]) if owner is not None else None
    if current is None or not same_identity(owner, current) or current["state"] in ("Z", "X"):
        return []
    candidates = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdecimal():
            row = process_identity(int(directory.name))
            if row is not None and row["uid"] == owner["uid"] and row["birth"] >= owner["birth"]:
                candidates.append(row)
    parents = {owner["pid"]: owner}
    descendants = []
    while True:
        found = []
        for candidate in candidates:
            parent = parents.get(candidate["ppid"])
            if candidate["pid"] in parents or parent is None:
                continue
            before = process_identity(parent["pid"])
            row = process_identity(candidate["pid"])
            after = process_identity(parent["pid"])
            if (
                same_identity(parent, before)
                and same_identity(parent, after)
                and same_identity(candidate, row)
                and row["ppid"] == parent["pid"]
                and row["birth"] >= parent["birth"]
            ):
                found.append(row)
        if not found:
            break
        descendants.extend(found)
        parents.update((row["pid"], row) for row in found)
    return descendants
