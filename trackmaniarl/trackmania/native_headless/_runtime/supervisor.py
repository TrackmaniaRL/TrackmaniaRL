"""Linux child supervisor, deliberately executable without importing the SDK.

The actor's private socket is a lifetime lease. Only this process launches the
native driver, after receiving the launch request. Losing the lease therefore
covers actor death even during the actor's own subprocess factory.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import select
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING or __name__ != "__main__":
    from trackmaniarl.trackmania.native_headless._runtime import ownership
    from trackmaniarl.trackmania.native_headless._runtime.posix_api import (
        linux_os,
        linux_signal,
        linux_socket,
    )
else:
    import ownership
    from posix_api import linux_os, linux_signal, linux_socket


def _send(control: socket.socket, value: dict[str, Any]) -> None:
    control.sendall(json.dumps(value, allow_nan=False).encode())


def _receive(control: socket.socket) -> dict[str, Any] | None:
    raw = control.recv(1024 * 1024)
    return json.loads(raw) if raw else None


def _key(row: dict[str, Any]) -> tuple[int, int, int]:
    return row["pid"], row["birth"], row["uid"]


def _cleanup(
    owner: dict[str, Any], driver: subprocess.Popen[bytes] | None, known: dict[Any, Any]
) -> dict[str, Any]:
    """Repeated discovery handles forks and subreaper adoption during teardown."""
    deadline = time.monotonic() + 3
    sent, errors = [], []
    driver_pid = None if driver is None else driver.pid
    while True:
        for row in ownership.owned_descendants(owner):
            known[_key(row)] = row
        if driver is not None:
            driver.poll()
        for row in sorted(known.values(), key=lambda item: item["pid"] == driver_pid):
            try:
                if ownership.signal_identity(row, linux_signal.SIGKILL):
                    sent.append(row)
            except OSError as error:
                errors.append(repr(error))
            if row["pid"] != driver_pid:
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(row["pid"], linux_os.WNOHANG)
        remaining = [
            row for row in ownership.owned_descendants(owner) if row["state"] not in ("Z", "X")
        ]
        if not remaining and (driver is None or driver.poll() is not None):
            # Reap adopted zombies after the final live process has exited.
            for row in ownership.owned_descendants(owner):
                if row["pid"] != driver_pid:
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(row["pid"], linux_os.WNOHANG)
            break
        if time.monotonic() >= deadline:
            break
        select.select([], [], [], 0.01)
    return {
        "cleanupPidfdSignals": sent,
        "cleanupErrors": errors,
        "remainingOwned": remaining,
        "returncode": None if driver is None else driver.poll(),
        "capturedOwned": list(known.values()),
    }


def _run(control: socket.socket, evidence: Path) -> int:
    owner = ownership.process_identity(os.getpid())
    if owner is None or owner["state"] in ("Z", "X"):
        raise RuntimeError("live supervisor identity required before launch")
    # Unsupported pidfd kernels fail before any native process can exist.
    preflight = linux_os.pidfd_open(os.getpid())
    os.close(preflight)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    request = _receive(control)
    if request is None:
        return 0
    if request["kind"] != "start":
        raise ValueError("supervisor requires a launch request")
    driver, driver_fd, identity = None, None, None
    known: dict[Any, Any] = {}
    reason = "driver-exited"
    supervisor_error = publication_error = None
    try:
        driver = subprocess.Popen(
            request["args"],
            cwd=request["cwd"],
            env=request["env"],
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            stderr=sys.stderr.buffer,
            start_new_session=True,
        )
        # Every post-launch operation is covered by cleanup, including identity
        # capture, pidfd allocation, pipe closing and the initial acknowledgement.
        os.close(0)
        os.close(1)
        identity = ownership.process_identity(driver.pid)
        if identity is None:
            raise RuntimeError("native identity unavailable after launch")
        known[_key(identity)] = identity
        driver_fd = linux_os.pidfd_open(driver.pid)
        _send(
            control,
            {"kind": "started", "driverIdentity": identity, "supervisorIdentity": owner},
        )
        while driver.poll() is None:
            # No /proc polling in the simulation loop. The subreaper retains
            # ownership until teardown; pidfd and lease wake this wait directly.
            if control in select.select([control, driver_fd], [], [])[0]:
                command = _receive(control)
                reason = "actor-lease-eof" if command is None else "actor-stop"
                if command is not None and command != {"kind": "stop"}:
                    reason = "invalid-actor-command"
                break
    except (BrokenPipeError, ConnectionResetError):
        reason = "actor-lease-eof"
    except BaseException as error:
        reason, supervisor_error = "supervisor-error", repr(error)
    finally:
        if driver_fd is not None:
            with contextlib.suppress(OSError):
                os.close(driver_fd)
        report = {
            "schema": "tmrl.native-headless.supervisor/v1",
            "reason": reason,
            "driverIdentity": identity,
            "supervisorIdentity": owner,
            "supervisorError": supervisor_error,
            **_cleanup(owner, driver, known),
        }
        # A failed publication must neither skip cleanup nor prevent the actor
        # from receiving the failure and exact cleanup result over its lease.
        try:
            with evidence.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            publication_error = repr(error)
            report["publicationError"] = publication_error
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            _send(control, {"kind": "exited", **report})
    return (
        0
        if (
            report["returncode"] is not None
            and not report["remainingOwned"]
            and supervisor_error is None
            and publication_error is None
        )
        else 1
    )


class SupervisedProcess:
    """Native PID and byte pipes with independently owned, bounded teardown."""

    def __init__(
        self, supervisor: subprocess.Popen[bytes], control: socket.socket, evidence: Path
    ) -> None:
        self.supervisor = supervisor
        self.control = control
        self.evidence = evidence
        self.stdin, self.stdout = supervisor.stdin, supervisor.stdout
        self.returncode: int | None = None
        self.report: dict[str, Any] | None = None
        started = _receive(control)
        if started is None or started.get("kind") != "started":
            raise RuntimeError("native supervisor did not acknowledge launch")
        self.driver_identity = started["driverIdentity"]
        self.supervisor_identity = started["supervisorIdentity"]
        if self.driver_identity is None:
            raise RuntimeError("native driver disappeared before identity capture")
        self.pid = self.driver_identity["pid"]

    def _collect(self) -> None:
        while select.select([self.control], [], [], 0)[0]:
            try:
                value = _receive(self.control)
            except ConnectionResetError:
                # A driver may exit while the actor sends stop. Closing the
                # peer with that command unread resets this socket, sometimes
                # after its final report was already received.
                break
            if value is None:
                break
            if value.get("kind") != "exited":
                raise RuntimeError("unexpected supervisor response")
            self.report = value
            self.returncode = value["returncode"]

    def _recover_report(self) -> None:
        """Use the supervisor's durable receipt only after it has been reaped."""
        if self.report is not None:
            return
        try:
            fd = os.open(self.evidence, os.O_RDONLY | linux_os.O_NOFOLLOW | linux_os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
                    raise ValueError("invalid native supervisor receipt file")
                report = json.loads(stream.read(1024 * 1024 + 1))
            if (
                report.get("schema") != "tmrl.native-headless.supervisor/v1"
                or not ownership.same_identity(report.get("driverIdentity"), self.driver_identity)
                or not ownership.same_identity(
                    report.get("supervisorIdentity"), self.supervisor_identity
                )
                or type(report.get("returncode")) is not int
                or not isinstance(report.get("remainingOwned"), list)
            ):
                raise ValueError("native supervisor receipt identity or schema mismatch")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(
                "native supervisor exited without matching cleanup evidence"
            ) from error
        self.report = {"kind": "exited", **report}
        self.returncode = report["returncode"]

    def poll(self) -> int | None:
        self._collect()
        if self.supervisor.poll() is not None:
            self._collect()
            self._recover_report()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.supervisor.wait(timeout=timeout)
        self._collect()
        self._recover_report()
        if (
            self.report is None
            or self.returncode is None
            or self.report["remainingOwned"]
            or self.report.get("supervisorError")
            or self.report.get("publicationError")
        ):
            raise RuntimeError("native supervisor could not prove complete cleanup")
        return self.returncode

    def stop(self, timeout: float = 5) -> dict[str, Any]:
        if self.poll() is None:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                _send(self.control, {"kind": "stop"})
        self.wait(timeout=timeout)
        assert self.report is not None
        return self.report

    def close_control(self) -> None:
        self.control.close()
        for stream in (self.stdin, self.stdout):
            if stream is not None:
                stream.close()


def spawn_supervised(  # noqa: PLR0913
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stderr: BinaryIO,
    evidence: Path,
    timeout: float,
) -> SupervisedProcess:
    if sys.platform != "linux":
        raise RuntimeError("native supervision requires Linux pidfd and subreaper support")
    control, child = socket.socketpair(linux_socket.AF_UNIX, socket.SOCK_SEQPACKET)
    control.settimeout(min(timeout, 10))
    supervisor = None
    try:
        supervisor = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(child.fileno()), str(evidence)],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            pass_fds=(child.fileno(),),
            start_new_session=True,
        )
        child.close()
        _send(control, {"kind": "start", "args": args, "cwd": str(cwd), "env": env})
        process = SupervisedProcess(supervisor, control, evidence)
        control.settimeout(5)
        return process
    except BaseException:
        control.close()
        if supervisor is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                supervisor.wait(timeout=5)
            for stream in (supervisor.stdin, supervisor.stdout):
                if stream is not None:
                    stream.close()
        raise
    finally:
        child.close()


if __name__ == "__main__":
    with socket.socket(fileno=int(sys.argv[1])) as lease:
        raise SystemExit(_run(lease, Path(sys.argv[2])))
