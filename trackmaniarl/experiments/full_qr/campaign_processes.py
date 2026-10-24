"""Own learner/actor process trees through the existing pidfd lease supervisor.

This is process lifecycle plumbing, not a qualified environment or a launcher.
Each child gets its own subreaper, so an actor crash cannot leave native forks
outside the parent's cleanup scope. Closing first requests the shared SDK stop.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from trackmaniarl.trackmania.native_headless._runtime import ownership
from trackmaniarl.trackmania.native_headless._runtime.supervisor import (
    SupervisedProcess,
    spawn_supervised,
)


@dataclass(slots=True)
class _Child:
    name: str
    process: SupervisedProcess
    stderr: BinaryIO
    stdout: BinaryIO
    drain: threading.Thread
    drain_errors: list[str]


class CampaignProcessCohort:
    """One fresh cohort; no historical PID, foreign process or old STOP access."""

    def __init__(self, directory: Path, request_stop: Callable[[], None]) -> None:
        if not directory.is_absolute() or directory.resolve() != directory:
            raise ValueError("canonical absolute campaign process directory required")
        directory.mkdir(exist_ok=False)
        self.directory = directory
        self.request_stop = request_stop
        self.children: list[_Child] = []
        self.closed = False

    def start(self, name: str, args: list[str], *, cwd: Path) -> dict[str, Any]:
        if (
            self.closed
            or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None
            or any(child.name == name for child in self.children)
            or not args
        ):
            raise ValueError("fresh uniquely named campaign child required")
        stderr = (self.directory / f"{name}.stderr").open("xb")
        stdout = (self.directory / f"{name}.stdout").open("xb")
        try:
            process = spawn_supervised(
                args,
                cwd=cwd,
                env=dict(os.environ),
                stderr=stderr,
                evidence=self.directory / f"{name}.closure.json",
                timeout=10,
            )
        except BaseException:
            stderr.close()
            stdout.close()
            raise
        errors: list[str] = []

        def drain_output() -> None:
            try:
                if process.stdout is not None:
                    shutil.copyfileobj(process.stdout, stdout)
            except BaseException as error:
                errors.append(repr(error))

        drain = threading.Thread(target=drain_output, name=f"{name}-output", daemon=True)
        try:
            drain.start()
        except BaseException:
            try:
                process.stop(timeout=5)
            finally:
                process.close_control()
                stderr.close()
                stdout.close()
            raise
        self.children.append(_Child(name, process, stderr, stdout, drain, errors))
        return {
            "name": name,
            "worker": process.driver_identity,
            "supervisor": process.supervisor_identity,
        }

    def exited(self) -> dict[str, int]:
        """Any exit before the shared stop is a campaign failure, even code zero."""
        result = {}
        for child in self.children:
            code = child.process.poll()
            if code is not None:
                result[child.name] = code
        return result

    def close(self, *, grace_seconds: float = 120) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("campaign cohort closure may only be recorded once")
        if not 0 <= grace_seconds <= 120:
            raise ValueError("bounded shared shutdown grace required")
        self.closed = True
        errors, forced, reports = [], [], {}
        try:
            self.request_stop()
        except BaseException as error:
            errors.append(f"shared stop: {error!r}")
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            try:
                if len(self.exited()) == len(self.children):
                    break
            except BaseException as error:
                errors.append(f"poll: {error!r}")
                break
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        for child in self.children:
            try:
                if child.process.poll() is None:
                    forced.append(child.name)
                report = child.process.stop(timeout=5)
                if report["remainingOwned"] or report["cleanupErrors"]:
                    raise RuntimeError("child supervisor did not prove clean teardown")
                reports[child.name] = report
            except BaseException as error:
                errors.append(f"{child.name}: {error!r}")
            finally:
                child.drain.join(timeout=5)
                if child.drain.is_alive():
                    errors.append(f"{child.name}: output drain still alive")
                else:
                    child.stdout.close()
                errors.extend(f"{child.name}: output: {value}" for value in child.drain_errors)
                child.stderr.close()
                child.process.close_control()
        remaining = []
        for child in self.children:
            for row in (
                child.process.driver_identity,
                child.process.supervisor_identity,
                *reports.get(child.name, {}).get("capturedOwned", []),
            ):
                if ownership.same_identity(row, ownership.process_identity(row["pid"])):
                    remaining.append(row)
        return {
            "schema": "tmrl.campaign-process-cohort/v1",
            "children": reports,
            "forced": forced,
            "cleanupErrors": errors,
            "remainingOwned": remaining,
            "final_checkpoint_verified": False,
            "training_completed": False,
        }
