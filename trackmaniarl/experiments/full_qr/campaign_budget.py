"""Wall-clock/space watchdog for a future qualified full-QR launcher.

This component requests a graceful SDK stop; it does not launch or qualify an
environment, kill processes, prune checkpoints, or declare training completed.
The launcher must verify its final checkpoint and owned-process cleanup.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from trackmaniarl.experiments.full_qr.campaign_clock import clock_seconds, validate_clock_source

PILOT_SECONDS = 14_400
SUPPORTED_PILOT_SECONDS = (PILOT_SECONDS, 25_200)
GIB = 1024**3


class StopSignal(Protocol):
    """An owned campaign's process-shared, one-way SDK shutdown flag."""

    def set(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StorageReserve:
    path: Path
    minimum_free_bytes: int

    def __post_init__(self) -> None:
        if self.minimum_free_bytes <= 0:
            raise ValueError("storage reserve must be positive")


@dataclass(frozen=True, slots=True)
class BudgetSettings:
    # Must be a fresh directory inside this new campaign's output directory.
    directory: Path
    reserves: tuple[StorageReserve, ...]
    duration_seconds: float = PILOT_SECONDS
    check_interval_seconds: float = 1.0
    clock_source: str = "monotonic"

    def __post_init__(self) -> None:
        for value in (self.duration_seconds, self.check_interval_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("duration and check interval must be finite and positive")
        if not self.reserves:
            raise ValueError("at least one storage reserve is required")
        validate_clock_source(self.clock_source)


def next_monitor_offset(
    elapsed_seconds: float, duration_seconds: int = PILOT_SECONDS
) -> int | None:
    """Next check strictly after elapsed time, measured from actual training start."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    if type(duration_seconds) is not int or duration_seconds not in SUPPORTED_PILOT_SECONDS:
        raise ValueError("explicit supported pilot duration required")
    offsets = (900, 1800, 2700, 3600, *range(5400, duration_seconds + 1, 1800))
    return next((value for value in offsets if value > elapsed_seconds), None)


class CampaignBudget:
    """Arm immediately at actual launch; STOP is consumed by the existing SDK.

    Elapsed wall time uses monotonic time, so changing the system clock cannot
    extend the experiment. UTC start/deadline are recorded for the operator.
    The watchdog uses an independent thread, including during a learner update.
    """

    def __init__(
        self,
        settings: BudgetSettings,
        *,
        free_bytes: Callable[[Path], int] | None = None,
        stop_signal: StopSignal | None = None,
    ) -> None:
        self.settings = settings
        self.stop_file = settings.directory / "STOP"
        self.state_file = settings.directory / "budget-state.json"
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self._stop_signal = stop_signal
        self._requested = threading.Event()
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {}
        self._deadline = 0.0

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def stop_requested(self) -> bool:
        """Work-boundary check; the in-memory stop survives a full output disk."""
        return self._requested.is_set() or self.limit_reached()

    def limit_reached(self) -> bool:
        """Use the same declared clock as the learner and parent deadline."""
        return bool(self._state) and clock_seconds(self.settings.clock_source) >= self._deadline

    def start(self) -> dict[str, Any]:
        """Called by the launcher only when its qualified training starts."""
        if self._state or self._closing.is_set():
            raise RuntimeError("campaign budget can only be started once")
        reserves = self._check_storage()
        # Never clear an old STOP, replace a prior run, or follow a directory link.
        started = clock_seconds(self.settings.clock_source)
        self.settings.directory.mkdir(exist_ok=False)
        utc = datetime.now(UTC)
        self._deadline = started + self.settings.duration_seconds
        self._state = {
            "schema": "tmrl.campaign-budget/v1",
            "phase": "armed",
            "started_utc": utc.isoformat(),
            "deadline_utc": (utc + timedelta(seconds=self.settings.duration_seconds)).isoformat(),
            "duration_seconds": self.settings.duration_seconds,
            "started_monotonic": started,
            "clock_source": self.settings.clock_source,
            "stop_reason": None,
            "storage": reserves,
            "final_checkpoint_verified": False,
            "process_cleanup_verified": False,
        }
        try:
            self._save()
            self._thread = threading.Thread(target=self._watch, name="full-qr-budget", daemon=True)
            self._thread.start()
        except BaseException:
            self.request_stop("watchdog_start_failed")
            raise
        return self.state

    def request_stop(self, reason: str) -> None:
        """Signal first, then preserve/write STOP and record the first reason.

        A full disk must not prevent the actor and learner's shared shutdown
        notification. Persistence errors remain explicit in the returned state.
        The caller must pass only this new campaign's own StopSignal.
        """
        if not reason:
            raise ValueError("stop reason must not be empty")
        with self._lock:
            if not self._state:
                raise RuntimeError("campaign budget has not started")
            if self._state["stop_reason"] is not None:
                return
            self._requested.set()
            if self._stop_signal is not None:
                try:
                    self._stop_signal.set()
                except Exception as exc:
                    self._state["shutdown_signal_error"] = repr(exc)
            try:
                fd = os.open(self.stop_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                reason = "external_stop"
            except OSError as exc:
                self._state["stop_file_error"] = repr(exc)
            else:
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        stream.write(reason + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as exc:
                    self._state["stop_file_error"] = repr(exc)
            self._state.update(
                phase="stop_requested",
                stop_reason=reason,
                stop_requested_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=(
                    clock_seconds(self.settings.clock_source) - self._state["started_monotonic"]
                ),
            )
            self._save_stop_state()

    def close(self) -> dict[str, Any]:
        """Stop this watchdog after the caller shuts down its training runtime.

        Closing early still writes STOP. A closed watchdog alone is never a
        completed experiment, nor evidence that a checkpoint or cleanup passed.
        """
        if not self._state:
            raise RuntimeError("campaign budget has not started")
        self.request_stop("launcher_closed")
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("campaign watchdog did not close")
        with self._lock:
            self._state["watchdog_closed"] = True
            self._save_stop_state()
        return self.state

    def _watch(self) -> None:
        while not self._closing.is_set():
            remaining = self._deadline - clock_seconds(self.settings.clock_source)
            if remaining <= 0:
                self.request_stop("wall_clock_limit")
                return
            if self.stop_file.exists():
                self.request_stop("external_stop")
                return
            try:
                self._check_storage()
            except Exception as exc:
                with self._lock:
                    self._state["storage_error"] = str(exc)
                self.request_stop("storage_reserve")
                return
            self._closing.wait(min(remaining, self.settings.check_interval_seconds))

    def _check_storage(self) -> list[dict[str, Any]]:
        result = []
        for reserve in self.settings.reserves:
            free = self._free_bytes(reserve.path)
            if free < reserve.minimum_free_bytes:
                raise RuntimeError(f"insufficient space at {reserve.path}: {free} bytes free")
            result.append(
                {
                    "path": str(reserve.path),
                    "free_bytes": free,
                    "minimum_free_bytes": reserve.minimum_free_bytes,
                }
            )
        return result

    def _save(self) -> None:
        temporary = self.state_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self._state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_file)

    def _save_stop_state(self) -> None:
        try:
            self._save()
        except OSError as exc:
            # The shared shutdown signal and in-memory reason were set before
            # file operations. Never claim durable evidence when writing fails.
            self._state["state_file_error"] = repr(exc)
