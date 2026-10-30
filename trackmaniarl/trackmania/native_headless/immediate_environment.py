"""Reusable, explicitly selected V7 physical-state facade with literal journals.

This adapter supplies measured engine endpoints and applied controls. Rewards,
common feature normalization and replay insertion belong to a separately bound
training contract; native44 pending-input features are unavailable here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from trackmaniarl.trackmania.native_headless.assets import verify_assets
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROFILE,
    IMMEDIATE_PROTOCOL,
    IMMEDIATE_RESET_PROFILE,
    NativeRuntimeConfig,
)
from trackmaniarl.trackmania.native_headless.protocols import (
    NativeExecutionDisabledError,
    NativeInconclusiveError,
    NativeLifecycleError,
)


class NativeImmediateEnvironment:
    def __init__(self, config: NativeRuntimeConfig, *, seed: int | None = None) -> None:
        if config.protocol_profile != IMMEDIATE_PROTOCOL:
            raise ValueError("explicit immediate-current/v7 configuration required")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError("seed must be an integer or None")
            seed = int(seed)
        self.config, self.seed = config, seed
        self._session: Any = None
        self._current: dict[str, Any] | None = None
        self._journal: BinaryIO | None = None
        self._journal_path: Path | None = None
        self._closed, self._ended, self._elapsed = False, True, 0
        self._duration_limit = config.duration_limit
        self._episode = 0

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "assetProfile": IMMEDIATE_PROFILE,
            "wireProfile": IMMEDIATE_PROTOCOL,
            "inputSchedule": "current-record-before-load/v1",
            "resetProfile": IMMEDIATE_RESET_PROFILE,
            "resetNativeT": 2200,
            "physicalObservation": "actual native76 POST state",
            "native44Compatible": False,
            "featureDimension": None,
            "referenceAssistance": False,
            "actionCount": 78,
            "nativeIterations": [1, 2, 3, 4, 5],
            "nativeIntervalRaw": 10,
            "nativeTimeIsRaceTime": False,
            "clockMapping": None,
            "clientParity": False,
            "arbitrarySnapshot": False,
            "runtimeRoot": str(self.config.runtime_root),
            "maxTraceBytes": self.config.max_trace_bytes or 64 * 1024 * 1024,
        }

    @property
    def journal_path(self) -> Path | None:
        return self._journal_path

    def session_reuse_status(
        self, *, minimum_remaining_seconds: float, additional_trace_bytes: int
    ) -> dict[str, Any]:
        """Fresh boundary-only quota/deadline check; never extends either limit."""
        if (
            type(minimum_remaining_seconds) not in (int, float)
            or not math.isfinite(minimum_remaining_seconds)
            or minimum_remaining_seconds < 0
            or type(additional_trace_bytes) is not int
            or additional_trace_bytes < 0
        ):
            raise ValueError(
                "finite session time reserve and nonnegative literal byte reserve required"
            )
        status: dict[str, Any] = {
            "eligible": False,
            "reason": "session unavailable or episode active",
            "minimum_remaining_seconds": minimum_remaining_seconds,
            "additional_trace_bytes": additional_trace_bytes,
        }
        if (
            self._closed
            or self._session is None
            or self._session.closed
            or not self._ended
            or self._session.episode is not None
        ):
            return status
        quota = self._session.quota_status()
        remaining = self._session.deadline - time.monotonic()
        status.update(quota, remaining_seconds=remaining)
        if remaining <= minimum_remaining_seconds:
            status["reason"] = "session deadline reserve"
        elif (
            quota["used_bytes"] + quota["close_reserve_bytes"] + additional_trace_bytes
            > quota["max_bytes"]
        ):
            status["reason"] = "session evidence reserve"
        else:
            status.update(eligible=True, reason=None)
        return status

    def _write(self, record: Mapping[str, Any]) -> str:
        if self._journal is None:
            raise NativeLifecycleError("V7 active journal required")
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._session.check_budget(len(raw) + 1)
        if self._journal.write(raw + b"\n") != len(raw) + 1:
            raise OSError("V7 short raw journal write")
        self._journal.flush()
        return hashlib.sha256(raw).hexdigest()

    def _fail(self, error: BaseException) -> None:
        try:
            self.close(error=True)
        except BaseException as cleanup:
            error.add_note(f"V7 cleanup also failed: {cleanup!r}")

    def reset(
        self, *, seed: int | None = None, duration_limit: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise NativeLifecycleError("create a fresh V7 environment after close/failure")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError("seed must be an integer or None")
            self.seed = int(seed)
        limit = self.config.duration_limit if duration_limit is None else duration_limit
        if limit is not None and (
            type(limit) is not int
            or not 10 <= limit <= 47770
            or limit % 10
            or (self.config.duration_limit is not None and limit > self.config.duration_limit)
        ):
            raise ValueError(
                "episode duration must stay within its provisioned whole native intervals"
            )
        if not self.config.allow_native:
            raise NativeExecutionDisabledError("set allow_native=True for an explicit V7 reset")
        from trackmaniarl.trackmania.native_headless._runtime.immediate import (
            ImmediateSession,
            physical_state,
        )

        try:
            if self._journal is not None:
                self._journal.close()
            if self._session is None:
                verified = verify_assets(self.config)
                self._session = ImmediateSession(self.config, verified)
            packet = self._session.reset()
            self._current = physical_state(packet)
            directory = self.config.trace_dir / "raw-journal-v7"
            if directory.is_symlink() or directory.resolve() != directory:
                raise NativeLifecycleError("V7 journal path must remain within its exact lease")
            directory.mkdir(exist_ok=True)
            self._journal_path = directory / f"episode-{self._episode:08d}.jsonl"
            self._journal = self._journal_path.open("xb")
            info = {
                "capabilities": self.capabilities,
                "clock": {
                    "source": "actual POST native counter",
                    "unit": "native-raw",
                    "raceTimeMapping": None,
                    "resetOrigin": self._current["nativeT"],
                },
                "requestedSeed": self.seed,
                "seedChangesEngineReset": False,
                "episodeDurationLimit": limit,
                "episodeProcessId": packet["pid"],
                "runtimeProvenance": packet["runtimeProvenance"],
            }
            self._write({"kind": "reset", "observation": self._current, "info": info})
            self._episode += 1
            self._ended, self._elapsed = False, 0
            self._duration_limit = limit
            return deepcopy(self._current), deepcopy(info)
        except BaseException as error:
            self._fail(error)
            raise

    def read_state(self) -> dict[str, Any]:
        if self._closed or self._current is None:
            raise NativeLifecycleError("successful V7 reset required")
        return deepcopy(self._current)

    def step_native(self, action: int, *, native_iterations: int = 5) -> dict[str, Any]:
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 78:
            raise ValueError("V7 action must follow the pinned 78-action ordering")
        if (
            isinstance(native_iterations, bool)
            or not isinstance(native_iterations, Integral)
            or not 1 <= native_iterations <= 5
        ):
            raise ValueError("V7 native_iterations must be 1..5")
        action, native_iterations = int(action), int(native_iterations)
        if self._closed or self._current is None or self._ended:
            raise NativeLifecycleError("active V7 reset required")
        from trackmaniarl.trackmania.native_headless._runtime.immediate import (
            TERMINAL_CONTRACT,
            physical_state,
            qualify_immediate,
        )

        before = self._current
        n = native_iterations
        if self._duration_limit is not None:
            n = min(n, (self._duration_limit - self._elapsed) // 10)
        if n <= 0:
            raise NativeLifecycleError("V7 explicit duration exhausted; reset required")
        try:
            packet = self._session.advance(action, n)
            events = qualify_immediate(packet, before, action, n)
            eof = packet["eof"]
            terminal = self._session.classify_terminal(packet) if eof else None
            if eof and (
                terminal is None
                or terminal.get("status") != "terminated"
                or terminal.get("terminated") is not True
                or terminal.get("truncated") is not False
            ):
                raise NativeInconclusiveError(
                    {
                        "status": "inconclusive",
                        "evidence": packet,
                        "outcome": terminal,
                        "trainingEligible": False,
                    }
                )
            state = physical_state(packet, terminal=eof)
            self._elapsed += len(events) * 10
            truncated = (
                not eof
                and self._duration_limit is not None
                and self._elapsed >= self._duration_limit
            )
            if truncated:
                self._session.end(truncated=True)
            result = {
                "status": "terminated" if eof else "truncated" if truncated else "continue",
                "terminalContract": TERMINAL_CONTRACT,
                "observation": state,
                "terminated": eof,
                "truncated": truncated,
                "actualIterations": len(events),
                "requestedIterations": native_iterations,
                "effectiveRequestedIterations": n,
                "deltaNative": len(events) * 10,
                "authorization": packet["authorization"],
                "responseFramesHex": packet["responseFramesHex"],
                "outcome": terminal,
                "eofStage": packet["eofStage"],
                "rawIntervalQualified": True,
                "trainingEligible": False,
                "reward": None,
                "truncationReason": "explicit native duration limit" if truncated else None,
            }
            digest = self._write(
                {"kind": "raw-transition", "before": before, "action": action, "result": result}
            )
            result["rawTransitionSha256"] = digest
            self._current, self._ended = state, eof or truncated
            return deepcopy(result)
        except BaseException as error:
            self._fail(error)
            raise

    def close(self, *, error: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._session is not None:
                self._session.close(error=error)
        finally:
            if self._journal is not None:
                self._journal.close()
                self._journal = None


class NativeImmediateEnvironmentFactory:
    def __init__(self, config: NativeRuntimeConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, NativeRuntimeConfig)
            else NativeRuntimeConfig.from_mapping(config)
        )
        if self.config.protocol_profile != IMMEDIATE_PROTOCOL:
            raise ValueError("explicit immediate-current/v7 factory configuration required")

    def create(self, *, seed: int) -> NativeImmediateEnvironment:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        lease = f"env-{uuid4().hex}"
        config = replace(
            self.config,
            trace_dir=self.config.trace_dir / lease,
            work_dir=self.config.work_dir / lease,
        )
        return NativeImmediateEnvironment(config, seed=int(seed))
