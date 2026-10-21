"""Actor-facing qualification and owned raw journals over the existing native facade."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, BinaryIO

from trackmaniarl.distributed.metric_diagnostics import METRIC_DIAGNOSTICS_KEY
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.environment import (
    NativeHeadlessEnvironment,
    NativeHeadlessEnvironmentFactory,
)
from trackmaniarl.trackmania.native_headless.protocols import (
    NativeExecutionDisabledError,
    NativeInconclusiveError,
    NativeLifecycleError,
    RawState,
)
from trackmaniarl.trackmania.native_headless.training_contract import (
    NATIVE_INTERVAL_KEY,
    NATIVE_INTERVAL_SCHEMA,
    NativeTensorFeaturePipeline,
    canonical_raw_transition,
    qualify_raw_transition,
)
from trackmaniarl.trackmania.native_headless.training_metrics import native_metric_diagnostics


class NativeTrainingEnvironment:
    """Delegate all engine work; qualify and journal before returning an actor row.

    Timeout/transport/storage failures abort this adapter rather than entering the
    generic actor's recoverable interruption-to-truncation path. No replay rows
    are created by reset or by writing journal records.
    """

    def __init__(self, environment: NativeHeadlessEnvironment) -> None:
        self.environment = environment
        self.config = environment.config
        self._before: RawState | None = None
        self._journal: BinaryIO | None = None
        self._journal_path: Path | None = None
        self._episode = 0
        self._closed = False
        self._ended = False
        self._steps = 0
        duration = self.config.duration_limit
        self._step_cap = None if duration is None else (duration + 49) // 50

    @property
    def journal_path(self) -> Path | None:
        return self._journal_path

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            **self.environment.capabilities,
            "trainingIntervalContract": NATIVE_INTERVAL_SCHEMA,
            "completeDistributedTrainingCompatible": False,
            "trainingEpisodeStepCap": self._step_cap,
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._journal is None:
            raise NativeLifecycleError("successful reset and journal required")
        raw = (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        written = self._journal.write(raw)
        if written != len(raw):
            raise OSError("short native training journal write")
        self._journal.flush()

    def _write_transition(self, before: RawState, action: int, result: Mapping[str, Any]) -> None:
        raw = canonical_raw_transition(before, action, result)
        self._write(
            {
                "kind": "raw-transition",
                "rawTransitionSha256": hashlib.sha256(raw).hexdigest(),
                "transition": {"before": before, "action": action, "result": result},
            }
        )

    def _close_journal(self) -> None:
        journal, self._journal = self._journal, None
        if journal is not None:
            journal.close()

    def _fail(self, error: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"native training cleanup also failed: {cleanup!r}")

    def _open_journal(self) -> None:
        trace = self.config.trace_dir
        if not trace.is_dir() or trace.resolve(strict=True) != trace:
            raise NativeLifecycleError("an exact owned native trace directory is required")
        directory = trace / "training-journal"
        if directory.is_symlink() or directory.resolve() != directory:
            raise NativeLifecycleError("native training journal must not be a symlink/alias")
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or directory.resolve(strict=True) != directory:
            raise NativeLifecycleError("native training journal ownership changed")
        self._journal_path = directory / f"episode-{self._episode:08d}.jsonl"
        self._journal = self._journal_path.open("xb")

    def reset(self, *, seed: int | None = None) -> tuple[RawState, dict[str, Any]]:
        if self._closed:
            raise NativeLifecycleError("create a fresh training environment after close/failure")
        try:
            self._close_journal()
            state, info = self.environment.reset(seed=seed)
            NativeTensorFeaturePipeline().transform_observation(state)
            self._open_journal()
            self._write({"kind": "reset", "observation": state, "info": info})
            self._episode += 1
            self._before, self._ended, self._steps = deepcopy(state), False, 0
            return state, info
        except BaseException as error:
            self._fail(error)
            if isinstance(
                error,
                (KeyboardInterrupt, SystemExit, NativeExecutionDisabledError, NativeLifecycleError),
            ):
                raise
            raise NativeLifecycleError("native training reset failed; session closed") from error

    def read_state(self) -> RawState:
        if self._closed or self._before is None:
            raise NativeLifecycleError("successful training reset required")
        return self.environment.read_state()

    def step(self, action: int) -> tuple[RawState, float, bool, bool, dict[str, Any]]:
        if type(action) is not int or not 0 <= action < 78:
            raise ValueError("action must be an integer in the pinned 78-action ordering")
        if self._closed or self._before is None or self._ended:
            raise NativeLifecycleError("active training reset required")
        before = deepcopy(self._before)
        result: Mapping[str, Any] | None = None
        try:
            result = self.environment.step_native(action)
            self._write_transition(before, action, result)
            record = qualify_raw_transition(before, action, result)
            self._steps += 1
            if (
                self._step_cap is not None
                and self._steps >= self._step_cap
                and record["status"] == "continue"
            ):
                raise NativeLifecycleError("native endpoint required before collector step limit")
            self._write({"kind": "qualified", NATIVE_INTERVAL_KEY: record})
            after = deepcopy(result["observation"])
            terminated, truncated = record["terminated"], record["truncated"]
            info: dict[str, Any] = {
                NATIVE_INTERVAL_KEY: record,
                METRIC_DIAGNOSTICS_KEY: native_metric_diagnostics(result, record),
            }
            if terminated:
                info.update(
                    termination_reason="finished",
                    race_time_ms=result["outcome"]["computedResult"]["Time"],
                )
            elif truncated:
                info["termination_reason"] = "native_duration_limit"
            self._before, self._ended = deepcopy(after), terminated or truncated
            return after, float(result["reward"]), terminated, truncated, info
        except BaseException as error:
            if result is None and isinstance(error, NativeInconclusiveError):
                try:
                    self._write_transition(before, action, error.raw_result)
                except BaseException as persistence:
                    error.add_note(f"inconclusive raw journal also failed: {persistence!r}")
            self._fail(error)
            if isinstance(
                error,
                (KeyboardInterrupt, SystemExit, NativeInconclusiveError, NativeLifecycleError),
            ):
                raise
            raise NativeLifecycleError(
                "native training step failed; inspect owned journal"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._before = None
        try:
            self._close_journal()
        finally:
            self.environment.close()


class NativeTrainingEnvironmentFactory:
    """Reuse the facade's per-instance UUID leases and lazy execution boundary."""

    def __init__(self, config: NativeRuntimeConfig | Mapping[str, Any]) -> None:
        self._factory = NativeHeadlessEnvironmentFactory(config)

    def create(self, *, seed: int) -> NativeTrainingEnvironment:
        return NativeTrainingEnvironment(self._factory.create(seed=seed))
