"""Opt-in SDK facade over the evidence-qualified, pinned native runtime."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets, verify_assets
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROTOCOL,
    NATIVE_PROFILE,
    NativeRuntimeConfig,
)
from trackmaniarl.trackmania.native_headless.protocols import (
    NativeBackend,
    NativeExecutionDisabledError,
    NativeInconclusiveError,
    NativeLifecycleError,
    NativeObservationUnavailableError,
    RawResult,
    RawState,
)

if TYPE_CHECKING:
    from trackmaniarl.core.data import Transition


def _create_backend(settings: NativeRuntimeConfig) -> NativeBackend:
    # The runtime module and all its process/file activity stay behind reset.
    from trackmaniarl.trackmania.native_headless._runtime import create_backend

    return cast(NativeBackend, create_backend(settings))


class NativeHeadlessEnvironment:
    """Raw-state Environment protocol; natural terminal features may be None.

    The standard EpisodeCollector is not asserted to support that terminal
    observation. Use the separately qualified ragged transition bridge there.
    """

    def __init__(self, config: NativeRuntimeConfig, *, seed: int | None = None) -> None:
        if config.protocol_profile == IMMEDIATE_PROTOCOL:
            raise ValueError("V7 requires NativeImmediateEnvironment; native44 semantics differ")
        self.config = config
        self._seed = seed
        self._backend: NativeBackend | None = None
        self._assets: VerifiedAssets | None = None
        self._last_result: RawResult | None = None
        self._closed = False
        self._failed = False
        self._ready = False
        self._ended = False

    @property
    def capabilities(self) -> dict[str, Any]:
        actual = {} if self._backend is None else deepcopy(self._backend.capabilities)
        return {
            **actual,
            "assetProfile": NATIVE_PROFILE,
            "clientParity": False,
            "completeStateSnapshot": False,
            "fullSACObservation": False,
            "referenceAssistance": True,
            "defaultCollectorTerminalNoneCompatible": False,
            "arbitraryMapOrBuildSupported": False,
            "resetProfile": self.config.reset_profile,
            "inputSchedule": "pending-then-future/v1",
            "readStateFresh": False,
        }

    @property
    def last_result(self) -> RawResult | None:
        return deepcopy(self._last_result)

    def reset(self, *, seed: int | None = None) -> tuple[RawState, dict[str, Any]]:
        if self._closed or self._failed:
            raise NativeLifecycleError("create a fresh environment after close or failure")
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("seed must be an integer or None")
            self._seed = seed
        if not self.config.allow_native:
            raise NativeExecutionDisabledError("set allow_native=True for an explicit native reset")
        self._ready = False
        if self._backend is None:
            self._assets = verify_assets(self.config)
            self._backend = _create_backend(self.config)
        try:
            state = self._backend.reset()
            NativeFeaturePipeline().transform_observation(state)
        except BaseException as error:
            self._failed = True
            try:
                self._backend.close()
            except BaseException as cleanup:
                error.add_note(f"native cleanup also failed: {cleanup!r}")
            raise
        self._ready = True
        self._ended = False
        self._last_result = None
        return deepcopy(state), {
            "resetProfile": self.config.reset_profile,
            "clock": {
                "source": "POST/reader native counter",
                "unit": "native-raw",
                "raceTimeMapping": None,
                "resetOrigin": state["nativeT"],
            },
            "capabilities": self.capabilities,
            "assetManifestSha256": None if self._assets is None else self._assets.manifest_sha256,
            "requestedSeed": self._seed,
            "seedChangesEngineReset": False,
        }

    def _active_backend(self) -> NativeBackend:
        if self._closed or self._failed or not self._ready or self._backend is None:
            raise NativeLifecycleError("a successful reset is required before reading or stepping")
        return self._backend

    def read_state(self) -> RawState:
        """Return the last qualified boundary; this does not advance or poll physics."""
        return deepcopy(self._active_backend().read_state())

    def _close_failed(self, error: BaseException) -> None:
        self._failed = True
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"native failure cleanup also failed: {cleanup!r}")

    def step_native(self, action: int, *, native_iterations: int = 5) -> RawResult:
        if type(action) is not int or not 0 <= action < 78:
            raise ValueError("action must be an integer in the pinned 78-action ordering")
        if type(native_iterations) is not int or not 1 <= native_iterations <= 5:
            raise ValueError("native_iterations must be 1..5")
        backend = self._active_backend()
        if self._ended:
            raise NativeLifecycleError("reset is required after an explicit episode boundary")
        try:
            result = backend.step(action, native_iterations=native_iterations)
        except BaseException as error:
            self._close_failed(error)
            raise
        self._last_result = deepcopy(result)  # Preserve literal evidence before qualification.
        terminated, truncated = result.get("terminated"), result.get("truncated")
        wanted_status = (
            "terminated" if terminated is True else "truncated" if truncated is True else "continue"
        )
        state = result.get("observation")
        reward = result.get("reward")
        if (
            result.get("trainingEligible") is not True
            or type(terminated) is not bool
            or type(truncated) is not bool
            or (terminated and truncated)
            or result.get("status") != wanted_status
            or not isinstance(state, dict)
            or "features" not in state
            or type(reward) not in (float, int)
            or not isfinite(cast(float, reward))
        ):
            inconclusive = NativeInconclusiveError(result)
            self._close_failed(inconclusive)
            raise inconclusive
        if not terminated or state.get("features") is not None:
            try:
                NativeFeaturePipeline().transform_observation(state)
            except NativeObservationUnavailableError as error:
                inconclusive = NativeInconclusiveError(result)
                self._close_failed(inconclusive)
                raise inconclusive from error
        self._ended = terminated or truncated
        return deepcopy(result)

    def step(self, action: int) -> tuple[RawState, float, bool, bool, dict[str, Any]]:
        result = self.step_native(action)
        return (
            deepcopy(result["observation"]),
            float(result["reward"]),
            result["terminated"],
            result["truncated"],
            {"nativeResult": result, "capabilities": self.capabilities},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False
        if self._backend is not None:
            self._backend.close()


class NativeHeadlessEnvironmentFactory:
    """Lease distinct paths per inert facade; native work starts at reset.

    A random UUID lease keeps factories and processes independent without
    touching the filesystem. The explicit trace/work settings are base folders.
    """

    def __init__(self, config: NativeRuntimeConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, NativeRuntimeConfig)
            else NativeRuntimeConfig.from_mapping(config)
        )

    def create(self, *, seed: int) -> NativeHeadlessEnvironment:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        lease = f"env-{uuid4().hex}"
        config = replace(
            self.config,
            trace_dir=self.config.trace_dir / lease,
            work_dir=self.config.work_dir / lease,
        )
        return NativeHeadlessEnvironment(config, seed=seed)


class NativeFeaturePipeline:
    """Expose only actual finite44; None remains unsupported for transformation."""

    def transform_observation(self, observation: Any) -> tuple[float, ...]:
        if (
            not isinstance(observation, dict)
            or observation.get("featureProfile") != "tmre." + NATIVE_PROFILE
        ):
            raise NativeObservationUnavailableError(
                "the pinned native44 feature profile is required"
            )
        values = observation.get("features") if isinstance(observation, dict) else None
        if not isinstance(values, (list, tuple)) or len(values) != 44:
            raise NativeObservationUnavailableError("an actual 44-feature observation is required")
        if any(type(v) not in (int, float) or not isfinite(v) for v in values):
            raise NativeObservationUnavailableError(
                "actual features must contain 44 finite numbers"
            )
        return tuple(float(v) for v in values)

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        """Keep explicit None next_observation; never manufacture terminal features."""
        return {
            "observations": [item.observation for item in transitions],
            "actions": [item.action for item in transitions],
            "rewards": [item.reward for item in transitions],
            "next_observations": [item.next_observation for item in transitions],
            "terminated": [item.terminated for item in transitions],
            "truncated": [item.truncated for item in transitions],
        }
