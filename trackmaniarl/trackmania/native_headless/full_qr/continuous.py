"""Experimental continuous actuator adapter with separately pinned sources.

This adapter does not certify analog controls or relax motor/finish asset pins.
Training requires independently qualified inputs, shim and event references.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from trackmaniarl.trackmania.native_headless._runtime.continuous import ContinuousControl
from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession,
    qualify_continuous_immediate,
)
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.full_qr.environment import FullQrEnvironment
from trackmaniarl.trackmania.native_headless.full_qr.factory import FullQrEnvironmentFactory
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import NativeFullQrProvider
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError


class ExperimentalContinuousProvider(NativeFullQrProvider):
    def _create_session(self, settings: NativeRuntimeConfig) -> ImmediateSession:
        session = super()._create_session(settings)
        session._allow_experimental_continuous = True
        return session

    def advance(self, action: Any) -> Any:
        if type(action) is not ContinuousControl:
            raise ValueError("a validated continuous control is required")
        if (
            self._closed
            or self._session is None
            or self._before is None
            or self._collector is None
            or self._identity is None
        ):
            raise NativeLifecycleError("an active native reset is required")
        try:
            return self._advance_decision(cast(Any, action))
        except BaseException as error:
            self._fail(error)
            raise

    def _decision_steering(self, action: Any) -> float:
        return float(action.steer)

    def _control_for_tick(self, action: Any, offset: int) -> Any:
        return action

    def _advance_tick(self, action: Any, stride: int) -> dict[str, Any]:
        assert self._session is not None
        return self._session.advance_continuous(action, stride)

    def _qualify_tick(  # noqa: PLR0913
        self, packet: dict[str, Any], before: dict[str, Any], action: Any, stride: int
    ) -> None:
        qualify_continuous_immediate(packet, before, action, stride)


class ExperimentalContinuousEnvironment(FullQrEnvironment):
    def step(self, action: Any) -> Any:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError("continuous action must have shape (3,) in gas, brake, steer order")
        return super().step(cast(Any, ContinuousControl(*(float(x) for x in array))))

    def _validate_action(self, action: Any) -> None:
        if type(action) is not ContinuousControl:
            raise ValueError("a validated continuous control is required")


class ExperimentalContinuousEnvironmentFactory(FullQrEnvironmentFactory):
    def validate_scheduled_evaluation_map(self, evaluation_map: Any, pipeline: Any) -> None:
        super().validate_scheduled_evaluation_map(evaluation_map, pipeline)

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("allow_experimental_continuous") is not True:
            raise ValueError("explicit experimental continuous actuator opt-in required")
        pin = config.get("continuous_shim_sha256")
        if (
            not isinstance(pin, str)
            or len(pin) != 64
            or any(x not in "0123456789abcdef" for x in pin)
        ):
            raise ValueError("explicit experimental continuous shim SHA-256 required")
        if pin == "ec861211a43ae675229ad954574d185ac1c1898e5a104ccd82a70169fad86bc0":
            raise ValueError("the original shim rejects analog pedal controls")
        self._continuous_shim_sha256 = pin
        super().__init__(config)

    def _create_provider(self, settings: Any, events: Any) -> ExperimentalContinuousProvider:
        with settings.assets.paths["shim"].open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != self._continuous_shim_sha256:
            raise ValueError("continuous shim pin differs from verified runtime")
        return ExperimentalContinuousProvider(settings, events)

    def _create_environment(
        self, provider: Any, pipeline: Any, settings: Any
    ) -> ExperimentalContinuousEnvironment:
        return ExperimentalContinuousEnvironment(provider, pipeline, settings)
