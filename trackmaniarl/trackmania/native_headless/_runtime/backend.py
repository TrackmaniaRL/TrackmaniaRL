from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from trackmaniarl.trackmania.native_headless._runtime.artifacts import ROLE_PATHS

if TYPE_CHECKING:
    from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


class RawNativeBackend:
    """Raw qualified mappings; the public facade supplies typed observations."""

    def __init__(self, settings: NativeRuntimeConfig) -> None:
        self._settings = settings
        self._env: Any = None
        self._closed = False

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "profile": "tmre.long4152-declared40000/v6",
            "terminalContract": "tmrl.native-headless.terminal/v2",
            "featureProfile": "tmre.native16-pending-reference78/44/v2",
            "featureDimension": 44,
            "actionCount": 78,
            "nativeIterations": [1, 2, 3, 4, 5],
            "nativeIntervalRaw": 10,
            "partiallyObserved": True,
            "referenceAssistance": True,
            "arbitrarySnapshot": False,
            "restoreSnapshot": False,
            "clientParity": False,
            "nativeTimeIsRaceTime": False,
            "terminalFlagsCanBeNone": True,
            "runtimeRoot": str(self._settings.runtime_root),
            "portablePreviewProfile": "tmre.preview200/root-configured/v1",
            "relocatedRuntimeQualifiedBySourceAlone": False,
        }

    def reset(self) -> dict[str, Any]:
        if self._closed:
            raise ValueError("native backend closed")
        if self._env is None:
            settings = self._settings
            if not settings.allow_native:
                raise ValueError("explicit allow_native required before reset")
            if sys.platform != "linux" or not settings.runtime_root.is_absolute():
                raise ValueError(
                    "experimental prebuilt driver requires its fixed Linux/WSL runtime_root"
                )
            from trackmaniarl.trackmania.native_headless._runtime.session import make_native_env
            from trackmaniarl.trackmania.native_headless.assets import verify_assets

            verified = verify_assets(settings)
            if any(
                (
                    verified.paths[role] != settings.runtime_root / path
                    for role, path in ROLE_PATHS.items()
                )
            ):
                raise ValueError("experimental fixed-layout profile role paths are unsupported")
            self._env = make_native_env(settings, verified)
        value: dict[str, Any] = self._env.reset()
        return value

    def read_state(self) -> dict[str, Any]:
        if self._closed or self._env is None:
            raise ValueError("active reset required before read_state")
        value: dict[str, Any] = self._env.read_state()
        return value

    def step(self, action: int, native_iterations: int = 5) -> dict[str, Any]:
        if self._closed or self._env is None:
            raise ValueError("active reset required before step")
        value: dict[str, Any] = self._env.step(action, native_iterations)
        return value

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._env is not None:
                self._env.close()


def create_backend(settings: NativeRuntimeConfig) -> RawNativeBackend:
    return RawNativeBackend(settings)
