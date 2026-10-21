"""Opt-in pinned native headless profile; construction never starts the engine."""

from trackmaniarl.trackmania.native_headless.assets import (
    NativeAssetError,
    VerifiedAssets,
    verify_assets,
)
from trackmaniarl.trackmania.native_headless.config import (
    ASSET_SCHEMA,
    NATIVE_PROFILE,
    NativeConfigurationError,
    NativeRuntimeConfig,
)
from trackmaniarl.trackmania.native_headless.environment import (
    NativeFeaturePipeline,
    NativeHeadlessEnvironment,
    NativeHeadlessEnvironmentFactory,
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

__all__ = [
    "ASSET_SCHEMA",
    "NATIVE_PROFILE",
    "NativeAssetError",
    "NativeBackend",
    "NativeConfigurationError",
    "NativeExecutionDisabledError",
    "NativeFeaturePipeline",
    "NativeHeadlessEnvironment",
    "NativeHeadlessEnvironmentFactory",
    "NativeInconclusiveError",
    "NativeLifecycleError",
    "NativeObservationUnavailableError",
    "NativeRuntimeConfig",
    "RawResult",
    "RawState",
    "VerifiedAssets",
    "verify_assets",
]
