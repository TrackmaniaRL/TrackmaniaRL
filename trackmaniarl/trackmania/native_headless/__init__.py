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
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment as NativeTelemetryEnvironment,
)
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    NativeObservationPipeline,
)
from trackmaniarl.trackmania.native_headless.full_qr.factory import (
    FullQrEnvironmentFactory as NativeTelemetryEnvironmentFactory,
)
from trackmaniarl.trackmania.native_headless.immediate_environment import (
    NativeImmediateEnvironment,
    NativeImmediateEnvironmentFactory,
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
    "NativeImmediateEnvironment",
    "NativeImmediateEnvironmentFactory",
    "NativeInconclusiveError",
    "NativeLifecycleError",
    "NativeObservationPipeline",
    "NativeObservationUnavailableError",
    "NativeRuntimeConfig",
    "NativeTelemetryEnvironment",
    "NativeTelemetryEnvironmentFactory",
    "RawResult",
    "RawState",
    "VerifiedAssets",
    "verify_assets",
]
