"""Explicit configuration for the pinned Linux native validator profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_SCHEMA = "tmrl.native-headless.assets/v1"
NATIVE_PROFILE = "native16-pending-reference78/44/v2"
IMMEDIATE_ASSET_SCHEMA = "tmrl.native-immediate.assets/v1"
IMMEDIATE_PROFILE = "native-immediate-kinematics6/v7"
IMMEDIATE_PROTOCOL = "immediate-current/v7"
IMMEDIATE_RESET_PROFILE = "neutral-race-start/v7"


class NativeConfigurationError(ValueError):
    """The requested profile or explicit artifact paths are invalid."""


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeRuntimeConfig:
    """Construction validates values only; reset performs all filesystem I/O.

    The initial driver profile supports its fixed Linux artifact layout only.
    This config does not imply portability to arbitrary servers, maps or clocks.
    """

    runtime_root: Path
    trace_dir: Path
    work_dir: Path
    asset_manifest: Path
    timeout_s: float = 240
    duration_limit: int | None = 39950
    allow_native: bool = False
    deployment_manifest: Path | None = None
    deployment_manifest_sha256: str | None = None
    reset_profile: str = "neutral-prelude/v1"
    protocol_profile: str = "pending-future/v6"
    max_trace_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_trace_bytes is not None and (
            type(self.max_trace_bytes) is not int or self.max_trace_bytes <= 0
        ):
            raise NativeConfigurationError("max_trace_bytes must be a positive integer or None")
        if self.protocol_profile == "pending-future/v6" and self.max_trace_bytes is not None:
            raise NativeConfigurationError("max_trace_bytes is supported only by explicit V7")
        if self.protocol_profile not in ("pending-future/v6", IMMEDIATE_PROTOCOL):
            raise NativeConfigurationError("unsupported native wire protocol profile")
        if self.protocol_profile == IMMEDIATE_PROTOCOL:
            if self.reset_profile != IMMEDIATE_RESET_PROFILE:
                raise NativeConfigurationError("V7 requires explicit neutral-race-start/v7 reset")
        elif self.reset_profile not in ("legacy-gas-prelude/v1", "neutral-prelude/v1"):
            raise NativeConfigurationError("unsupported reset profile")
        for name in ("runtime_root", "trace_dir", "work_dir", "asset_manifest"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise NativeConfigurationError(f"{name} must be an explicit pathlib.Path")
            if not (value.is_absolute() or PurePosixPath(value.as_posix()).is_absolute()):
                raise NativeConfigurationError(f"{name} must be absolute")
            if name != "asset_manifest" and str(value).replace("\\", "/").startswith("/mnt/"):
                raise NativeConfigurationError(f"{name} must live on Linux/ext4, outside /mnt")
        if self.deployment_manifest is not None and not isinstance(self.deployment_manifest, Path):
            raise NativeConfigurationError("deployment_manifest must be pathlib.Path")
        if self.deployment_manifest_sha256 is not None and (
            not isinstance(self.deployment_manifest_sha256, str)
            or len(self.deployment_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.deployment_manifest_sha256)
        ):
            raise NativeConfigurationError("deployment_manifest_sha256 must be lowercase SHA256")
        if self.trace_dir == self.work_dir:
            raise NativeConfigurationError("trace_dir and work_dir must differ")
        if type(self.allow_native) is not bool:
            raise NativeConfigurationError("allow_native must be an explicit boolean")
        if type(self.timeout_s) not in (int, float) or not 0 < self.timeout_s <= 240:
            raise NativeConfigurationError("the pinned session deadline is at most 240 seconds")
        duration = self.duration_limit
        if duration is not None and (
            type(duration) is not int or not 0 < duration <= 47770 or duration % 10
        ):
            raise NativeConfigurationError(
                "duration_limit must be None or 10..47770 in steps of 10 raw units"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeRuntimeConfig:
        """Parse component kwargs without reading paths or optional dependencies."""
        fields = dict(value)
        for name in ("runtime_root", "trace_dir", "work_dir", "asset_manifest"):
            path = fields.get(name)
            if not isinstance(path, (str, Path)):
                raise NativeConfigurationError(f"{name} must be an explicit path")
            fields[name] = Path(path)
        if fields.get("deployment_manifest") is not None:
            fields["deployment_manifest"] = Path(fields["deployment_manifest"])
        try:
            return cls(**fields)
        except TypeError as error:
            raise NativeConfigurationError(f"invalid native configuration: {error}") from error
