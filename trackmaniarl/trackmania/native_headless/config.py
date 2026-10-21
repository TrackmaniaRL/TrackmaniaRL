"""Explicit configuration for the pinned Linux native validator profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_SCHEMA = "tmrl.native-headless.assets/v1"
NATIVE_PROFILE = "native16-pending-reference78/44/v2"


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

    def __post_init__(self) -> None:
        for name in ("runtime_root", "trace_dir", "work_dir", "asset_manifest"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise NativeConfigurationError(f"{name} must be an explicit pathlib.Path")
            if not (value.is_absolute() or PurePosixPath(str(value)).is_absolute()):
                raise NativeConfigurationError(f"{name} must be absolute")
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
