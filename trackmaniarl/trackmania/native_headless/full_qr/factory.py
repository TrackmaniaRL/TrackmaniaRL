"""Explicit SDK factory over a separately pinned V8 deployment and event source.

No default motor/UI values are provided. The campaign must independently
qualify and explicitly bind its source before setting this factory in a preset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from trackmaniarl.core.runtime import import_symbol
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless.assets import (
    VerifiedAssets,
    _relative_path,
    _unique_object,
)
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment,
    FullQrEnvironmentSettings,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGraphFeaturePipeline,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import (
    NativeFullQrProvider,
    NativeFullQrProviderSettings,
)
from trackmaniarl.trackmania.native_headless.full_qr.profile import ASSET_SCHEMA, PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.full_qr.trace_retention import (
    ClosedTraceRetention,
    TraceRetentionSettings,
)
from trackmaniarl.trackmania.reward_config import RewardConfig


def verify_bound_assets(settings: NativeRuntimeConfig, expected_sha256: str) -> VerifiedAssets:
    """Verify this new profile against an explicitly supplied whole-manifest pin.

    This does not change the fixed old R7/R10 verifier or grant source parity.
    The caller must obtain its expected SHA from the reviewed deployment.
    """
    if (
        settings.protocol_profile != PROTOCOL
        or type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(x not in "0123456789abcdef" for x in expected_sha256)
    ):
        raise ValueError("explicit full QR V8 protocol and manifest SHA required")
    root = settings.runtime_root
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("exact canonical native runtime directory required")
    if settings.asset_manifest.resolve(strict=True) != settings.asset_manifest:
        raise ValueError("canonical deployment manifest required")
    raw = settings.asset_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("full QR deployment manifest changed")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        manifest.get("profile") != PROFILE
        or manifest.get("schema") != ASSET_SCHEMA
        or manifest.get("runtime_root") != str(root)
        or set(manifest.get("roles", {})) != {"server", "driver", "shim", "map", "tape"}
    ):
        raise ValueError("exact full QR V8 asset schema/root/roles required")
    paths = {}
    for role, relative in manifest["roles"].items():
        relative = _relative_path(relative)
        path = root / relative
        if path.resolve(strict=True) != path or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("asset must be a canonical file inside the runtime")
        record = manifest["assets"][relative]
        if (
            path.stat().st_size != record["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"pinned full QR {role} asset differs")
        paths[role] = path
    return VerifiedAssets(PROFILE, root, MappingProxyType(paths), expected_sha256)


class FullQrEnvironmentFactory:
    """SDK create(seed) entry point with separate state and output per actor."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        source = config.get("event_source")
        if not isinstance(source, Mapping) or not source.get("class_path"):
            raise ValueError("a separately qualified motor/UI event-source component is required")
        self._source_class = import_symbol(source["class_path"])
        self._source_kwargs = dict(source.get("kwargs", {}))
        self._runtime = NativeRuntimeConfig.from_mapping(config["runtime"])
        if self._runtime.protocol_profile != PROTOCOL or not self._runtime.allow_native:
            raise ValueError("explicit separately deployed full QR V8 runtime required")
        if self._runtime.max_trace_bytes is None:
            raise ValueError("an explicit native trace quota is required")
        self._driver_episode_limit = int(config.get("driver_episode_limit", 8))
        self._episode_reserve = int(config.get("next_episode_reserve_bytes", 192 * 1024**2))
        self._manifest_sha = str(config["asset_manifest_sha256"])
        self._geometry_path = Path(config["geometry_path"])
        self._map_uid = str(config["expected_map_uid"])
        self._reward = RewardConfig(**config["reward"])
        self._archive_root = Path(config["trace_archive_root"])
        if (
            not self._archive_root.is_absolute()
            or self._archive_root.resolve() != self._archive_root
        ):
            raise ValueError(
                "an explicitly bound canonical absolute trace archive root is required"
            )
        if any(
            self._archive_root.is_relative_to(root)
            for root in (
                self._runtime.runtime_root,
                self._runtime.trace_dir,
                self._runtime.work_dir,
            )
        ):
            raise ValueError("trace archives must remain outside native runtime and session roots")
        self._keep_archives = int(config.get("keep_trace_archives", 3))
        if self._reward.maximum_race_time_s != 150:
            raise ValueError("full QR requires the historical150s reward horizon")

    def create(self, *, seed: int) -> FullQrEnvironment:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        geometry = BoundaryGeometry(self._geometry_path)
        if geometry.map_uid != self._map_uid or not geometry.map_sha256:
            raise ValueError("the historical map UID and source geometry checksum are required")
        lease = "full-qr-" + uuid4().hex
        settings = replace(
            self._runtime,
            trace_dir=self._runtime.trace_dir / lease,
            work_dir=self._runtime.work_dir / lease,
        )
        assets = verify_bound_assets(settings, self._manifest_sha)
        geometry.validate_map(assets.paths["map"])
        pipeline = NativeGraphFeaturePipeline(self._geometry_path, self._map_uid)
        retention = ClosedTraceRetention(
            TraceRetentionSettings(
                trace_parent=settings.trace_dir,
                work_parent=settings.work_dir,
                archive_root=self._archive_root / lease,
                keep_archives=self._keep_archives,
            )
        )
        events = self._source_class(**self._source_kwargs)
        provider = NativeFullQrProvider(
            NativeFullQrProviderSettings(
                settings,
                assets,
                lease,
                retention,
                self._driver_episode_limit,
                self._episode_reserve,
            ),
            events,
        )
        return FullQrEnvironment(
            provider,
            pipeline,
            FullQrEnvironmentSettings(geometry.reward_center, self._reward, lease),
        )
