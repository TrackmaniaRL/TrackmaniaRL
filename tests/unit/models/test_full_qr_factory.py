"""Asset/source refusals; synthetic files never qualify a native deployment."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trackmaniarl.core.spec import ComponentSpec
from trackmaniarl.experiments.full_qr.runtime_profile import ASSET_SCHEMA, PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROTOCOL,
    IMMEDIATE_RESET_PROFILE,
)
from trackmaniarl.trackmania.native_headless.full_qr import factory as factory_module
from trackmaniarl.trackmania.native_headless.full_qr.factory import (
    FullQrEnvironmentFactory,
    verify_bound_assets,
)


def assets(tmp_path: Path) -> tuple[SimpleNamespace, dict, Path]:
    root = tmp_path / "runtime"
    root.mkdir()
    records = {}
    roles = {name: name + ".bin" for name in ("server", "driver", "shim", "map", "tape")}
    for name in roles.values():
        path = root / name
        path.write_bytes(b"synthetic fixture only: " + name.encode())
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": ASSET_SCHEMA,
        "profile": PROFILE,
        "runtime_root": str(root),
        "roles": roles,
        "assets": records,
    }
    path = root / "assets.json"
    path.write_text(json.dumps(manifest))
    return (
        SimpleNamespace(protocol_profile=PROTOCOL, runtime_root=root, asset_manifest=path),
        manifest,
        path,
    )


def test_missing_event_source_is_rejected_before_any_runtime_or_path_access() -> None:
    with pytest.raises(ValueError, match="event-source"):
        FullQrEnvironmentFactory({})


class RecordingSource:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.closed = 0
        self.fail_close = fail_close

    def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("source cleanup failed")


def factory_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "event_source": {"class_path": "unused:source"},
        "runtime": {
            "runtime_root": str(tmp_path / "runtime"),
            "trace_dir": str(tmp_path / "trace"),
            "work_dir": str(tmp_path / "work"),
            "asset_manifest": str(tmp_path / "manifest.json"),
            "protocol_profile": IMMEDIATE_PROTOCOL,
            "reset_profile": IMMEDIATE_RESET_PROFILE,
            "max_trace_bytes": 1048576,
            "allow_native": True,
        },
        "asset_manifest_sha256": "0" * 64,
        "geometry_path": str(tmp_path / "geometry.json"),
        "expected_map_uid": "test-map",
        "reward": {"maximum_race_time_s": 150},
        "trace_archive_root": str(tmp_path / "archives"),
    }


@pytest.mark.parametrize("setting", ["driver_episode_limit", "next_episode_reserve_bytes"])
def test_invalid_provider_budget_is_rejected_before_source_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    created: list[RecordingSource] = []

    def make_source() -> RecordingSource:
        source = RecordingSource()
        created.append(source)
        return source

    monkeypatch.setattr(factory_module, "PROTOCOL", IMMEDIATE_PROTOCOL)
    monkeypatch.setattr(factory_module, "import_symbol", lambda _: make_source)
    config = factory_config(tmp_path)
    config[setting] = 0
    with pytest.raises(ValueError, match="positive driver cohort"):
        FullQrEnvironmentFactory(config)
    assert created == []


@pytest.mark.parametrize(
    "case", [("provider", False), ("provider", True), ("environment", False), ("environment", True)]
)
def test_failed_factory_create_closes_source_once_and_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[str, bool]
) -> None:
    stage, fail_close = case
    source = RecordingSource(fail_close=fail_close)
    monkeypatch.setattr(factory_module, "PROTOCOL", IMMEDIATE_PROTOCOL)
    monkeypatch.setattr(factory_module, "import_symbol", lambda _: lambda: source)
    monkeypatch.setattr(
        factory_module,
        "BoundaryGeometry",
        lambda _: SimpleNamespace(
            map_uid="test-map", map_sha256="sha", validate_map=lambda _: None, reward_center=[]
        ),
    )
    monkeypatch.setattr(
        factory_module, "verify_bound_assets", lambda *_: SimpleNamespace(paths={"map": tmp_path})
    )
    monkeypatch.setattr(FullQrEnvironmentFactory, "_create_pipeline", lambda _: object())

    class FakeProvider:
        def __init__(self, _settings: Any, events: RecordingSource) -> None:
            if stage == "provider":
                raise ValueError("provider failed")
            self.events = events

        def close(self) -> None:
            self.events.close()

    def fail_environment(*_: Any) -> None:
        raise ValueError("environment failed")

    monkeypatch.setattr(factory_module, "NativeFullQrProvider", FakeProvider)
    monkeypatch.setattr(factory_module, "FullQrEnvironment", fail_environment)
    factory = FullQrEnvironmentFactory(factory_config(tmp_path))
    with pytest.raises(ValueError, match=f"{stage} failed") as error:
        factory.create(seed=0)
    assert source.closed == 1
    if fail_close:
        assert any("source cleanup failed" in note for note in error.value.__notes__)


@pytest.mark.parametrize("seed", [0, np.int32(-1), np.random.default_rng(42).integers(100)])
def test_successful_factory_transfers_source_ownership_to_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: Any
) -> None:
    from trackmaniarl.trackmania.native_headless.full_qr import native_provider

    source = RecordingSource()
    monkeypatch.setattr(factory_module, "PROTOCOL", IMMEDIATE_PROTOCOL)
    monkeypatch.setattr(native_provider, "PROTOCOL", IMMEDIATE_PROTOCOL)
    monkeypatch.setattr(factory_module, "import_symbol", lambda _: lambda: source)
    monkeypatch.setattr(
        factory_module,
        "BoundaryGeometry",
        lambda _: SimpleNamespace(
            map_uid="test-map", map_sha256="sha", validate_map=lambda _: None, reward_center=[]
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "verify_bound_assets",
        lambda *_: SimpleNamespace(paths={"map": tmp_path}, profile=PROFILE),
    )
    monkeypatch.setattr(FullQrEnvironmentFactory, "_create_pipeline", lambda _: object())
    environment = FullQrEnvironmentFactory(factory_config(tmp_path)).create(seed=seed)
    assert source.closed == 0
    environment.close()
    environment.close()
    assert source.closed == 1


@pytest.mark.parametrize("seed", [True, False, np.bool_(True), 1.0, "1", None])
def test_invalid_factory_seed_rejected_before_accessing_geometry(seed: Any) -> None:
    factory = object.__new__(FullQrEnvironmentFactory)
    with pytest.raises(ValueError, match="seed must be an integer"):
        factory.create(seed=seed)


def test_custom_pipeline_is_fresh_per_environment_and_requires_all_terminal_methods() -> None:
    from trackmaniarl.trackmania.native_headless import NativeTelemetryEnvironmentFactory

    assert NativeTelemetryEnvironmentFactory is FullQrEnvironmentFactory
    # Exercise component creation independently of asset deployment. No fake
    # asset or stub is treated as permission to start the engine.
    factory = object.__new__(NativeTelemetryEnvironmentFactory)
    factory._pipeline_spec = ComponentSpec(
        class_path="tests.unit.models.test_full_qr_factory:CustomPipeline"
    )
    assert factory._create_pipeline() is not factory._create_pipeline()
    factory._pipeline_spec = ComponentSpec(class_path="builtins:dict")
    with pytest.raises(TypeError, match="all frame forms"):
        factory._create_pipeline()


class CustomPipeline:
    def reset_episode(self) -> None:
        pass

    def transform_observation(self, observation: Any) -> dict[str, np.ndarray]:
        return {"telemetry": np.asarray(observation.channels, dtype=np.float32)}

    def transform_terminal(self, observation: Any) -> dict[str, np.ndarray]:
        return self.transform_observation(observation)

    def transform_committed_terminal(self, observation: Any) -> dict[str, np.ndarray]:
        return self.transform_observation(observation)


def test_manifest_pin_and_actual_asset_bytes_both_remain_required(tmp_path: Path) -> None:
    settings, _, path = assets(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = verify_bound_assets(settings, expected)
    with pytest.raises(TypeError):
        verified.paths["server"] = tmp_path
    with pytest.raises(ValueError, match="manifest changed"):
        verify_bound_assets(settings, "0" * 64)
    verified.paths["map"].write_bytes(b"changed actual bytes")
    with pytest.raises(ValueError, match="asset differs"):
        verify_bound_assets(settings, expected)


def test_correctly_hashed_old_diagnostic_manifest_is_not_a_full_v8_manifest(tmp_path: Path) -> None:
    settings, manifest, path = assets(tmp_path)
    manifest["schema"] = "tmrl.experimental-long150-diagnostic.assets/v1"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema/root/roles"):
        verify_bound_assets(settings, hashlib.sha256(path.read_bytes()).hexdigest())


def test_duplicate_json_keys_cannot_override_the_pinned_deployment_schema(tmp_path: Path) -> None:
    settings, _, path = assets(tmp_path)
    raw = path.read_text()
    path.write_text('{"schema": "wrong",' + raw[1:])
    with pytest.raises(ValueError, match="duplicate manifest key"):
        verify_bound_assets(settings, hashlib.sha256(path.read_bytes()).hexdigest())
