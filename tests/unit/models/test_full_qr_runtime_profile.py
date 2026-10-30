"""Separate V8 domain and control semantics; real150s proof is external."""

import ast
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from trackmaniarl.experiments.full_qr import runtime_profile
from trackmaniarl.experiments.full_qr.runtime_profile import PROTOCOL, transform_source
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig

ROOT = Path(__file__).parents[3]
PREFIX = "trackmaniarl/trackmania/native_headless/"


@pytest.fixture
def external_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="tmrl-v8-profile-test-") as tmp:
        yield Path(tmp).resolve()


def test_fresh_deployment_factory_accepts_v8_without_protocol_monkeypatch(
    external_tmp_path: Path,
) -> None:
    """A fresh interpreter must load the transformed package, not the original V7 tree."""
    destination = external_tmp_path / "deployment"
    receipt = runtime_profile.build_source_deployment(ROOT, destination)
    assert receipt["qualification"] is False
    script = """
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import trackmaniarl
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.full_qr.factory import FullQrEnvironmentFactory
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import (
    NativeFullQrProvider, NativeFullQrProviderSettings,
)
from trackmaniarl.trackmania.native_headless.full_qr.profile import PROFILE, PROTOCOL
from types import SimpleNamespace
assert pathlib.Path(trackmaniarl.__file__).resolve().is_relative_to(root)
assert PROTOCOL == "full-qr-current150/v8"
base = pathlib.Path(sys.argv[2]).resolve()
config = {
    "event_source": {"class_path": "builtins:object"},
    "runtime": {"runtime_root": str(base / "runtime"), "trace_dir": str(base / "trace"),
        "work_dir": str(base / "work"), "asset_manifest": str(base / "assets.json"),
        "protocol_profile": PROTOCOL, "reset_profile": "neutral-race-start/v7",
        "duration_limit": 150000, "max_trace_bytes": 1048576, "allow_native": True},
    "asset_manifest_sha256": "0" * 64, "geometry_path": str(base / "geometry.json"),
    "expected_map_uid": "test-map", "reward": {"maximum_race_time_s": 150},
    "trace_archive_root": str(base / "archives"),
}
factory = FullQrEnvironmentFactory(config)
assert factory._runtime.protocol_profile == PROTOCOL
from trackmaniarl.trackmania.native_headless._runtime import immediate
assert immediate.IMMEDIATE_PROTOCOL == PROTOCOL
assert immediate.IMMEDIATE_PROFILE == PROFILE
session = immediate.ImmediateSession.__new__(immediate.ImmediateSession)
session.events = {10: b'a', 30: b'c', 20: b'b'}
assert session._decision_events(10, 30) == [b'b', b'c']
try:
    session._decision_events(10, 20)
except Exception:
    pass
else:
    raise AssertionError("future native boundary was not rejected")
provider = NativeFullQrProvider(
    NativeFullQrProviderSettings(factory._runtime, VerifiedAssets(PROFILE, base, {}, "0" * 64),
        "test-actor", SimpleNamespace()), SimpleNamespace()
)
assert provider.settings.protocol_profile == PROTOCOL
try:
    NativeRuntimeConfig.from_mapping(
        {**config["runtime"], "protocol_profile": "immediate-current/v7"}
    )
except ValueError:
    pass
else:
    raise AssertionError("V8 deployment unexpectedly accepted V7")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(destination), str(external_tmp_path)],
        cwd=external_tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert NativeRuntimeConfig.__module__ == "trackmaniarl.trackmania.native_headless.config"


def deployment_source(root: Path) -> Path:
    """Minimal, independently writable source tree with valid V8 prerequisites."""
    source = root / "source"
    for relative, substitutions in runtime_profile._REPLACEMENTS.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(before for before, _ in substitutions), encoding="utf-8")
    return source


def test_source_deployment_reads_low_space_source_without_modifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = deployment_source(tmp_path)
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    target_parent = tmp_path / "other-volume"
    target_parent.mkdir()
    target = target_parent / "deployment"
    queried = []

    def free_space(path: Path) -> Any:
        queried.append(path)
        return type("Usage", (), {"free": 0 if path == source else 5 * 1024**3})()

    monkeypatch.setattr(runtime_profile.shutil, "disk_usage", free_space)
    receipt = runtime_profile.build_source_deployment(source, target)

    assert queried == [target_parent]
    assert receipt["qualification"] is False
    assert (target / "source-deployment.json").is_file()
    assert {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    } == before


def test_source_deployment_requires_destination_reserve_on_same_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = deployment_source(tmp_path)
    destination = tmp_path / "deployment"
    monkeypatch.setattr(
        runtime_profile.shutil,
        "disk_usage",
        lambda _: type("Usage", (), {"free": 4 * 1024**3 - 1})(),
    )
    with pytest.raises(OSError, match="deployment volume requires"):
        runtime_profile.build_source_deployment(source, destination)
    assert not destination.exists()


def test_source_deployment_allows_same_volume_with_destination_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = deployment_source(tmp_path)
    destination = tmp_path / "deployment"
    monkeypatch.setattr(
        runtime_profile.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 5 * 1024**3})()
    )
    receipt = runtime_profile.build_source_deployment(source, destination)
    assert (destination / "source-deployment.json").is_file()
    assert receipt["qualification"] is False


def test_source_deployment_rejects_a_destination_inside_its_source(tmp_path: Path) -> None:
    source = deployment_source(tmp_path)
    with pytest.raises(ValueError, match="outside the source"):
        runtime_profile.build_source_deployment(source, source / "deployment")
    assert not (source / "deployment").exists()


def test_v8_domain_cannot_relax_the_original_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = PREFIX + "config.py"
    original = (ROOT / relative).read_text()
    module = ModuleType("full_qr_config_domain_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(transform_source(relative, original), relative, "exec"), module.__dict__)
    paths = {
        name: tmp_path / name
        for name in ("runtime_root", "trace_dir", "work_dir", "asset_manifest")
    }
    with pytest.raises(ValueError, match="unsupported native wire"):
        NativeRuntimeConfig(**paths, protocol_profile=PROTOCOL)
    with pytest.raises(ValueError, match="duration_limit"):
        NativeRuntimeConfig(
            **paths,
            protocol_profile="immediate-current/v7",
            reset_profile="neutral-race-start/v7",
            duration_limit=150000,
        )
    settings = module.NativeRuntimeConfig(
        **paths,
        protocol_profile=PROTOCOL,
        reset_profile="neutral-race-start/v7",
        duration_limit=150000,
    )
    assert settings.duration_limit == 150000
    with pytest.raises(ValueError, match="duration_limit"):
        module.NativeRuntimeConfig(
            **paths,
            protocol_profile=PROTOCOL,
            reset_profile="neutral-race-start/v7",
            duration_limit=150010,
        )
    assert (ROOT / relative).read_text() == original


def test_calibrated_78_actions_preserve_gas_and_brake_tap_semantics() -> None:
    relative = PREFIX + "_runtime/features.py"
    transformed = transform_source(relative, (ROOT / relative).read_text())
    node = next(
        x
        for x in ast.parse(transformed).body
        if isinstance(x, ast.FunctionDef) and x.name == "action_sgb"
    )
    namespace = {"struct": struct, "Any": Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    calibrated = namespace["action_sgb"]
    steering = []
    for action in range(78):
        before, after = action_sgb(action), calibrated(action)
        assert len(after) == 5
        assert [x[1:] for x in after] == [x[1:] for x in before]
        assert len({x[0] for x in after}) == 1
        if action % 6 == 0:
            steering.append(after[0][0])
    assert len(set(steering)) == 13
    assert steering == sorted(steering)
    assert steering[0] == -1.0
    assert steering[-1] == 1.0
    assert steering[6] == 0.0
    # Measured Windows plateau: requested0.5 becomes signed byte63, not64.
    assert steering[9] == struct.unpack("<f", struct.pack("<f", 63 / 127))[0]
