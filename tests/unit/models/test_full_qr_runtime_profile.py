"""Separate V8 domain and control semantics; real150s proof is external."""

import ast
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from trackmaniarl.experiments.full_qr.runtime_profile import PROTOCOL, transform_source
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig

ROOT = Path(__file__).parents[3]
PREFIX = "trackmaniarl/trackmania/native_headless/"


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
