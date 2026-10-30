"""Contract checks only; no native engine or historical qualification is run."""

import hashlib
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


@contextmanager
def snapshot_modules() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Import the snapshot without leaking generic module names to other tests."""
    names = ("environment", "orientation", "multi_goal")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    original_path = sys.path[:]
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(HERE))
        env = importlib.import_module("environment")
        orientation = importlib.import_module("orientation")
        yield env, orientation
    finally:
        sys.path[:] = original_path
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = cast(ModuleType, module)


def test_historical_snapshot_bytes_are_unchanged() -> None:
    expected = {
        "environment.py": "0bfe69968b36856bb47c4296dcf6dabfebd5be40b0c5057f3ff9b91d03997188",
        "multi_goal.py": "b1501bd83eff6da58217a905fbe6bfbc1d685453260b9214fa68e28c4598f129",
        "orientation.py": "4b7d07323b577cca8a929f16d9ed08c158bcf63cad2d88693bcb04631a19f746",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest


def test_observation_contract_and_rotation_sign() -> None:
    with snapshot_modules() as (environment, orientation):
        _check_observation(environment, orientation)


def _check_observation(environment: ModuleType, orientation: ModuleType) -> None:
    physical = {
        "quaternionWXYZ": [1.0, 0.0, 0.0, 0.0],
        "physicalPosition": [0.0, 0.0, 0.0],
        "physicalVelocity": [0.0, 0.0, 0.0],
        "rpm": 0.0,
        "gearRaw": 1,
        "gas": 0.0,
        "brake": 0.0,
        "steer": 0.0,
        "filteredSteer": 0.0,
    }
    result = environment.observation({"physical": physical, "nativeT": 2200})
    assert result.shape == (22,)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[1:10], np.eye(3).reshape(-1))
    np.testing.assert_array_equal(
        orientation.rotation_matrix([1, 0, 0, 0]), orientation.rotation_matrix([-1, 0, 0, 0])
    )
    with pytest.raises(ValueError, match="zero quaternion"):
        orientation.rotation_matrix([0, 0, 0, 0])


def test_float_session_keeps_required_base_methods() -> None:
    with snapshot_modules() as (environment, _):
        _check_methods(environment)


def _check_methods(environment: ModuleType) -> None:
    for name in (
        "command",
        "arm",
        "request",
        "response",
        "poll_posts",
        "packet",
        "read_mem",
        "end",
    ):
        assert callable(getattr(environment.FloatSession, name, None)), name


def test_imports_restore_prior_modules_and_path_even_on_error() -> None:
    prior = {name: ModuleType(name) for name in ("environment", "orientation", "multi_goal")}
    with patch.dict(sys.modules, prior):
        original_path = sys.path[:]

        def fail_during_import() -> None:
            with snapshot_modules() as (env, _):
                assert env is not prior["environment"]
                raise RuntimeError("test failure")

        with pytest.raises(RuntimeError, match="test failure"):
            fail_during_import()
        assert sys.path == original_path
        assert all(sys.modules[name] is module for name, module in prior.items())
