"""Native map binding is checked before a provider is created, without starting it."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.trackmania.native_headless.full_qr import factory as factory_module
from trackmaniarl.trackmania.reward_config import RewardConfig


@dataclass(frozen=True)
class Runtime:
    trace_dir: Path
    work_dir: Path


@pytest.fixture
def factory_and_map(tmp_path, monkeypatch):
    geometry = _asset(tmp_path)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    instance = object.__new__(factory_module.FullQrEnvironmentFactory)
    instance._geometry_path = geometry
    instance._map_uid = "trackmaniarl-test"
    instance._runtime = Runtime(tmp_path / "trace", tmp_path / "work")
    instance._manifest_sha = "ignored"
    instance._reward_reference = None
    instance._archive_root = tmp_path / "archives"
    instance._keep_archives = 1
    instance._driver_episode_limit = 1
    instance._episode_reserve = 1
    instance._reward = RewardConfig(maximum_race_time_s=150)
    instance._source_class = Mock(return_value=object())
    instance._source_kwargs = {}
    instance._create_pipeline = Mock(return_value=object())
    instance._create_provider = Mock(return_value=object())
    instance._create_environment = Mock(return_value="environment")
    monkeypatch.setattr(
        factory_module, "verify_bound_assets", lambda *_: SimpleNamespace(paths={"map": map_path})
    )
    spec = EvaluationMapSpec(
        id="map", map_path=map_path, geometry_path=geometry, expected_map_uid="trackmaniarl-test"
    )
    return instance, spec


def test_evaluation_map_and_training_create_use_same_factory(factory_and_map):
    factory, spec = factory_and_map
    assert factory.create(seed=17, evaluation_map=spec) == "environment"
    assert factory.create(seed=17) == "environment"
    assert factory._create_provider.call_count == 2


@pytest.mark.parametrize("field", ["expected_map_uid", "geometry_path", "map_path"])
def test_rejects_wrong_map_binding_before_provider(factory_and_map, tmp_path, field):
    factory, spec = factory_and_map
    other = tmp_path / "other"
    other.write_bytes(b"different map or geometry")
    changes = {field: "other-map" if field == "expected_map_uid" else other}
    wrong = spec.model_copy(update=changes)
    with pytest.raises(ValueError):
        factory.create(seed=17, evaluation_map=wrong)
    factory._create_provider.assert_not_called()
    factory._source_class.assert_not_called()


def test_rejects_evaluation_with_only_reward_points(factory_and_map, tmp_path):
    factory, _ = factory_and_map
    spec = EvaluationMapSpec(
        id="map", reward_points_path=tmp_path / "points", expected_map_uid="trackmaniarl-test"
    )
    with pytest.raises(ValueError, match="pinned map and geometry"):
        factory.create(seed=17, evaluation_map=spec)
    factory._create_provider.assert_not_called()


def test_rejects_runtime_map_that_differs_from_evaluation(factory_and_map, tmp_path, monkeypatch):
    factory, spec = factory_and_map
    different = tmp_path / "runtime-map"
    different.write_bytes(b"different runtime map")
    monkeypatch.setattr(
        factory_module, "verify_bound_assets", lambda *_: SimpleNamespace(paths={"map": different})
    )
    with pytest.raises(ValueError, match="checksum"):
        factory.create(seed=17, evaluation_map=spec)
    factory._create_provider.assert_not_called()
