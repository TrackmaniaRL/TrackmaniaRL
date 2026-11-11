"""Native map binding is checked before a provider is created, without starting it."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.spec import ComponentSpec, EvaluationMapSpec
from trackmaniarl.trackmania.native_headless.full_qr import factory as factory_module
from trackmaniarl.trackmania.reward_config import RewardConfig


@dataclass(frozen=True)
class Runtime:
    trace_dir: Path
    work_dir: Path


@pytest.fixture
def factory_and_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec]:
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
    instance._pipeline_spec = None
    instance._create_provider = Mock(return_value=object())
    instance._create_environment = Mock(return_value="environment")
    monkeypatch.setattr(
        factory_module, "verify_bound_assets", lambda *_: SimpleNamespace(paths={"map": map_path})
    )
    spec = EvaluationMapSpec(
        id="map", map_path=map_path, geometry_path=geometry, expected_map_uid="trackmaniarl-test"
    )
    return instance, spec


def test_evaluation_map_and_training_create_use_same_factory(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
) -> None:
    factory, spec = factory_and_map
    assert factory.create(seed=17, evaluation_map=spec) == "environment"
    assert factory.create(seed=17) == "environment"
    assert factory._create_provider.call_count == 2


def test_scheduled_evaluation_attests_pinned_map_without_starting_provider(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
) -> None:
    factory, spec = factory_and_map
    factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    factory._create_provider.assert_not_called()
    wrong = tmp_path / "wrong.Map.Gbx"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        factory.validate_scheduled_evaluation_map(
            spec.model_copy(update={"map_path": wrong}), IdentityFeaturePipeline()
        )
    factory._create_provider.assert_not_called()


def test_scheduled_evaluation_supports_native_vision_pipelines(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
) -> None:
    from trackmaniarl.trackmania.native_headless.vision import RibbonVisionFeaturePipeline
    from trackmaniarl.trackmania.vision import PreparedVisionFeaturePipeline

    factory, spec = factory_and_map
    for pipeline in (PreparedVisionFeaturePipeline(), RibbonVisionFeaturePipeline()):
        factory.validate_scheduled_evaluation_map(spec, pipeline)
    factory._create_provider.assert_not_called()


def test_scheduled_evaluation_verifies_actual_native_vision_geometry(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
) -> None:
    from trackmaniarl.trackmania.native_headless.vision import NativeSchematicVisionPipeline

    factory, spec = factory_and_map
    correct = NativeSchematicVisionPipeline(spec.geometry_path, "trackmaniarl-test")
    other_dir = tmp_path / "other-geometry"
    other_dir.mkdir()
    wrong = NativeSchematicVisionPipeline(
        _asset(other_dir, lookahead_points=3), "trackmaniarl-test"
    )
    factory._create_pipeline = Mock(side_effect=[correct, wrong])
    factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    with pytest.raises(ValueError, match="native observation pipeline geometry differs"):
        factory.create(seed=17)
    factory._create_provider.assert_not_called()
    factory._source_class.assert_not_called()


def test_custom_native_pipeline_must_explicitly_attest(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
) -> None:
    factory, spec = factory_and_map
    factory._create_pipeline = Mock(return_value=object())
    with pytest.raises(ValueError, match="native observation pipeline must attest"):
        factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    attest = Mock()
    factory._create_pipeline = Mock(
        return_value=SimpleNamespace(validate_scheduled_evaluation_map=attest)
    )
    factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    attest.assert_called_once_with(spec)


@pytest.mark.parametrize(
    "class_path",
    [
        "trackmaniarl.trackmania.native_headless.full_qr.native_features:NativeGraphFeaturePipeline",
        "trackmaniarl.trackmania.native_headless.vision:NativeSchematicVisionPipeline",
    ],
)
def test_configured_native_pipeline_must_use_pinned_geometry(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
    class_path: str,
) -> None:
    factory, spec = factory_and_map

    def configure(path: Path) -> None:
        factory._pipeline_spec = ComponentSpec(
            class_path=class_path,
            kwargs={"geometry_path": path, "expected_map_uid": "trackmaniarl-test"},
        )

    assert spec.geometry_path is not None
    configure(spec.geometry_path)
    factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    other_dir = tmp_path / "other-pipeline"
    other_dir.mkdir()
    configure(_asset(other_dir, lookahead_points=3))
    with pytest.raises(ValueError, match="native observation pipeline geometry differs"):
        factory.validate_scheduled_evaluation_map(spec, IdentityFeaturePipeline())
    factory._create_provider.assert_not_called()


@pytest.mark.parametrize("field", ["expected_map_uid", "geometry_path", "map_path"])
def test_rejects_wrong_map_binding_before_provider(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
    field: str,
) -> None:
    factory, spec = factory_and_map
    other = tmp_path / "other"
    other.write_bytes(b"different map or geometry")
    changes = {field: "other-map" if field == "expected_map_uid" else other}
    wrong = spec.model_copy(update=changes)
    with pytest.raises(ValueError, match=r"pinned map and geometry|checksum"):
        factory.create(seed=17, evaluation_map=wrong)
    factory._create_provider.assert_not_called()
    factory._source_class.assert_not_called()


def test_rejects_evaluation_with_only_reward_points(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
) -> None:
    factory, _ = factory_and_map
    spec = EvaluationMapSpec(
        id="map", reward_points_path=tmp_path / "points", expected_map_uid="trackmaniarl-test"
    )
    with pytest.raises(ValueError, match="pinned map and geometry"):
        factory.create(seed=17, evaluation_map=spec)
    factory._create_provider.assert_not_called()


def test_rejects_runtime_map_that_differs_from_evaluation(
    factory_and_map: tuple[factory_module.FullQrEnvironmentFactory, EvaluationMapSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, spec = factory_and_map
    different = tmp_path / "runtime-map"
    different.write_bytes(b"different runtime map")
    monkeypatch.setattr(
        factory_module, "verify_bound_assets", lambda *_: SimpleNamespace(paths={"map": different})
    )
    with pytest.raises(ValueError, match="checksum"):
        factory.create(seed=17, evaluation_map=spec)
    factory._create_provider.assert_not_called()
