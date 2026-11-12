from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.evaluation_contract import validate_scheduled_evaluation
from trackmaniarl.trackmania.environment import OpenPlanetEnvironmentFactory
from trackmaniarl.trackmania.features import LidarFeaturePipeline, TelemetryFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.multimodal import LidarVisionFeaturePipeline
from trackmaniarl.trackmania.reward_points import save_reward_points
from trackmaniarl.trackmania.vision import VisionFeaturePipeline
from trackmaniarl.trackmania.vision_environment import VisionEnvironmentFactory


def _spec(maps: tuple[EvaluationMapSpec, ...], *, scheduled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(evaluate_every_episodes=1 if scheduled else None),
        evaluation=SimpleNamespace(maps=maps),
    )


def test_multi_map_scheduled_evaluation_is_rejected_before_attestation(tmp_path: Path) -> None:
    path = _asset(tmp_path)
    item = EvaluationMapSpec(
        id="a",
        geometry_path=path,
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        expected_map_uid="trackmaniarl-test",
    )
    factory = SimpleNamespace(validate_scheduled_evaluation_map=Mock())
    with pytest.raises(ValueError, match="does not support multiple maps"):
        validate_scheduled_evaluation(_spec((item, item)), factory, object(), tmp_path)
    factory.validate_scheduled_evaluation_map.assert_not_called()


def test_coordinator_rejects_multi_map_before_creating_journal(tmp_path: Path) -> None:
    path = _asset(tmp_path)
    item = EvaluationMapSpec(
        id="a",
        geometry_path=path,
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        expected_map_uid="trackmaniarl-test",
    )
    run = SimpleNamespace(
        spec=_spec((item, item)),
        learner=object(),
        environment_factory=object(),
        feature_pipeline=object(),
        base_dir=tmp_path,
        run_dir=tmp_path / "run",
    )
    config = CoordinatorConfig("127.0.0.1:0", "x" * 32, "fingerprint")
    with pytest.raises(ValueError, match="does not support multiple maps"):
        Coordinator(run, config)
    assert not run.run_dir.exists()


def test_actor_rejects_multi_map_before_registering_or_creating_environment(tmp_path: Path) -> None:
    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    actor = object.__new__(ActorRuntime)
    actor.spec = _spec((item, item))
    actor.spec.distributed = SimpleNamespace(actor_execution=None)
    actor.spec.components = SimpleNamespace(feature_pipeline=object())
    actor.base_dir = tmp_path
    factory = Mock()
    actor._environment_factory = Mock(return_value=factory)
    actor._model_factory = Mock()
    from unittest.mock import patch

    with (
        patch("trackmaniarl.distributed.actor._instantiate", return_value=object()),
        pytest.raises(ValueError, match="does not support multiple maps"),
    ):
        actor._components()
    factory.create.assert_not_called()
    actor._model_factory.assert_not_called()


def test_actor_rejects_fingerprint_change_during_multi_map_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.distributed import actor as actor_module
    from trackmaniarl.distributed import evaluation_contract

    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    actor = object.__new__(ActorRuntime)
    actor.spec = _spec((item, item))
    actor.spec.distributed = SimpleNamespace(actor_execution=None, actor_stall_timeout_s=None)
    actor.spec.components = SimpleNamespace(feature_pipeline=object())
    actor.base_dir = tmp_path
    actor.fingerprint = "original"
    actor._environment_factory = Mock(return_value=object())
    actor._model_factory = Mock()
    monkeypatch.setattr(actor_module, "_instantiate", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation_contract, "validate_scheduled_evaluation", lambda *args: None)
    monkeypatch.setattr(actor_module, "run_fingerprint", lambda *args: "changed")
    with pytest.raises(RuntimeError, match="assets changed"):
        actor._components()
    actor._model_factory.assert_not_called()


def test_coordinator_rejects_fingerprint_change_before_creating_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.distributed import evaluation_contract, protocol

    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    run = SimpleNamespace(
        spec=_spec((item, item)),
        learner=object(),
        environment_factory=object(),
        feature_pipeline=object(),
        base_dir=tmp_path,
        run_dir=tmp_path / "run",
    )
    monkeypatch.setattr(evaluation_contract, "validate_scheduled_evaluation", lambda *args: None)
    monkeypatch.setattr(protocol, "run_fingerprint", lambda *args: "changed")
    with pytest.raises(RuntimeError, match="assets changed"):
        Coordinator(run, CoordinatorConfig("127.0.0.1:0", "x" * 32, "original"))
    assert not run.run_dir.exists()


def test_no_scheduled_suite_leaves_custom_game_free_runtime_unchanged(tmp_path: Path) -> None:
    spec = SimpleNamespace(training=SimpleNamespace(evaluate_every_episodes=1), evaluation=None)
    validate_scheduled_evaluation(spec, object(), object(), tmp_path)


def test_disabled_scheduled_evaluation_keeps_multi_map_local_suite(tmp_path: Path) -> None:
    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    validate_scheduled_evaluation(
        _spec((item, item), scheduled=False), object(), object(), tmp_path
    )


def test_unattested_custom_factory_is_rejected(tmp_path: Path) -> None:
    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    with pytest.raises(ValueError, match="attests its evaluation map"):
        validate_scheduled_evaluation(_spec((item,)), object(), object(), tmp_path)


def test_custom_factory_receives_resolved_map_and_pipeline(tmp_path: Path) -> None:
    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    hook = Mock()
    pipeline = object()
    validate_scheduled_evaluation(
        _spec((item,)), SimpleNamespace(validate_scheduled_evaluation_map=hook), pipeline, tmp_path
    )
    resolved, actual_pipeline = hook.call_args.args
    assert resolved.geometry_path == tmp_path / "geometry.npz"
    assert resolved.map_path == tmp_path / "map.gbx"
    assert actual_pipeline is pipeline


def test_openplanet_attests_exact_map_and_rejects_different_identity(tmp_path: Path) -> None:
    geometry = _asset(tmp_path)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    item = EvaluationMapSpec(
        id="a", geometry_path=geometry, map_path=map_path, expected_map_uid="trackmaniarl-test"
    )
    factory = OpenPlanetEnvironmentFactory(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"}
    )
    factory.validate_scheduled_evaluation_map(item, TelemetryFeaturePipeline())
    lidar = object.__new__(LidarFeaturePipeline)
    lidar.geometry = BoundaryGeometry(geometry)
    factory.validate_scheduled_evaluation_map(item, lidar)
    lidar.geometry = SimpleNamespace(map_uid="other", sha256=lidar.geometry.sha256)
    with pytest.raises(ValueError, match="pipeline geometry differs"):
        factory.validate_scheduled_evaluation_map(item, lidar)
    with pytest.raises(ValueError, match="UID differs"):
        factory.validate_scheduled_evaluation_map(
            item.model_copy(update={"expected_map_uid": "other"}), TelemetryFeaturePipeline()
        )
    wrong_map = tmp_path / "wrong.gbx"
    wrong_map.write_bytes(b"different map")
    with pytest.raises(ValueError, match="checksum"):
        factory.validate_scheduled_evaluation_map(
            item.model_copy(update={"map_path": wrong_map}), TelemetryFeaturePipeline()
        )


def test_reward_points_and_lidar_geometry_must_bind_the_same_map(tmp_path: Path) -> None:
    geometry = _asset(tmp_path)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    reward = save_reward_points(
        tmp_path / "reward.npz", points, map_uid="trackmaniarl-test", map_path=map_path
    )
    item = EvaluationMapSpec(
        id="a", reward_points_path=reward, map_path=map_path, expected_map_uid="trackmaniarl-test"
    )
    factory = OpenPlanetEnvironmentFactory(
        {"reward_points_path": reward, "expected_map_uid": "trackmaniarl-test"}
    )
    lidar = LidarFeaturePipeline(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"}
    )
    validate_scheduled_evaluation(_spec((item,)), factory, lidar, tmp_path)

    lidar.geometry.map_sha256 = "f" * 64
    with pytest.raises(ValueError, match="pipeline geometry differs"):
        validate_scheduled_evaluation(_spec((item,)), factory, lidar, tmp_path)
    lidar.geometry.map_sha256 = BoundaryGeometry(geometry).map_sha256
    lidar.geometry.map_uid = "other"
    with pytest.raises(ValueError, match="pipeline geometry differs"):
        validate_scheduled_evaluation(_spec((item,)), factory, lidar, tmp_path)

    uid_only = save_reward_points(tmp_path / "uid-only.npz", points, map_uid="trackmaniarl-test")
    uid_only_map = item.model_copy(update={"reward_points_path": uid_only})
    uid_only_factory = OpenPlanetEnvironmentFactory(
        {"reward_points_path": uid_only, "expected_map_uid": "trackmaniarl-test"}
    )
    with pytest.raises(ValueError, match="pipeline geometry differs"):
        validate_scheduled_evaluation(
            _spec((uid_only_map,)),
            uid_only_factory,
            LidarFeaturePipeline({"geometry_path": geometry}),
            tmp_path,
        )


def test_vision_factory_checks_map_and_lidar_branch_without_opening_capture(tmp_path: Path) -> None:
    geometry = _asset(tmp_path)
    item = EvaluationMapSpec(
        id="a",
        geometry_path=geometry,
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        expected_map_uid="trackmaniarl-test",
    )
    factory = VisionEnvironmentFactory(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"},
        include_telemetry=True,
    )
    pipeline = LidarVisionFeaturePipeline({"geometry_path": geometry})
    validate_scheduled_evaluation(_spec((item,)), factory, pipeline, tmp_path)
    validate_scheduled_evaluation(_spec((item,)), factory, VisionFeaturePipeline(), tmp_path)
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    wrong_geometry = _asset(wrong_dir, lookahead_points=3)
    pipeline.lidar.geometry = BoundaryGeometry(wrong_geometry)
    with pytest.raises(ValueError, match="pipeline geometry differs"):
        validate_scheduled_evaluation(_spec((item,)), factory, pipeline, tmp_path)
    with pytest.raises(ValueError, match="UID differs"):
        validate_scheduled_evaluation(
            _spec((item.model_copy(update={"expected_map_uid": "wrong"}),)),
            factory,
            VisionFeaturePipeline(),
            tmp_path,
        )


@pytest.mark.parametrize("base", [OpenPlanetEnvironmentFactory, VisionEnvironmentFactory])
def test_factory_subclass_cannot_inherit_unrelated_map_attestation(
    tmp_path: Path, base: type
) -> None:
    class OtherMapFactory(base):
        def create(self, *, seed: int, evaluation_map: object = None) -> object:
            return object()

    geometry = _asset(tmp_path)
    item = EvaluationMapSpec(
        id="a",
        geometry_path=geometry,
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        expected_map_uid="trackmaniarl-test",
    )
    factory = OtherMapFactory({"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"})
    with pytest.raises(ValueError, match="subclasses must explicitly attest"):
        validate_scheduled_evaluation(_spec((item,)), factory, VisionFeaturePipeline(), tmp_path)


def test_multi_map_actor_refuses_watchdog_without_progress_contract(tmp_path: Path) -> None:
    from unittest.mock import patch

    item = EvaluationMapSpec(
        id="a", geometry_path="geometry.npz", map_path="map.gbx", expected_map_uid="test"
    )
    actor = object.__new__(ActorRuntime)
    actor.spec = _spec((item, item))
    actor.spec.distributed = SimpleNamespace(actor_execution=None, actor_stall_timeout_s=1.0)
    actor.spec.components = SimpleNamespace(feature_pipeline=object())
    actor.base_dir = tmp_path
    actor._environment_factory = Mock(return_value=object())
    actor._model_factory = Mock()
    with (
        patch("trackmaniarl.distributed.actor._instantiate", return_value=object()),
        pytest.raises(ValueError, match="cannot safely use actor_stall_timeout_s"),
    ):
        actor._components()
    actor._model_factory.assert_not_called()
