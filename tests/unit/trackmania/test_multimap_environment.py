from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.trackmania import multimap_environment as module
from trackmaniarl.trackmania.reward_points import save_reward_points


def _assets(tmp_path: Path, name: str) -> tuple[Path, Path]:
    track = tmp_path / f"{name}.gbx"
    track.write_bytes(name.encode())
    reward = tmp_path / f"{name}.npz"
    save_reward_points(
        reward, np.asarray([[0, 0, 0], [20, 0, 0]], dtype=np.float32), map_uid=name, map_path=track
    )
    return track, reward


def _factory(
    tmp_path: Path,
    *,
    enabled: bool = False,
) -> tuple[module.OpenPlanetMultiMapEnvironmentFactory, EvaluationMapSpec]:
    track, reward = _assets(tmp_path, "training")
    other_track, other_reward = _assets(tmp_path, "other")
    factory = module.OpenPlanetMultiMapEnvironmentFactory(
        {"reward_points_path": reward, "expected_map_uid": "training"},
        training_map_path=track,
        journal_path=tmp_path / "journal.sqlite",
        instance_id="instance",
        staging_root=tmp_path,
        enable_experimental_multimap=enabled,
    )
    return factory, EvaluationMapSpec(
        id="other", expected_map_uid="other", map_path=other_track, reward_points_path=other_reward
    )


def test_multi_map_admission_requires_explicit_factory_opt_in(tmp_path: Path) -> None:
    from trackmaniarl.distributed.evaluation_contract import validate_scheduled_evaluation

    factory, other = _factory(tmp_path)
    # Distinct IDs and the same valid file are sufficient to exercise the
    # admission contract; the backend itself still validates every asset.
    second = other.model_copy(update={"id": "second"})
    spec = SimpleNamespace(
        training=SimpleNamespace(evaluate_every_episodes=1),
        evaluation=SimpleNamespace(maps=(other, second)),
    )
    with pytest.raises(ValueError, match="explicitly enabled"):
        validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)

    factory.enable_experimental_multimap = True
    validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)
    assert factory._training_map_sha256 is not None

    other.map_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)


def test_multi_map_opt_in_rejects_non_boolean(tmp_path: Path) -> None:
    track, reward = _assets(tmp_path, "training")
    with pytest.raises(ValueError, match="must be a boolean"):
        module.OpenPlanetMultiMapEnvironmentFactory(
            {"reward_points_path": reward, "expected_map_uid": "training"},
            training_map_path=track,
            journal_path=tmp_path / "journal.sqlite",
            instance_id="instance",
            staging_root=tmp_path,
            enable_experimental_multimap=1,  # type: ignore[arg-type]
        )


def test_validates_each_map_and_isolates_pipeline(tmp_path: Path) -> None:
    factory, other = _factory(tmp_path)
    suite = SimpleNamespace(maps=(other,))
    source = IdentityFeaturePipeline()
    factory.validate_scheduled_evaluation_suite(suite, source, tmp_path)
    assert factory.evaluation_pipeline(source, other) is not source
    other.map_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        factory.validate_scheduled_evaluation_suite(suite, source, tmp_path)


def test_training_pipeline_must_match_current_geometry_bytes(tmp_path: Path) -> None:
    from tests.unit.trackmania._lidar_fixtures import _asset
    from trackmaniarl.trackmania.features import LidarFeaturePipeline

    geometry = _asset(tmp_path)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    pipeline = LidarFeaturePipeline(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"}
    )
    factory = module.OpenPlanetMultiMapEnvironmentFactory(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"},
        training_map_path=map_path,
        journal_path=tmp_path / "journal.sqlite",
        instance_id="instance",
        staging_root=tmp_path,
    )
    item = EvaluationMapSpec(
        id="a", expected_map_uid="trackmaniarl-test", map_path=map_path, geometry_path=geometry
    )
    suite = SimpleNamespace(maps=(item,))
    factory.validate_scheduled_evaluation_suite(suite, pipeline, tmp_path)
    # npz remains readable, but the already loaded pipeline is now stale.
    with geometry.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="training feature geometry"):
        factory.validate_scheduled_evaluation_suite(suite, pipeline, tmp_path)


def test_training_pace_reference_is_rejected_before_any_switch(tmp_path: Path) -> None:
    from tests.unit.trackmania._lidar_fixtures import _asset

    geometry = _asset(tmp_path)
    track = tmp_path / "trackmaniarl-test.Map.Gbx"
    factory = module.OpenPlanetMultiMapEnvironmentFactory(
        {
            "geometry_path": geometry,
            "expected_map_uid": "trackmaniarl-test",
            "pace_reference_path": tmp_path / "training-pace.npz",
        },
        training_map_path=track,
        journal_path=tmp_path / "journal.sqlite",
        instance_id="instance",
        staging_root=tmp_path,
    )
    with pytest.raises(ValueError, match="map-specific pace"):
        factory.validate_scheduled_evaluation_suite(
            SimpleNamespace(maps=()), IdentityFeaturePipeline(), tmp_path
        )


@pytest.mark.skipif(os.name != "nt", reason="staging requires Windows")
def test_switch_evaluation_and_restore_keep_the_same_owned_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, other = _factory(tmp_path)
    events: list[str] = []

    class FakeClient:
        def close(self) -> None:
            events.append("telemetry closed")

    class FakeEnvironment:
        def __init__(
            self, config: Any = None, controller: Any = None, evaluation_map: Any = None
        ) -> None:
            self.controller = controller if controller is not None else object()
            self.client = FakeClient()
            self.map = evaluation_map

        def release_controls(self) -> None:
            events.append("released")

        def close(self) -> None:
            events.append("training closed")

        def reset(self, *, seed: int | None = None) -> tuple[int, dict]:
            events.append("reset")
            return 0, {}

    class FakeSwitch:
        def __init__(self, *_: Any) -> None:
            self.uid = ""

        def switch(self, path: Path, uid: str, *, before_dispatch: Any = None) -> None:
            if before_dispatch is not None:
                before_dispatch()
            if uid == "training":
                assert path.read_bytes() == b"training"
            self.uid = uid
            events.append("switch " + uid)

        def verify_active_map(self, uid: str, *, require_player: bool = True) -> None:
            assert self.uid == uid
            events.append("verify " + uid)

        def close(self) -> None:
            events.append("lock closed")

    monkeypatch.setattr(
        module, "MapSwitchJournal", lambda *_: SimpleNamespace(pending=lambda: None)
    )
    monkeypatch.setattr(module, "OpenPlanetSwitchTransport", lambda *_: None)
    monkeypatch.setattr(module, "MapSwitchController", FakeSwitch)
    monkeypatch.setattr(
        module.OpenPlanetEnvironmentFactory, "create", lambda *a, **k: FakeEnvironment()
    )
    monkeypatch.setattr(module, "OpenPlanetEnvironment", FakeEnvironment)
    factory.validate_scheduled_evaluation_suite(
        SimpleNamespace(maps=(other,)), IdentityFeaturePipeline(), tmp_path
    )
    training = factory.create(seed=0)
    with factory.evaluation_map(training, other) as evaluation:
        assert evaluation.controller is training.environment.controller
        factory.verify_evaluation_map(training, other)
    factory.training_map_path.write_bytes(b"changed original")
    factory.restore_training(training)
    training.reset(seed=0)
    assert [e for e in events if e.startswith("switch ")] == [
        "switch training",
        "switch other",
        "switch training",
    ]
    assert events[-2:] == ["verify training", "reset"]
    training.close()
    assert events[-2:] == ["training closed", "lock closed"]


@pytest.mark.skipif(os.name != "nt", reason="staging requires Windows")
def test_changed_training_file_is_rejected_before_plugin_use(tmp_path: Path) -> None:
    factory, other = _factory(tmp_path)
    factory.validate_scheduled_evaluation_suite(
        SimpleNamespace(maps=(other,)), IdentityFeaturePipeline(), tmp_path
    )
    factory.training_map_path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checksum"):
        factory.create(seed=0)


@pytest.mark.skipif(os.name != "nt", reason="staging requires Windows")
def test_failed_claim_never_creates_or_touches_game_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    factory, other = _factory(tmp_path)
    factory.validate_scheduled_evaluation_suite(
        SimpleNamespace(maps=(other,)), IdentityFeaturePipeline(), tmp_path
    )

    class RejectedSwitch:
        def __init__(self, *_: Any) -> None:
            pass

        def switch(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("claim refused")

        def close(self) -> None:
            pass

    create = Mock()
    monkeypatch.setattr(
        module, "MapSwitchJournal", lambda *_: SimpleNamespace(pending=lambda: None)
    )
    monkeypatch.setattr(module, "OpenPlanetSwitchTransport", lambda *_: None)
    monkeypatch.setattr(module, "MapSwitchController", RejectedSwitch)
    monkeypatch.setattr(module.OpenPlanetEnvironmentFactory, "create", create)
    with pytest.raises(RuntimeError, match="claim refused"):
        factory.create(seed=0)
    create.assert_not_called()


def test_pipeline_rejects_unreviewed_subclasses(tmp_path: Path) -> None:
    factory, other = _factory(tmp_path)

    class SharedPipeline(IdentityFeaturePipeline):
        def __deepcopy__(self, memo: Any) -> Any:
            return self

    with pytest.raises(ValueError, match="reviewed telemetry feature pipeline"):
        factory.evaluation_pipeline(SharedPipeline(), other)


def test_lidar_pipeline_is_copied_and_rebound_to_evaluation_geometry(tmp_path: Path) -> None:
    from tests.unit.trackmania._lidar_fixtures import _asset
    from trackmaniarl.trackmania.features import LidarFeaturePipeline

    train_dir, eval_dir = tmp_path / "first", tmp_path / "second"
    train_dir.mkdir()
    eval_dir.mkdir()
    first = _asset(train_dir)
    second = _asset(eval_dir, lookahead_points=3)
    pipeline = LidarFeaturePipeline(
        {"geometry_path": first, "expected_map_uid": "trackmaniarl-test"}
    )
    item = EvaluationMapSpec(
        id="map",
        expected_map_uid="trackmaniarl-test",
        geometry_path=second,
        map_path=eval_dir / "trackmaniarl-test.Map.Gbx",
    )
    factory, _ = _factory(tmp_path)
    copied = factory.evaluation_pipeline(pipeline, item)
    assert copied is not pipeline
    assert copied.geometry is not pipeline.geometry
    assert copied.geometry.sha256 != pipeline.geometry.sha256
    assert not np.shares_memory(copied.geometry.reward_center, pipeline.geometry.reward_center)


@pytest.mark.skipif(os.name != "nt", reason="staging requires Windows")
def test_repeated_suite_reuses_locked_content_addressed_copy(tmp_path: Path) -> None:
    from trackmaniarl.core.spec import EvaluationSuiteSpec
    from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest

    factory, other = _factory(tmp_path)
    suite = EvaluationSuiteSpec(name="suite", version="1", maps=(other,))
    digest = evaluation_assets_digest(suite, tmp_path)
    with factory.staged_suite(suite, tmp_path, digest) as first:
        first_path = first.maps[0].map_path
        assert first_path is not None
        assert first_path.read_bytes() == b"other"
    with factory.staged_suite(suite, tmp_path, digest) as second:
        assert second.maps[0].map_path == first_path
        with pytest.raises(PermissionError):
            first_path.write_bytes(b"forbidden")
    directories = [p for p in tmp_path.iterdir() if p.name.startswith("trackmaniarl-map-")]
    assert len(directories) == 1
    factory._release_suite_cache()
    assert first_path.read_bytes() == b"other"


def test_lidar_guidance_from_training_map_is_rejected(tmp_path: Path) -> None:
    from tests.unit.trackmania._lidar_fixtures import _asset
    from trackmaniarl.trackmania.features import LidarFeaturePipeline

    geometry = _asset(tmp_path)
    pipeline = LidarFeaturePipeline(
        {"geometry_path": geometry, "expected_map_uid": "trackmaniarl-test"}
    )
    pipeline._expert_guidance_line = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    item = EvaluationMapSpec(
        id="map",
        expected_map_uid="trackmaniarl-test",
        geometry_path=geometry,
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
    )
    factory, _ = _factory(tmp_path)
    with pytest.raises(ValueError, match="expert guidance"):
        factory.evaluation_pipeline(pipeline, item)
