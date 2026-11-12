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
    enabled: bool = True,
    assume_exclusive: bool = True,
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
        assume_exclusive_no_unobserved_map_changes=assume_exclusive,
    )
    return factory, EvaluationMapSpec(
        id="other", expected_map_uid="other", map_path=other_track, reward_points_path=other_reward
    )


@pytest.mark.parametrize(
    ("enabled", "assume_exclusive"), [(False, False), (True, False), (False, True)]
)
def test_multi_map_admission_requires_both_explicit_consents(
    tmp_path: Path, *, enabled: bool, assume_exclusive: bool
) -> None:
    from trackmaniarl.distributed.evaluation_contract import validate_scheduled_evaluation

    factory, other = _factory(tmp_path, enabled=enabled, assume_exclusive=assume_exclusive)
    # Distinct IDs and the same valid file are sufficient to exercise the
    # admission contract; the backend itself still validates every asset.
    second = other.model_copy(update={"id": "second"})
    spec = SimpleNamespace(
        training=SimpleNamespace(evaluate_every_episodes=1),
        evaluation=SimpleNamespace(maps=(other, second)),
    )
    with pytest.raises(ValueError, match="explicit experimental opt-in"):
        validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)
    with pytest.raises(ValueError, match="explicit experimental opt-in"):
        factory.create(seed=0)


def test_multi_map_admission_validates_assets_with_both_consents(tmp_path: Path) -> None:
    from trackmaniarl.distributed.evaluation_contract import validate_scheduled_evaluation

    factory, other = _factory(tmp_path)
    spec = SimpleNamespace(
        training=SimpleNamespace(evaluate_every_episodes=1),
        evaluation=SimpleNamespace(maps=(other, other.model_copy(update={"id": "second"}))),
    )
    validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)
    assert factory._training_map_sha256 is not None

    other.map_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_scheduled_evaluation(spec, factory, IdentityFeaturePipeline(), tmp_path)
    with pytest.raises(RuntimeError, match="training assets must be attested"):
        factory.create(seed=0)


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

    with pytest.raises(ValueError, match="must be a boolean"):
        module.OpenPlanetMultiMapEnvironmentFactory(
            {"reward_points_path": reward, "expected_map_uid": "training"},
            training_map_path=track,
            journal_path=tmp_path / "journal.sqlite",
            instance_id="instance",
            staging_root=tmp_path,
            enable_experimental_multimap=True,
            assume_exclusive_no_unobserved_map_changes=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "option", ["enable_experimental_multimap", "assume_exclusive_no_unobserved_map_changes"]
)
def test_multi_map_consent_changes_run_fingerprint(tmp_path: Path, option: str) -> None:
    from tests.integration.runtime.core_runtime_support import runtime_spec
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.spec import ComponentSpec

    spec = runtime_spec(tmp_path)
    environment = ComponentSpec(
        class_path=(
            "trackmaniarl.trackmania.multimap_environment:OpenPlanetMultiMapEnvironmentFactory"
        ),
        kwargs={option: False},
    )
    disabled = spec.model_copy(
        update={"components": spec.components.model_copy(update={"environment": environment})}
    )
    enabled = disabled.model_copy(
        update={
            "components": disabled.components.model_copy(
                update={"environment": environment.model_copy(update={"kwargs": {option: True}})}
            )
        }
    )
    assert run_fingerprint(disabled, tmp_path) != run_fingerprint(enabled, tmp_path)


def test_validates_each_map_and_isolates_pipeline(tmp_path: Path) -> None:
    factory, other = _factory(tmp_path)
    suite = SimpleNamespace(maps=(other,))
    source = IdentityFeaturePipeline()
    factory.validate_scheduled_evaluation_suite(suite, source, tmp_path)
    assert factory.evaluation_pipeline(source, other) is not source
    other.map_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        factory.validate_scheduled_evaluation_suite(suite, source, tmp_path)


def test_bound_client_is_replaced_for_each_verified_map_operation(tmp_path: Path) -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.bound_telemetry import (
        BoundOpenPlanetClient,
        BoundTelemetryIdentity,
    )

    factory, _ = _factory(tmp_path)
    old = Mock()
    environment = SimpleNamespace(client=old)
    identity = BoundTelemetryIdentity("epoch", "map-operation", "map-uid")
    switch = SimpleNamespace(telemetry_identity=Mock(return_value=identity))

    factory._bind_telemetry(environment, switch, "map-uid")

    old.close.assert_called_once_with()
    switch.telemetry_identity.assert_called_once_with("map-uid")
    assert isinstance(environment.client, BoundOpenPlanetClient)
    assert environment.client.identity == identity
    assert (environment.client.host, environment.client.port) == ("127.0.0.1", 9002)
    environment.client.close()


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
        enable_experimental_multimap=True,
        assume_exclusive_no_unobserved_map_changes=True,
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
        enable_experimental_multimap=True,
        assume_exclusive_no_unobserved_map_changes=True,
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

        def telemetry_identity(self, uid: str) -> Any:
            from trackmaniarl.trackmania.bound_telemetry import BoundTelemetryIdentity

            assert self.uid == uid
            return BoundTelemetryIdentity("epoch", "operation", uid)

        def begin_evaluation_guard(self) -> None:
            events.append("guard armed")

        def finish_evaluation_guard(self) -> None:
            events.append("guard cleared")

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
        with pytest.raises(RuntimeError, match="training is suspended"):
            training.reset(seed=0)
        with (
            pytest.raises(RuntimeError, match="already open"),
            factory.evaluation_map(training, other),
        ):
            pass
        with pytest.raises(RuntimeError, match="close the evaluation"):
            factory.restore_training(training)
        factory.verify_evaluation_map(training, other)
        evaluation.close()
        assert "training closed" not in events
    with pytest.raises(RuntimeError, match="evaluation session is closed"):
        evaluation.reset(seed=0)
    with pytest.raises(RuntimeError, match="training is suspended"):
        training.reset(seed=0)
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
    assert events[-4:] == ["training closed", "verify training", "guard cleared", "lock closed"]


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


def test_training_reset_verifies_switch_lease_before_reset_controls() -> None:
    from unittest.mock import Mock, call

    events = Mock()
    environment = SimpleNamespace(reset=events.reset)
    switch = SimpleNamespace(verify_active_map=events.verify_active_map)
    asset = SimpleNamespace(validate_map=events.validate_map)
    training_path = Path("training.Map.Gbx")
    training = module._TrainingSession(
        environment, switch, "training-uid", SimpleNamespace(), training_path, asset, lambda: None
    )

    training.reset(seed=123)

    assert events.mock_calls == [
        call.validate_map(training_path),
        call.verify_active_map("training-uid", require_player=False),
        call.reset(seed=123),
    ]


def test_training_reset_rejects_restarted_plugin_before_reset_controls() -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.map_switch_controller import MapSwitchIndeterminateError

    environment = SimpleNamespace(reset=Mock(), release_controls=Mock())
    switch = SimpleNamespace(
        verify_active_map=Mock(side_effect=MapSwitchIndeterminateError("plugin restarted"))
    )
    training = module._TrainingSession(
        environment,
        switch,
        "training-uid",
        SimpleNamespace(),
        Path("training.Map.Gbx"),
        SimpleNamespace(validate_map=Mock()),
        lambda: None,
    )

    with pytest.raises(MapSwitchIndeterminateError, match="plugin restarted"):
        training.reset(seed=123)

    switch.verify_active_map.assert_called_once_with("training-uid", require_player=False)
    environment.reset.assert_not_called()
    environment.release_controls.assert_called_once_with()


def test_training_reset_preserves_verification_error_if_release_fails() -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.map_switch_controller import MapSwitchIndeterminateError

    environment = SimpleNamespace(
        reset=Mock(), release_controls=Mock(side_effect=OSError("release"))
    )
    switch = SimpleNamespace(
        verify_active_map=Mock(side_effect=MapSwitchIndeterminateError("plugin restarted"))
    )
    training = module._TrainingSession(
        environment,
        switch,
        "training-uid",
        SimpleNamespace(),
        Path("training.Map.Gbx"),
        SimpleNamespace(validate_map=Mock()),
        lambda: None,
    )

    with pytest.raises(MapSwitchIndeterminateError, match="plugin restarted") as captured:
        training.reset(seed=123)

    assert any("release" in note for note in captured.value.__notes__)
    environment.reset.assert_not_called()
    environment.release_controls.assert_called_once_with()


def test_training_step_telemetry_failure_neutralizes_and_keeps_original_error() -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.bound_telemetry import BoundTelemetryError

    error = BoundTelemetryError("wrong map")
    environment = SimpleNamespace(
        step=Mock(side_effect=error), release_controls=Mock(side_effect=OSError("release failed"))
    )
    switch = SimpleNamespace(quarantine_uncertain_telemetry=Mock())
    training = module._TrainingSession(
        environment,
        switch,
        "training-uid",
        SimpleNamespace(),
        Path("training.Map.Gbx"),
        SimpleNamespace(),
        lambda: None,
    )
    with pytest.raises(BoundTelemetryError, match="wrong map") as captured:
        training.step(0)
    assert captured.value is error
    assert any("release failed" in note for note in error.__notes__)
    environment.release_controls.assert_called_once_with()
    switch.quarantine_uncertain_telemetry.assert_called_once_with()
    assert training._failed
    with pytest.raises(RuntimeError, match="session is quarantined"):
        training.step(1)
    with pytest.raises(RuntimeError, match="session is quarantined"):
        training.reset(seed=1)
    environment.step.assert_called_once_with(0)


def test_evaluation_telemetry_failure_neutralizes_and_keeps_original_error(
    tmp_path: Path,
) -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.bound_telemetry import BoundTelemetryError

    factory, _ = _factory(tmp_path)
    error = BoundTelemetryError("wrong map")
    environment = SimpleNamespace(release_controls=Mock(side_effect=OSError("release failed")))
    switch = SimpleNamespace(quarantine_uncertain_telemetry=Mock())
    training = module._TrainingSession(
        environment, switch, "training-uid", SimpleNamespace(), Path("training"), None, lambda: None
    )
    factory.abort_evaluation(training, error)
    environment.release_controls.assert_called_once_with()
    switch.quarantine_uncertain_telemetry.assert_called_once_with()
    assert training._failed
    assert any("release failed" in note for note in error.__notes__)


def test_close_before_restoration_keeps_guard_even_if_uid_might_match() -> None:
    from unittest.mock import Mock

    environment = SimpleNamespace(release_controls=Mock(), close=Mock())
    switch = SimpleNamespace(finish_evaluation_guard=Mock(), close=Mock())
    staged = SimpleNamespace(close=Mock())
    training = module._TrainingSession(
        environment, switch, "uid", staged, Path("training"), None, lambda: None
    )
    training._evaluation_active = True
    with pytest.raises(RuntimeError, match="before training map restoration"):
        training.close()
    environment.release_controls.assert_called_once_with()
    environment.close.assert_called_once_with()
    switch.finish_evaluation_guard.assert_not_called()
    switch.close.assert_called_once_with()


def test_failed_close_cannot_clear_guard_on_a_later_retry() -> None:
    from unittest.mock import Mock

    environment = SimpleNamespace(
        release_controls=Mock(side_effect=[OSError("release failed"), None]), close=Mock()
    )
    switch = SimpleNamespace(finish_evaluation_guard=Mock(), close=Mock())
    training = module._TrainingSession(
        environment,
        switch,
        "uid",
        SimpleNamespace(close=Mock()),
        Path("training"),
        None,
        lambda: None,
    )
    with pytest.raises(OSError, match="release failed"):
        training.close()
    training.close()
    switch.finish_evaluation_guard.assert_not_called()


def test_close_preserves_first_failure_and_reports_other_cleanup_errors() -> None:
    from unittest.mock import Mock

    environment = SimpleNamespace(
        release_controls=Mock(side_effect=OSError("release failed")),
        close=Mock(side_effect=RuntimeError("controller close failed")),
    )
    switch = SimpleNamespace(
        finish_evaluation_guard=Mock(), close=Mock(side_effect=OSError("switch close failed"))
    )
    staged = SimpleNamespace(close=Mock(side_effect=OSError("staging close failed")))
    release_suite = Mock(side_effect=OSError("suite close failed"))
    training = module._TrainingSession(
        environment, switch, "uid", staged, Path("training"), None, release_suite
    )
    with pytest.raises(OSError, match="release failed") as captured:
        training.close()
    for text in (
        "controller close failed",
        "suite close failed",
        "staging close failed",
        "switch close failed",
    ):
        assert any(text in note for note in captured.value.__notes__)
    release_suite.assert_called_once_with()
    switch.finish_evaluation_guard.assert_not_called()


def test_direct_evaluation_cannot_resume_after_caught_telemetry_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from trackmaniarl.trackmania.bound_telemetry import BoundTelemetryError

    factory, other = _factory(tmp_path)
    error = BoundTelemetryError("wrong map")
    environment = SimpleNamespace(release_controls=Mock(), controller=object())
    switch = SimpleNamespace(switch=Mock(), quarantine_uncertain_telemetry=Mock())
    training = module._TrainingSession(
        environment, switch, "training-uid", SimpleNamespace(), Path("training"), None, lambda: None
    )
    evaluation = SimpleNamespace(client=Mock(), step=Mock(side_effect=error))
    monkeypatch.setattr(factory, "_asset", Mock())
    monkeypatch.setattr(factory, "_bind_telemetry", Mock())
    monkeypatch.setattr(module, "OpenPlanetEnvironment", lambda *args, **kwargs: evaluation)

    with factory.evaluation_map(training, other) as session:
        with pytest.raises(BoundTelemetryError, match="wrong map"):
            session.step(0)
        with pytest.raises(RuntimeError, match="session is quarantined"):
            session.step(1)
    with pytest.raises(RuntimeError, match="session is quarantined"):
        factory.restore_training(training)
    evaluation.step.assert_called_once_with(0)
    assert environment.release_controls.call_count >= 1
    switch.quarantine_uncertain_telemetry.assert_called_once_with()
    assert switch.switch.call_count == 1


@pytest.mark.parametrize("safe_stop", [False, True])
def test_direct_evaluation_context_exit_neutralizes_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, safe_stop: bool
) -> None:
    from unittest.mock import Mock

    from trackmaniarl.distributed.actor_evaluation import _EvaluationStoppedError

    factory, other = _factory(tmp_path)
    error = (
        _EvaluationStoppedError(safe_boundary=True) if safe_stop else ValueError("policy failed")
    )
    environment = SimpleNamespace(release_controls=Mock(), controller=object())
    switch = SimpleNamespace(switch=Mock(), quarantine_uncertain_telemetry=Mock())
    training = module._TrainingSession(
        environment, switch, "training-uid", SimpleNamespace(), Path("training"), None, lambda: None
    )
    evaluation = SimpleNamespace(client=Mock())
    monkeypatch.setattr(factory, "_asset", Mock())
    monkeypatch.setattr(factory, "_bind_telemetry", Mock())
    monkeypatch.setattr(module, "OpenPlanetEnvironment", lambda *args, **kwargs: evaluation)
    with pytest.raises(type(error)) as captured, factory.evaluation_map(training, other):
        raise error
    assert captured.value is error
    assert environment.release_controls.call_count >= 1
    assert training._failed is not safe_stop
    assert switch.quarantine_uncertain_telemetry.call_count == (0 if safe_stop else 1)


def test_direct_evaluation_context_preserves_policy_error_when_release_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    factory, other = _factory(tmp_path)
    error = ValueError("policy failed")
    environment = SimpleNamespace(
        release_controls=Mock(side_effect=OSError("release failed")), controller=object()
    )
    switch = SimpleNamespace(switch=Mock(), quarantine_uncertain_telemetry=Mock())
    training = module._TrainingSession(
        environment, switch, "training-uid", SimpleNamespace(), Path("training"), None, lambda: None
    )
    evaluation = SimpleNamespace(client=Mock())
    monkeypatch.setattr(factory, "_asset", Mock())
    monkeypatch.setattr(factory, "_bind_telemetry", Mock())
    monkeypatch.setattr(module, "OpenPlanetEnvironment", lambda *args, **kwargs: evaluation)
    with (
        pytest.raises(ValueError, match="policy failed") as captured,
        factory.evaluation_map(training, other),
    ):
        raise error
    assert captured.value is error
    assert training._failed
    assert any("release failed" in note for note in error.__notes__)


def test_training_reset_releases_controls_when_asset_validation_fails() -> None:
    from unittest.mock import Mock

    environment = SimpleNamespace(reset=Mock(), release_controls=Mock())
    switch = SimpleNamespace(verify_active_map=Mock())
    training = module._TrainingSession(
        environment,
        switch,
        "training-uid",
        SimpleNamespace(),
        Path("training.Map.Gbx"),
        SimpleNamespace(validate_map=Mock(side_effect=ValueError("asset changed"))),
        lambda: None,
    )

    with pytest.raises(ValueError, match="asset changed"):
        training.reset()

    switch.verify_active_map.assert_not_called()
    environment.reset.assert_not_called()
    environment.release_controls.assert_called_once_with()
