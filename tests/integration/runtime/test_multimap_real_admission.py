"""No-game admission of the documented factory and first-party lidar policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from trackmaniarl.commands.parser import entrypoint
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_evaluation import _isolated_policy, _supports_multimap_policy
from trackmaniarl.distributed.protocol import run_fingerprint
from trackmaniarl.trackmania.encoders import LidarSensorEncoder
from trackmaniarl.trackmania.geometry import build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.reward_points import save_reward_points


def _synthetic_config(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "isolated" / "staging").mkdir(parents=True)
    for name, width in (("training", 5), ("one", 7), ("two", 9)):
        left = assets / f"{name}-left.npy"
        right = assets / f"{name}-right.npy"
        np.save(left, np.array([[0, 0, -width], [20, 0, -width]], dtype=np.float32))
        np.save(right, np.array([[0, 0, width], [20, 0, width]], dtype=np.float32))
        track = assets / f"{name}.Map.Gbx"
        track.write_bytes(f"synthetic-{name}".encode())
        save_reward_points(
            assets / f"{name}-reward.npz",
            np.array([[0, 0, 0], [20, 0, 0]], dtype=np.float32),
            map_uid=name,
            map_path=track,
        )
        build_geometry_asset(
            GeometryBuildRequest(
                assets / f"{name}-geometry.npz",
                left,
                right,
                name,
                track,
                spacing_m=1.0,
                lookahead_points=0,
            )
        )

    example = Path(__file__).resolve().parents[3] / "examples" / "own-map.yaml"
    config = yaml.safe_load(example.read_text(encoding="utf-8"))
    config["run_id"] = "isolated-multimap-admission"
    config["artifacts_dir"] = "artifacts"
    # This is the environment component documented in distributed-map-switch-design.md.
    config["components"]["environment"] = {
        "class_path": (
            "trackmaniarl.trackmania.multimap_environment:OpenPlanetMultiMapEnvironmentFactory"
        ),
        "kwargs": {
            "config": {
                "expected_map_uid": "training",
                "reward_points_path": "assets/training-reward.npz",
            },
            "training_map_path": "assets/training.Map.Gbx",
            "journal_path": "isolated/switch.sqlite",
            "instance_id": "isolated-game",
            "staging_root": "isolated/staging",
            "enable_experimental_multimap": True,
            "assume_exclusive_no_unobserved_map_changes": True,
        },
    }
    config["components"]["feature_pipeline"]["kwargs"]["config"].update(
        geometry_path="assets/training-geometry.npz", expected_map_uid="training"
    )
    config["components"]["learner"]["kwargs"]["execution"] = {
        "device": "cpu",
        "precision": "auto",
    }
    config["training"].update(batch_size=2, warmup_transitions=2, evaluate_every_episodes=1)
    config["evaluation"].update(
        name="synthetic",
        maps=[
            {
                "id": name,
                "map_path": f"assets/{name}.Map.Gbx",
                "geometry_path": f"assets/{name}-geometry.npz",
                "expected_map_uid": name,
            }
            for name in ("one", "two")
        ],
        trials_per_map=2,
    )
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_documented_lidar_configuration_is_admitted_without_game(tmp_path: Path) -> None:
    path = _synthetic_config(tmp_path)

    # CLI validation performs its actual synthetic update and checkpoint in tmp_path.
    entrypoint(["validate", str(path)])
    spec = RunSpec.from_yaml(path)
    actor = object.__new__(ActorRuntime)
    actor.spec, actor.base_dir = spec, tmp_path
    actor.actor_id = "isolated-admission"
    actor.fingerprint = run_fingerprint(spec, tmp_path)
    # Only instantiate/admit: do not open a transport, create a session, or touch the game.
    _, factory = actor._components()
    assert factory._training_asset_sha256
    assert factory._training_map_sha256
    policy = actor._new_policy()
    assert type(policy.model.encoder) is LidarSensorEncoder
    assert _supports_multimap_policy(policy)
    copied = _isolated_policy(policy)
    for source, target in zip(policy.model.parameters(), copied.model.parameters(), strict=True):
        assert source.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
    assert copied.action_selector is not policy.action_selector
    assert copied.action_selector.weights.data_ptr() != policy.action_selector.weights.data_ptr()
    before = next(policy.model.parameters()).detach().clone()
    with torch.no_grad():
        next(copied.model.parameters()).add_(1)
    assert torch.equal(next(policy.model.parameters()), before)

    class UnknownLidarEncoder(LidarSensorEncoder):
        pass

    policy.model.encoder = UnknownLidarEncoder({"output_dim": 256})
    assert not _supports_multimap_policy(policy)


@pytest.mark.skipif(os.name != "nt", reason="read-locked staging requires Windows")
def test_real_lidar_suite_with_simulated_game_and_no_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real actor/policy/factory with synthetic assets and in-memory game fakes."""
    from tests.integration.runtime.distributed_runtime_support import _DISTRIBUTED_TOKEN, _Logger
    from tests.integration.runtime.test_distributed_evaluation_integration import (
        _close,
        _submit,
    )
    from tests.unit.trackmania.test_map_switch_controller import FakePlugin
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
    from trackmaniarl.core.runtime import ResolvedRun
    from trackmaniarl.distributed.actor_background import _accept_evaluation_request
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_submission import _policy_states_equal
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
    from trackmaniarl.trackmania import map_switch_controller, multimap_environment
    from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
    from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal
    from trackmaniarl.trackmania.telemetry import TelemetryFrame

    path = _synthetic_config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["components"]["environment"]["kwargs"]["config"].update(
        action_repeat_frames=1, minimum_finish_steps=1
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    actor = object.__new__(ActorRuntime)
    actor.spec = RunSpec.from_yaml(path)
    actor.base_dir = tmp_path
    actor.actor_id = "isolated-lidar-suite"
    actor._initialize_runtime_state()
    actor._initialize_spool()
    pipeline, factory = actor._components()
    plugin = FakePlugin()

    class FakeController:
        frames = 0

        def __init__(self) -> None:
            self.actions: list[Any] = []

        def reset(self) -> None:
            self.frames = 0

        def apply(self, action: Any) -> None:
            pass

        def apply_discrete(self, action: Any) -> None:
            self.actions.append(action)

        def consume_collision(self) -> bool:
            return False

        def confirm_finish(self) -> None:
            pass

        def close(self) -> None:
            pass

    controller = FakeController()

    class FakeSession:
        def verify_loaded_map(self, uid: str) -> None:
            assert plugin.uid == uid

        def confirm_ready(self, uid: str) -> None:
            assert plugin.uid == uid

    observed: list[tuple[str, str]] = []

    class FakeBoundTelemetry:
        def __init__(self, config: Any, identity: Any) -> None:
            self.identity = identity

        def begin_reset(self) -> None:
            pass

        def finish_reset(self) -> None:
            pass

        def close(self) -> None:
            pass

        def read(self) -> TelemetryFrame:
            assert self.identity.map_uid == plugin.uid
            assert self.identity.operation_id == plugin.operation
            observed.append((plugin.uid, plugin.operation))
            controller.frames += 1
            values = np.zeros(33, dtype=np.float32)
            values[2] = float(controller.frames >= 3)
            values[3] = 100.0 * controller.frames
            values[4] = 20.0 if controller.frames >= 3 else 0.0
            values[10] = 1.0
            values[14] = 1.0
            return TelemetryFrame(values)

        read_next = read

    monkeypatch.setattr(map_switch_controller, "_registry_root", lambda: tmp_path / "registry")
    monkeypatch.setattr(multimap_environment, "OpenPlanetSwitchTransport", lambda *_: plugin)
    monkeypatch.setattr(multimap_environment, "BoundOpenPlanetClient", FakeBoundTelemetry)
    monkeypatch.setattr(
        OpenPlanetEnvironment, "_create_session", staticmethod(lambda _: FakeSession())
    )
    monkeypatch.setattr(
        multimap_environment.OpenPlanetEnvironmentFactory,
        "create",
        lambda self, **kwargs: OpenPlanetEnvironment(self.config, controller),
    )
    MapSwitchJournal(factory.journal_path, factory.instance_id, create=True)
    run = ResolvedRun(
        spec=actor.spec,
        run_dir=tmp_path / "coordinator",
        learner=actor._replica_learner,
        environment_factory=factory,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
        base_dir=tmp_path,
    )
    coordinator = Coordinator(
        run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, actor.fingerprint)
    )
    training_geometry = pipeline.geometry
    evaluation_pipeline = factory.evaluation_pipeline
    observed_geometry: list[tuple[str, str, str]] = []
    lidar_samples: dict[str, torch.Tensor] = {}

    def capture_pipeline(training_pipeline: Any, item: Any) -> Any:
        result = evaluation_pipeline(training_pipeline, item)
        assert result is not training_pipeline
        assert result.geometry is not training_geometry
        observed_geometry.append((item.id, result.geometry.map_uid, result.geometry.sha256))
        transform = result.transform_observation

        def capture(observation: Any) -> Any:
            transformed = transform(observation)
            lidar_samples.setdefault(item.id, transformed["lidar"].clone())
            return transformed

        result.transform_observation = capture
        return result

    monkeypatch.setattr(factory, "evaluation_pipeline", capture_pipeline)
    training = None
    try:
        coordinator._evaluation_due.add(actor.actor_id)
        coordinator.counters.policy_version = 1
        coordinator._policy_payload = coordinator.codec.encode(
            dict(actor._new_policy().export_state())
        )
        assignment = _submit(
            coordinator,
            coordinator.codec.encode(
                {
                    **actor._request_base(),
                    "sequence": 0,
                    "policy_version": 1,
                    "transitions": [],
                    "episodes": [],
                    "evaluations": [],
                    "evaluation_snapshot": b"",
                }
            ),
        )
        _accept_evaluation_request(actor, assignment)
        actor.sequence = 1
        training = factory.create(seed=0)
        actor._evaluate(training, pipeline)
        result = actor.codec.decode(actor.queue.get_nowait().read_bytes())
        assert [(item["map_id"], item["trial_index"]) for item in result["evaluations"]] == [
            ("one", 0),
            ("one", 1),
            ("two", 0),
            ("two", 1),
        ]
        assert {uid for uid, _ in observed} == {"one", "two"}
        assert all(item["termination"] == "finished" for item in result["evaluations"])
        assert [item[:2] for item in observed_geometry] == [("one", "one"), ("two", "two")]
        assert len({training_geometry.sha256, *(item[2] for item in observed_geometry)}) == 3
        assert not torch.equal(lidar_samples["one"], lidar_samples["two"])
        assert len(controller.actions) >= 4
        operations = {
            uid: {op for map_uid, op in observed if uid == map_uid} for uid in ("one", "two")
        }
        assert all(len(ids) == 1 for ids in operations.values())
        assert operations["one"] != operations["two"]
        assert plugin.uid == "training"
        assert plugin.switch_count == 4  # training, two evaluation maps, restore
        assert all(
            item["evaluation_assignment_id"] == assignment["evaluation_assignment_id"]
            and item["evaluation_assets_sha256"] == assignment["evaluation_assets_sha256"]
            for item in result["evaluations"]
        )
        assert _policy_states_equal(
            actor.codec.decode(result["evaluation_snapshot"]),
            actor.codec.decode(assignment["evaluation_snapshot"]),
        )
        assert _submit(coordinator, coordinator.codec.encode(result))["accepted"] is True
        coordinator._recover_journal(0)
        assert coordinator.counters.evaluations == 4
        assert coordinator.counters.evaluation_finishes == 4
        observation, _ = training.reset(seed=0)
        assert pipeline.transform_observation(observation)
        assert plugin.uid == "training"
    finally:
        try:
            if training is not None:
                training.close()
        finally:
            _close(coordinator)
