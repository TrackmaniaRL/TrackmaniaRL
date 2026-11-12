"""No-game admission of the documented factory and first-party lidar policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_documented_lidar_configuration_is_admitted_without_game(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "isolated" / "staging").mkdir(parents=True)
    np.save(assets / "left.npy", np.array([[0, 0, -5], [20, 0, -5]], dtype=np.float32))
    np.save(assets / "right.npy", np.array([[0, 0, 5], [20, 0, 5]], dtype=np.float32))
    for name in ("training", "one", "two"):
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
                assets / "left.npy",
                assets / "right.npy",
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
