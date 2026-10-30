"""Bounded CPU audit of a PRIVATE checkpoint copy, never a live replay or learner.

The saved SDK's loss implementation is reused without calling update/optimize.
The output directory must be separate from the running campaign.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from tools.ghost_divergence1103.diagnose_ghost_divergence import file_path
from tools.gradient_audit1104.gradients import compare_gradients, gradients, model_digest


def sha(path: Path) -> str:
    with file_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_path(path: Path) -> Path:
    value = os.path.abspath(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def windows_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: windows_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [windows_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/mnt/f/"):
        return str(file_path(Path("F:/" + value[7:])))
    return value


def write_new(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with file_path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


@dataclass(frozen=True)
class Inputs:
    checkpoint: Path
    spec: Path
    source: Path
    study: Path
    output: Path
    batches: int


def reward_audit(spec: dict[str, Any], study: Path) -> dict[str, Any]:
    from trackmaniarl.trackmania.reward import TrajectoryReward
    from trackmaniarl.trackmania.reward_config import RewardConfig
    from trackmaniarl.trackmania.reward_types import TransitionInput

    factory = spec["components"]["environment"]["kwargs"]["config"]
    reference = file_path(Path(windows_paths(factory["reward_reference"]["path"])))
    if sha(reference) != factory["reward_reference"]["sha256"]:
        raise ValueError("reward reference SHA differs")
    with np.load(reference, allow_pickle=False) as data:
        points = data["position_m"].copy()
    reward = TrajectoryReward(points, RewardConfig(**factory["reward"]))
    trajectory = file_path(study / "ghost-divergence1103/jimmy-headless-final.json")
    frames = {row["time_ms"]: row for row in json.loads(trajectory.read_text())["frames"]}
    demo = file_path(study / "training-data1091" /
                     "bcfe2be9-e460-45b4-a6b9-fa48c3609b30-native-demonstration.npz")
    with np.load(demo, allow_pickle=False) as data:
        infos = json.loads(str(data["infos_json"].item()))
        recorded_rewards = data["rewards"].copy()
    first = frames[0]
    reward.reset(np.asarray(first["position_m"], dtype=np.float32),
                 velocity=np.asarray(first["velocity_m_s"], dtype=np.float32), race_time_ms=0)
    prefix_errors = []
    for index, info in enumerate(infos[:-1]):
        frame = frames[int(info["observed_physics_time_ms"])]
        result = reward.step(TransitionInput(
            position=np.asarray(frame["position_m"], dtype=np.float32),
            velocity=np.asarray(frame["velocity_m_s"], dtype=np.float32),
            race_time_ms=float(info["race_time_ms"]),
            finish_ui_active=False,
            collision=bool(info["collision_detected"]), steering=float(info["control_steer"]),
        ))
        if result.terminated:
            raise ValueError("archived reward prefix terminated early")
        prefix_errors.append(abs(result.reward - float(recorded_rewards[index])))
    final = infos[-1]
    frame = frames[int(final["observed_physics_time_ms"])]
    probes: list[dict[str, Any]] = []
    for goal in (17713, 17714, 17715, 17716, 17720, 17749):
        cloned = copy.deepcopy(reward)
        scored = cloned.step(TransitionInput(
            position=np.asarray(frame["position_m"], dtype=np.float32),
            velocity=np.asarray(frame["velocity_m_s"], dtype=np.float32),
            race_time_ms=float(goal), finish_ui_active=True,
            collision=bool(final["collision_detected"]), steering=float(final["control_steer"]),
        ))
        if scored.reason != "finished":
            raise ValueError("counterfactual reward probe did not score Finish")
        probes.append({"counterfactual_goal_ms": goal, "physical_observation_ms": 17750,
                       "reward": scored.reward, "reward_float32": float(np.float32(scored.reward)),
                       "time_reward": scored.time_reward,
                       "time_attack_terminal_reward": scored.time_attack_terminal_reward})
    actual = next(row for row in probes if row["counterfactual_goal_ms"] == 17715)
    faster = next(row for row in probes if row["counterfactual_goal_ms"] == 17714)
    if max(prefix_errors) > 1e-5 or abs(actual["reward"] - float(recorded_rewards[-1])) > 1e-5:
        raise ValueError("reward replay does not reproduce the archived demonstration")
    delta = faster["reward"] - actual["reward"]
    return {
        "method": "real archived prefix; last-step time-only counterfactual, NOT new driving",
        "reference_sha256": sha(reference), "trajectory_sha256": sha(trajectory),
        "demonstration_sha256": sha(demo), "prefix_max_reward_error": max(prefix_errors),
        "actual_final_reward_error": abs(actual["reward"] - float(recorded_rewards[-1])),
        "configuration": factory["reward"], "probes": probes,
        "gain_for_one_ms_same_observed_final_state": delta,
        "float32_gain": faster["reward_float32"] - actual["reward_float32"],
        "discounted_gain_at_start_354_steps": delta * 0.995**354,
        "does_not_prove_a_faster_trajectory_exists": True,
    }


def run(inputs: Inputs) -> None:
    import trackmaniarl
    from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner
    from trackmaniarl.algorithms.value_based.updater import ValueUpdater
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.data import BatchRequest
    from trackmaniarl.core.replay import InMemoryReplayStore
    from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
    from trackmaniarl.experiments.shortcut_multi import PaceAwareMixedSampler
    from trackmaniarl.models.factory import CompositeValueModelFactory

    loaded = Path(trackmaniarl.__file__)
    if loaded.parent != inputs.source / "trackmaniarl":
        raise ValueError(f"refuse different SDK source: {loaded}")

    started = time.monotonic()
    checkpoint_sha = sha(inputs.checkpoint)
    state = TorchCheckpointCodec(max_decompressed_bytes=4 * 1024**3).load(
        checkpoint_path(inputs.checkpoint)
    )
    spec = yaml.safe_load(file_path(inputs.spec).read_text())
    components = windows_paths(spec["components"])
    model = CompositeValueModelFactory(**components["model_factory"]["kwargs"]).build()
    options = dict(components["learner"]["kwargs"])
    options.update(execution={"device": "cpu", "precision": "float32", "torch_threads": 1},
                   model_initialization_checkpoint=None)
    learner = DiscreteValueLearner(model, **options)
    learner.setup({"seed": 17})
    if model.architecture_fingerprint() != state["learner"]["architecture_fingerprint"]:
        raise ValueError("checkpoint architecture differs")
    learner._load_modules(model, state["learner"]["online"])
    learner._load_modules(learner.target_model, state["learner"]["target"])
    learner.update_count = state["distributed"]["updates"]
    frozen = model_digest(model), model_digest(learner.target_model)
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    store = InMemoryReplayStore(capacity=state["replay_store"]["capacity"])
    store.load_state_dict(state["replay_store"])
    pipeline = BoundaryGraphFeaturePipelineV5(**components["feature_pipeline"]["kwargs"])
    sampler = PaceAwareMixedSampler(pipeline, **components["sampler"]["kwargs"])
    sampler.load_state_dict(state["sampler"])
    count = state["distributed"]["transitions"]
    weight = float(learner.objectives[0].weight_at(count))
    request = BatchRequest(256, gamma=0.995, transition_count=count)
    updater = ValueUpdater(learner)
    batches: list[dict[str, Any]] = []
    for index in range(inputs.batches):
        if time.monotonic() - started > 300:
            raise TimeoutError("bounded offline gradient audit exceeded300seconds")
        batch = sampler.sample(store, request)
        _, losses = updater._forward_losses(updater._batch_step(batch))
        td, bc = gradients(losses.value, parameters), gradients(losses.objective, parameters)
        measurements = compare_gradients(td, bc)
        for values in measurements.values():
            norm, ratio = values["bc_weighted_norm"], values["weighted_bc_to_td_ratio"]
            values["bc_unweighted_norm"] = None if norm is None else norm / weight
            values["ratio_if_lambda16_same_weights_and_batch"] = (
                None if ratio is None else ratio * 16 / weight
            )
        batches.append({
            "index": index, "transition_ids": list(batch.transition_ids),
            "td_loss": float(losses.value.detach()), "bc_weighted_loss": float(
                losses.objective.detach()), "groups": measurements,
            "demo_sample_fraction": batch.metadata.get("replay/demo_sample_fraction"),
        })
        if index == 0:
            _, repeated = updater._forward_losses(updater._batch_step(batch))
            check = compare_gradients(gradients(repeated.value, parameters),
                                      gradients(repeated.objective, parameters))
            for group, values in check.items():
                for key, value in values.items():
                    if value != measurements[group][key]:
                        raise ValueError("same frozen CPU batch gradient did not repeat exactly")
        del td, bc, losses
    summary: dict[str, Any] = {}
    for group in ("all", "encoder", "head"):
        summary[group] = {}
        for key in batches[0]["groups"][group]:
            samples = [float(row["groups"][group][key]) for row in batches
                       if row["groups"][group][key] is not None]
            summary[group][key] = {
                "minimum": min(samples), "median": statistics.median(samples),
                "maximum": max(samples)
            } if samples else None
    reward = reward_audit(spec, inputs.study)
    if (model_digest(model), model_digest(learner.target_model)) != frozen:
        raise ValueError("frozen online/target tensors changed")
    if any(p.grad is not None for p in parameters.values()) or learner.optimizer.state:
        raise ValueError("diagnostic unexpectedly accumulated .grad or optimizer state")
    cuda_initialized = bool(cast(Any, torch.cuda).is_initialized())
    if sha(inputs.checkpoint) != checkpoint_sha or cuda_initialized:
        raise ValueError("checkpoint changed or CUDA initialized")
    report = {
        "status": "PASS_FROZEN_CPU_GRADIENT_AND_REWARD_AUDIT", "checkpoint_sha256": checkpoint_sha,
        "snapshot_updates": learner.update_count, "snapshot_transitions": count,
        "lambda": weight, "batch_size": 256, "batch_count": len(batches),
        "sampling": "restored snapshot replay/sampler RNG; diagnostic draws, not logged batches",
        "device": "cpu", "torch_threads": 1, "cuda_initialized": False,
        "online_target_unchanged": True, "optimizer_steps": 0,
        "first_batch_exactly_repeatable": True, "summary": summary,
        "batches": batches, "reward_audit": reward, "elapsed_seconds": time.monotonic() - started,
        "limitations": ["CPU float32, not a bitwise reproduction of CUDA backward.",
                        "Raw gradients before clipping/Adam; no optimizer step was applied.",
                        "A lambda16 rescaling on this snapshot does not predict later training.",
                        "Finite snapshot minibatches do not establish causality for plateau."],
    }
    write_new(inputs.output / "result.json", report)
    print(json.dumps({"status": report["status"], "lambda": weight, "summary": summary,
                      "reward_gain_one_ms": reward["gain_for_one_ms_same_observed_final_state"],
                      "seconds": report["elapsed_seconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "spec", "source", "study", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--batches", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.batches <= 16:
        raise ValueError("one to sixteen diagnostic batches only")
    if file_path(args.checkpoint).parent != file_path(args.output):
        raise ValueError("checkpoint must be a private copy inside the diagnostic output directory")
    proof = json.loads(file_path(args.output / "snapshot.json").read_text())
    if (proof["copy_sha256"] != sha(args.checkpoint)
            or checkpoint_path(Path(proof["source"])) == checkpoint_path(args.checkpoint)):
        raise ValueError("snapshot proof must identify a matching copy and a different source")
    if file_path(args.output / "result.json").exists():
        raise FileExistsError("completed diagnostic result already exists")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.source))
    print(json.dumps({"source": str(args.source), "exists": args.source.is_dir()}), flush=True)
    run(Inputs(args.checkpoint, args.spec, args.source, args.study, args.output, args.batches))


if __name__ == "__main__":
    main()
