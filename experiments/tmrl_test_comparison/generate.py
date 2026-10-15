"""Generate standalone configurations; never starts Trackmania or training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.tmrl_test_comparison.policy import EXPERIMENTAL_ALGORITHMS, STANDARD_ALGORITHMS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ALGORITHMS = STANDARD_ALGORITHMS
DISCRETE_ALGORITHMS = frozenset({"iqn", "q", "qr", "fqf", "sd-sac"})
SEEDS = (17, 29, 43)


def configuration(algorithm: str, seed: int, stage: str) -> dict:
    algorithm = {"discrete-sac": "sd-sac", "dsac": "sd-sac"}.get(algorithm, algorithm)
    base = json.loads((ROOT / "docs/benchmarks/2026-09-08-v108-config.json").read_text())
    base["distributed"] = {
        "epsilon_profiles": [1.0],
        "epsilon_start": 0.3,
        "epsilon_final": 0.01,
        "epsilon_decay_transitions": 500_000,
        "actor_execution": {"device": "cpu", "precision": "float32", "torch_threads": 2},
    }
    base["run_id"] = f"tmrl-test-{stage}-{algorithm}-s{seed}"
    base["seed"] = seed
    base["distributed"].update(strict_update_budget=True, max_update_credit=512)
    base["artifacts_dir"] = "../../../../artifacts/tmrl-test-comparison"
    components = base["components"]
    geometry = "../../../../my-trackmania-agent/assets/trackmaniarl-test.geometry.npz"
    map_path = "../../../../my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx"
    components["environment"]["kwargs"]["config"]["geometry_path"] = geometry
    components["feature_pipeline"]["kwargs"]["geometry_path"] = geometry
    components["additional_loggers"] = [
        {
            "class_path": "trackmaniarl.observability.trackers:WandbTracker",
            "kwargs": {"project": "my-trackmania-agent"},
        }
    ]
    components["replay_store"]["kwargs"] = {"capacity": 1_000_000}
    components["sampler"] = {
        "class_path": "trackmaniarl.core.replay:UniformSampler",
        "kwargs": {"seed": seed},
    }
    encoder = copy.deepcopy(components["model_factory"]["kwargs"]["encoder"])
    encoder["class_path"] = "experiments.tmrl_test_comparison.components:BatchedIncidentEncoder"
    components["model_factory"]["kwargs"]["encoder"] = encoder
    execution = {"device": "cuda", "precision": "float32", "torch_threads": 2}
    components["environment"]["kwargs"]["config"]["steering_curve_path"] = (
        "../../../../artifacts/tmrl-test-comparison/steering-local.json"
    )
    learner = components["learner"]
    learner["class_path"] = "trackmaniarl.algorithms.value_based:DiscreteValueLearner"
    learner["kwargs"] = {
        "learning_rate": 1e-4,
        "gradient_clip_norm": 10.0,
        "target_update_interval": 1000,
        "target_tau": 0.0,
        "exploration_epsilon": 0.1,
        "value_rescaling": False,
        "execution": execution,
    }
    model = components["model_factory"]["kwargs"]
    if algorithm == "q":
        model["head"] = {
            "class_path": "trackmaniarl.models.heads:ScalarQHead",
            "kwargs": {"feature_dim": 192, "action_count": 78},
        }
        model["strategy"] = {"class_path": "trackmaniarl.models.strategies:ScalarValueStrategy"}
    elif algorithm == "qr":
        model["head"] = {
            "class_path": "trackmaniarl.models.heads:FixedQuantileHead",
            "kwargs": {
                "config": {
                    "feature_dim": 192,
                    "action_count": 78,
                    "quantile_count": 64,
                    "dueling": True,
                }
            },
        }
        model["strategy"] = {
            "class_path": "trackmaniarl.models.strategies:FixedQuantileStrategy",
            "kwargs": {"quantile_count": 64},
        }
    elif algorithm == "fqf":
        model["strategy"] = {
            "class_path": "trackmaniarl.models.strategies:LearnedFractionStrategy",
            "kwargs": {"feature_dim": 192, "fraction_count": 64},
        }
    elif algorithm not in {"iqn"}:
        names = {
            "sac": "SoftActorCritic",
            "redq": "RandomizedEnsembleSAC",
            "tqc": "TruncatedQuantileCritic",
            "sd-sac": "StableDiscreteSoftActorCritic",
            "ppo": "ProximalPolicyOptimization",
        }
        components["learner"] = {
            "class_path": "trackmaniarl.algorithms:" + names[algorithm],
            "kwargs": {"execution": execution},
        }
        if algorithm in {"sac", "tqc", "sd-sac"}:
            components["learner"]["kwargs"].update(
                learning_rate=3e-4,
                target_tau=0.005,
                entropy_coefficient=0.2,
                learn_entropy_coefficient=True,
            )
        if algorithm in {"sac", "tqc"}:
            components["learner"]["kwargs"]["target_entropy"] = -3.0
        if algorithm == "tqc":
            components["learner"]["kwargs"]["top_quantiles_to_drop_per_critic"] = 2
        components["model_factory"] = {
            "class_path": "trackmaniarl.models.sensor_actor_critic:SensorActorCriticModelFactory",
            "kwargs": {
                "algorithm": algorithm,
                "encoder": encoder,
                "config": {
                    "feature_dim": 192,
                    "action_count": 78,
                    "critic_count": 10 if algorithm == "redq" else 2,
                },
            },
        }
    # Multiples of PPO's rollout length: identical interaction budgets for all runs.
    budget = 145_408 if stage == "pilot" else 2_048_000
    if stage == "pilot":
        components["evaluator"] = None
    base["training"] = {
        "total_transitions": budget,
        "max_episode_steps": 3000,
        "batch_size": 256,
        "sequence_length": 1,
        "n_step": 1,
        "gamma": 0.995,
        "warmup_transitions": 10_000,
        "updates_per_transition": 0.25,
        "offline_pretrain_updates": 0,
        "checkpoint_interval_updates": 5000,
        "checkpoint_keep_last": 3,
        "save_final_checkpoint": True,
        "metrics_interval_updates": 100,
        "evaluate_every_episodes": None,
    }
    if algorithm == "ppo":
        components["learner"]["kwargs"].update(
            learning_rate=3e-4,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            gae_lambda=0.95,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_gradient_norm=0.5,
            target_kl=0.02,
            normalize_observations=False,
            normalize_rewards=False,
            update_epochs=10,
            minibatch_size=256,
        )
        components["sampler"] = {
            "class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler",
            "kwargs": {"seed": seed},
        }
        components["replay_store"]["kwargs"] = {"capacity": 2048}
        base["training"].update(
            batch_size=1,
            sequence_length=2048,
            warmup_transitions=0,
            checkpoint_interval_updates=10,
            reset_after_on_policy_rollout=True,
        )
    base["evaluation"].update(name="tmrl-test-comparison", version="1", trials_per_map=30)
    base["evaluation"]["maps"][0].update(map_path=map_path, geometry_path=geometry)
    base["metadata"] = {
        "algorithm": algorithm,
        "stage": stage,
        "initialization": "random; no demos; no reference filter",
        "group": "discrete-78" if algorithm in DISCRETE_ALGORITHMS else "continuous-3",
        "encoder_source": "V107I/V108 incident-gated GNN Simba V6",
        "protocol": "experiments/tmrl_test_comparison/README.md",
    }
    if algorithm in EXPERIMENTAL_ALGORITHMS:
        base["metadata"].update(support_tier="experimental", included_in_campaign=False)
    return base


def main() -> None:
    from trackmaniarl import RunSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-experimental", action="store_true")
    args = parser.parse_args()
    paths = []
    experimental = []
    for stage, seeds in (("pilot", SEEDS[:1]), ("full", SEEDS)):
        directory = HERE / "configs" / stage
        directory.mkdir(parents=True, exist_ok=True)
        algorithms = ALGORITHMS + (EXPERIMENTAL_ALGORITHMS if args.include_experimental else ())
        for algorithm in algorithms:
            for seed in seeds:
                config = configuration(algorithm, seed, stage)
                RunSpec.model_validate(config)
                path = directory / f"{algorithm}-s{seed}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                destination = experimental if algorithm in EXPERIMENTAL_ALGORITHMS else paths
                destination.append(path.relative_to(ROOT).as_posix())
    (HERE / "manifest.json").write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    if args.include_experimental:
        (HERE / "experimental-manifest.json").write_text(
            json.dumps(experimental, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Generated {len(paths)} standard and {len(experimental)} experimental configs; "
        "no training started."
    )


if __name__ == "__main__":
    main()
