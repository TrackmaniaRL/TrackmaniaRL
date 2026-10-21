"""Exercise the training lifecycle with a tiny custom environment and reward.

SmokeLearner counts updates; it does not learn to drive. No game or GPU is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trackmaniarl import RunSpec, Trainer, resolve_run


class ToyEnvironment:
    """Eight-step episodes with a reward for matching a target action."""

    def __init__(self, target_action: float) -> None:
        self.target_action = target_action
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, float]]:
        del seed
        self.steps = 0
        return 0.0, {}

    def step(self, action: float) -> tuple[float, float, bool, bool, dict[str, float]]:
        self.steps += 1
        reward = 1.0 - abs(action - self.target_action)
        return float(self.steps), reward, self.steps == 8, False, {}


class ToyEnvironmentFactory:
    """The runtime creates environments through this public component contract."""

    def __init__(self, target_action: float = 0.0) -> None:
        self.target_action = target_action

    def create(self, *, seed: int) -> ToyEnvironment:
        del seed
        return ToyEnvironment(self.target_action)


def make_spec(output: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "python-example",
            "artifacts_dir": output.resolve(),
            "components": {
                "environment": {
                    "class_path": f"{__name__}:ToyEnvironmentFactory",
                    "kwargs": {"target_action": 0.0},
                },
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
                "replay_store": {
                    "class_path": "trackmaniarl.core.replay:InMemoryReplayStore",
                    "kwargs": {"capacity": 64},
                },
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
            },
            "training": {
                "total_transitions": 32,
                "max_episode_steps": 8,
                "batch_size": 4,
                "warmup_transitions": 4,
                "checkpoint_interval_updates": 16,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples"))
    args = parser.parse_args()
    run = resolve_run(make_spec(args.output))
    try:
        result = Trainer(run).train()
        print(result)
    finally:
        run.logger.close()


if __name__ == "__main__":
    main()
