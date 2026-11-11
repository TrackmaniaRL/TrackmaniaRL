"""Run a bounded, local, game-free recurrent IQN training measurement on CPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import trackmaniarl
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.project.scaffold_run_templates import _trackmania_vision_config


class ToyEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.steps = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        return np.full((8, 8, 3), self.steps, dtype=np.uint8), 1.0, self.steps == 8, False, {}

    def close(self) -> None:
        pass


class ToyFactory:
    def create(self, *, seed: int, evaluation_map: Any = None) -> ToyEnvironment:
        return ToyEnvironment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--transitions", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"benchmark output already exists: {args.output}")
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
            raise OSError("Failed to reduce benchmark process priority")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    config = yaml.safe_load(_trackmania_vision_config("iqn"))
    config["run_id"] = "isolated-benchmark"
    config["artifacts_dir"] = str(args.output.resolve())
    config["seed"] = 42
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components["environment"] = {"class_path": "__main__:ToyFactory"}
    components["feature_pipeline"]["kwargs"] = {"config": {"width": 8, "height": 8}}
    components["model_factory"]["kwargs"]["temporal"] = {
        "class_path": "trackmaniarl.models.temporal:GruTemporalCore",
        "kwargs": {"input_dim": 256, "hidden_dim": 256},
    }
    components["learner"]["kwargs"]["execution"] = {"device": "cpu", "precision": "float32"}
    components["learner"]["kwargs"]["burn_in"] = 1
    components["replay_store"]["kwargs"] = {"capacity": 128}
    config["training"].update(
        total_transitions=args.transitions,
        max_episode_steps=8,
        batch_size=2,
        sequence_length=4,
        n_step=1,
        warmup_transitions=8,
        updates_per_transition=0.5,
        checkpoint_interval_updates=8,
    )
    spec = RunSpec.model_validate(config)
    start = time.perf_counter()
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "transitions": result.transitions,
                "updates": result.updates,
                "package_path": trackmaniarl.__file__,
            }
        )
    )


if __name__ == "__main__":
    main()
