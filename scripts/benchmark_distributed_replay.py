"""Bounded game-free distributed replay measurement; use an isolated output parent."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import trackmaniarl
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import run_fingerprint
from trackmaniarl.project.scaffold_run_templates import _trackmania_vision_config


class ToyEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.steps = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Bound actor production independently of learner and disk scheduling.
        time.sleep(0.01)
        self.steps += 1
        return np.full((8, 8, 3), self.steps, dtype=np.uint8), 1.0, self.steps == 8, False, {}

    def close(self) -> None:
        pass


class ToyFactory:
    def create(self, *, seed: int, evaluation_map: Any = None) -> ToyEnvironment:
        return ToyEnvironment()


def _limit_process() -> None:
    if sys.platform != "win32":
        raise RuntimeError("this measurement requires Windows priority and affinity controls")
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    process = kernel.GetCurrentProcess()
    mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(mask), ctypes.byref(system_mask)):
        raise OSError("cannot read process affinity")
    if not kernel.SetProcessAffinityMask(process, mask.value & -mask.value):
        raise OSError("cannot constrain process affinity")
    if not kernel.SetPriorityClass(process, 0x00004000):
        raise OSError("cannot lower priority")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--transitions", type=int, default=64)
    args = parser.parse_args()
    _limit_process()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    torch.manual_seed(42)
    config = yaml.safe_load(_trackmania_vision_config("iqn"))
    config.update(run_id="isolated-distributed-benchmark", artifacts_dir=str(output), seed=42)
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
    config["distributed"] = {
        "rollout_chunk_transitions": 8,
        "rollout_flush_s": 0.1,
        "policy_refresh_s": 0.2,
        "heartbeat_s": 0.2,
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = RunSpec.model_validate(config)
    run = resolve_run(spec, base_dir=Path(trackmaniarl.__file__).resolve().parent.parent)
    fingerprint = run_fingerprint(spec, output)
    token = "isolated-benchmark-token-20261112"
    coordinator = Coordinator(run, CoordinatorConfig("127.0.0.1:0", token, fingerprint))
    external_stop = threading.Event()
    coordinator.external_stop = external_stop
    errors: list[BaseException] = []
    results: list[Any] = []

    def serve() -> None:
        try:
            results.append(coordinator.run_forever())
        except BaseException as exc:
            errors.append(exc)

    def drive(actor: ActorRuntime) -> None:
        try:
            actor.run_forever()
        except BaseException as exc:
            errors.append(exc)

    started = time.perf_counter()
    server = threading.Thread(target=serve, daemon=True)
    server.start()
    deadline = time.monotonic() + 90
    actor = None
    actor_thread = None
    try:
        while coordinator._bound_port is None:
            if errors or not server.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"coordinator failed to start: {errors}")
            time.sleep(0.01)
        actor = ActorRuntime(
            ActorProcessRequest(
                config_path, f"127.0.0.1:{coordinator.bound_port}", "benchmark-actor", token
            )
        )
        actor_thread = threading.Thread(target=drive, args=(actor,), daemon=True)
        actor_thread.start()
        server.join(max(0.0, deadline - time.monotonic()))
        if server.is_alive() or errors:
            raise RuntimeError(f"distributed benchmark did not complete: {errors}")
    finally:
        if actor is not None:
            actor.stop.set()
        if actor_thread is not None:
            actor_thread.join(timeout=10)
        if server.is_alive():
            external_stop.set()
            server.join(timeout=10)
    if not results or errors or (actor_thread is not None and actor_thread.is_alive()):
        raise RuntimeError(f"incomplete benchmark: {errors}")
    result = results[0]
    print(
        json.dumps(
            {
                "elapsed_s": time.perf_counter() - started,
                "transitions": result.transitions,
                "updates": result.updates,
                "package_path": trackmaniarl.__file__,
                "port": coordinator.bound_port,
            }
        )
    )


if __name__ == "__main__":
    main()
