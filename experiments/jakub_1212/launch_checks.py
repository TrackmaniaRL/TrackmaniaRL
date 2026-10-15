"""Verify completed runs and retain final measurements without selecting checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.commands.evaluation import (
    _benchmark,
    _benchmark_gate,
    _benchmark_gate_failure,
    _benchmark_gate_passed,
    _BenchmarkArtifact,
    _has_runtime_errors,
    _runtime_trial_statistics,
    _validate_benchmark_artifact,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.fingerprint import run_fingerprint
from trackmaniarl.core.spec import RunSpec

_TRAINING_CHECKPOINT = re.compile(r"(?:distributed-)?update-([0-9]+)\.pt")


@dataclass(frozen=True)
class LaunchRun:
    config: Path
    spec: RunSpec
    directory: Path

    @classmethod
    def load(cls, config: Path) -> LaunchRun:
        config = config.resolve()
        spec = RunSpec.from_yaml(config)
        return cls(config, spec, (config.parent / spec.artifacts_dir / spec.run_id).resolve())


def final_checkpoint(run: LaunchRun) -> Path:
    """Require the last requested training save to have completed successfully."""
    requested = completed = None
    with (run.directory / "events.jsonl").open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            if event["event"] not in {"train/checkpoint", "train/checkpoint_completed"}:
                continue
            path = Path(event["payload"]["path"]).resolve()
            if not _TRAINING_CHECKPOINT.fullmatch(path.name):
                continue
            if path.parent != run.directory / "checkpoints":
                raise RuntimeError("Checkpoint event refers outside this run")
            if event["event"] == "train/checkpoint":
                requested, completed = path, None
            elif path == requested:
                completed = path
    if requested is None or completed != requested or not requested.is_file():
        raise RuntimeError("The final requested training checkpoint was not successfully saved")
    state = TorchCheckpointCodec().load(requested)
    _validate_final_state(run, requested, state)
    return requested


def _validate_final_state(run: LaunchRun, path: Path, state: Mapping[str, Any]) -> None:
    if state.get("run_fingerprint") != run_fingerprint(run.spec, run.config.parent):
        raise RuntimeError("Final checkpoint fingerprint differs; resume only with its frozen code")
    if state.get("evaluated_policy_version") is not None:
        raise RuntimeError("An evaluated-policy checkpoint cannot replace the final training state")
    distributed = "distributed" in state
    counters = state["distributed" if distributed else "counters"]
    transitions, updates = counters["transitions"], counters["updates"]
    if (
        type(transitions) is not int
        or type(updates) is not int
        or transitions < run.spec.training.total_transitions
        or updates < 1
        or int(path.stem.rsplit("-", 1)[1]) != updates
    ):
        raise RuntimeError("Checkpoint does not contain the completed training budget")
    if distributed:
        credit = float(counters["update_credit"])
        training = run.spec.training
        expected = math.floor(
            max(0, transitions - training.warmup_transitions) * training.updates_per_transition
        )
        if not 0 <= credit < 1 or updates != expected:
            raise RuntimeError("Final checkpoint has missing updates or undrained update credit")
    else:
        rollout = run.spec.training.sequence_length
        processed = state["learner"].get("processed_transitions")
        if (
            transitions % rollout
            or updates != transitions // rollout
            or counters.get("fractional_updates") != 0.0
            or type(processed) is not int
            or processed != transitions
        ):
            raise RuntimeError("Final PPO checkpoint has an unprocessed or incomplete rollout")
    _finite_learner(state["learner"])


def _finite_learner(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError("Final checkpoint contains non-finite learner tensors")
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite_learner(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_learner(child)


def measurement_gate(run: LaunchRun, checkpoint: Path, artifact: Path) -> Any:
    evaluation = run.spec.evaluation
    if evaluation is None or evaluation.trials_per_map != 30:
        raise RuntimeError("Full comparison requires the configured 30-trial evaluation")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    expected = 30 * len(evaluation.maps)
    if payload.get("status") != "complete" or payload.get("expected_trials") != expected:
        raise RuntimeError("Evaluation is incomplete or cancelled; retain it and resume explicitly")
    recorded = payload.get("checkpoint")
    if recorded is None or Path(recorded).resolve() != checkpoint.resolve():
        raise RuntimeError("Evaluation is not bound to the final checkpoint")
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if payload.get("checkpoint_sha256") != digest:
        raise RuntimeError("Evaluation checkpoint hash does not match")
    trials = payload["trials"]
    _validate_benchmark_artifact(_BenchmarkArtifact(trials, recorded, evaluation, Path(recorded)))
    _, timing_valid = _runtime_trial_statistics(trials)
    interrupted = {"operator_interruption", "capture_interruption", "telemetry_error"}
    if (
        _has_runtime_errors(trials)
        or not timing_valid
        or any(trial.get("termination_reason") in interrupted for trial in trials)
    ):
        raise RuntimeError(
            "Evaluation has a controller, telemetry, timing, or interruption failure"
        )
    gate = _benchmark_gate(trials, payload["metrics"], evaluation)
    runtime_spec = evaluation.model_copy(
        update={"target_median_s": None, "target_mean_s": None, "min_finish_rate": 0.0}
    )
    if not _benchmark_gate_passed(_benchmark_gate(trials, payload["metrics"], runtime_spec)):
        raise RuntimeError("Evaluation failed the configured runtime checks")
    return gate


def report_measurement(run: LaunchRun, checkpoint: Path, artifact: Path) -> None:
    gate = measurement_gate(run, checkpoint, artifact)
    passed = _benchmark_gate_passed(gate)
    status = "passed" if passed else "failed (measurement retained; next seed may proceed)"
    print(f"Complete final measurement: {artifact}; configured benchmark gate {status}")
    if not passed:
        print(_benchmark_gate_failure(gate))


def benchmark_final(run: LaunchRun, checkpoint: Path, stop_file: Path) -> None:
    """Run the normal CLI benchmark; tolerate only its verified performance-gate failure."""
    if stop_file.exists():
        raise RuntimeError("STOP exists; benchmark was not started")
    pattern = f"{run.spec.run_id}-benchmark-*/evaluation.json"
    previous = set(run.directory.parent.glob(pattern))
    failure = None
    try:
        _benchmark(
            argparse.Namespace(
                config=run.config, checkpoint=checkpoint, trials=30, stop_file=stop_file
            )
        )
    except RuntimeError as exc:
        failure = exc
    created = set(run.directory.parent.glob(pattern)) - previous
    if len(created) != 1:
        raise RuntimeError(
            "Benchmark did not produce exactly one new final measurement"
        ) from failure
    artifact = created.pop()
    gate = measurement_gate(run, checkpoint, artifact)
    if failure is not None and (
        _benchmark_gate_passed(gate) or str(failure) != _benchmark_gate_failure(gate)
    ):
        raise failure
    if stop_file.exists():
        raise RuntimeError("STOP received; no further seed will start")
    report_measurement(run, checkpoint, artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checkpoint", "evaluation", "benchmark"))
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = LaunchRun.load(args.config)
    if args.command == "checkpoint":
        print(final_checkpoint(run))
    elif args.checkpoint is None:
        parser.error("checkpoint is required")
    elif args.command == "evaluation":
        report_measurement(run, args.checkpoint, run.directory / "evaluation.json")
    elif args.stop_file is None:
        parser.error("--stop-file is required for benchmark")
    else:
        benchmark_final(run, args.checkpoint.resolve(), args.stop_file.resolve())


if __name__ == "__main__":
    main()
