"""Bounded recurrent behavior cloning using the library's checkpoint/learner contract."""

from __future__ import annotations

import json
from pathlib import Path
from time import time_ns
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.runtime import prepare_run, record_run_attempt, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.driving_bc_data import load_episodes, sample_endpoints, sequence_batch
from trackmaniarl.trackmania.driving_recording import recording_contract
from trackmaniarl.trackmania.driving_vision import DrivingVisionFeaturePipeline
from trackmaniarl.trackmania.geometry import file_sha256
from trackmaniarl.trackmania.imitation_learning import class_weights


def train_driving_bc(config_path: Path, demos: Path, *, steps: int | None = None) -> Path:
    config_path = config_path.resolve()
    spec = RunSpec.from_yaml(config_path)
    spec = spec.model_copy(update={"run_id": f"{spec.run_id}-{time_ns()}"})
    run = resolve_run(spec, base_dir=config_path.parent)
    try:
        prepare_run(run)
        learner: Any = run.learner
        learner.setup(
            {"seed": spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
        )
        record_run_attempt(run)
        if not isinstance(run.feature_pipeline, DrivingVisionFeaturePipeline):
            raise ValueError("camera BC requires DrivingVisionFeaturePipeline")
        episodes = load_episodes(demos, run.feature_pipeline, recording_contract(config_path))
        generator = np.random.default_rng(spec.seed)
        order = generator.permutation(len(episodes))
        validation_count = max(1, round(len(episodes) * 0.2))
        validation = [episodes[i] for i in order[:validation_count]]
        training = [episodes[i] for i in order[validation_count:]]
        manifest = {
            "training": [str(e.path.resolve()) for e in training],
            "validation": [str(e.path.resolve()) for e in validation],
            "sha256": {str(e.path.resolve()): file_sha256(e.path) for e in episodes},
        }
        (run.run_dir / "bc-dataset.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        weights = class_weights(torch.cat([e.actions for e in training]), 78, power=0.25)
        batch_size, length = spec.training.batch_size, spec.training.sequence_length
        # Evaluate every held-out frame, not a noisy subset dominated by straight driving.
        validation_points = [
            (index, end)
            for index, episode in enumerate(validation)
            for end in range(len(episode.actions))
        ]
        best_loss, stale = float("inf"), 0
        best_path = run.run_dir / "checkpoints" / "bc-best-validation.pt"
        total_steps = learner.max_steps if steps is None else steps
        if total_steps < 1 or length <= learner.model.burn_in:
            raise ValueError("steps must be positive and sequence length must exceed burn-in")
        for step in range(1, total_steps + 1):
            endpoints = sample_endpoints(training, batch_size, generator)
            observations, labels = sequence_batch(training, endpoints, length)
            metrics = learner.train_batch(observations, labels, weights)
            if step % spec.training.metrics_interval_updates == 0:
                run.logger.log("bc/train", metrics, step=step)
            if step % learner.validation_interval and step != total_steps:
                continue
            numerator, denominator, correct, count = 0.0, 0.0, 0, 0
            per_action_correct, per_action_count = torch.zeros(78), torch.zeros(78)
            switch_correct, switch_count = 0, 0
            for offset in range(0, len(validation_points), batch_size):
                observations, labels = sequence_batch(
                    validation, validation_points[offset : offset + batch_size], length
                )
                report = learner.evaluate_batch(observations, labels, weights)
                numerator += report.loss_numerator
                denominator += report.loss_denominator
                correct += report.correct
                count += report.total
                per_action_correct += report.per_action_correct.cpu()
                per_action_count += report.per_action_count.cpu()
                switch_correct += report.steering_transition_correct
                switch_count += report.steering_transition_count
            loss = numerator / max(denominator, 1e-8)
            balanced = float(
                (per_action_correct / per_action_count.clamp_min(1))[per_action_count > 0].mean()
            )
            print(
                f"BC {step}/{total_steps}: loss={loss:.4f}, accuracy={correct / count:.3f}",
                flush=True,
            )
            run.logger.log(
                "bc/validation",
                {
                    "loss": loss,
                    "accuracy": correct / count,
                    "balanced_accuracy": balanced,
                    "steering_switch_accuracy": switch_correct / max(1, switch_count),
                    "steering_switch_count": switch_count,
                    "frames": count,
                },
                step=step,
            )
            learner.scheduler.step(loss)
            if loss < best_loss:
                best_loss, stale = loss, 0
                run.checkpoint_codec.save(
                    {
                        "schema_version": "trackmaniarl-bc-policy-v2",
                        "learner": learner.state_dict(),
                    },
                    best_path,
                )
            else:
                stale += 1
            if stale >= learner.early_stopping_patience:
                break
        print(f"BC checkpoint: {best_path}", flush=True)
        return best_path
    finally:
        run.logger.close()
