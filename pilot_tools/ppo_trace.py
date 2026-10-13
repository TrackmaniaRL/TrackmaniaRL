"""Bounded in-memory evaluation traces, flushed only between races."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator

TRACE_FIELDS = (
    "race_time_ms", "position", "progress_pct", "progress_m", "projected_velocity_mps",
    "nearest_distance_m", "steps_since_progress", "window_progress_m", "termination_reason",
    "step_race_time_ms", "telemetry_skipped_frames", "controller_apply_ms",
)


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


class TraceEvaluator(TrackmaniaEvaluator):
    def __init__(self, request: Any) -> None:
        super().__init__(request)
        if self.run_dir is None:
            raise ValueError("Trace evaluator requires a run directory")
        self.root = self.run_dir
        self.rows: list[dict[str, Any]] = []

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if path.name not in {"update-00000050.pt", "update-00000100.pt"}:
            raise ValueError("Only the two preregistered checkpoints may be evaluated")
        if path.name == "update-00000050.pt":
            archive = self.root / "evaluation-checkpoints" / path.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source, archive.open("xb") as target:
                shutil.copyfileobj(source, target)
            path = archive
        super().set_checkpoint(path)
        self.run_dir = self.root / "evaluations" / path.stem

    def _start_episode(self, request: Any, context: Any) -> Any:
        loop = super()._start_episode(request, context)
        self.rows = [{"kind": "reset", "raw_observation": serializable(loop.observation)}]
        return loop

    def _take_step(self, loop: Any) -> Any:
        current = serializable(loop.observation)
        step = super()._take_step(loop)
        if len(self.rows) > self.max_episode_steps:
            raise RuntimeError("Evaluation trace exceeds the bounded episode budget")
        self.rows.append({
            "kind": "step", "step": len(self.rows) - 1,
            "raw_current": current, "raw_next": serializable(step.observation),
            "issued_action": serializable(step.action),
            "action_latency_ms": step.action_duration_ms,
            "terminated": step.terminated, "truncated": step.truncated,
            "after_action": {key: serializable(step.info[key]) for key in TRACE_FIELDS
                             if key in step.info},
        })
        return step

    def _run_episode(self, request: Any, context: Any, state: Any) -> None:
        try:
            super()._run_episode(request, context, state)
        finally:
            assert self.run_dir is not None
            self.run_dir.mkdir(parents=True, exist_ok=True)
            path = self.run_dir / f"trial-{request.trial_index:02d}.json.gz"
            payload = {
                "schema_version": 1, "checkpoint": self.checkpoint,
                "checkpoint_sha256": self.checkpoint_sha256,
                "trial_index": request.trial_index, "map_uid": request.map_spec.expected_map_uid,
                "rows": self.rows,
            }
            with path.open("xb") as target:
                target.write(gzip.compress(json.dumps(payload, allow_nan=False).encode()))
            self.rows = []

    def _write_artifact(self, results: list[Any], metrics: dict[str, float], **kwargs: Any) -> None:
        super()._write_artifact(results, metrics, **kwargs)
        if (self.checkpoint is not None and Path(self.checkpoint).name == "update-00000100.pt"
                and kwargs.get("status", "complete") == "complete"):
            assert self.run_dir is not None
            with (
                (self.run_dir / "evaluation.json").open("rb") as source,
                (self.root / "evaluation.json").open("xb") as target,
            ):
                shutil.copyfileobj(source, target)
