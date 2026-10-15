"""Validate real evaluation artifacts and preserve every registered trial."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from trackmaniarl.research.manifest import sha256


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self, root: Path) -> Path:
        path = (root / self.path).resolve()
        if not path.is_file():
            raise ValueError(f"missing registered artifact: {path}")
        if sha256(path) != self.sha256:
            raise ValueError(f"registered artifact checksum mismatch: {path}")
        return path


class RegisteredRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    scenario: Literal["per-map", "multi-map", "zero-shot", "fine-tune", "historical"]
    map_id: str
    map_uid: str
    split: Literal["development", "train", "validation", "test"]
    training_seed: StrictInt | None
    timeout_s: float = Field(gt=0, allow_inf_nan=False)
    expected_trials: StrictInt = Field(gt=0)
    evaluation: Artifact
    config: Artifact | None = None
    checkpoint: Artifact | None = None
    source_snapshot: Artifact | None = None
    map_asset: Artifact | None = None
    provenance_gaps: list[str] = Field(default_factory=list)


class ResultRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"]
    evidence: Literal["confirmatory", "exploratory"]
    runs: list[RegisteredRun] = Field(min_length=1)


class Trial(BaseModel):
    model_config = ConfigDict(extra="ignore")
    map_id: str
    map_uid: str
    trial_index: StrictInt = Field(ge=0)
    finished: StrictBool
    finish_time_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    progress_pct: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    crashed: StrictBool | None = None
    telemetry_error: str | None = None
    controller_error: str | None = None
    termination_reason: str | None = None
    finish_time_source: Literal["race_clock", "elapsed_fallback"] | None = None
    action_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    action_latency_ms_p95: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_capture_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_capture_ms_p95: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_capture_measurement_count: int | None = Field(default=None, ge=0)
    vision_pairing_delay_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_pairing_delay_ms_p95: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_pairing_measurement_count: int | None = Field(default=None, ge=0)


def wilson_interval(successes: int, count: int) -> list[float]:
    """Descriptive 95% binomial interval within one checkpoint, not training uncertainty."""
    z = 1.959963984540054
    rate, correction = successes / count, z * z / count
    center = (rate + correction / 2) / (1 + correction)
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2))
    radius /= 1 + correction
    return [max(0.0, min(rate, center - radius)), min(1.0, max(rate, center + radius))]


def _check_provenance(run: RegisteredRun, root: Path, *, strict: bool) -> list[str]:
    missing = list(run.provenance_gaps)
    if run.training_seed is None:
        missing.append("training_seed")
    for name in ("config", "checkpoint", "source_snapshot", "map_asset"):
        artifact = getattr(run, name)
        if artifact is None:
            missing.append(name)
            continue
        artifact_path = artifact.verify(root)
        if name == "source_snapshot":
            from trackmaniarl.observability.provenance import validate_source_snapshot

            validate_source_snapshot(artifact_path)
        if name == "config" and run.training_seed is not None:
            config = yaml.safe_load(artifact_path.read_text("utf-8"))
            if not isinstance(config, dict) or type(config.get("seed")) is not int:
                raise ValueError("registered configuration must contain its integer training seed")
            if config["seed"] != run.training_seed:
                raise ValueError("declared training seed differs from registered configuration")
    if strict and missing:
        raise ValueError(f"confirmatory run {run.run_id} has provenance gaps: {missing}")
    return missing


def _load_trials(run: RegisteredRun, root: Path) -> tuple[list[Trial], list[str]]:
    path = run.evaluation.verify(root)
    payload = json.loads(path.read_text("utf-8"))
    missing = _check_evaluation_binding(run, payload)
    trials = [Trial.model_validate(row) for row in payload["trials"]]
    if len(trials) != run.expected_trials:
        raise ValueError(f"trial count mismatch for {run.run_id}; do not omit failed trials")
    identities = set()
    for trial in trials:
        if (trial.map_id, trial.map_uid) != (run.map_id, run.map_uid):
            raise ValueError(f"trial map identity mismatch for {run.run_id}")
        if trial.trial_index in identities:
            raise ValueError(f"duplicate trial index in {run.run_id}")
        identities.add(trial.trial_index)
        if trial.finished != (trial.finish_time_s is not None):
            raise ValueError("finished requires a time; a failed trial must not have a finish time")
        if trial.termination_reason is not None and (
            (trial.termination_reason == "finished") != trial.finished
        ):
            raise ValueError("trial finish status contradicts its termination reason")
        if trial.finished and (trial.telemetry_error or trial.controller_error):
            raise ValueError("a finished trial cannot also report a runtime error")
        if trial.finish_time_s is not None and trial.finish_time_s > run.timeout_s:
            raise ValueError(
                "recorded finish exceeds registered timeout; review protocol explicitly"
            )
    if identities != set(range(run.expected_trials)):
        raise ValueError("trial indices must cover exactly range(expected_trials)")
    return trials, missing


def _check_evaluation_binding(run: RegisteredRun, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("evaluation artifact must be an object")
    missing = []
    status = payload.get("status")
    if status is None:
        missing.append("evaluation_completion_status")
    elif status != "complete":
        raise ValueError("incomplete evaluation cannot be aggregated as a completed benchmark")
    expected_trials = payload.get("expected_trials")
    if expected_trials is None:
        missing.append("evaluation_trial_budget")
    elif type(expected_trials) is not int or expected_trials != run.expected_trials:
        raise ValueError("registered trial budget differs from the evaluation's declared budget")
    recorded = payload.get("checkpoint_sha256")
    if recorded is not None:
        if run.checkpoint is None or recorded != run.checkpoint.sha256:
            raise ValueError("evaluated checkpoint checksum differs from registered checkpoint")
    else:
        missing.append("evaluation_checkpoint_binding")
    return missing


def _failure_reason(trial: Trial) -> str:
    if trial.finished:
        return "finished"
    if trial.controller_error:
        return "controller_error"
    if trial.telemetry_error:
        return "telemetry_error"
    if trial.termination_reason:
        return trial.termination_reason
    return "crash" if trial.crashed else "unspecified_dnf"


def _summarize(run: RegisteredRun, trials: list[Trial], missing: list[str]) -> dict[str, Any]:
    times = [trial.finish_time_s for trial in trials if trial.finish_time_s is not None]
    costs = [
        trial.finish_time_s / run.timeout_s
        if trial.finished and trial.finish_time_s is not None
        else 1.0
        for trial in trials
    ]
    progress = [trial.progress_pct for trial in trials if trial.progress_pct is not None]
    latency = [trial.action_latency_ms for trial in trials if trial.action_latency_ms is not None]
    return {
        "run_id": run.run_id,
        "condition": run.condition,
        "scenario": run.scenario,
        "map_id": run.map_id,
        "map_uid": run.map_uid,
        "split": run.split,
        "training_seed": run.training_seed,
        "trials": len(trials),
        "finishes": len(times),
        "finish_rate": len(times) / len(trials),
        "finish_rate_trial_wilson95": wilson_interval(len(times), len(trials)),
        "timeout_s": run.timeout_s,
        "normalized_cost": mean(costs),
        "best_finish_s": min(times) if times else None,
        "median_finish_s": median(times) if times else None,
        "mean_finish_s": mean(times) if times else None,
        "progress_pct": mean(progress) if len(progress) == len(trials) else None,
        "action_latency_ms": mean(latency) if len(latency) == len(trials) else None,
        "outcomes": dict(Counter(_failure_reason(trial) for trial in trials)),
        "runtime_errors": sum(bool(t.telemetry_error or t.controller_error) for t in trials),
        "runtime_error_status_unknown_trials": sum(
            not {"telemetry_error", "controller_error"}.issubset(t.model_fields_set) for t in trials
        ),
        "provenance_gaps": missing,
        "missing_trial_metrics": [
            name
            for name in ("progress_pct", "action_latency_ms", "finish_time_source")
            if any(getattr(t, name) is None for t in trials)
        ],
    }


def _seed_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep maps and splits separate; bootstrap independent training seeds, never trials."""
    import numpy as np

    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["scenario"], row["map_id"], row["split"])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        if len({row["timeout_s"] for row in group}) != 1:
            raise ValueError("a seed group cannot combine different evaluation timeouts")
        seeds = [row["training_seed"] for row in group]
        if len([s for s in seeds if s is not None]) != len({s for s in seeds if s is not None}):
            raise ValueError(
                "multiple evaluations of a training seed cannot count as independent runs"
            )
        known = all(seed is not None for seed in seeds)
        interval = None
        if known and len(group) >= 2:
            values = np.array([row["normalized_cost"] for row in group])
            rng = np.random.default_rng(20260922)
            estimates = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            interval = np.quantile(estimates, [0.025, 0.975]).tolist()
        output.append(
            {
                "condition": key[0],
                "scenario": key[1],
                "map_id": key[2],
                "map_uid": group[0]["map_uid"],
                "split": key[3],
                "timeout_s": group[0]["timeout_s"],
                "independent_training_seeds": len(seeds) if known else None,
                "registered_evaluations": len(group),
                "finish_rate_seed_mean": mean(r["finish_rate"] for r in group),
                "normalized_cost_seed_mean": mean(r["normalized_cost"] for r in group),
                "normalized_cost_seed_bootstrap95": interval,
                "uncertainty_note": "Descriptive only; fewer than five seeds is unstable."
                if interval
                else "Training uncertainty unavailable; trials are not seeds.",
            }
        )
    return output


def _check_map_identities(runs: list[RegisteredRun]) -> None:
    """A map label must identify one UID and one split throughout the registry."""
    by_id: dict[str, str] = {}
    by_uid: dict[str, tuple[str, str]] = {}
    for run in runs:
        if by_id.setdefault(run.map_id, run.map_uid) != run.map_uid:
            raise ValueError("one map_id cannot identify different map UIDs")
        identity = (run.map_id, run.split)
        if by_uid.setdefault(run.map_uid, identity) != identity:
            raise ValueError("one map UID cannot use different map IDs or data splits")


def aggregate_registry(path: Path) -> dict[str, Any]:
    """Read only registered, hash-verified artifacts; fail instead of dropping missing runs."""
    registry = ResultRegistry.model_validate_json(path.read_text("utf-8"))
    _check_map_identities(registry.runs)
    rows: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    artifacts: set[str] = set()
    checkpoint_seeds: dict[str, int] = {}
    for run in registry.runs:
        if run.run_id in seen or run.evaluation.sha256 in artifacts:
            raise ValueError("duplicate run or evaluation artifact in registry")
        seen.add(run.run_id)
        artifacts.add(run.evaluation.sha256)
        if run.checkpoint is not None and run.training_seed is not None:
            previous_seed = checkpoint_seeds.setdefault(run.checkpoint.sha256, run.training_seed)
            if previous_seed != run.training_seed:
                raise ValueError("the same checkpoint cannot represent independent training seeds")
        strict = registry.evidence == "confirmatory"
        missing = _check_provenance(run, path.parent, strict=strict)
        trials, evaluation_gaps = _load_trials(run, path.parent)
        missing.extend(evaluation_gaps)
        if strict and evaluation_gaps:
            raise ValueError(f"confirmatory evaluation has provenance gaps: {evaluation_gaps}")
        if strict and any(
            trial.progress_pct is None
            or not trial.termination_reason
            or not {"telemetry_error", "controller_error"}.issubset(trial.model_fields_set)
            or (trial.finished and trial.finish_time_source != "race_clock")
            for trial in trials
        ):
            raise ValueError(
                "confirmatory trials require progress, termination, runtime error status, "
                "and race-clock times"
            )
        rows.append(_summarize(run, trials, missing))
        trial_records.extend({"run_id": run.run_id, **trial.model_dump()} for trial in trials)
    return {
        "schema_version": "1",
        "evidence": registry.evidence,
        "registry_sha256": sha256(path),
        "runs": rows,
        "groups": _seed_groups(rows),
        "trials": trial_records,
        "registry": registry.model_dump(),
    }
