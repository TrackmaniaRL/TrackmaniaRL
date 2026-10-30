from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.contracts import CheckpointSnapshotProvider
from trackmaniarl.core.data import BatchRequest, TrainingBatch
from trackmaniarl.core.runtime import ResolvedRun

ROLLOUT_QUEUE_MAXSIZE = 64


@dataclass(slots=True)
class _Counters:
    transitions: int = 0
    episodes: int = 0
    finishes: int = 0
    best_finish_time_s: float = 0.0
    evaluations: int = 0
    evaluation_finishes: int = 0
    evaluation_bucket_finishes: dict[str, int] = field(default_factory=dict)
    updates: int = 0
    update_credit: float = 0.0
    journal_applied_frontier: int = 0
    policy_version: int = 0
    actor_sequences: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    batch: TrainingBatch
    preparation_s: float


class _BatchPrefetcher:
    """Batch source; optional samplers own and quiesce their bounded lookahead."""

    def __init__(self, run: ResolvedRun) -> None:
        self.run = run
        binder = getattr(type(run.sampler), "bind_batch_preparer", None)
        if callable(binder):
            learner = getattr(run, "learner", None)
            binder(run.sampler, getattr(learner, "prepare_batch", None))

    def next(self, request: BatchRequest) -> tuple[TrainingBatch, float, float]:
        prepared = self._prepare(request)
        timings = getattr(type(self.run.sampler), "prefetch_timings", None)
        if callable(timings):
            preparation_s, wait_s = timings(self.run.sampler)
            return prepared.batch, preparation_s, wait_s
        return prepared.batch, prepared.preparation_s, 0.0

    def _prepare(self, request: BatchRequest) -> _PreparedBatch:
        started = perf_counter()
        batch = self.run.sampler.sample(self.run.replay_store, request)
        return _PreparedBatch(batch, perf_counter() - started)

    def close(self) -> None:
        closer = getattr(type(self.run.sampler), "close_prefetch", None)
        if callable(closer):
            closer(self.run.sampler)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.maximums: dict[str, float] = {}
        self.terminal_error_sum = 0.0
        self.terminal_samples = 0.0
        self.terminal_observations = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        terminal_fields = (
            "critic/terminal_td_mae",
            "critic/terminal_td_abs_error_sum",
            "critic/terminal_samples",
        )
        if all(key in metrics for key in terminal_fields):
            self.terminal_error_sum += float(metrics["critic/terminal_td_abs_error_sum"])
            self.terminal_samples += float(metrics["critic/terminal_samples"])
            self.terminal_observations += 1
        for key, value in metrics.items():
            numeric = float(value)
            if key.endswith("_max"):
                self.maximums[key] = max(self.maximums.get(key, numeric), numeric)
            else:
                self.values[key] = self.values.get(key, 0.0) + numeric
                self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self) -> dict[str, float]:
        if not self.values and not self.maximums:
            return {}
        output = {key: value / self.counts[key] for key, value in self.values.items()}
        output.update(self.maximums)
        terminal_mae = "critic/terminal_td_mae"
        if terminal_mae in output:
            # Empty batches must not dilute the error on actual terminal samples.
            if self.terminal_observations == self.counts[terminal_mae]:
                output[terminal_mae] = (
                    self.terminal_error_sum / self.terminal_samples
                    if self.terminal_samples > 0
                    else 0.0
                )
                output["critic/terminal_samples_total"] = self.terminal_samples
            else:
                # Unpaired statistics cannot support a sample-weighted mean.
                del output[terminal_mae]
        self.values.clear()
        self.counts.clear()
        self.maximums.clear()
        self.terminal_error_sum = 0.0
        self.terminal_samples = 0.0
        self.terminal_observations = 0
        return output


@dataclass(frozen=True, slots=True)
class _CheckpointWrite:
    state: Mapping[str, Any]
    path: Path
    on_saved: Callable[[], None] | None
    on_failed: Callable[[BaseException], None] | None


@dataclass(frozen=True, slots=True)
class _RolloutRejection:
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)


class _AsyncCheckpointWriter:
    def __init__(self, codec: Any) -> None:
        self.codec = codec
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trackmaniarl-checkpoint"
        )
        self.pending: Any = None

    def submit(self, request: _CheckpointWrite) -> None:
        self.wait()
        self.pending = self.executor.submit(self._save, request)

    def _save(self, request: _CheckpointWrite) -> None:
        try:
            self.codec.save(request.state, request.path)
            sync_checkpoint_path(request.path)
            if request.on_saved is not None:
                request.on_saved()
        except BaseException as exc:
            if request.on_failed is not None:
                request.on_failed(exc)
            raise

    def wait(self) -> None:
        pending = self.pending
        if pending is None:
            return
        try:
            pending.result()
        finally:
            if pending.done() and self.pending is pending:
                self.pending = None

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)


def state_dict(component: object) -> Mapping[str, object]:
    method = getattr(component, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{type(component).__name__}.state_dict() must return a mapping")
    return cast(Mapping[str, object], state)


def load_state_dict(component: object, state: object) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint component state must be a mapping")
    method = getattr(component, "load_state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no load_state_dict()")
    method(state)


def snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_value(item) for item in value)
    return deepcopy(value)


def snapshot_state_dict(component: object) -> Mapping[str, object]:
    """Capture detached CPU state, honoring an explicit component capability.

    An opt-in ``snapshot_state_dict`` must return caller-owned mutable values;
    the component may continue changing its live state while the writer runs.
    Other components retain the defensive recursive-copy path.
    """
    if not isinstance(component, CheckpointSnapshotProvider):
        return cast(Mapping[str, object], snapshot_value(state_dict(component)))
    state = component.snapshot_state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{type(component).__name__}.snapshot_state_dict() must return a mapping")
    return cast(Mapping[str, object], state)
