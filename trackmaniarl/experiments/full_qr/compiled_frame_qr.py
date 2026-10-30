"""Opt-in CUDA compilation for stateless QR frame batches, independent of the sensor."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from threading import Thread, current_thread
from time import perf_counter
from typing import Any, Unpack, cast

import torch

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.config import DiscreteValueOptions
from trackmaniarl.algorithms.value_based.updater import (
    ValueUpdater,
    _BatchStep,
    _ForwardStep,
    _LossStep,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.core.pytree import tree_map
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.heads import FixedQuantileHead
from trackmaniarl.models.temporal import IdentityTemporalCore

type _ForwardLoss = Callable[[_BatchStep], tuple[_ForwardStep, _LossStep]]
_WARMUP_SCHEMA = "tmrl.compiler-frame-batch/v1"
_WARMUP_MAX_BYTES = 256 * 1024**2
_BATCH_FIELDS = frozenset(field.name for field in fields(TrainingBatch))


def _load_warmup(pin: Mapping[str, str] | None) -> TrainingBatch:
    if pin is None or set(pin) != {"path", "sha256"}:
        raise ValueError("compiler_warmup requires an actual batch path and sha256")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError("compiler warmup requires a canonical file")
    if path.stat().st_size > _WARMUP_MAX_BYTES:
        raise ValueError("compiler warmup input exceeds256MiB")
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError("compiler warmup input hash changed")
    state = TorchCheckpointCodec(max_decompressed_bytes=_WARMUP_MAX_BYTES).load(path)
    if state.get("schema") != _WARMUP_SCHEMA:
        raise ValueError("compiler warmup frame-batch schema required")
    batch = state.get("batch")
    if not isinstance(batch, Mapping) or set(batch) != _BATCH_FIELDS:
        raise ValueError("compiler warmup requires every TrainingBatch field")
    return TrainingBatch(**dict(batch))


def _batch_signature(prepared: _BatchStep) -> object:
    view, batch = prepared.view, prepared.batch
    if view.sequence or view.n_step != 1:
        raise ValueError("compiled frame QR requires non-sequential one-step batches")

    def describe(value: Any) -> object:
        if isinstance(value, torch.Tensor):
            return tuple(value.shape), value.dtype, value.device.type
        if value is not None:
            raise TypeError("compiled frame QR requires tensor observation trees")
        return None

    values = (
        batch.observations,
        batch.next_observations,
        batch.actions,
        batch.rewards,
        batch.bootstrap_discounts,
        batch.masks,
        batch.importance_weights,
    )
    return view.batch_size, view.gamma, tree_map(describe, values)


class _CompiledFrameUpdater(ValueUpdater):
    def __init__(self, learner: DiscreteValueLearner, evaluate: _ForwardLoss) -> None:
        super().__init__(learner)
        self._evaluate = evaluate

    def _forward_losses(self, batch: _BatchStep) -> tuple[_ForwardStep, _LossStep]:
        return self._evaluate(batch)


class CompiledFrameQrLearner(DiscreteValueLearner):
    """Compile forward/loss on Linux CUDA float32; retain ordinary CPU replicas.

    A pinned, weights-only frame-batch archive warms the compiler without an
    optimizer step. Its shape, dtype and importance-weight layout must match
    training. Replay draws, priorities, diagnostics, Adam and target updates
    retain the regular value learner's validation and order.
    Setup and updates must share one thread, including CUDA graph ownership.
    """

    def __init__(
        self,
        model: CompositeValueModel | None = None,
        *,
        compiler_warmup: Mapping[str, str] | None = None,
        forward_compile_mode: str = "reduce-overhead",
        **options: Unpack[DiscreteValueOptions],
    ) -> None:
        if forward_compile_mode not in ("default", "reduce-overhead"):
            raise ValueError("forward_compile_mode must be default or reduce-overhead")
        self._compiler_warmup = None if compiler_warmup is None else dict(compiler_warmup)
        self._forward_compile_mode = forward_compile_mode
        self._forward_kernel: _ForwardLoss | None = None
        self._frame_signature: object = None
        self._compiler_warmup_s = 0.0
        self._execution_thread: Thread | None = None
        super().__init__(model, **options)

    def _setup_model(self) -> None:
        self._forward_kernel = None
        self._frame_signature = None
        self._compiler_warmup_s = 0.0
        self._execution_thread = None
        super()._setup_model()
        if not isinstance(self.model.head, FixedQuantileHead) or not isinstance(
            self.model.temporal, IdentityTemporalCore
        ):
            raise TypeError("compiled frame QR requires a fixed-quantile stateless model")
        if self.burn_in or self.objectives or self.fraction_optimizer is not None:
            raise ValueError("compiled frame QR does not support recurrent or auxiliary losses")
        if self.device.type != "cuda":
            return
        if (
            sys.platform != "linux"
            or self.resolved_execution is None
            or (self.resolved_execution.precision != "float32")
        ):
            raise ValueError("compiled frame QR requires Linux CUDA float32")
        for model in (self.model, self.target_model):
            model.head.dense_cuda_selected = True
        eager = ValueUpdater(self)
        self._execution_thread = current_thread()
        self._forward_kernel = cast(
            _ForwardLoss,
            torch.compile(
                eager._forward_losses,
                fullgraph=True,
                dynamic=False,
                mode=self._forward_compile_mode,
            ),
        )
        self._warmup(eager)

    def _warmup(self, updater: ValueUpdater) -> None:
        batch = _load_warmup(self._compiler_warmup)
        prepared = updater._batch_step(batch)
        self._frame_signature = _batch_signature(prepared)
        models = (self.model, self.target_model)
        before = [
            {key: value.clone() for key, value in model.state_dict().items()} for model in models
        ]
        rng = self._rng_state()
        started = perf_counter()
        try:
            self._mark_step()
            _, losses = self._evaluate_compiled(prepared)
            cast(Any, losses.total).backward()
            torch.cuda.synchronize(self.device)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._restore_rng(rng)
        if self.update_count != 0 or self.optimizer.state:
            raise RuntimeError("compiler warmup advanced the optimizer")
        for model, expected in zip(models, before, strict=True):
            if any(
                not torch.equal(value, expected[key]) for key, value in model.state_dict().items()
            ):
                raise RuntimeError("compiler warmup changed model parameters or buffers")
        self._compiler_warmup_s = perf_counter() - started

    def _mark_step(self) -> None:
        if self._forward_compile_mode == "reduce-overhead":
            cast(Callable[[], None], torch.compiler.cudagraph_mark_step_begin)()

    def _evaluate_compiled(self, batch: _BatchStep) -> tuple[_ForwardStep, _LossStep]:
        if _batch_signature(batch) != self._frame_signature:
            raise ValueError(
                "training batch differs from compiler warmup shape/dtype/weight layout"
            )
        assert self._forward_kernel is not None
        return self._forward_kernel(batch)

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        if self._forward_kernel is None:
            return super().update(batch)
        if current_thread() is not self._execution_thread:
            raise RuntimeError("compiled learner setup and updates must run on the same thread")
        self._mark_step()
        return _CompiledFrameUpdater(self, self._evaluate_compiled).update(batch)

    def execution_manifest(self) -> dict[str, object]:
        return {
            **super().execution_manifest(),
            "forward_loss_compilation": self._forward_compile_mode
            if self._forward_kernel is not None
            else "eager",
            "compiler_warmup_s": self._compiler_warmup_s,
            "compiler_warmup_optimizer_updates": 0,
            "compiler_warmup_sha256": None
            if self._compiler_warmup is None
            else self._compiler_warmup.get("sha256"),
            "cuda_selected_action_head": "dense" if self.device.type == "cuda" else "cpu_reference",
        }
