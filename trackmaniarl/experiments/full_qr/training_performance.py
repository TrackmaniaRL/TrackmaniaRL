"""Bounded CPU batch lookahead for uniform replay, with exact sampler checkpoints."""

import hashlib
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Unpack, cast

import torch

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.config import DiscreteValueOptions
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.algorithms.value_based.updater import ValueUpdater
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch
from trackmaniarl.core.replay import UniformSampler
from trackmaniarl.experiments.full_qr.components import BatchedIncidentEncoder
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.heads import FixedQuantileHead


class CompiledEncoderQrLearner(DiscreteValueLearner):
    """Compile CUDA encoders in place; retain eager CPU actors and checkpoint keys."""

    def __init__(  # noqa: PLR0913 -- model and independently qualified execution options
        self,
        model: CompositeValueModel | None = None,
        *,
        compiler_warmup: Mapping[str, str] | None = None,
        encoder_compile_mode: str = "default",
        batch_transfer_mode: str = "blocking",
        **options: Unpack[DiscreteValueOptions],
    ) -> None:
        if encoder_compile_mode not in ("default", "reduce-overhead"):
            raise ValueError("qualified encoder compile mode required")
        if batch_transfer_mode not in ("blocking", "pinned"):
            raise ValueError("batch transfer mode must be blocking or pinned")
        self._encoder_compile_mode = encoder_compile_mode
        self._batch_transfer_mode = batch_transfer_mode
        self._compiler_warmup = compiler_warmup
        self._compiler_warmup_s = 0.0
        super().__init__(model, **options)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.device.type == "cuda":
            if not isinstance(self.model.encoder, BatchedIncidentEncoder):
                raise TypeError("compiled full QR requires the qualified batched V6 encoder")
            if not isinstance(self.model.head, FixedQuantileHead):
                raise TypeError("compiled full QR requires the fixed quantile head")
            for model in (self.model, self.target_model):
                model.head.dense_cuda_selected = True
                model.encoder.compile(
                    backend="inductor",
                    mode=self._encoder_compile_mode,
                    fullgraph=True,
                    dynamic=False,
                )
            if self._compiler_warmup is None:
                raise ValueError("compiled training requires a pinned real startup batch")
            self._warmup_compiler()

    def _warmup_compiler(self) -> None:
        assert self._compiler_warmup is not None
        path = Path(self._compiler_warmup["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("canonical real compiler warmup input required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self._compiler_warmup["sha256"]:
            raise ValueError("compiler warmup input hash changed")
        batch = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(batch, TrainingBatch) or len(batch.transition_ids) != 256:
            raise ValueError("actual full QR batch256 required for compiler warmup")
        models = (self.model, self.target_model)
        before = [
            {key: value.clone() for key, value in model.state_dict().items()} for model in models
        ]
        rng_state = self._rng_state()
        started = perf_counter()
        try:
            updater = ValueUpdater(self)
            prepared = updater._batch_step(batch)
            losses = updater._loss_step(updater._forward_step(prepared))
            cast(Any, losses.total).backward()
            torch.cuda.synchronize(self.device)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._restore_rng(rng_state)
        if self.update_count != 0 or self.optimizer.state:
            raise RuntimeError("compiler warmup must not advance the optimizer")
        for model, expected in zip(models, before, strict=True):
            if any(
                not torch.equal(value, expected[key]) for key, value in model.state_dict().items()
            ):
                raise RuntimeError("compiler warmup changed model parameters or buffers")
        self._compiler_warmup_s = perf_counter() - started

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        if self.device.type == "cuda" and self._encoder_compile_mode == "reduce-overhead":
            # Online and target encoder calls belong to this same training step.
            # Their graph-owned outputs may only be recycled on the next step.
            cast(Callable[[], None], torch.compiler.cudagraph_mark_step_begin)()
        if (
            self.device.type == "cuda"
            and self._batch_transfer_mode == "pinned"
            and "_trackmaniarl_transfer_event" not in batch.metadata
        ):
            # Use the existing pinned-copy stream and consumer-event protocol.
            # _batch still waits for that event and records every CUDA tensor
            # on the consumer stream before any feature/value validation.
            batch = self.prepare_batch(batch)
        return super().update(batch)

    def policy(self) -> DiscreteValuePolicy:
        policy = super().policy()
        # Module.compile leaves parameter names intact, but its execution-only
        # callable must not follow a copied policy to another device or shape.
        policy.model.encoder._compiled_call_impl = None
        return policy

    def execution_manifest(self) -> dict[str, object]:
        return {
            **super().execution_manifest(),
            "encoder_compilation": (
                f"inductor/{self._encoder_compile_mode}" if self.device.type == "cuda" else "eager"
            ),
            "compiler_warmup_s": self._compiler_warmup_s,
            "compiler_warmup_optimizer_updates": 0,
            "cuda_selected_action_head": "dense" if self.device.type == "cuda" else "sparse",
            "batch_transfer": (
                "pinned_async"
                if self.device.type == "cuda" and self._batch_transfer_mode == "pinned"
                else "blocking"
            ),
        }


@dataclass(frozen=True, slots=True)
class _PreparedUniformBatch:
    batch: TrainingBatch
    preparation_s: float


class PrefetchedUniformSampler(UniformSampler):
    """Own at most one fully materialized batch while the learner updates.

    Uniform sampling does not use beta or transition_count. Its next draw may
    see replay one learner update earlier; no rewards, targets or IDs change
    after materialization. Checkpoints discard this unused draw and rewind its
    RNG, so there is no hidden speculative state to lose on restore.
    """

    def __init__(self, pipeline: Any, seed: int = 0, *, transfer_prefetch: bool = False) -> None:
        if type(transfer_prefetch) is not bool:
            raise TypeError("transfer_prefetch must be an explicit bool")
        super().__init__(pipeline, seed)
        self._transfer_prefetch = transfer_prefetch
        self._batch_preparer: Callable[[TrainingBatch], TrainingBatch] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._pending: Future[_PreparedUniformBatch] | None = None
        self._pending_request: tuple[Any, ...] | None = None
        self._before_pending: dict[str, Any] | None = None
        self._closed = False
        self._last_timings = (0.0, 0.0)

    def bind_batch_preparer(self, preparer: Any) -> None:
        """Bind the learner's existing cross-stream transfer, only by opt-in."""
        if not self._transfer_prefetch:
            return
        if self._pending is not None or self._closed:
            raise RuntimeError("bind batch transfer before starting the sampler")
        if not callable(preparer):
            raise TypeError("transfer prefetch requires a learner batch preparer")
        self._batch_preparer = preparer

    @staticmethod
    def _request_key(store: Any, request: BatchRequest) -> tuple[Any, ...]:
        return (
            id(store),
            request.batch_size,
            request.sequence_length,
            request.n_step,
            request.gamma,
        )

    def sample(self, store: Any, request: BatchRequest) -> TrainingBatch:
        started = perf_counter()
        if self._closed:
            raise RuntimeError("uniform batch prefetcher is closed")
        key = self._request_key(store, request)
        if self._pending is not None and self._pending_request != key:
            self._discard_pending()
        if self._pending is None:
            prepared = self._sample_locked(store, request)
        else:
            # A failed speculative draw remains pending until close/state_dict
            # rewinds it; never silently replace a source failure with a retry.
            prepared = self._pending.result()
            self._pending = None
            self._before_pending = None
        self._before_pending = super().state_dict()
        self._pending_request = key
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qr-batch")
        self._pending = self._executor.submit(self._sample_locked, store, request)
        self._last_timings = (prepared.preparation_s, perf_counter() - started)
        return prepared.batch

    def _sample_locked(self, store: Any, request: BatchRequest) -> _PreparedUniformBatch:
        # Keep ID selection and ownership copies in one transaction. FIFO
        # eviction cannot invalidate selected rows before materialization.
        started = perf_counter()
        if self._transfer_prefetch and self._batch_preparer is None:
            raise RuntimeError("transfer prefetch must be bound before sampling")
        with store.sampling_transaction():
            batch = super().sample(store, request)
        # Pinning and H2D enqueueing must never hold the replay transaction.
        # The consumer waits for the existing CUDA event and records streams;
        # this worker owns at most the same one speculative batch as before.
        if self._transfer_prefetch:
            assert self._batch_preparer is not None
            batch = self._batch_preparer(batch)
        return _PreparedUniformBatch(batch, perf_counter() - started)

    def prefetch_timings(self) -> tuple[float, float]:
        """Full consumed-batch CPU cost and caller wait, rather than hidden work."""
        return self._last_timings

    def _discard_pending(self) -> None:
        if self._pending is None:
            return
        future, state = self._pending, self._before_pending
        try:
            future.result()
        finally:
            self._pending = None
            self._pending_request = None
            self._before_pending = None
            assert state is not None
            super().load_state_dict(state)

    def state_dict(self) -> dict[str, Any]:
        self._discard_pending()
        return super().state_dict()

    def load_state_dict(self, state: Any) -> None:
        self._discard_pending()
        super().load_state_dict(state)

    def close_prefetch(self) -> None:
        try:
            self._discard_pending()
        finally:
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
