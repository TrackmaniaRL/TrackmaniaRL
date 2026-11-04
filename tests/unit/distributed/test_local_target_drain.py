"""The local learner must keep accepting WAL until its actor has exited."""

import multiprocessing
from types import SimpleNamespace

from trackmaniarl.distributed.coordinator_learning import _learning_needed, _LearningLoop
from trackmaniarl.distributed.stop_signal import ProcessStopSignal


def _coordinator(stop: ProcessStopSignal) -> SimpleNamespace:
    return SimpleNamespace(
        external_stop=stop,
        _external_stop_requested=stop.is_set,
        _evaluation_stop_reason=None,
        counters=SimpleNamespace(transitions=100, update_credit=0.0, journal_applied_frontier=0),
        run=SimpleNamespace(
            spec=SimpleNamespace(
                training=SimpleNamespace(
                    total_transitions=100,
                    batch_size=1,
                    sequence_length=1,
                    n_step=1,
                    warmup_transitions=1,
                )
            ),
            replay_store=[],
        ),
        journal=SimpleNamespace(has_rows_after=lambda _: False),
        _should_stop=lambda: True,
    )


def test_local_target_waits_until_actor_has_finished_submitting() -> None:
    stop = ProcessStopSignal(multiprocessing.get_context("spawn"), wait_for_actor_drain=True)
    coordinator = _coordinator(stop)
    loop = _LearningLoop(coordinator, None)  # type: ignore[arg-type]
    assert _learning_needed(loop)
    stop.request_drain()
    assert not _learning_needed(loop)


def test_nonlocal_coordinator_keeps_previous_stop_behavior() -> None:
    stop = ProcessStopSignal(multiprocessing.get_context("spawn"))
    loop = _LearningLoop(_coordinator(stop), None)  # type: ignore[arg-type]
    assert not _learning_needed(loop)


def test_late_journal_is_drained_after_actor_exit() -> None:
    stop = ProcessStopSignal(multiprocessing.get_context("spawn"), wait_for_actor_drain=True)
    coordinator = _coordinator(stop)
    coordinator.journal = SimpleNamespace(has_rows_after=lambda _: True)
    stop.request_drain()
    loop = _LearningLoop(coordinator, None)  # type: ignore[arg-type]
    assert _learning_needed(loop)
    coordinator.journal = SimpleNamespace(has_rows_after=lambda _: False)
    assert not _learning_needed(loop)
