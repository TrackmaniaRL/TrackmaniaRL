import logging
import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.integration.runtime.distributed_evaluation_support import (
    _stopped_coordinator,
)
from tests.integration.runtime.distributed_runtime_support import (
    _crash_submit_request,
    _transition,
    _TransitionSpec,
)
from trackmaniarl.commands.training import (
    _LocalProcesses,
    _shutdown_local_processes,
    _stop_file_requested,
    _stop_local_processes_with_notice,
    _wait_for_processes,
)
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_runtime import (
    _run_learning,
    _run_offline_pretraining,
    close_runtime,
)
from trackmaniarl.distributed.coordinator_support import _AsyncCheckpointWriter
from trackmaniarl.distributed.protocol import transition_to_wire
from trackmaniarl.distributed.stop_signal import ProcessStopSignal


def test_stop_file_requests_shutdown_without_console_signal(tmp_path: Path) -> None:
    flag = tmp_path / "stop"
    event = threading.Event()
    process = SimpleNamespace(is_alive=lambda: True)
    processes = _LocalProcesses(process, process, event, "unused", flag)
    assert not _stop_file_requested(processes)
    assert not event.is_set()
    flag.touch()
    _wait_for_processes(processes)
    assert event.is_set()


def test_stop_file_stops_only_collection_until_actor_has_closed(tmp_path: Path) -> None:
    flag = tmp_path / "stop"
    flag.touch()
    learner_stop, actor_stop = threading.Event(), threading.Event()
    process = SimpleNamespace(is_alive=lambda: True)
    processes = _LocalProcesses(process, process, learner_stop, "unused", flag, actor_stop)
    assert _wait_for_processes(processes)
    assert actor_stop.is_set()
    assert not learner_stop.is_set()


def test_graceful_stop_drains_journal_and_credit_before_partial_budget_save(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = _stopped_coordinator(tmp_path, events)
    stop = ProcessStopSignal(multiprocessing.get_context("spawn"))
    stop.request_drain()
    coordinator.external_stop = stop
    coordinator._prepare_training()
    for index in range(3):
        payload = dict(coordinator.codec.decode(_crash_submit_request(coordinator).value))
        payload["sequence"] = index
        payload["transitions"] = [
            transition_to_wire(_transition(_TransitionSpec("actor", index, 1.0)))
        ]
        row, _ = coordinator.journal.append(
            "crashed-session", index, coordinator.codec.encode(payload)
        )
        coordinator._rollouts.put((row, 0.0))
    try:
        result = _run_learning(coordinator)
    finally:
        close_runtime(coordinator)
    assert coordinator.counters.transitions < coordinator.run.spec.training.total_transitions
    assert coordinator.counters.transitions == 3
    assert coordinator.counters.updates == 2
    assert coordinator.counters.update_credit == 0
    checkpoint = coordinator.run.checkpoint_codec.load(result.checkpoints[-1])
    assert checkpoint["distributed"]["updates"] == 2
    assert checkpoint["distributed"]["update_credit"] == 0
    assert checkpoint["distributed"]["journal_applied_frontier"] == 3


def test_actor_closes_before_drain_is_requested() -> None:
    events: list[str] = []
    learner_stop = ProcessStopSignal(multiprocessing.get_context("spawn"))
    actor_stop = threading.Event()

    class Actor(_ShutdownProcess):
        def join(self, *, timeout: float) -> None:
            assert actor_stop.is_set()
            assert not learner_stop.drain_requested()
            events.append("actor_closed")
            super().join(timeout=timeout)

    class Learner(_ShutdownProcess):
        def join(self, *, timeout: float) -> None:
            assert learner_stop.drain_requested()
            assert not learner_stop.is_set()
            events.append("learner_drained_and_saved")
            super().join(timeout=timeout)

    processes = _LocalProcesses(
        Learner(), Actor(), learner_stop, "unused", actor_shutdown=actor_stop
    )
    assert not _shutdown_local_processes(processes, drain=True)
    assert events == ["actor_closed", "learner_drained_and_saved"]


def test_failed_actor_requests_emergency_save_instead_of_drain() -> None:
    stop = ProcessStopSignal(multiprocessing.get_context("spawn"))
    actor = _ShutdownProcess()
    actor.exitcode = 7
    processes = _LocalProcesses(
        _ShutdownProcess(), actor, stop, "unused", actor_shutdown=threading.Event()
    )
    assert not _shutdown_local_processes(processes, drain=True)
    assert stop.is_set()
    assert not stop.drain_requested()


def test_shutdown_timeout_is_not_reported_as_a_successful_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(range(1000))
    monkeypatch.setattr("trackmaniarl.commands.training.monotonic", lambda: float(next(clock)))

    class StuckLearner(_ShutdownProcess):
        def join(self, *, timeout: float) -> None:
            del timeout

    processes = _LocalProcesses(
        StuckLearner(),
        _ShutdownProcess(),
        ProcessStopSignal(multiprocessing.get_context("spawn")),
        "unused",
        actor_shutdown=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="did not confirm a clean final checkpoint"):
        _stop_local_processes_with_notice(processes)
    assert not processes.learner.is_alive()


def test_actor_failure_during_stop_is_not_reported_as_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    actor = _ShutdownProcess()
    actor.exitcode = 7
    processes = _LocalProcesses(
        _ShutdownProcess(),
        actor,
        ProcessStopSignal(multiprocessing.get_context("spawn")),
        "unused",
        actor_shutdown=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="actor process exited with code 7"):
        _stop_local_processes_with_notice(processes)
    assert "Safe shutdown complete" not in capsys.readouterr().out


class _ShutdownProcess:
    exitcode = 0
    pid = 1

    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, *, timeout: float) -> None:
        del timeout
        self.alive = False

    def terminate(self) -> None:
        self.alive = False


class _InterruptingShutdownProcess(_ShutdownProcess):
    def __init__(self) -> None:
        super().__init__()
        self.pending_interrupt = True

    def join(self, *, timeout: float) -> None:
        if self.pending_interrupt:
            self.pending_interrupt = False
            raise KeyboardInterrupt
        super().join(timeout=timeout)


class _InterruptedWait:
    def __init__(self) -> None:
        self.calls = 0

    def result(self) -> None:
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt

    def done(self) -> bool:
        return self.calls > 1


class _CloseProbe:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.failure is not None:
            raise self.failure


def test_graceful_shutdown_ignores_repeated_interrupt_until_checkpoint_is_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    processes = _LocalProcesses(
        learner=_ShutdownProcess(),
        actor=_InterruptingShutdownProcess(),
        shutdown=SimpleNamespace(set=lambda: None),
        endpoint="127.0.0.1:8787",
    )

    _stop_local_processes_with_notice(processes)

    output = capsys.readouterr().out
    assert "Ctrl+C was ignored to protect the checkpoint" in output
    assert "Safe shutdown complete" in output


def test_checkpoint_writer_retains_an_in_progress_write_after_interrupted_wait() -> None:
    writer = _AsyncCheckpointWriter(SimpleNamespace())
    pending = _InterruptedWait()
    writer.pending = pending
    try:
        with pytest.raises(KeyboardInterrupt):
            writer.wait()
        assert writer.pending is pending
        writer.wait()
        assert writer.pending is None
    finally:
        writer.close()


def test_runtime_closes_the_journal_after_a_checkpoint_writer_failure() -> None:
    writer = _CloseProbe(OSError("checkpoint failed"))
    journal = _CloseProbe()
    coordinator = cast(
        Coordinator,
        SimpleNamespace(
            _server=None,
            _rpc_executor=None,
            _checkpoint_writer=writer,
            journal=journal,
        ),
    )

    with pytest.raises(OSError, match="checkpoint failed"):
        close_runtime(coordinator)

    assert writer.closed
    assert journal.closed


def _offline_spec() -> SimpleNamespace:
    training = SimpleNamespace(offline_pretrain_updates=1, save_final_checkpoint=True)
    return SimpleNamespace(training=training)


class _OfflineInterruptHarness:
    def __init__(self, tmp_path: Path, captured: dict[str, int]) -> None:
        self.tmp_path = tmp_path
        self.captured = captured

    def interrupt(self) -> None:
        raise KeyboardInterrupt

    def save_checkpoint(self) -> Path:
        self.captured["checkpoints"] += 1
        return self.tmp_path / "interrupted.pt"

    def wait(self) -> None:
        self.captured["waits"] += 1


def _interrupted_coordinator(tmp_path: Path, captured: dict[str, int]) -> Coordinator:
    harness = _OfflineInterruptHarness(tmp_path, captured)
    coordinator = SimpleNamespace(
        run=SimpleNamespace(spec=_offline_spec()),
        demo_paths=(tmp_path / "demo.npz",),
        journal=SimpleNamespace(has_history=lambda: False),
        _prepare_training=lambda: None,
        _import_demonstrations=lambda: None,
        _offline_pretrain=harness.interrupt,
        _checkpoint=harness.save_checkpoint,
        _checkpoint_writer=SimpleNamespace(wait=harness.wait),
    )
    return cast(Coordinator, coordinator)


def test_interrupted_offline_pretraining_writes_a_final_checkpoint(tmp_path: Path) -> None:
    captured = {"checkpoints": 0, "waits": 0}
    coordinator = _interrupted_coordinator(tmp_path, captured)

    with pytest.raises(KeyboardInterrupt):
        _run_offline_pretraining(coordinator)

    assert captured == {"checkpoints": 1, "waits": 1}


def test_interrupted_offline_pretraining_reports_disabled_final_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    captured = {"checkpoints": 0, "waits": 0}
    coordinator = _interrupted_coordinator(tmp_path, captured)
    coordinator.run.spec.training.save_final_checkpoint = False

    with caplog.at_level(logging.WARNING), pytest.raises(KeyboardInterrupt):
        _run_offline_pretraining(coordinator)

    assert captured == {"checkpoints": 0, "waits": 1}
    assert "Final checkpoint is disabled" in caplog.text
    assert "Saving a final checkpoint" not in caplog.text
