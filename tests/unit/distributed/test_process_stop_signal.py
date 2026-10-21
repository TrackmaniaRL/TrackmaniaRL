"""Spawned-process cancellation stays usable after another child exits abruptly."""

from __future__ import annotations

import multiprocessing
import os
from multiprocessing.connection import Connection

from trackmaniarl.distributed.stop_signal import ProcessStopSignal


def _wait_for_stop(signal: ProcessStopSignal, connection: Connection) -> None:
    connection.send("ready")
    connection.send(signal.wait(10.0))
    connection.close()


def _failed_child(signal: ProcessStopSignal) -> None:
    signal.is_set()
    os._exit(3)


def test_stop_signal_wait_reports_timeout_and_remains_set() -> None:
    signal = ProcessStopSignal(multiprocessing.get_context("spawn"))
    assert not signal.wait(0.0)
    signal.set()
    assert signal.wait(0.0)
    signal.set()
    assert signal.is_set()


def test_dead_child_does_not_block_notifying_surviving_child() -> None:
    context = multiprocessing.get_context("spawn")
    signal = ProcessStopSignal(context)
    receiver, sender = context.Pipe(duplex=False)
    failed = context.Process(target=_failed_child, args=(signal,))
    survivor = context.Process(target=_wait_for_stop, args=(signal, sender))
    try:
        survivor.start()
        failed.start()
        assert receiver.poll(10.0)
        assert receiver.recv() == "ready"
        failed.join(10.0)
        assert failed.exitcode == 3
        signal.set()
        assert receiver.poll(10.0)
        assert receiver.recv() is True
        survivor.join(10.0)
        assert survivor.exitcode == 0
    finally:
        for process in (failed, survivor):
            if process.is_alive():
                process.terminate()
            process.join(5.0)
            process.close()
        receiver.close()
        sender.close()
