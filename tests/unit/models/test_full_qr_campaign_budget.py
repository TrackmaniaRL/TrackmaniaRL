"""Real watchdog deadlines and SDK STOP interoperability; no game or training."""

import errno
import json
import multiprocessing
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from trackmaniarl.distributed.stop_signal import ProcessStopSignal
from trackmaniarl.experiments.full_qr.campaign_budget import (
    GIB,
    PILOT_SECONDS,
    BudgetSettings,
    CampaignBudget,
    StorageReserve,
    next_monitor_offset,
)


def _settings(tmp_path: Path, **kwargs: float) -> BudgetSettings:
    return BudgetSettings(tmp_path / "new-budget", (StorageReserve(tmp_path, 10 * GIB),), **kwargs)


def _wait_for_stop(budget: CampaignBudget) -> None:
    until = time.monotonic() + 3
    while not budget.stop_file.exists() and time.monotonic() < until:
        time.sleep(0.005)
    assert budget.stop_file.exists()


def test_diagnostic_monitor_offsets_stop_at_deadline() -> None:
    assert next_monitor_offset(0, 1800) == 900
    assert next_monitor_offset(900, 1800) == 1800
    assert next_monitor_offset(1800, 1800) is None


def test_deadline_stops_while_the_launcher_is_busy(tmp_path: Path) -> None:
    budget = CampaignBudget(
        _settings(tmp_path, duration_seconds=0.1), free_bytes=lambda _: 20 * GIB
    )
    state = budget.start()
    # The main training thread need not poll the budget for STOP to arrive.
    threading.Event().wait(0.3)
    assert budget.stop_file.read_text().strip() == "wall_clock_limit"
    closed = budget.close()
    assert 0.09 <= closed["elapsed_seconds"] < 2
    assert closed["watchdog_closed"]
    assert closed["phase"] == "stop_requested"
    assert not closed["final_checkpoint_verified"]
    assert not closed["process_cleanup_verified"]
    assert datetime.fromisoformat(state["deadline_utc"]) > datetime.fromisoformat(
        state["started_utc"]
    )
    assert json.loads(budget.state_file.read_text())["stop_reason"] == "wall_clock_limit"


def test_storage_failure_stops_without_waiting_for_four_hours(tmp_path: Path) -> None:
    free = [20 * GIB]
    budget = CampaignBudget(
        _settings(tmp_path, check_interval_seconds=0.01), free_bytes=lambda _: free[0]
    )
    budget.start()
    try:
        free[0] = 9 * GIB
        _wait_for_stop(budget)
    finally:
        state = budget.close()
    assert state["duration_seconds"] == PILOT_SECONDS == 14_400
    assert state["stop_reason"] == "storage_reserve"
    assert "insufficient space" in state["storage_error"]


def test_old_stop_and_run_are_preserved(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.directory.mkdir()
    old_stop = settings.directory / "STOP"
    old_stop.write_text("old campaign\n")
    budget = CampaignBudget(settings, free_bytes=lambda _: 20 * GIB)
    with pytest.raises(FileExistsError):
        budget.start()
    assert old_stop.read_text() == "old campaign\n"
    assert not budget.state


def test_existing_external_stop_is_not_replaced(tmp_path: Path) -> None:
    budget = CampaignBudget(_settings(tmp_path), free_bytes=lambda _: 20 * GIB)
    budget.start()
    budget.stop_file.write_text("operator stop\n")
    state = budget.close()
    assert state["stop_reason"] == "external_stop"
    assert budget.stop_file.read_text() == "operator stop\n"
    with pytest.raises(RuntimeError, match="only be started once"):
        budget.start()


def test_bad_initial_storage_does_not_create_a_run(tmp_path: Path) -> None:
    budget = CampaignBudget(_settings(tmp_path), free_bytes=lambda _: 1)
    with pytest.raises(RuntimeError, match="insufficient space"):
        budget.start()
    assert not budget.settings.directory.exists()


def test_monitor_checks_are_relative_to_actual_start() -> None:
    assert [next_monitor_offset(t) for t in (0, 900, 1800, 2700, 3600, 5400)] == [
        900,
        1800,
        2700,
        3600,
        5400,
        7200,
    ]
    assert next_monitor_offset(14_399) == 14_400
    assert next_monitor_offset(14_400) is None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _settings(tmp_path, duration_seconds=value)


def _wait_in_child(signal: ProcessStopSignal, ready: object) -> None:
    ready.send("ready")
    ready.close()
    if not signal.wait(5):
        raise RuntimeError("owned test child did not receive shutdown")


def test_deadline_notifies_the_real_sdk_shared_signal_in_a_spawned_child(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    signal = ProcessStopSignal(context)
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_wait_in_child, args=(signal, send))
    child.start()
    send.close()
    budget = CampaignBudget(
        _settings(tmp_path, duration_seconds=0.1),
        free_bytes=lambda _: 20 * GIB,
        stop_signal=signal,
    )
    try:
        assert receive.poll(10), "owned test child failed to start"
        assert receive.recv() == "ready"
        budget.start()
        child.join(timeout=8)
        assert not child.is_alive()
        assert child.exitcode == 0
        assert budget.stop_requested()
        assert signal.is_set()
        assert budget.state["stop_reason"] == "wall_clock_limit"
    finally:
        signal.set()
        child.join(timeout=8)
        receive.close()
        if budget.state:
            budget.close()
    assert not child.is_alive()


def test_full_disk_does_not_prevent_shared_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = ProcessStopSignal(multiprocessing.get_context("spawn"))
    budget = CampaignBudget(
        _settings(tmp_path, duration_seconds=0.1),
        free_bytes=lambda _: 20 * GIB,
        stop_signal=signal,
    )
    budget.start()
    original = os.open

    def full_stop_disk(path: Path, *args: object, **kwargs: object) -> int:
        if path == budget.stop_file:
            raise OSError(errno.ENOSPC, "injected full disk")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", full_stop_disk)
    try:
        assert signal.wait(3)
    finally:
        state = budget.close()
    assert budget.stop_requested()
    assert state["stop_reason"] == "wall_clock_limit"
    assert "injected full disk" in state["stop_file_error"]
    assert not budget.stop_file.exists()
    assert state["watchdog_closed"]
    assert not state["final_checkpoint_verified"]


def test_atomic_state_write_failure_also_leaves_shutdown_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = ProcessStopSignal(multiprocessing.get_context("spawn"))
    budget = CampaignBudget(_settings(tmp_path), free_bytes=lambda _: 20 * GIB, stop_signal=signal)
    budget.start()

    def full_state_disk() -> None:
        raise OSError(errno.ENOSPC, "injected state persistence failure")

    monkeypatch.setattr(budget, "_save", full_state_disk)
    budget.request_stop("storage_reserve")
    state = budget.close()
    assert signal.is_set()
    assert budget.stop_requested()
    assert state["stop_reason"] == "storage_reserve"
    assert "injected state persistence failure" in state["state_file_error"]
    assert budget.stop_file.read_text().strip() == "storage_reserve"


def test_shutdown_signal_failure_still_creates_the_stop_file(tmp_path: Path) -> None:
    class BrokenSignal:
        def set(self) -> None:
            raise RuntimeError("injected shutdown signal failure")

    budget = CampaignBudget(
        _settings(tmp_path), free_bytes=lambda _: 20 * GIB, stop_signal=BrokenSignal()
    )
    budget.start()
    state = budget.close()
    assert budget.stop_requested()
    assert state["stop_reason"] == "launcher_closed"
    assert "injected shutdown signal failure" in state["shutdown_signal_error"]
    assert budget.stop_file.read_text().strip() == "launcher_closed"
