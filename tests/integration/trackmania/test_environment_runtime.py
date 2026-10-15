"""Release contracts for environment and telemetry."""

from __future__ import annotations

import socket
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import build_brake_tap_action_table
from trackmaniarl.trackmania.control import RecordingController
from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
from trackmaniarl.trackmania.environment_step import _score_step
from trackmaniarl.trackmania.reward import RewardResult
from trackmaniarl.trackmania.reward_types import TransitionInput
from trackmaniarl.trackmania.telemetry import (
    OpenPlanetClient,
    OpenPlanetClientConfig,
    TelemetryFrame,
)


class _TimedClient:
    def __init__(self) -> None:
        self.race_times_ms = iter((105.0, 112.0, 121.0))

    def read(self) -> TelemetryFrame:
        values = np.zeros(33, dtype=np.float32)
        values[3] = next(self.race_times_ms)
        skipped = 2 if values[3] == 121.0 else int(values[3] == 105.0)
        return TelemetryFrame(values, skipped_frames=skipped)


class _StaticReward:
    progress_m = 12.0
    progress_pct = 0.5

    def step(self, transition: TransitionInput) -> RewardResult:
        del transition
        return RewardResult(1.0, False, None)


class _RestartClient:
    def __init__(self) -> None:
        self.race_times_ms = iter((1_000.0, 0.0, 50.0))

    def read(self) -> TelemetryFrame:
        values = np.asarray([0.0, 0.0, 0.0, next(self.race_times_ms)], dtype=np.float32)
        return TelemetryFrame(values)


class _RecoveryClient:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close(self) -> None:
        self.calls.append("close")


class _RecoveryController:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def confirm_finish(self) -> None:
        self.calls.append("enter")

    def reset(self) -> None:
        self.calls.append("delete")


class _BufferedSocket:
    def __init__(self, packets: list[bytes]) -> None:
        self.chunks: list[bytes | type[BlockingIOError]] = [
            packets[0],
            packets[1] + packets[2] + packets[3][:4],
            BlockingIOError,
            packets[3][4:],
            BlockingIOError,
        ]
        self.timeout = 1.0

    def recv(self, size: int) -> bytes:
        del size
        chunk = self.chunks.pop(0)
        if chunk is BlockingIOError:
            raise BlockingIOError
        return chunk

    def gettimeout(self) -> float:
        return self.timeout

    def setblocking(self, value: object) -> None:
        del value

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def close(self) -> None:
        pass


def _telemetry_packet(value: float) -> bytes:
    return struct.pack("<33f", *([value] * 33))


def _step_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = SimpleNamespace(
        action_repeat_frames=2,
        decision_interval_ms=20.0,
        position_indices=(4, 5, 6),
        velocity_indices=(7, 8, 9),
    )
    environment.client = _TimedClient()
    environment.controller = RecordingController()
    environment.reward = _StaticReward()
    environment._episode_started_at = 0.0
    environment._last_race_time_ms = 100.0
    environment._action_count, environment._action_table = build_brake_tap_action_table()
    times = iter((1.0, 1.002, 1.002, 1.012))
    monkeypatch.setattr("trackmaniarl.trackmania.environment.perf_counter", lambda: next(times))
    return environment.step(3)[-1]


def _assert_control_info(info: dict[str, object]) -> None:
    assert info["control_gas"] == 1.0
    assert info["control_brake"] == 0.0
    assert info["control_steer"] == -1.0


def _assert_timing_info(info: dict[str, object]) -> None:
    assert info["step_race_time_ms"] == pytest.approx(21.0)
    assert info["decision_interval_error_ms"] == pytest.approx(1.0)
    assert info["controller_apply_ms"] == pytest.approx(2.0)
    assert info["telemetry_wait_ms"] == pytest.approx(10.0)
    assert info["telemetry_skipped_frames"] == 3
    assert info["race_time_ms"] == pytest.approx(121.0)


def test_environment_step_reports_applied_control_and_race_time_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _step_environment(monkeypatch)
    _assert_control_info(info)
    _assert_timing_info(info)


def test_environment_preserves_regressing_clock_in_step_diagnostics() -> None:
    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = SimpleNamespace(position_indices=(4, 5, 6), velocity_indices=(7, 8, 9))
    environment.controller = RecordingController()
    environment.reward = _StaticReward()
    environment._last_race_time_ms = 100.0
    values = np.zeros(33, dtype=np.float32)
    values[3] = 90.0
    outcome = _score_step(environment, np.array([1.0, 0.0, 0.0]), TelemetryFrame(values))
    assert outcome.step_race_time_ms == -10.0


def test_environment_waits_for_race_timer_restart() -> None:
    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = _RestartClient()
    environment.config = SimpleNamespace(
        start_timeout_s=1.0, start_poll_s=0.0, start_race_time_ms=0.0
    )
    frame = environment._wait_for_active_run(500.0)
    assert float(frame.values[3]) == 50.0


def test_environment_can_align_first_observation_to_race_clock() -> None:
    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = _RestartClient()
    environment.client.race_times_ms = iter((1000.0, 0.0, 10.0, 50.0, 100.0))
    environment.config = SimpleNamespace(
        start_timeout_s=1.0, start_poll_s=0.0, start_race_time_ms=100.0
    )
    frame = environment._wait_for_active_run(500.0)
    assert float(frame.values[3]) == 100.0


def test_environment_confirms_ready_after_observing_timer_restart() -> None:
    calls: list[str] = []
    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = _RestartClient()
    environment.controller = SimpleNamespace(reset=lambda: calls.append("reset"))
    environment._session = SimpleNamespace(confirm_ready=lambda _: calls.append("ready"))
    environment.config = SimpleNamespace(
        start_timeout_s=1.0, start_poll_s=0.0, start_race_time_ms=0.0
    )
    environment._expected_map_uid = "map"
    environment._last_race_time_ms = 500.0
    environment._finish_confirmation_pending = False

    frame = environment._restart_race()

    assert float(frame.values[3]) == 50.0
    assert calls == ["reset", "ready"]


def test_environment_recovers_reset_timeout_with_finish_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = _RecoveryClient(calls)
    environment.controller = _RecoveryController(calls)
    environment.config = SimpleNamespace(confirm_finish_before_reset=True)
    environment._finish_confirmation_pending = True
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)
    environment._recover_reset_timeout()
    assert calls == ["close", "enter", "delete"]


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_first_reset_recovers_a_missing_finish_screen_frame(
    monkeypatch: pytest.MonkeyPatch,
    failures: int,
) -> None:
    calls: list[str] = []
    remaining_failures = failures
    race_times = iter((1000.0, 0.0, 50.0))

    def read() -> TelemetryFrame:
        nonlocal remaining_failures
        calls.append("read")
        if remaining_failures:
            remaining_failures -= 1
            raise TimeoutError("finish screen")
        values = np.zeros(33, dtype=np.float32)
        values[3] = next(race_times)
        return TelemetryFrame(values)

    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = SimpleNamespace(read=read, close=lambda: calls.append("close"))
    environment.controller = _RecoveryController(calls)
    environment._session = SimpleNamespace(confirm_ready=lambda _: calls.append("ready"))
    environment.config = SimpleNamespace(
        confirm_finish_before_reset=True,
        start_timeout_s=1.0,
        start_poll_s=0.0,
        start_race_time_ms=0.0,
    )
    environment._expected_map_uid = "map"
    environment._last_race_time_ms = None
    environment._finish_confirmation_pending = True
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)

    if failures == 2:
        with pytest.raises(TimeoutError, match="finish screen"):
            environment._restart_race()
        assert "ready" not in calls
    else:
        assert float(environment._restart_race().values[3]) == 50.0
        assert calls[-1] == "ready"
    if failures:
        assert calls[:6] == ["enter", "close", "delete", "read", "close", "enter"]
        assert "delete" in calls
    else:
        assert calls[:4] == ["enter", "close", "delete", "read"]


def test_first_reset_starts_telemetry_before_waiting_for_a_frame() -> None:
    calls: list[str] = []

    def read() -> TelemetryFrame:
        assert "delete" in calls, "Finish screen produces no telemetry until reset"
        calls.append("read")
        values = np.zeros(33, dtype=np.float32)
        values[3] = 50.0
        return TelemetryFrame(values)

    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = SimpleNamespace(read=read, close=lambda: calls.append("close"))
    environment.controller = _RecoveryController(calls)
    environment._session = SimpleNamespace(confirm_ready=lambda _: calls.append("ready"))
    environment.config = SimpleNamespace(
        confirm_finish_before_reset=True,
        start_timeout_s=1.0,
        start_poll_s=0.0,
        start_race_time_ms=0.0,
    )
    environment._expected_map_uid = "map"
    environment._last_race_time_ms = None
    environment._finish_confirmation_pending = True
    assert float(environment._restart_race().values[3]) == 50.0
    assert calls == ["enter", "close", "delete", "read", "ready"]


def test_first_reset_ignores_a_finished_frame() -> None:
    environment = object.__new__(OpenPlanetEnvironment)
    values = np.zeros(33, dtype=np.float32)
    values[2:4] = (1.0, 20.0)
    active = values.copy()
    active[2:4] = (0.0, 50.0)
    frames = iter((TelemetryFrame(values), TelemetryFrame(active)))
    environment.client = SimpleNamespace(read=lambda: next(frames))
    environment.config = SimpleNamespace(
        start_timeout_s=1.0, start_poll_s=0.0, start_race_time_ms=0.0
    )
    assert float(environment._wait_for_active_run(None).values[3]) == 50.0


def test_environment_closes_telemetry_when_controller_close_fails() -> None:
    calls: list[str] = []

    def close_controller() -> None:
        raise RuntimeError("controller close")

    environment = object.__new__(OpenPlanetEnvironment)
    environment.controller = SimpleNamespace(close=close_controller)
    environment.client = SimpleNamespace(close=lambda: calls.append("telemetry closed"))
    with pytest.raises(RuntimeError, match="controller close"):
        environment.close()
    assert calls == ["telemetry closed"]


def test_environment_recovery_skips_finish_confirmation_for_editor_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = _RecoveryClient(calls)
    environment.controller = _RecoveryController(calls)
    environment.config = SimpleNamespace(confirm_finish_before_reset=False)
    environment._finish_confirmation_pending = True
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)

    environment._recover_reset_timeout()

    assert calls == ["close", "delete"]


def test_environment_can_skip_finish_confirmation_for_editor_validation() -> None:
    calls: list[str] = []
    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = SimpleNamespace(confirm_finish_before_reset=False)
    environment.controller = _RecoveryController(calls)
    environment._finish_confirmation_pending = True

    environment._confirm_finish_if_needed()

    assert calls == []


def test_openplanet_client_counts_only_complete_skipped_frames() -> None:
    packets = [_telemetry_packet(value) for value in (1.0, 2.0, 3.0, 4.0)]
    client = OpenPlanetClient()
    client._socket = _BufferedSocket(packets)
    newest, completed_fragment = client.read(), client.read()
    assert np.array_equal(newest.values, np.full(33, 3.0, dtype=np.float32))
    assert newest.skipped_frames == 2
    assert np.array_equal(completed_fragment.values, np.full(33, 4.0, dtype=np.float32))
    assert completed_fragment.skipped_frames == 0


def _serve_after_disconnect(server: socket.socket, payload: bytes) -> None:
    first, _ = server.accept()
    first.close()
    second, _ = server.accept()
    with second:
        second.sendall(payload)
    server.close()


def _read_after_disconnect(payload: bytes) -> np.ndarray:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    host, port = server.getsockname()
    thread = threading.Thread(target=_serve_after_disconnect, args=(server, payload))
    thread.start()
    client = OpenPlanetClient(OpenPlanetClientConfig(host, port, 1.0))
    try:
        return client.read().values
    finally:
        client.close()
        thread.join(timeout=1)


def test_openplanet_client_reconnects_after_the_producer_closes() -> None:
    values = _read_after_disconnect(_telemetry_packet(4.0))
    assert np.array_equal(values, np.full(33, 4.0, dtype=np.float32))
