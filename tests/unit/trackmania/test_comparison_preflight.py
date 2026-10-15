from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.tmrl_test_comparison import preflight
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig, TelemetryFrame


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.clock = _Clock()
        self.path = tmp_path / "config.yaml"
        self.stop = tmp_path / "STOP"
        self.receipt = tmp_path / "preflight.json"
        self.events: list[str] = []
        self.frames = [(100.0, False)]
        self.hook: Callable[[str], None] = lambda _: None
        self.config = TrackmaniaEnvironmentConfig(
            geometry_path=Path("unused.npz"),
            expected_map_uid="expected-map",
            restart_input="editor_validation",
            start_timeout_s=1.0,
            start_poll_s=0.1,
        )

    def event(self, event: str) -> None:
        self.events.append(event)
        self.hook(event)

    def run(self) -> dict[str, Any]:
        return preflight.run_preflight(
            self.path, stop_files=(self.stop,), receipt_path=self.receipt
        )


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    harness = _Harness(tmp_path)

    class Session:
        def __init__(self, *_: object) -> None:
            harness.event("session")
            self.timeout_s = 1.0
            self.confirmations = 0

        def verify_loaded_map(self, uid: str) -> None:
            assert uid == "expected-map"
            harness.event("verify")

        def confirm_ready(self, uid: str) -> None:
            assert uid == "expected-map"
            self.confirmations += 1
            harness.event(f"ready{self.confirmations}")

    class Controller:
        def __init__(self, **kwargs: str) -> None:
            assert kwargs == {"restart_input": "editor_validation"}
            harness.event("controller")

        def reset(self, *, guard: preflight.ResetGuard) -> None:
            guard.check()
            harness.event("reset")

        def close(self) -> None:
            harness.event("controller.close")

        def confirm_finish(self) -> None:
            raise AssertionError("Preflight must never dismiss a menu")

        def apply(self, _: object) -> None:
            raise AssertionError("Preflight must never drive")

    class Client:
        def __init__(self, config: OpenPlanetClientConfig, budget: preflight._Budget) -> None:
            assert config.port == 9000
            self._budget = budget
            harness.event("client")

        def read(self) -> TelemetryFrame:
            harness.event("read")
            race_time, finished = harness.frames[0]
            if len(harness.frames) > 1:
                harness.frames.pop(0)
            values = np.zeros(33, dtype=np.float32)
            values[2:4] = [float(finished), race_time]
            return TelemetryFrame(values)

        def close(self) -> None:
            harness.event("client.close")

    monkeypatch.setattr(preflight, "monotonic", lambda: harness.clock.now)
    monkeypatch.setattr(preflight, "sleep", harness.clock.advance)
    monkeypatch.setattr(preflight, "_load_config", lambda *_args, **_kwargs: harness.config)
    monkeypatch.setattr(preflight, "_PreflightSessionClient", Session)
    monkeypatch.setattr(preflight, "GamepadController", Controller)
    monkeypatch.setattr(preflight, "_PreflightTelemetryClient", Client)
    return harness


def test_preflight_waits_through_stale_countdown_and_finished_frames(
    fake_runtime: _Harness,
) -> None:
    fake_runtime.frames = [(6000, False), (-500, False), (0, False), (50, True), (100, False)]
    result = fake_runtime.run()

    assert result["status"] == "ready"
    assert result["race_time_ms"] == 100
    assert result["finished"] is False
    assert result["read_count"] == 5
    assert fake_runtime.events == [
        "session",
        "verify",
        "ready1",
        "controller",
        "reset",
        "client",
        "read",
        "read",
        "read",
        "read",
        "read",
        "ready2",
        "client.close",
        "controller.close",
    ]
    assert json.loads(fake_runtime.receipt.read_text()) == result


def test_preflight_respects_configured_minimum_start_clock(fake_runtime: _Harness) -> None:
    fake_runtime.config = fake_runtime.config.model_copy(update={"start_race_time_ms": 200.0})
    fake_runtime.frames = [(100, False), (200, False)]
    assert fake_runtime.run()["read_count"] == 2


@pytest.mark.parametrize("phase", ["verify", "ready1"])
def test_failed_map_or_player_readiness_creates_no_controller(
    fake_runtime: _Harness, phase: str
) -> None:
    def fail(event: str) -> None:
        if event == phase:
            raise RuntimeError("not ready")

    fake_runtime.hook = fail
    with pytest.raises(preflight.PreflightError, match="not ready"):
        fake_runtime.run()

    assert "controller" not in fake_runtime.events
    assert "reset" not in fake_runtime.events
    receipt = json.loads(fake_runtime.receipt.read_text())
    assert receipt["race_time_ms"] is None
    assert receipt["finished"] is None
    assert receipt["read_count"] == 0


def test_stop_before_preflight_performs_no_external_action(fake_runtime: _Harness) -> None:
    fake_runtime.stop.touch()
    with pytest.raises(preflight.PreflightError, match="cancelled"):
        fake_runtime.run()
    assert not fake_runtime.events
    assert json.loads(fake_runtime.receipt.read_text())["status"] == "cancelled"


@pytest.mark.parametrize("phase", ["ready1", "controller", "reset", "read", "ready2"])
def test_cancellation_at_barriers_never_reports_ready_or_retries_input(
    fake_runtime: _Harness, phase: str
) -> None:
    def stop(event: str) -> None:
        if event == phase:
            fake_runtime.stop.touch()

    fake_runtime.hook = stop
    with pytest.raises(preflight.PreflightError, match="cancelled"):
        fake_runtime.run()

    assert fake_runtime.events.count("reset") <= 1
    if "controller" in fake_runtime.events:
        assert fake_runtime.events[-1] == "controller.close"
    if phase == "ready1":
        assert "controller" not in fake_runtime.events
    if phase == "controller":
        assert "reset" not in fake_runtime.events
    if phase == "read":
        assert "ready2" not in fake_runtime.events
    assert json.loads(fake_runtime.receipt.read_text())["status"] == "cancelled"


def test_stale_clock_timeout_records_observation_without_repeating_reset(
    fake_runtime: _Harness,
) -> None:
    fake_runtime.frames = [(5000, False)]
    with pytest.raises(preflight.PreflightError, match="start_timeout_s"):
        fake_runtime.run()
    receipt = json.loads(fake_runtime.receipt.read_text())
    assert receipt["status"] == "failed"
    assert receipt["race_time_ms"] == 5000
    assert receipt["finished"] is False
    assert receipt["read_count"] > 1
    assert receipt["phase"] == "telemetry"
    assert fake_runtime.events.count("reset") == 1
    assert fake_runtime.events[-2:] == ["client.close", "controller.close"]


def test_frame_arriving_after_deadline_cannot_pass(fake_runtime: _Harness) -> None:
    fake_runtime.hook = lambda event: fake_runtime.clock.advance(1.1) if event == "read" else None
    with pytest.raises(preflight.PreflightError, match="start_timeout_s"):
        fake_runtime.run()
    receipt = json.loads(fake_runtime.receipt.read_text())
    assert receipt["race_time_ms"] == 100
    assert receipt["read_count"] == 1
    assert "ready2" not in fake_runtime.events


def test_cancel_during_poll_wait_closes_both_resources(
    fake_runtime: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_runtime.frames = [(0, False)]

    def cancel(seconds: float) -> None:
        fake_runtime.clock.advance(seconds)
        fake_runtime.stop.touch()

    monkeypatch.setattr(preflight, "sleep", cancel)
    with pytest.raises(preflight.PreflightError, match="cancelled"):
        fake_runtime.run()
    assert fake_runtime.events.count("read") == 1
    assert fake_runtime.events[-2:] == ["client.close", "controller.close"]


@pytest.mark.parametrize("phase", ["reset", "read", "client.close"])
def test_controller_is_closed_even_when_operation_or_client_cleanup_fails(
    fake_runtime: _Harness, phase: str
) -> None:
    def fail(event: str) -> None:
        if event == phase:
            raise RuntimeError(f"failure in {phase}")

    fake_runtime.hook = fail
    with pytest.raises(preflight.PreflightError, match=f"failure in {phase}"):
        fake_runtime.run()
    assert fake_runtime.events[-1] == "controller.close"
    assert fake_runtime.events.count("reset") == 1


class _Socket:
    def __init__(self, receive: Callable[[], bytes]) -> None:
        self.receive = receive
        self.timeouts: list[float] = []
        self.closed = False
        self.sent: list[bytes] = []

    def __enter__(self) -> _Socket:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def gettimeout(self) -> float:
        return self.timeouts[-1]

    def setblocking(self, blocking: bool) -> None:  # noqa: FBT001
        assert not blocking

    def recv(self, _: int) -> bytes:
        return self.receive()

    def sendall(self, request: bytes) -> None:
        self.sent.append(request)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("channel", ["session", "telemetry"])
@pytest.mark.parametrize("failure", ["trickle", "timeout", "cancel"])
def test_partial_transport_reads_obey_total_budget_and_cancellation(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: str, failure: str
) -> None:
    clock = _Clock()
    stop = tmp_path / "STOP"
    budget = preflight._Budget(1.0, (stop,))
    monkeypatch.setattr(preflight, "monotonic", lambda: clock.now)

    def receive() -> bytes:
        clock.advance(0.2)
        if failure == "cancel":
            stop.touch()
        if failure == "timeout":
            raise TimeoutError("fake timeout")
        return b"x"

    connection = _Socket(receive)

    def connect(*_args: object, **kwargs: float) -> _Socket:
        assert 0 < kwargs["timeout"] <= 0.25
        return connection

    monkeypatch.setattr(preflight.socket, "create_connection", connect)
    if channel == "session":
        config = _Harness(tmp_path).config
        client = preflight._PreflightSessionClient(config, budget)

        def operation() -> object:
            return client.verify_loaded_map("expected-map")
    else:
        client = preflight._PreflightTelemetryClient(OpenPlanetClientConfig(), budget)
        operation = client.read
    try:
        with pytest.raises((TimeoutError, preflight._PreflightCancelledError)):
            operation()
    finally:
        if channel == "telemetry":
            client.close()

    assert connection.closed
    assert connection.timeouts
    assert all(0 < value <= 0.25 for value in connection.timeouts)
    assert clock.now <= 1.0
    if failure == "cancel":
        assert clock.now == 0.2


def test_complete_start_frame_followed_by_disconnect_cannot_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(tmp_path)
    budget = preflight._Budget(1.0, ())
    monkeypatch.setattr(preflight, "monotonic", lambda: harness.clock.now)
    client = preflight._PreflightTelemetryClient(OpenPlanetClientConfig(), budget)
    values = [0.0] * 33
    values[3] = 100.0
    chunks = iter([client._packet.pack(*values), b""])
    connection = _Socket(lambda: next(chunks))
    monkeypatch.setattr(preflight.socket, "create_connection", lambda *_a, **_kw: connection)
    state: dict[str, Any] = {"read_count": 0}

    try:
        with pytest.raises(ConnectionError, match="closed the preflight telemetry"):
            preflight._wait_for_start(client, harness.config, state)
    finally:
        client.close()

    assert "race_time_ms" not in state
    assert state["read_count"] == 1
    assert connection.closed


def test_existing_run_guard_precedes_config_and_all_external_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "artifacts" / "existing"
    directory.mkdir(parents=True)
    spec = SimpleNamespace(artifacts_dir="artifacts", run_id="existing")
    monkeypatch.setattr(preflight.RunSpec, "from_yaml", lambda _: spec)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("No session or controller is allowed")

    monkeypatch.setattr(preflight, "_PreflightSessionClient", forbidden)
    monkeypatch.setattr(preflight, "GamepadController", forbidden)
    with pytest.raises(preflight.PreflightError, match="Run already exists"):
        preflight.run_preflight(tmp_path / "config.yaml", require_new_run=True)


@pytest.mark.parametrize("failure", ["trickle", "timeout", "cancel"])
def test_bounded_session_failure_in_full_preflight_never_creates_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    harness = _Harness(tmp_path)
    monkeypatch.setattr(preflight, "monotonic", lambda: harness.clock.now)
    monkeypatch.setattr(preflight, "_load_config", lambda *_args, **_kwargs: harness.config)

    def receive() -> bytes:
        harness.clock.advance(0.2)
        if failure == "cancel":
            harness.stop.touch()
        if failure == "timeout":
            raise TimeoutError("fake timeout")
        return b"x"

    connection = _Socket(receive)
    monkeypatch.setattr(preflight.socket, "create_connection", lambda *_a, **_kw: connection)

    def forbidden(**_kwargs: object) -> None:
        raise AssertionError("Readiness failure must not create a controller")

    monkeypatch.setattr(preflight, "GamepadController", forbidden)
    with pytest.raises(preflight.PreflightError, match=r"cancelled|start_timeout_s"):
        harness.run()

    receipt = json.loads(harness.receipt.read_text())
    assert receipt["phase"] == "verify_map"
    assert receipt["read_count"] == 0
    assert receipt["race_time_ms"] is None
    assert connection.closed
