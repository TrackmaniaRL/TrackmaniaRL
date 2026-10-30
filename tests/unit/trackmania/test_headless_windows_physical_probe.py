"""Pure/socket diagnostic tests; never touch the game, UI or Windows mutex."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.headless_windows_physical_probe as physical_probe
from scripts.headless_windows_physical_probe import (
    EXECUTABLE_SHA256,
    PROTOCOL,
    PhysicalProbeClient,
    drive_two_phases,
    require_game_directory,
    run_live,
    select_actions,
    verify_capture,
    verify_restoration,
)

ORIGINAL_HEX = bytes(range(64)).hex()


@pytest.fixture(autouse=True)
def synthetic_expected_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        physical_probe, "ORIGINAL_SHA256", hashlib.sha256(bytes.fromhex(ORIGINAL_HEX)).hexdigest()
    )


def test_installation_directory_must_be_explicit_and_absolute(tmp_path: Path) -> None:
    for value in (None, Path("relative"), tmp_path / "missing"):
        with pytest.raises(ValueError, match="existing absolute"):
            require_game_directory(value)
    assert require_game_directory(tmp_path) == tmp_path


def valid_report(nonce: str) -> dict[str, Any]:
    return {
        "schema": "tmrl.headless-physical-copy-probe/v5",
        "nonce": nonce,
        "target_rva": "0x7d0cb0",
        "expected_executable_sha256": EXECUTABLE_SHA256,
        "observed_exe_version": "2026-02-02_17_51",
        "openplanet_version": "1.29.14",
        "openplanet_git_commit": "6fd4200f",
        "expected_original_64_hex": ORIGINAL_HEX,
        "observed_before_64_hex": ORIGINAL_HEX,
        "observed_after_unhook_64_hex": ORIGINAL_HEX,
        "observed_original_trampoline_7_hex": ORIGINAL_HEX[:14],
        "hook_ever_installed": True,
        "code_hook_attempted": True,
        "unhooked": True,
        "original_64_bytes_restored": True,
        "exact_original_prefix_verified": True,
        "unhook_error": False,
        "record_count": 1,
        "callback_reentered": False,
        "callback_pointer_identity_lost": False,
        "destination_samples_are_PRE_copy": True,
        "Time_Now_is_simulation_clock": False,
        "applied_input_acknowledgement": False,
        "game_data_or_controls_written": False,
        "stop_reason": "explicit-stop",
        "expected_participant_address": "0x10000",
        "records": [
            {
                "sequence": 0,
                "Time_Now_raw": 123,
                "pose_destination_address": "0x10e14",
                "velocity_destination_address": "0x10e30",
                "raw_body_bits_u32": [0x3F800000] * 10,
                "destination_PRE_copy_bits_u32": [0] * 10,
            }
        ],
    }


class ProbeServer:
    def __init__(self, mode: str = "stop") -> None:
        self.mode = mode
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.socket.listen(2)
        self.socket.settimeout(0.02)
        self.stop = threading.Event()
        self.requests: list[str] = []
        self.nonce = ""
        self.restored = False
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self.run)

    def __enter__(self) -> ProbeServer:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(2)
        self.socket.close()
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error

    @staticmethod
    def send(connection: socket.socket, value: dict[str, Any]) -> None:
        connection.sendall(json.dumps({"protocol": PROTOCOL, **value}).encode() + b"\n")

    def receive(self, connection: socket.socket) -> dict[str, Any] | None:
        buffer = bytearray()
        while not self.stop.is_set():
            try:
                data = connection.recv(4096)
            except TimeoutError:
                continue
            except ConnectionResetError:
                # Windows may reset, rather than EOF, when the client correctly
                # aborts after rejecting the deliberately bad restoration proof.
                return None
            if not data:
                return None
            buffer.extend(data)
            if b"\n" in buffer:
                value = json.loads(buffer.split(b"\n", 1)[0])
                assert isinstance(value, dict)
                return value
        return None

    def complete(self, connection: socket.socket) -> None:
        self.restored = True
        report = valid_report(self.nonce)
        if self.mode == "bad-restoration":
            report["observed_after_unhook_64_hex"] = "e9" + ORIGINAL_HEX[2:]
        self.send(connection, {"op": "complete", "nonce": self.nonce, "report": report})

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    connection, _ = self.socket.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(0.02)
                    self.send(
                        connection,
                        {
                            "op": "hello",
                            "hook_installed": False,
                            "max_duration_ms": 5000,
                            "max_records": 256,
                        },
                    )
                    request = self.receive(connection)
                    if request is None:
                        continue
                    self.requests.append(request["op"])
                    if request["op"] == "status":
                        assert self.restored
                        self.complete(connection)
                        continue
                    assert request["op"] == "arm"
                    assert request["acknowledge_temporary_code_hook"] is True
                    self.nonce = request["nonce"]
                    acknowledged_nonce = "bad" if self.mode == "bad-arm-ack" else self.nonce
                    self.send(connection, {"op": "armed", "nonce": acknowledged_nonce})
                    if self.mode == "eof":
                        self.restored = True
                        continue
                    if self.mode == "async":
                        self.stop.wait(0.12)
                        with contextlib.suppress(OSError):
                            self.complete(connection)
                        continue
                    request = self.receive(connection)
                    if request is not None:
                        self.requests.append(request["op"])
                        assert request["op"] == "stop"
                        assert request["nonce"] == self.nonce
                        self.complete(connection)
                    else:
                        self.restored = True
        except BaseException as error:
            self.error = error


def test_reader_survives_per_receive_timeouts_until_async_completion() -> None:
    with ProbeServer("async") as server:
        client = PhysicalProbeClient(server.port, timeout_seconds=0.02)
        with client:
            time.sleep(0.2)
            assert client.restored_verified
        assert server.requests == ["arm"]
        assert client.report is not None
        verify_capture(client.report)


def test_body_failure_stops_probe_and_verifies_original_bytes() -> None:
    with ProbeServer() as server:
        client = PhysicalProbeClient(server.port, timeout_seconds=0.05)
        with pytest.raises(RuntimeError, match="body failed"), client:
            raise RuntimeError("body failed")
        assert client.restored_verified
        assert server.restored
        assert server.requests == ["arm", "stop"]
        assert client._thread is not None
        assert not client._thread.is_alive()


def test_eof_requires_fresh_status_restoration_proof() -> None:
    with ProbeServer("eof") as server:
        client = PhysicalProbeClient(server.port, timeout_seconds=0.05)
        with client:
            pass
        assert client.restored_verified
        assert server.requests == ["arm", "status"]
        assert isinstance(client._reader_error, ConnectionError)


def test_uncertain_arm_ack_closes_and_recovers_original_byte_proof() -> None:
    with ProbeServer("bad-arm-ack") as server:
        client = PhysicalProbeClient(server.port, timeout_seconds=0.05)
        with pytest.raises(RuntimeError, match="failed to arm"), client:
            pytest.fail("uncertain arm must not enter the driving body")
        assert client.restored_verified
        assert client._thread is None
        assert server.requests == ["arm", "status"]


def test_bad_restoration_evidence_cannot_complete_successfully() -> None:
    with ProbeServer("bad-restoration") as server:
        client = PhysicalProbeClient(server.port, timeout_seconds=0.05)
        with pytest.raises(RuntimeError, match="restoration could not be verified"), client:
            pass
        assert not client.restored_verified
        assert client.report is not None
        assert client.report["observed_after_unhook_64_hex"] != ORIGINAL_HEX
        assert client._socket is None
        assert client._thread is not None
        assert not client._thread.is_alive()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unhooked", False),
        ("observed_after_unhook_64_hex", "e9" + ORIGINAL_HEX[2:]),
        ("observed_original_trampoline_7_hex", ORIGINAL_HEX[2:16]),
        ("openplanet_git_commit", "different"),
        ("original_64_bytes_restored", 1),
    ],
)
def test_unhook_claim_requires_exact_independent_bytes_and_build(field: str, value: Any) -> None:
    report = valid_report("a" * 32)
    report[field] = value
    with pytest.raises(ValueError, match="restoration evidence"):
        verify_restoration(report, "a" * 32)


@pytest.mark.parametrize(
    "field", ["expected_original_64_hex", "observed_before_64_hex", "observed_after_unhook_64_hex"]
)
@pytest.mark.parametrize("value", [ORIGINAL_HEX[:-2], ORIGINAL_HEX.upper(), "z" * 128])
def test_code_evidence_rejects_noncanonical_or_incomplete_observations(
    field: str, value: str
) -> None:
    report = valid_report("a" * 32)
    report[field] = value
    with pytest.raises(ValueError, match="restoration evidence"):
        verify_restoration(report, "a" * 32)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination_samples_are_PRE_copy", False),
        ("Time_Now_is_simulation_clock", True),
        ("applied_input_acknowledgement", True),
        ("stop_reason", "focus-lost"),
    ],
)
def test_capture_does_not_accept_fabricated_phase_clock_or_ack(field: str, value: Any) -> None:
    report = valid_report("a" * 32)
    report[field] = value
    with pytest.raises(ValueError, match="005"):
        verify_capture(report)


def packet(timestamp: float) -> list[float]:
    values = [0.0] * 33
    values[3] = timestamp
    return values


class FakeEnvironment:
    def __init__(self, timestamps: list[float]) -> None:
        self._action_table = [[1, 0, 0], [1, 0, -1 / 6], [1, 0, -1]]
        self.applied: list[list[float]] = []
        self.releases = 0
        samples = iter(timestamps)
        self.controller = SimpleNamespace(apply_discrete=self.applied.append)
        self.client = SimpleNamespace(
            read_next=lambda: SimpleNamespace(values=packet(next(samples)))
        )

    def release_controls(self) -> None:
        self.releases += 1


def test_drive_retains_literal_observed_packets_and_issued_actions_separately() -> None:
    env = FakeEnvironment([float(value) for value in range(100, 2100, 100)])
    trace: dict[str, Any] = {"initial_raw33": packet(0)}
    drive_two_phases(env, trace, lambda: None)
    assert env.releases == 1
    assert env.applied == [[1, 0, 0], [1, 0, -1 / 6]]
    assert [row["observed_race_delta_ms"] for row in trace["phases"]] == [1000, 1000]
    assert all(row["raw33"][30:33] == [0, 0, 0] for row in trace["packets"])
    assert trace["phases"][1]["issued_gbs"] == [1, 0, -1 / 6]


def test_drive_interruption_preserves_partial_evidence_and_releases_controls() -> None:
    env = FakeEnvironment([100, 200, 300])
    trace: dict[str, Any] = {"initial_raw33": packet(0)}
    checks = 0

    def stop_guard() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TimeoutError("STOP")

    with pytest.raises(TimeoutError, match="STOP"):
        drive_two_phases(env, trace, stop_guard)
    assert env.releases == 1
    assert len(trace["packets"]) == 2
    assert len(trace["phases"]) == 1
    assert "completed" not in trace["phases"][0]


def test_drive_rejects_clock_reset_and_unsupported_actions() -> None:
    env = FakeEnvironment([-10])
    with pytest.raises(RuntimeError, match="regressed"):
        drive_two_phases(env, {"initial_raw33": packet(0)}, lambda: None)
    assert env.releases == 1
    with pytest.raises(ValueError, match="lacks"):
        select_actions([[1, 0, 0], [1, 0, -1]])


def test_live_guard_rejects_non_windows_before_creating_output(tmp_path: Path) -> None:
    import sys

    if sys.platform == "win32":
        pytest.skip("this guard is exercised only on the Linux test executor")
    with pytest.raises(RuntimeError, match="requires Windows"):
        run_live(tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
