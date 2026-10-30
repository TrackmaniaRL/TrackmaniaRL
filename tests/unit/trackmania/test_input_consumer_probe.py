"""007 pure/socket contract checks. No game, UI or live plugin operations."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.input_consumer_probe as consumer_probe
from scripts.headless_windows_physical_probe import (
    EXECUTABLE_SHA256,
    PhysicalProbeClient,
    ProbeRunContract,
    drive_two_phases,
)
from scripts.input_consumer_live import drive_three_phases
from scripts.input_consumer_probe import (
    PHASE,
    PROTOCOL,
    InputConsumerProbeClient,
    verify_capture,
    verify_restoration,
)

ORIGINAL_HEX = bytes(range(64)).hex()


@pytest.fixture(autouse=True)
def synthetic_expected_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        consumer_probe, "ORIGINAL_SHA256", hashlib.sha256(bytes.fromhex(ORIGINAL_HEX)).hexdigest()
    )


def report(nonce: str) -> dict[str, Any]:
    return {
        "schema": "tmrl.headless-input-consumer-probe/v7",
        "nonce": nonce,
        "target_rva": "0x841734",
        "expected_executable_sha256": EXECUTABLE_SHA256,
        "observed_exe_version": "2026-02-02_17_51",
        "openplanet_version": "1.29.14",
        "openplanet_git_commit": "6fd4200f",
        "expected_original_64_hex": ORIGINAL_HEX,
        "observed_before_64_hex": ORIGINAL_HEX,
        "observed_after_unhook_64_hex": ORIGINAL_HEX,
        "observed_original_trampoline_8_hex": ORIGINAL_HEX[:16],
        "padding": 3,
        "original_instruction_bytes_copied": 8,
        "hook_ever_installed": True,
        "code_hook_attempted": True,
        "unhooked": True,
        "original_64_bytes_restored": True,
        "exact_original_prefix_verified": True,
        "unhook_error": False,
        "phase": PHASE,
        "sampled_actual_consumer_state": True,
        "same_record_API_getters_invoked": False,
        "issued_command_consumption_acknowledged": False,
        "R13D_is_qualified_simulation_clock": False,
        "Time_Now_is_simulation_clock": False,
        "applied_input_acknowledgement": False,
        "game_data_or_controls_written": False,
        "callback_reentered": False,
        "owner_vehicle_identity_lost": False,
        "expected_vehicle_entity_class": "0x0a020000",
        "expected_vehicle_address": "0x20000",
        "stop_reason": "explicit-stop",
        "record_count": 1,
        "records": [
            {
                "sequence": 0,
                "raw_body_address": "0x20000",
                "Time_Now_raw": 100,
                "R13D_step_argument_raw": 0xFFFFFFFF,
                "actual_control_bits_u32": [0x3F800000, 0, 0xBE2AAAAB, 0xBE2AAAAB],
                "vehicle_observed_input_bits_u32": [0xBE2AAAAA, 0x3F800000, 0],
                "participant_cached_input_bits_u32": [0xBE2AAAAA, 0x3F800000, 0],
                "raw_body_bits_u32": [0] * 10,
            }
        ],
    }


@contextlib.contextmanager
def producer(*, wrong_protocol: bool = False) -> Iterator[tuple[int, list[str]]]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    requests: list[str] = []
    errors: list[BaseException] = []
    stopping = threading.Event()

    def send(connection: socket.socket, value: dict[str, Any]) -> None:
        connection.sendall(json.dumps({"protocol": PROTOCOL, **value}).encode() + b"\n")

    def read(connection: socket.socket) -> dict[str, Any] | None:
        pending = bytearray()
        while not stopping.is_set():
            try:
                part = connection.recv(4096)
            except TimeoutError:
                continue
            if not part:
                return None
            pending.extend(part)
            if b"\n" in pending:
                value = json.loads(pending.partition(b"\n")[0])
                assert isinstance(value, dict)
                assert value["protocol"] == PROTOCOL
                requests.append(value["op"])
                return value
        return None

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(0.02)
                send(
                    connection,
                    {
                        "protocol": "tmrl.physical-probe/v5" if wrong_protocol else PROTOCOL,
                        "op": "hello",
                        "hook_installed": False,
                        "max_duration_ms": 5000,
                        "max_records": 256,
                    },
                )
                arm = read(connection)
                if wrong_protocol:
                    assert arm is None
                    return
                assert arm is not None
                assert arm["op"] == "arm"
                assert arm["acknowledge_temporary_code_hook"] is True
                nonce = arm["nonce"]
                send(connection, {"op": "armed", "nonce": nonce})
                stop = read(connection)
                assert stop is not None
                assert stop["op"] == "stop"
                assert stop["nonce"] == nonce
                send(connection, {"op": "complete", "nonce": nonce, "report": report(nonce)})
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield listener.getsockname()[1], requests
    finally:
        stopping.set()
        thread.join(3)
        listener.close()
        assert not thread.is_alive()
        if errors:
            raise errors[0]


def test_separate_007_wire_stops_and_verifies_after_body_failure() -> None:
    with producer() as (port, requests):
        client = InputConsumerProbeClient(port, timeout_seconds=0.05)
        with pytest.raises(RuntimeError, match="STOP"), client:
            raise RuntimeError("STOP")
        assert requests == ["arm", "stop"]
        assert client.restored_verified
        assert client.report is not None
        verify_capture(client.report)
        assert client._socket is None
        assert client._thread is not None
        assert not client._thread.is_alive()


def test_005_cannot_accidentally_arm_the_separate_007_client() -> None:
    with producer(wrong_protocol=True) as (port, requests):
        with pytest.raises(ValueError, match="007 JSON-lines"), InputConsumerProbeClient(port):
            pytest.fail("the wrong probe protocol must not arm")
        assert requests == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("observed_after_unhook_64_hex", "e9" + ORIGINAL_HEX[2:]),
        ("observed_original_trampoline_8_hex", ORIGINAL_HEX[:14]),
        ("padding", 2),
        ("original_64_bytes_restored", 1),
    ],
)
def test_exact_007_instruction_and_restoration_are_mandatory(key: str, value: Any) -> None:
    actual = report("a" * 32)
    actual[key] = value
    with pytest.raises(ValueError, match="007 restoration"):
        verify_restoration(actual, "a" * 32)


@pytest.mark.parametrize(
    "key",
    [
        "issued_command_consumption_acknowledged",
        "R13D_is_qualified_simulation_clock",
        "applied_input_acknowledgement",
        "owner_vehicle_identity_lost",
    ],
)
def test_observation_cannot_silently_become_a_clock_or_issued_ack(key: str) -> None:
    actual = report("a" * 32)
    actual[key] = True
    with pytest.raises(ValueError, match="007 capture"):
        verify_capture(actual)


def test_actual_observed_and_cached_bits_remain_distinct_with_raw_step_argument() -> None:
    actual = report("a" * 32)
    verify_capture(actual)
    row = actual["records"][0]
    assert row["actual_control_bits_u32"][2] != row["vehicle_observed_input_bits_u32"][0]
    assert row["R13D_step_argument_raw"] == 0xFFFFFFFF
    row["raw_body_address"] = "0x30000"
    with pytest.raises(ValueError, match="owner mismatch"):
        verify_capture(actual)


def test_negative_host_metadata_stamp_is_rejected() -> None:
    actual = report("a" * 32)
    actual["records"][0]["Time_Now_raw"] = -1
    with pytest.raises(ValueError, match="timestamp"):
        verify_capture(actual)


def test_explicit_probe_contract_preserves_the_default_005_wire_and_driver() -> None:
    contract = ProbeRunContract()
    assert contract.client_factory is PhysicalProbeClient
    assert contract.drive is drive_two_phases
    assert contract.report_name == "physical005.json"
    with pytest.raises(ValueError, match="plain JSON filename"):
        ProbeRunContract(report_name="../consumer007.json")


@pytest.mark.parametrize(
    "name",
    ["diagnostic.json", "original-connect.json", "input-profile-client.json", "Preflight.json"],
)
def test_probe_report_cannot_replace_guard_or_cleanup_evidence(name: str) -> None:
    with pytest.raises(ValueError, match="reserved diagnostic filename"):
        ProbeRunContract(report_name=name)


@pytest.mark.parametrize(
    ("profile", "durations", "endpoint"),
    [("default", [500, 800, 800], 2100), ("long-mild-left", [200, 200, 2000], 2400)],
)
def test_three_phases_preserve_real_duration_overshoot_and_release_controls(
    profile: str, durations: list[int], endpoint: int
) -> None:
    def packet(timestamp: int) -> list[float]:
        values = [0.0] * 33
        values[3] = timestamp
        return values

    applied: list[list[float]] = []
    released: list[bool] = []
    stamps = iter(range(100, 2500, 100))
    env = SimpleNamespace(
        _action_table=[[0, 0, 0], [1, 0, 0], [1, 0, -1 / 6]],
        controller=SimpleNamespace(apply_discrete=applied.append),
        client=SimpleNamespace(read_next=lambda: SimpleNamespace(values=packet(next(stamps)))),
        release_controls=lambda: released.append(True),
    )
    journal: dict[str, Any] = {"initial_raw33": packet(0)}
    drive_three_phases(env, journal, lambda: None, phase_profile=profile)
    assert applied == [[0, 0, 0], [1, 0, 0], [1, 0, -1 / 6]]
    assert released == [True]
    assert [phase["observed_race_delta_ms"] for phase in journal["phases"]] == durations
    assert journal["packets"][-1]["raw33"][3] == endpoint
    assert journal["issued_command_ack_claimed"] is False
    assert journal["fresh006"] == []
