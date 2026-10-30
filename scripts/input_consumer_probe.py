"""Separate 007 wire contract for bounded physical input-consumer observations.

No game controls or live connections are created by importing this module.
The caller owns the global controller lease, focus/map/reset guards and binary
identity checks, and must retain issued commands separately from this report.
"""

from __future__ import annotations

import json
import time
from typing import Any

from scripts.headless_windows_physical_probe import (
    EXECUTABLE_SHA256,
    MAX_RESPONSE_BYTES,
    PhysicalProbeClient,
    verify_code_evidence,
)

PROTOCOL = "tmrl.input-consumer-probe/v7"
ORIGINAL_SHA256 = "394d3821b576a7db12a9e7d826c556ba45510481889f2ceaa8a5d4aa55240b43"
PHASE = (
    "after original filtered-steer store in physical vehicle update; "
    "whole-step PRE/POST unqualified"
)


def verify_restoration(report: dict[str, Any], nonce: str) -> None:
    expected = {
        "schema": "tmrl.headless-input-consumer-probe/v7",
        "nonce": nonce,
        "target_rva": "0x841734",
        "expected_executable_sha256": EXECUTABLE_SHA256,
        "observed_exe_version": "2026-02-02_17_51",
        "openplanet_version": "1.29.14",
        "openplanet_git_commit": "6fd4200f",
        "padding": 3,
        "original_instruction_bytes_copied": 8,
        "hook_ever_installed": True,
        "code_hook_attempted": True,
        "unhooked": True,
        "original_64_bytes_restored": True,
        "exact_original_prefix_verified": True,
        "unhook_error": False,
    }
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError("007 restoration evidence mismatch: " + key)
    verify_code_evidence(report, ORIGINAL_SHA256, prefix_bytes=8, label="007")


def verify_capture(report: dict[str, Any]) -> None:
    """Check literal source/ownership without qualifying clock or command ACK."""
    for key, expected in {
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
    }.items():
        if type(report.get(key)) is not type(expected) or report[key] != expected:
            raise ValueError("007 capture evidence mismatch: " + key)
    if report.get("stop_reason") not in ("explicit-stop", "record-cap", "duration-cap"):
        raise ValueError("007 unexpected capture stop")
    count, rows = report.get("record_count"), report.get("records")
    if type(count) is not int or not 1 <= count <= 256 or not isinstance(rows, list):
        raise ValueError("007 requires 1..256 actual records")
    if len(rows) != count:
        raise ValueError("007 records differ from the retained count")
    owner = int(report["expected_vehicle_address"], 0)
    if not 0x10000 <= owner <= 0x00007FFFFFFF0000:
        raise ValueError("007 qualified vehicle pointer required")
    previous_time = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("sequence")) is not int:
            raise ValueError("007 actual record sequence required")
        if row["sequence"] != index or int(row["raw_body_address"], 0) != owner:
            raise ValueError("007 record sequence or owner mismatch")
        stamp, step = row.get("Time_Now_raw"), row.get("R13D_step_argument_raw")
        if type(stamp) is not int or stamp < 0 or stamp < previous_time:
            raise ValueError("007 invalid host metadata timestamp")
        previous_time = stamp
        if type(step) is not int or not 0 <= step <= 0xFFFFFFFF:
            raise ValueError("007 literal uint32 step argument required")
        for key, length in (
            ("actual_control_bits_u32", 4),
            ("vehicle_observed_input_bits_u32", 3),
            ("participant_cached_input_bits_u32", 3),
            ("raw_body_bits_u32", 10),
        ):
            words = row.get(key)
            if (
                not isinstance(words, list)
                or len(words) != length
                or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words)
            ):
                raise ValueError("007 literal source word array required: " + key)


class InputConsumerProbeClient(PhysicalProbeClient):
    """Reuse bounded 005 lifetime/cleanup with an incompatible explicit 007 wire.

    ``with client: ...; report = client.finish(); verify_capture(report)``.
    Completion requires exact restoration. Capture checks remain explicit so a
    zero-record failure can retain independently verified cleanup evidence.
    """

    def __init__(self, port: int = 19180, *, timeout_seconds: float = 1.0) -> None:
        super().__init__(port, timeout_seconds=timeout_seconds)

    def _receive(self, deadline: float) -> dict[str, Any]:
        assert self._socket is not None
        connection = self._socket
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("007 response deadline exceeded")
            connection.settimeout(min(self.timeout_seconds, remaining))
            try:
                data = connection.recv(4096)
            except TimeoutError:
                continue
            if not data:
                raise ConnectionError("007 socket EOF")
            self._buffer.extend(data)
            if len(self._buffer) > MAX_RESPONSE_BYTES:
                raise ValueError("007 response byte budget exceeded")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
            raise ValueError("007 JSON-lines protocol mismatch")
        if len(self.messages) >= 16:
            raise ValueError("007 protocol message budget exhausted")
        self.messages.append({"host_perf_counter_ns": time.perf_counter_ns(), **value})
        return value

    def _send(self, operation: str) -> None:
        assert self._socket is not None
        value: dict[str, Any] = {"protocol": PROTOCOL, "op": operation, "nonce": self.nonce}
        if operation == "arm":
            value["acknowledge_temporary_code_hook"] = True
        self._socket.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")

    def _accept_result(self, value: dict[str, Any]) -> None:
        if value.get("nonce") != self.nonce or value.get("op") not in ("complete", "status"):
            raise ValueError("007 completion nonce/operation mismatch")
        report = value.get("report")
        if not isinstance(report, dict):
            raise ValueError("007 actual report missing")
        self.report = report
        verify_restoration(report, self.nonce)
        self.restored_verified = True
