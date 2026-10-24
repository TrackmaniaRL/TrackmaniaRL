"""Opt-in fresh request/response telemetry; no game controls or UI operations.

This transport eliminates producer-pushed socket queues. Producer host timestamps
remain separate from the race clock and do not qualify simulation phase or input
consumption. The matching source-only plugin must be installed independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import socket
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from trackmaniarl.trackmania.telemetry import TelemetryFrame

PROTOCOL = "tmrl.fresh-telemetry/v6"
FIELD_SCHEMA = "tmrl.connect33-float32/v2"
COLLECTOR_VERSION = "0.0.6"
PAYLOAD_BYTES = 33 * 4
MAX_HEADER_BYTES = 4096


class Telemetry006ProtocolError(ValueError):
    """A response failed the diagnostic wire contract."""


class Telemetry006UnavailableError(RuntimeError):
    """The producer explicitly had no complete current observation."""


@dataclass(frozen=True, kw_only=True)
class Telemetry006Config:
    port: int = 19179
    timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("valid explicit loopback port required")
        if (
            type(self.timeout_s) not in (int, float)
            or not math.isfinite(self.timeout_s)
            or not 0 < self.timeout_s <= 5
        ):
            raise ValueError("finite positive timeout of at most 5 seconds required")


class Telemetry006Client:
    """One newly requested TelemetryFrame per read, with retained raw evidence.

    ``read``, ``read_next`` and ``read_latest`` have identical fresh-request
    semantics: this producer never streams or queues unsolicited observations.
    ``last_metadata`` contains the raw producer header, host request/receive
    perf_counter_ns and payload SHA-256. ``last_packet`` is the exact binary
    payload. Neither field is silently replaced with a previous valid sample.
    Explicit unavailability raises Telemetry006UnavailableError without fabricating a
    frame. Protocol, timeout and EOF failures close the connection; no request is
    silently replayed. The next explicit read may create a fresh connection.
    """

    def __init__(self, config: Telemetry006Config | None = None) -> None:
        self.config = config or Telemetry006Config()
        self.host, self.port, self.timeout_s = "127.0.0.1", self.config.port, self.config.timeout_s
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._sequence = 0
        self._nonce = ""
        self.hello: dict[str, Any] | None = None
        self.last_metadata: dict[str, Any] | None = None
        self.last_packet: bytes | None = None

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fresh telemetry operation exceeded its total deadline")
        return remaining

    def _receive_more(self, deadline: float) -> None:
        assert self._socket is not None
        self._socket.settimeout(self._remaining(deadline))
        chunk = self._socket.recv(4096)
        if not chunk:
            raise ConnectionError("fresh telemetry socket EOF before a complete response")
        self._buffer.extend(chunk)

    def _header(self, deadline: float) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            if len(self._buffer) > MAX_HEADER_BYTES:
                raise Telemetry006ProtocolError("fresh telemetry header exceeds its byte budget")
            self._receive_more(deadline)
        line, _, remaining = self._buffer.partition(b"\n")
        if len(line) > MAX_HEADER_BYTES:
            raise Telemetry006ProtocolError("fresh telemetry header exceeds its byte budget")
        self._buffer = bytearray(remaining)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Telemetry006ProtocolError("fresh telemetry response must be an object")
        expected = {
            "protocol": PROTOCOL,
            "collector_version": COLLECTOR_VERSION,
            "field_schema": FIELD_SCHEMA,
            "encoding": "little-endian-float32",
            "producer_clock": "Openplanet.Time.Now/raw",
            "simulation_phase_qualified": False,
            "applied_input_acknowledgement": False,
            "atomic_snapshot_claimed": False,
        }
        if any(key not in value or value[key] != item for key, item in expected.items()):
            raise Telemetry006ProtocolError("fresh telemetry producer contract mismatch")
        for key in (
            "simulation_phase_qualified",
            "applied_input_acknowledgement",
            "atomic_snapshot_claimed",
        ):
            if value[key] is not False:
                raise Telemetry006ProtocolError("unqualified phase/acknowledgement flags required")
        for key in ("sequence", "payload_bytes", "field_count"):
            if type(value.get(key)) is not int or value[key] < 0:
                raise Telemetry006ProtocolError("integral nonnegative response counters required")
        return value

    def connect(self) -> None:
        if self._socket is not None:
            return
        deadline = time.monotonic() + self.timeout_s
        self._nonce, self._sequence = secrets.token_hex(16), 0
        try:
            self._socket = socket.create_connection(
                (self.host, self.port), self._remaining(deadline)
            )
            hello = self._header(deadline)
            if (
                hello.get("status") != "hello"
                or hello.get("nonce") != ""
                or hello["sequence"] != 0
                or hello["payload_bytes"] != 0
                or hello["field_count"] != 0
                or self._buffer
            ):
                raise Telemetry006ProtocolError("fresh idle-producer hello required")
            self.hello = hello
        except BaseException:
            self.close()
            raise

    def read(self) -> TelemetryFrame:
        self.last_metadata = None
        self.last_packet = None
        self.connect()
        assert self._socket is not None
        deadline = time.monotonic() + self.timeout_s
        try:
            if self._buffer:
                raise Telemetry006ProtocolError("unsolicited telemetry response bytes")
            self._sequence += 1
            request = {
                "protocol": PROTOCOL,
                "op": "sample",
                "nonce": self._nonce,
                "sequence": self._sequence,
            }
            encoded = json.dumps(request, separators=(",", ":")).encode() + b"\n"
            host_request = time.perf_counter_ns()
            self._socket.settimeout(self._remaining(deadline))
            self._socket.sendall(encoded)
            header = self._header(deadline)
            if header.get("nonce") != self._nonce or header["sequence"] != self._sequence:
                raise Telemetry006ProtocolError("fresh telemetry nonce or sequence mismatch")
            for key in ("producer_begin_Time_Now_raw", "producer_end_Time_Now_raw"):
                if type(header.get(key)) is not int or header[key] < 0:
                    raise Telemetry006ProtocolError("actual integral producer timestamps required")
            if header["producer_end_Time_Now_raw"] < header["producer_begin_Time_Now_raw"]:
                raise Telemetry006ProtocolError("producer interval is reversed")
            self.last_metadata = {
                "producer": header,
                "host_request_perf_counter_ns": host_request,
            }
            if header.get("status") == "unavailable":
                if header["payload_bytes"] != 0 or header["field_count"] != 0 or self._buffer:
                    raise Telemetry006ProtocolError("unavailable response must contain no payload")
                self.last_metadata["host_received_perf_counter_ns"] = time.perf_counter_ns()
                raise Telemetry006UnavailableError(str(header.get("error", "snapshot-unavailable")))
            if (
                header.get("status") != "ok"
                or header["payload_bytes"] != PAYLOAD_BYTES
                or header["field_count"] != 33
            ):
                raise Telemetry006ProtocolError("complete raw33 success response required")
            while len(self._buffer) < PAYLOAD_BYTES:
                self._receive_more(deadline)
            if len(self._buffer) != PAYLOAD_BYTES:
                raise Telemetry006ProtocolError(
                    "unsolicited bytes after the requested raw33 payload"
                )
            packet = bytes(self._buffer)
            self._buffer.clear()
            values = np.frombuffer(packet, dtype="<f4").copy()
            # Same finite-vector and speed-domain constraints as Connect's client.
            frame = TelemetryFrame(values, skipped_frames=0)
            if not 0 <= float(values[16]) <= 2500:
                raise ValueError("fresh telemetry speed outside its declared domain")
            values.setflags(write=False)
            self.last_packet = packet
            self.last_metadata.update(
                host_received_perf_counter_ns=time.perf_counter_ns(),
                payload_sha256=hashlib.sha256(packet).hexdigest(),
            )
            return frame
        except Telemetry006UnavailableError:
            raise
        except BaseException:
            self.close()
            raise

    def read_next(self) -> TelemetryFrame:
        return self.read()

    def read_latest(self) -> TelemetryFrame:
        return self.read()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()
