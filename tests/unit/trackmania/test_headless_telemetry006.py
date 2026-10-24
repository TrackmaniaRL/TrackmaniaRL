"""Linux-safe mock socket tests; never load a plugin or contact the game."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
import time
from typing import Any

import numpy as np
import pytest

from scripts.headless_telemetry006 import (
    COLLECTOR_VERSION,
    FIELD_SCHEMA,
    PROTOCOL,
    Telemetry006Client,
    Telemetry006Config,
    Telemetry006ProtocolError,
    Telemetry006UnavailableError,
)


def _header(**fields: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "collector_version": COLLECTOR_VERSION,
        "field_schema": FIELD_SCHEMA,
        "encoding": "little-endian-float32",
        "producer_clock": "Openplanet.Time.Now/raw",
        "simulation_phase_qualified": False,
        "applied_input_acknowledgement": False,
        "atomic_snapshot_claimed": False,
        "status": "hello",
        "nonce": "",
        "sequence": 0,
        "payload_bytes": 0,
        "field_count": 0,
        **fields,
    }


class Fake006Server:
    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.requests: list[dict[str, Any]] = []
        self.packets: list[bytes] = []
        self.stop = threading.Event()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.sock.settimeout(0.02)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self) -> Fake006Server:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(2)
        self.sock.close()
        assert not self.thread.is_alive()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                connection, _ = self.sock.accept()
            except TimeoutError:
                continue
            connection.settimeout(0.02)
            try:
                self.respond(connection)
            except (OSError, ConnectionError):
                pass
            finally:
                connection.close()

    def respond(self, connection: socket.socket) -> None:
        connection.sendall(json.dumps(_header()).encode() + b"\n")
        pending = bytearray()
        while not self.stop.is_set():
            try:
                data = connection.recv(4096)
            except TimeoutError:
                continue
            if not data:
                return
            pending.extend(data)
            while b"\n" in pending:
                line, _, remaining = pending.partition(b"\n")
                pending = bytearray(remaining)
                request = json.loads(line)
                self.requests.append(request)
                sequence = request["sequence"]
                values = np.arange(33, dtype="<f4")
                values[0] = -0.0  # Raw bit preservation, including embedded NULs.
                values[3] = sequence * 10
                if self.mode == "nan":
                    values[4] = float("nan")
                packet = values.tobytes()
                self.packets.append(packet)
                header = _header(
                    status="ok",
                    nonce=request["nonce"],
                    sequence=sequence,
                    payload_bytes=132,
                    field_count=33,
                    producer_begin_Time_Now_raw=1000 + sequence,
                    producer_end_Time_Now_raw=1001 + sequence,
                    actual_map_uid="mock-local-map",
                )
                if (self.mode == "unavailable-first" and sequence == 1) or (
                    self.mode == "unavailable-second" and sequence == 2
                ):
                    header.update(status="unavailable", error="snapshot-unavailable")
                    header.update(payload_bytes=0, field_count=0)
                    packet = b""
                elif self.mode == "wrong-nonce":
                    header["nonce"] = "0" * 32
                elif self.mode == "stale-sequence":
                    header["sequence"] = sequence - 1
                elif self.mode == "reversed-time":
                    header["producer_end_Time_Now_raw"] = 0
                elif self.mode == "wrong-payload-length":
                    header["payload_bytes"] = 128
                prefix = json.dumps(header).encode() + b"\n"
                if self.mode == "partial-eof":
                    connection.sendall(prefix + packet[:7])
                    return
                if self.mode == "trickle":
                    connection.sendall(prefix)
                    for byte in packet:
                        if self.stop.wait(0.02):
                            return
                        connection.sendall(bytes((byte,)))
                    continue
                wire = prefix + packet
                if self.mode == "fragmented":
                    for start in range(0, len(wire), 7):
                        connection.sendall(wire[start : start + 7])
                else:
                    connection.sendall(wire)


def _client(server: Fake006Server, timeout: float = 0.5) -> Telemetry006Client:
    return Telemetry006Client(Telemetry006Config(port=server.port, timeout_s=timeout))


@pytest.mark.parametrize("mode", ["normal", "fragmented"])
def test_every_read_surface_requests_fresh_exact_raw33(mode: str) -> None:
    with Fake006Server(mode) as server:
        client = _client(server)
        try:
            client.connect()
            time.sleep(0.03)
            assert server.requests == []  # Producer emits no unsolicited frames.
            for sequence, read in enumerate((client.read, client.read_next, client.read_latest), 1):
                frame = read()
                assert frame.values.shape == (33,)
                assert frame.values.dtype == np.float32
                assert not frame.values.flags.writeable
                assert frame.skipped_frames == 0
                assert frame.values[3] == sequence * 10
                assert client.last_packet == server.packets[-1]
                assert client.last_packet[:4] == struct.pack("<I", 0x80000000)
                metadata = client.last_metadata
                assert metadata is not None
                assert metadata["producer"]["sequence"] == sequence
                assert metadata["producer"]["producer_begin_Time_Now_raw"] == 1000 + sequence
                assert (
                    metadata["host_request_perf_counter_ns"]
                    < metadata["host_received_perf_counter_ns"]
                )
                assert metadata["payload_sha256"] == hashlib.sha256(client.last_packet).hexdigest()
            assert [row["sequence"] for row in server.requests] == [1, 2, 3]
            assert len({row["nonce"] for row in server.requests}) == 1
        finally:
            client.close()


def test_unavailable_is_explicit_and_next_request_can_obtain_real_packet() -> None:
    with Fake006Server("unavailable-first") as server:
        client = _client(server)
        try:
            with pytest.raises(Telemetry006UnavailableError, match="snapshot-unavailable"):
                client.read()
            assert client.last_packet is None
            assert client.last_metadata is not None
            assert client.last_metadata["producer"]["status"] == "unavailable"
            assert client.read_next().values[3] == 20
        finally:
            client.close()


def test_unavailable_cannot_return_or_retain_previous_valid_packet() -> None:
    with Fake006Server("unavailable-second") as server:
        client = _client(server)
        try:
            assert client.read().values[3] == 10
            assert client.last_packet is not None
            with pytest.raises(Telemetry006UnavailableError):
                client.read_next()
            assert client.last_packet is None
            assert client.last_metadata is not None
            assert client.last_metadata["producer"]["sequence"] == 2
            assert client.last_metadata["producer"]["status"] == "unavailable"
            assert client.read_latest().values[3] == 30
        finally:
            client.close()


@pytest.mark.parametrize(
    "mode", ["wrong-nonce", "stale-sequence", "reversed-time", "wrong-payload-length"]
)
def test_invalid_or_stale_response_fails_closed(mode: str) -> None:
    with Fake006Server(mode) as server:
        client = _client(server)
        with pytest.raises(Telemetry006ProtocolError):
            client.read()
        assert client.last_packet is None
        assert client._socket is None


def test_partial_payload_eof_never_returns_an_observation() -> None:
    with Fake006Server("partial-eof") as server:
        client = _client(server)
        with pytest.raises(ConnectionError, match="EOF"):
            client.read()
        assert client.last_packet is None
        assert client._socket is None


def test_trickling_payload_cannot_extend_the_total_receive_deadline() -> None:
    with Fake006Server("trickle") as server:
        client = _client(server, timeout=0.1)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            client.read()
        assert time.monotonic() - started < 0.5
        assert client.last_packet is None
        assert client._socket is None


def test_nonfinite_actual_payload_is_rejected_without_silent_repair() -> None:
    with Fake006Server("nan") as server:
        client = _client(server)
        with pytest.raises(ValueError, match="finite"):
            client.read()
        assert client.last_packet is None
        assert client._socket is None


@pytest.mark.parametrize("timeout", [0, -1, 6, float("nan"), float("inf"), True])
def test_unbounded_or_invalid_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        Telemetry006Config(timeout_s=timeout)
