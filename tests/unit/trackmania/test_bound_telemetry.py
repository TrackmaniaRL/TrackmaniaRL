from __future__ import annotations

import json
import socket
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Thread
from time import monotonic, sleep
from typing import Any

import pytest

from trackmaniarl.trackmania.bound_telemetry import (
    BoundOpenPlanetClient,
    BoundTelemetryError,
    BoundTelemetryIdentity,
)
from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig

_IDENTITY = BoundTelemetryIdentity("epoch", "operation", "map-a")
_PLACEHOLDER = "0" * 32


def _header(**changes: Any) -> bytes:
    value: dict[str, Any] = {
        "protocol_version": "3",
        "request_id": _PLACEHOLDER,
        "instance_epoch": "epoch",
        "operation_id": "operation",
        "map_uid": "map-a",
        "state": "ready",
        "payload_bytes": 132,
    }
    value.update(changes)
    return json.dumps(value).encode() + b"\n"


_PAYLOAD = struct.pack("<33f", *([0.0] * 33))


@pytest.mark.parametrize(("keep_listener", "expect_disconnect"), [(False, True), (True, False)])
def test_reset_reconnect_with_pending_accept_backlog(
    monkeypatch: pytest.MonkeyPatch, *, keep_listener: bool, expect_disconnect: bool
) -> None:
    """Model listener lifetime without binding a port or claiming plugin execution."""
    requests: list[dict[str, Any]] = []
    connections: list[FakeSocket] = []

    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False
            self.output = b""

        def settimeout(self, _timeout: float) -> None:
            pass

        def sendall(self, data: bytes) -> None:
            request = json.loads(data)
            requests.append(request)
            if len(connections) == 2:
                # The second connect succeeded while the server had not yet
                # observed the first connection's FIN. Closing that listener
                # rejects the queued connection; retaining it accepts the next.
                assert connections[0].closed
                if not keep_listener:
                    raise ConnectionResetError("old listener closed with pending accept")
            self.output = _header(request_id=request["request_id"]) + _PAYLOAD

        def recv(self, limit: int) -> bytes:
            result, self.output = self.output[:limit], self.output[limit:]
            return result

        def close(self) -> None:
            self.closed = True

    def connect(_address: Any, *, timeout: float) -> FakeSocket:
        assert timeout > 0
        connection = FakeSocket()
        connections.append(connection)
        return connection

    monkeypatch.setattr(socket, "create_connection", connect)
    client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=1, timeout_s=1), _IDENTITY)
    try:
        client.read()
        client.begin_reset()
        if expect_disconnect:
            with pytest.raises(BoundTelemetryError):
                client.read()
        else:
            client.read()
        assert len(connections) == 2
        assert [request["wait_ready"] for request in requests] == [False, True]
        assert requests[0]["request_id"] != requests[1]["request_id"]
    finally:
        client.close()


def _read_request(connection: socket.socket) -> tuple[str, bool]:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = connection.recv(1)
        if not chunk or len(data) >= 1024:
            raise ValueError("missing or oversized request")
        data.extend(chunk)
    value = json.loads(data)
    assert set(value) == {"command", "request_id", "wait_ready"}
    assert value["command"] == "frame"
    assert type(value["wait_ready"]) is bool
    return str(value["request_id"]), value["wait_ready"]


@contextmanager
def _server(chunks: list[bytes]) -> Iterator[int]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                request_id, _ = _read_request(connection)
                wire = b"".join(chunks).replace(_PLACEHOLDER.encode(), request_id.encode())
                offset = 0
                for chunk in chunks:
                    try:
                        connection.sendall(wire[offset : offset + len(chunk)])
                        offset += len(chunk)
                    except OSError:
                        break
        finally:
            listener.close()

    worker = Thread(target=serve, daemon=True)
    port = listener.getsockname()[1]
    worker.start()
    try:
        yield port
    finally:
        worker.join(timeout=5)
        assert not worker.is_alive()


@contextmanager
def _sequential_server(maps: list[str], *, replay_first: bool = False) -> Iterator[int]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                first_id = ""
                for index, uid in enumerate(maps):
                    try:
                        request_id, _ = _read_request(connection)
                        if index == 0:
                            first_id = request_id
                        if index and replay_first:
                            request_id = first_id
                        connection.sendall(_header(map_uid=uid, request_id=request_id) + _PAYLOAD)
                    except (OSError, ValueError):
                        break
        finally:
            listener.close()

    worker = Thread(target=serve, daemon=True)
    port = listener.getsockname()[1]
    worker.start()
    try:
        yield port
    finally:
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_fragmented_bound_frame_is_read_without_changing_legacy_wire() -> None:
    wire = _header() + _PAYLOAD
    with _server([wire[:3], wire[3:70], wire[70:120], wire[120:]]) as port:
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        assert client.read().values.shape == (33,)
        client.close()


def test_transient_wrong_map_frame_fails_even_when_next_frame_returns_to_a() -> None:
    with _sequential_server(["map-a", "map-b", "map-a"]) as port:
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        client.read()
        with pytest.raises(BoundTelemetryError, match="identity differs"):
            client.read()


def test_old_response_cannot_be_replayed_for_next_request() -> None:
    with _sequential_server(["map-a", "map-a"], replay_first=True) as port:
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        client.read()
        with pytest.raises(BoundTelemetryError, match="identity differs"):
            client.read()


def test_only_first_reset_request_waits_for_readiness() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    requests: list[tuple[str, bool]] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                for _ in range(2):
                    request_id, wait_ready = _read_request(connection)
                    requests.append((request_id, wait_ready))
                    connection.sendall(_header(request_id=request_id) + _PAYLOAD)
        finally:
            listener.close()

    port = listener.getsockname()[1]
    worker = Thread(target=serve, daemon=True)
    worker.start()
    client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
    client.begin_reset()
    client.read()
    client.read()
    client.finish_reset()
    client.close()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert [flag for _, flag in requests] == [True, False]
    assert requests[0][0] != requests[1][0]


@pytest.mark.parametrize(
    "wire",
    [
        _header(instance_epoch="restarted") + _PAYLOAD,
        _header(operation_id="other") + _PAYLOAD,
        _header(state="loading") + _PAYLOAD,
        _header(payload_bytes=True) + _PAYLOAD,
        b'{"map_uid":"map-a","map_uid":"map-b"}\n' + _PAYLOAD,
        _header() + _PAYLOAD[:10],
        b"{" + b"a" * 4096,
    ],
)
def test_unbound_malformed_or_partial_frame_is_never_returned(wire: bytes) -> None:
    with _server([wire]) as port:
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        with pytest.raises(BoundTelemetryError):
            client.read()


def test_reset_waits_for_first_frame_after_player_is_temporarily_unavailable() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    attempts: list[int] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            attempts.append(1)
            with connection:
                request_id, wait_ready = _read_request(connection)
                assert wait_ready is True
                # The plugin waits for readiness without changing connection.
                sleep(0.1)
                connection.sendall(_header(request_id=request_id) + _PAYLOAD)
        finally:
            listener.close()

    port = listener.getsockname()[1]
    worker = Thread(target=serve, daemon=True)
    worker.start()
    client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
    client._buffer.extend(b"stale")
    client.begin_reset()
    assert client._buffer == b""
    assert client.read().values.shape == (33,)
    client.finish_reset()
    # Once one valid frame has been consumed, even a reset may not hide a gap.
    with pytest.raises(BoundTelemetryError):
        client.read()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert attempts == [1]


def test_partial_frame_during_reset_is_not_retried() -> None:
    with _server([_header() + _PAYLOAD[:10]]) as port:
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        client.begin_reset()
        with pytest.raises(BoundTelemetryError, match="complete frame"):
            client.read()


def test_eof_before_first_reset_frame_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.create_connection
    connections: list[object] = []

    def connect(*args: Any, **kwargs: Any) -> socket.socket:
        connections.append(args)
        return original_connect(*args, **kwargs)

    with _server([]) as port:
        monkeypatch.setattr(socket, "create_connection", connect)
        client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=2), _IDENTITY)
        client.begin_reset()
        with pytest.raises(BoundTelemetryError):
            client.read()
    assert len(connections) == 1


def test_reset_connect_uses_one_absolute_deadline_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trackmaniarl.trackmania import bound_telemetry

    timeouts: list[float] = []
    clock = [100.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    def unavailable(address: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        advance(timeout)
        raise TimeoutError("connection unavailable")

    monkeypatch.setattr(bound_telemetry, "monotonic", lambda: clock[0])
    monkeypatch.setattr(socket, "create_connection", unavailable)
    client = BoundOpenPlanetClient(OpenPlanetClientConfig(timeout_s=0.05), _IDENTITY)
    client.begin_reset()
    with pytest.raises(BoundTelemetryError, match="connection unavailable"):
        client.read()
    assert timeouts == pytest.approx([0.05])


def test_fragmented_wrong_uid_is_rejected_before_any_payload_is_returned() -> None:
    class FragmentedSocket:
        def __init__(self) -> None:
            wire = _header(map_uid="map-b") + _PAYLOAD
            split = wire.index(b"map-b") + 2
            self.chunks = [wire[:split], wire[split:]]

        def settimeout(self, value: float) -> None:
            pass

        def sendall(self, data: bytes) -> None:
            request_id = json.loads(data)["request_id"].encode()
            self.chunks = [
                chunk.replace(_PLACEHOLDER.encode(), request_id) for chunk in self.chunks
            ]

        def recv(self, size: int) -> bytes:
            return self.chunks.pop(0)

        def close(self) -> None:
            pass

    client = BoundOpenPlanetClient(OpenPlanetClientConfig(timeout_s=2), _IDENTITY)
    client._socket = FragmentedSocket()  # type: ignore[assignment]
    with pytest.raises(BoundTelemetryError, match="identity differs"):
        client.read()


def test_partial_header_has_absolute_deadline() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                _read_request(connection)
                for _ in range(20):
                    try:
                        connection.sendall(b"a")
                    except OSError:
                        break
                    sleep(0.02)
        finally:
            listener.close()

    port = listener.getsockname()[1]
    worker = Thread(target=serve, daemon=True)
    worker.start()
    client = BoundOpenPlanetClient(OpenPlanetClientConfig(port=port, timeout_s=0.08), _IDENTITY)
    started = monotonic()
    with pytest.raises(BoundTelemetryError, match="interrupted"):
        client.read()
    assert monotonic() - started < 1.0
    worker.join(timeout=5)
    assert not worker.is_alive()
