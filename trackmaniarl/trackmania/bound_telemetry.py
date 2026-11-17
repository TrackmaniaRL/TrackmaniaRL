"""Experimental protocol-3 frames bound to a map-switch operation.

The frame header and its binary telemetry are emitted together by the plugin.
This validates their declared identity, not the actual map bytes loaded by the
game or an atomic snapshot of internal engine state.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4

import numpy as np

from trackmaniarl.trackmania.telemetry import (
    OpenPlanetClient,
    OpenPlanetClientConfig,
    TelemetryFrame,
)

_MAX_HEADER_BYTES = 4096


class BoundTelemetryError(RuntimeError):
    """A telemetry frame cannot be tied to the assigned map operation."""


class _BoundTelemetryUnavailableError(BoundTelemetryError):
    """No bytes of a new frame were received before a disconnect."""


@dataclass(frozen=True, slots=True)
class BoundTelemetryIdentity:
    instance_epoch: str
    operation_id: str
    map_uid: str


def _unique_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate bound telemetry header field")
        result[key] = value
    return result


class BoundOpenPlanetClient(OpenPlanetClient):
    """Request fresh frames; uncertain or mismatched responses fail closed."""

    def __init__(self, config: OpenPlanetClientConfig, identity: BoundTelemetryIdentity) -> None:
        super().__init__(config)
        self.identity = identity
        self._reset_first_frame = False

    def begin_reset(self) -> None:
        """Drop old packets and request readiness for the first reset frame."""
        self.close()
        self._reset_first_frame = True

    def finish_reset(self) -> None:
        self._reset_first_frame = False

    def read(self) -> TelemetryFrame:
        deadline = monotonic() + self.timeout_s
        frame = self._read_bound_frame(deadline)
        self._reset_first_frame = False
        return frame

    def _read_bound_frame(self, deadline: float) -> TelemetryFrame:
        try:
            try:
                remaining = self._remaining(deadline)
                if self._socket is None:
                    connection = socket.create_connection((self.host, self.port), timeout=remaining)
                    self._socket = connection
                    connection.settimeout(remaining)
            except OSError as error:
                raise _BoundTelemetryUnavailableError(
                    "bound telemetry connection unavailable"
                ) from error
            if self._buffer:
                raise BoundTelemetryError("unexpected buffered telemetry before request")
            request_id = uuid4().hex
            assert self._socket is not None
            try:
                self._socket.settimeout(self._remaining(deadline))
                request = (
                    json.dumps(
                        {
                            "command": "frame",
                            "request_id": request_id,
                            "wait_ready": self._reset_first_frame,
                        },
                        separators=(",", ":"),
                    ).encode("ascii")
                    + b"\n"
                )
                self._socket.sendall(request)
            except OSError as error:
                raise _BoundTelemetryUnavailableError(
                    "bound telemetry request unavailable"
                ) from error
            header = self._read_header(deadline)
            expected = {
                "protocol_version": "3",
                "request_id": request_id,
                "instance_epoch": self.identity.instance_epoch,
                "operation_id": self.identity.operation_id,
                "map_uid": self.identity.map_uid,
                "state": "ready",
                "payload_bytes": self._packet.size,
            }
            if header != expected or type(header.get("payload_bytes")) is not int:
                raise BoundTelemetryError("telemetry frame identity differs from map assignment")
            packet = self._read_exact(self._packet.size, deadline)
            if self._buffer:
                raise BoundTelemetryError("unexpected trailing bound telemetry data")
            values = np.asarray(self._packet.unpack(packet), dtype=np.float32)
            self._validate_grab_data(values)
            return TelemetryFrame(values)
        except BoundTelemetryError:
            self.close()
            raise
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            self.close()
            raise BoundTelemetryError("bound telemetry frame unavailable or malformed") from error

    def read_next(self) -> TelemetryFrame:
        return self.read()

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("bound telemetry frame deadline exceeded")
        return remaining

    def _receive(self, deadline: float, *, payload: bool = False) -> None:
        assert self._socket is not None
        try:
            self._socket.settimeout(self._remaining(deadline))
            chunk = self._socket.recv(4096)
        except OSError as error:
            if not self._buffer and not payload:
                raise _BoundTelemetryUnavailableError(
                    "bound telemetry frame unavailable"
                ) from error
            raise BoundTelemetryError("bound telemetry frame was interrupted") from error
        if not chunk:
            if not self._buffer and not payload:
                raise _BoundTelemetryUnavailableError("bound telemetry frame unavailable")
            raise BoundTelemetryError("bound telemetry disconnected before a complete frame")
        self._buffer.extend(chunk)

    def _read_header(self, deadline: float) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            if len(self._buffer) >= _MAX_HEADER_BYTES:
                raise BoundTelemetryError("bound telemetry header exceeds limit")
            self._receive(deadline)
        index = self._buffer.index(b"\n")
        if index + 1 > _MAX_HEADER_BYTES:
            raise BoundTelemetryError("bound telemetry header exceeds limit")
        raw = bytes(self._buffer[:index])
        del self._buffer[: index + 1]
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_fields)
        if not isinstance(value, dict):
            raise BoundTelemetryError("bound telemetry header must be an object")
        return value

    def _read_exact(self, size: int, deadline: float) -> bytes:
        while len(self._buffer) < size:
            self._receive(deadline, payload=True)
        value = bytes(self._buffer[:size])
        del self._buffer[:size]
        return value
