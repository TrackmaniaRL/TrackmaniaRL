"""Pure request packing and explicit host gates, with a byte sink, never a server."""

from __future__ import annotations

import ast
import struct
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import host
from trackmaniarl.trackmania.native_headless._runtime.shapes import RequestSpec


class ByteSink:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def settimeout(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("positive test timeout required")

    def sendall(self, raw: bytes) -> None:
        self.sent.append(raw)


class PackingHost(host.NativeSession):
    """No base constructor: no files/process/socket is opened."""

    def __init__(self) -> None:
        self.seq = 0
        self.epoch = 7
        self.requests = []
        self.connection = ByteSink()

    def remaining(self) -> float:
        return 5.0


def test_request192_is_literal_and_invalid_plan_sends_nothing() -> None:
    transport = PackingHost()
    before = struct.pack("<3f", 0.0, 1.0, 0.0)
    after = struct.pack("<3f", 1.0, 0.0, 1.0)
    transport.request(RequestSpec(4, 5, 155, 156, [before] * 5, [after] * 5))
    expected = bytearray(192)
    expected[:8] = b"TMA6\x06\x04\0\0"
    struct.pack_into("<IIQIII", expected, 8, 1, 5, 7, 155, 156, 5)
    for index in range(5):
        expected[40 + 24 * index : 52 + 24 * index] = before
        expected[52 + 24 * index : 64 + 24 * index] = after
    assert transport.connection.sent == [bytes(expected)]
    assert transport.requests[0]["hex"] == bytes(expected).hex()
    second = PackingHost()
    try:
        second.request(RequestSpec(4, 5, 155, 156, [before], []))
    except ValueError:
        assert not second.connection.sent
    else:
        raise AssertionError("mismatched source/desired channels accepted")


def test_runtime_gates_do_not_depend_on_python_assert() -> None:
    source = Path(host.__file__).read_text()
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    raises: list[Any] = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    assert len(raises) >= 22
