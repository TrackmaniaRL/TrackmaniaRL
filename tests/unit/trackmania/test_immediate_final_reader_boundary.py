"""Synthetic final-reader events reject unsupported requests before native operations."""

from __future__ import annotations

import struct
from typing import Any

import pytest

from tests.unit.trackmania._immediate_v7_synthetic import META, PID, reader
from trackmaniarl.trackmania.native_headless._runtime import env
from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime.shapes import QualifyReaderSpec

# Synthetic reader: invented owner addresses and deliberately chosen boundary index.
SAVED_READER = reader(4147, 42150)


class NativeOperationError(Exception):
    pass


def session_at_saved_tail() -> tuple[ImmediateSession, list[str]]:
    session = ImmediateSession.__new__(ImmediateSession)
    session.reader, session.pid, session.meta = SAVED_READER, PID, META.copy()
    boundary = bytearray(256)
    struct.pack_into("<I", boundary, 112, 42150)
    session.boundary = bytes(boundary)
    calls: list[str] = []

    def read_native(address: Any, count: Any) -> bytes:
        calls.append("read-native")
        raise NativeOperationError

    def request_native(request: Any) -> None:
        calls.append("request-native")
        raise NativeOperationError

    session.read_mem, session.request = read_native, request_native
    return session, calls


def test_synthetic_reader_qualifies_at_v7_tail_boundary() -> None:
    env.qualify_reader(QualifyReaderSpec(raw=SAVED_READER, pid=PID, meta=META, anchor=42150))
    assert env.u(SAVED_READER, 112) == 4147
    assert env.u(SAVED_READER, 116) == 4152


def test_last_five_record_request_is_rejected_whole_before_native_reads_or_writes() -> None:
    session, calls = session_at_saved_tail()
    with pytest.raises(ValueError, match="unsupported-final-reader-boundary"):
        session.advance(39, 5)
    assert calls == []
    assert session.reader == SAVED_READER
    assert env.u(session.boundary, 112) == 42150


def test_request_that_leaves_a_next_reader_is_not_blocked_by_tail_guard() -> None:
    session, calls = session_at_saved_tail()
    with pytest.raises(NativeOperationError):
        session.advance(39, 4)
    assert calls == ["read-native"]


def test_single_last_record_is_also_rejected_before_any_native_operation() -> None:
    session, calls = session_at_saved_tail()
    reader = bytearray(SAVED_READER)
    struct.pack_into("<III", reader, 112, 4151, 4152, 42200)
    struct.pack_into("<I", reader, 124, 42190)
    boundary = bytearray(session.boundary)
    struct.pack_into("<I", boundary, 112, 42190)
    session.reader, session.boundary = bytes(reader), bytes(boundary)
    with pytest.raises(ValueError, match="unsupported-final-reader-boundary"):
        session.advance(39, 1)
    assert calls == []
