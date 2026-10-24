"""Saved final tape reader must reject unsupported requests before any native operation."""

from __future__ import annotations

import struct
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import env
from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime.shapes import QualifyReaderSpec

# Literal V7 committed reader from finish002, sequence1602, p4147/count4152.
# The original run accepted N5, ACKed native42200, then ARM caused an extra
# POST42210 and FKPOST reject337. No fabricated terminal or changed tolerance.
SAVED_READER = bytes.fromhex(
    "544d5245414430310001000001000000a4010000a40100005f2f000000000000"
    "ec4110000000000040d3ab76105a0000803a9478105a00006099c1cffc7f0000"
    "009bc1cffc7f00008e2f3743105a0000504b2979105a0000544b2979105a0000"
    "b4512b79105a000038100000000000003310000038100000b0a40000a6a40000"
    "02000000ffa70000000000000200000024faffffbf0100000200000000000000"
    "000000000000000000000000b6fa42f900000000020000000000000038100000"
    "60492879105a0000d0172879105a0000010000000b0000006099c1cffc7f0000"
    "9302000000000000e84a2879105a0000504b2979105a00001afafffffea10000"
)
META = {
    "controller": 99026756948800,
    "sim": 99026788956800,
    "input_base": 99026798725972,
    "input_count": 4152,
}


class NativeOperationError(Exception):
    pass


def session_at_saved_tail() -> tuple[ImmediateSession, list[str]]:
    session = ImmediateSession.__new__(ImmediateSession)
    session.reader, session.pid, session.meta = SAVED_READER, 420, META.copy()
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


def test_saved_reader_is_a_real_qualified_v7_tail_boundary() -> None:
    env.qualify_reader(QualifyReaderSpec(raw=SAVED_READER, pid=420, meta=META, anchor=42150))
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
