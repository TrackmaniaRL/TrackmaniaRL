"""Synthetic protocol/transport fixtures, never native physics or captured user data."""

from __future__ import annotations

import ast
import json
import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import (
    codec_v6,
    env,
    features,
    host,
    outcome,
    session,
)
from trackmaniarl.trackmania.native_headless._runtime.artifacts import PINS
from trackmaniarl.trackmania.native_headless._runtime.backend import RawNativeBackend
from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    QualifyPlanSpec,
    QualifyReaderSpec,
    RequestSpec,
    ResponseSpec,
    StateSpec,
)
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig

_CONTRACT = "tmrl.native-headless.terminal/v2"
_META = {"controller": 100, "sim": 200, "input_base": 300, "input_count": 4152}
_PENDING = struct.pack("<3f", 0.0, 1.0, 0.0)


def _reader(clock: int, index: int) -> bytes:
    raw = bytearray(256)
    raw[:8] = b"TMREAD01"
    for offset, value in {
        8: 256,
        12: 1,
        16: 42,
        20: 42,
        112: index,
        116: 4152,
        120: clock + 10,
        124: clock,
        148: 447,
        152: 2,
        188: 4152,
        208: 1,
        212: 11,
    }.items():
        struct.pack_into("<I", raw, offset, value)
    for offset, value in {40: 100, 48: 200, 88: 300, 104: 4152}.items():
        struct.pack_into("<Q", raw, offset, value)
    raw[156:168] = _PENDING if index == 155 else bytes(12)
    return bytes(raw)


def _post(clock: int, *, gas: float = 0.0) -> bytes:
    raw = bytearray(256)
    raw[:8] = b"TMPOST03"
    for offset, value in {
        8: 256,
        12: 3,
        16: 42,
        20: 42,
        112: clock,
        116: clock,
        136: 10,
        140: 10,
        144: clock,
        148: 75,
        212: 10,
    }.items():
        struct.pack_into("<I", raw, offset, value)
    for offset, value in {24: clock, 40: 100, 48: 200}.items():
        struct.pack_into("<Q", raw, offset, value)
    struct.pack_into("<4f", raw, 88, gas, 0.0, 0.0, 0.0)
    struct.pack_into("<10ffIfII", raw, 152, 1.0, *([0.0] * 9), 1000.0, 1, 1000.0, 1, 0)
    return bytes(raw)


def _frame(spec: ResponseSpec, event: bytes, receipt: bytes = bytes(160)) -> bytes:
    raw = bytearray(448)
    raw[:8] = b"TMR6\x06" + bytes([spec.kind]) + bytes(2)
    struct.pack_into("<IIQII", raw, 8, spec.seq, spec.n, spec.epoch, 416, int(spec.kind == 4))
    raw[32:288], raw[288:] = event, receipt
    return bytes(raw)


def _result_evidence(clock: int) -> dict[str, Any]:
    value = {"Time": 17715, "Score": 0, "NbRespawns": 0, "NbCheckpoints": 1}
    validator = {
        "FileName": "g.Ghost.Gbx",
        "MapUid": outcome.MAP_UID,
        "IsValid": False,
        "ValidatedResult": value,
        "DeclaredResult": dict(value, Time=40000),
        "Desc": "synthetic protocol fixture; not a real finish",
    }
    result = bytearray(768)
    result[:8] = b"TMRESULT"
    for offset, number in {
        8: 768,
        12: 1,
        16: 42,
        20: 42,
        128: clock,
        148: 24575,
        152: 17715,
        156: 0,
        160: 0,
        184: 1,
        192: 8,
        240: 13,
        248: 1,
        256: 0,
        260: 17715,
    }.items():
        struct.pack_into("<I", result, offset, number)
    for offset, number in {24: 1, 40: 100, 48: 200}.items():
        struct.pack_into("<Q", result, offset, number)
    return {
        "profile": env.RUNTIME_PROFILE,
        "pins": PINS,
        "reaped": {
            "pid": 42,
            "kind": "reaped",
            "process_gone": True,
            "parent_unchanged": True,
            "exit_status": None,
            "truncated_by_driver": False,
        },
        "validatorArrayHex": json.dumps([validator]).encode().hex(),
        "collectorHex": result.hex(),
    }


@dataclass
class ProtocolFixture:
    packet: dict[str, Any]
    before: dict[str, Any]

    def qualify(self) -> tuple[Any, Any]:
        return env.qualify_plan(
            QualifyPlanSpec(self.packet, self.before, 36, 5, features, codec_v6)
        )


def _fixture(stage: str | None, *, actual: int = 5, legacy: bool = False) -> ProtocolFixture:
    reader = _reader(2230, 155)
    receipt = struct.pack("<4I", 155, 156, 5, 0) + (_PENDING + bytes(12)) * 5 + bytes(24)
    audits = []
    for ordinal in range(5):
        raw = bytearray(256)
        raw[:8] = b"TMAUD002"
        for offset, number in {
            8: 256,
            12: 2,
            16: 42,
            20: 42,
            176: 155,
            180: 156 + ordinal,
            184: 5,
            188: 2230,
            200: 11,
            204: 1,
            208: 155,
            212: 12,
            248: 5,
            252: ordinal,
        }.items():
            struct.pack_into("<I", raw, offset, number)
        for offset, number in {
            24: ordinal + 1,
            192: 7,
            216: 300,
            232: 300 - 4 + (156 + ordinal) * 32,
            240: 300 - 4 + 155 * 32,
        }.items():
            struct.pack_into("<Q", raw, offset, number)
        struct.pack_into("<I", raw, 40, 2)
        struct.pack_into("<I", raw, 72, 2)
        raw[44:56] = _PENDING
        raw[104:136] = raw[136:168] = reader[152:184]
        audits.append(bytes(raw).hex())
    events = [_post(2240 + j * 10, gas=1.0 if j == 0 else 0.0) for j in range(actual)]
    boundary = events[-1]
    next_reader = _reader(2280, 160)
    frames = [_frame(ResponseSpec(4, 11, 5, 7), reader, receipt)]
    if stage != "boundary":
        frames.append(_frame(ResponseSpec(1, 11, 5, 8), _post(2280)))
    if stage is None:
        frames.append(_frame(ResponseSpec(3, 12, 0, 9), next_reader))
    packet = {
        "profile": env.RUNTIME_PROFILE,
        "terminalContract": _CONTRACT,
        "eofStage": stage,
        "pid": 42,
        "meta": _META.copy(),
        "eof": stage is not None,
        "epoch": 9 if stage is None else 8 if stage == "reader" else 7,
        "boundaryHex": boundary.hex(),
        "readerHex": (next_reader if stage is None else reader).hex(),
        "pendingProvenance": {"actionId": 36},
        "responseFramesHex": [b.hex() for b in frames],
        "authorization": {
            "anchor": 2230,
            "end": 2230 + actual * 10,
            "ownP": 155,
            "first": 156,
            "epoch": 7,
            "sequence": 11,
            "actionId": 36,
            "policyDecision": True,
            "pendingSGB": [0.0, 1.0, 0.0],
            "plannedSGB": features.action_sgb(36),
            "receiptHex": receipt.hex(),
            "auditEventHex": audits,
            "nativeEventHex": [b.hex() for b in events],
        },
    }
    if stage is not None:
        packet["resultEvidence"] = _result_evidence(2230 + actual * 10)
        packet["receiveError"] = {
            "reason": "EOFError: synthetic EOF",
            "currentResponseBytes": 0,
            "currentResponseHex": "",
        }
    if legacy:
        del packet["terminalContract"]
        del packet["eofStage"]
    before = {
        "nativeT": 2230,
        "readerHex": reader.hex(),
        "pendingOwnP": 155,
        "epoch": 7,
        "historyActionIds": [],
    }
    return ProtocolFixture(packet, before)


@pytest.mark.parametrize("stage", [None, "boundary", "reader"])
def test_stage_accepts_exact_continue_boundary_or_reader_frames(stage: str | None) -> None:
    fixture = _fixture(stage, actual=3 if stage == "boundary" else 5)
    events, credit = fixture.qualify()
    assert len(events) == credit["intervals"] == (3 if stage == "boundary" else 5)
    assert len(fixture.packet["responseFramesHex"]) == {None: 3, "boundary": 1, "reader": 2}[stage]


@pytest.mark.parametrize("stage", [None, "boundary", "reader"])
def test_stage_rejects_wrong_frame_count(stage: str | None) -> None:
    fixture = _fixture(stage)
    frames = fixture.packet["responseFramesHex"]
    if stage == "boundary":
        frames.append(_frame(ResponseSpec(1, 11, 5, 8), _post(2280)).hex())
    else:
        frames.pop()
    with pytest.raises(ValueError, match="frame count"):
        fixture.qualify()


@pytest.mark.parametrize("stage", [None, "unknown", "boundary"])
def test_versioned_eof_rejects_stage_mismatch(stage: str | None) -> None:
    fixture = _fixture("reader")
    fixture.packet["eofStage"] = stage
    with pytest.raises(ValueError, match=r"EOF transport stage|frame count"):
        fixture.qualify()


def test_reader_stage_requires_version_and_fully_confirmed_ack() -> None:
    fixture = _fixture("reader")
    del fixture.packet["terminalContract"]
    with pytest.raises(ValueError, match="versioned"):
        fixture.qualify()
    fixture = _fixture("reader")
    fixture.packet["boundaryHex"] = _post(2270).hex()
    fixture.packet["responseFramesHex"][1] = _frame(ResponseSpec(1, 11, 5, 8), _post(2270)).hex()
    with pytest.raises(ValueError, match="ACK"):
        fixture.qualify()
    fixture = _fixture("reader", actual=3)
    fixture.packet["boundaryHex"] = _post(2280).hex()
    with pytest.raises(ValueError, match="every ACKed interval"):
        fixture.qualify()


@pytest.mark.parametrize("stage", [None, "boundary", "reader"])
def test_contract_version_and_confirmed_epoch_are_not_optional(stage: str | None) -> None:
    fixture = _fixture(stage)
    fixture.packet["terminalContract"] = "tmrl.native-headless.terminal/unknown"
    with pytest.raises(ValueError, match="contract version"):
        fixture.qualify()
    fixture = _fixture(stage)
    fixture.packet["epoch"] += 1
    with pytest.raises(ValueError, match="epoch"):
        fixture.qualify()


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        [],
        {},
        {"reason": None, "currentResponseBytes": 0, "currentResponseHex": ""},
        {"reason": "TimeoutError: synthetic", "currentResponseBytes": 0, "currentResponseHex": ""},
        {"reason": "EOFError: partial", "currentResponseBytes": 1, "currentResponseHex": "aa"},
        {
            "reason": "EOFError: wrong count type",
            "currentResponseBytes": "0",
            "currentResponseHex": "",
        },
        {
            "reason": "EOFError: bool is not count",
            "currentResponseBytes": False,
            "currentResponseHex": "",
        },
        {
            "reason": "EOFError: wrong tail type",
            "currentResponseBytes": 0,
            "currentResponseHex": b"",
        },
    ],
)
def test_versioned_terminal_requires_literal_zero_tail_evidence(evidence: Any) -> None:
    fixture = _fixture("reader")
    fixture.packet["receiveError"] = evidence
    with pytest.raises(ValueError, match=r"EOF receive evidence|zero-tail"):
        fixture.qualify()


@pytest.mark.parametrize("stage", [None, "boundary"])
def test_legacy_captured_packets_keep_old_three_or_one_frame_contract(stage: str | None) -> None:
    fixture = _fixture(stage, actual=3 if stage else 5, legacy=True)
    events, _ = fixture.qualify()
    assert len(events) == (3 if stage else 5)


class PacketSession:
    """A predetermined protocol packet, no file/process/socket or dynamics."""

    def __init__(self, packet: dict[str, Any]) -> None:
        self.packet = packet

    def advance(self, action: int, n: int) -> dict[str, Any]:
        assert (action, n) == (36, 5)
        return deepcopy(self.packet)


def _backend(fixture: ProtocolFixture) -> env.Native16Env:
    backend = env.Native16Env(
        lambda: None, duration_limit=None, root=Path("/unused-native-fixture")
    )
    backend.pid, backend.meta = 42, _META.copy()
    backend.core, backend.codec, backend.outcomes = features, codec_v6, outcome
    backend.reference = features.Reference(
        [(float(i), 0.0, 0.0) for i in range(100)], {"sha256": "synthetic-only"}
    )
    initial = dict(
        fixture.packet,
        boundaryHex=_post(2230, gas=1.0).hex(),
        readerHex=fixture.before["readerHex"],
        epoch=7,
    )
    backend.current = backend._state(StateSpec(initial, [], 0, False))
    backend.session = PacketSession(fixture.packet)
    backend.elapsed, backend.ended = 0, False
    return backend


@pytest.mark.parametrize("stage", [None, "boundary", "reader"])
def test_completed_boundary_is_ack_semantics_not_eof_boolean(stage: str | None) -> None:
    backend = _backend(_fixture(stage, actual=3 if stage == "boundary" else 5))
    result = backend.step(36)
    assert backend.capabilities["terminalContract"] == _CONTRACT
    assert result["terminalContract"] == _CONTRACT
    assert result["eofStage"] == stage
    assert result["completedRequestedBoundary"] is (stage != "boundary")
    assert result["terminated"] is (stage is not None)
    if stage is not None:
        assert result["outcome"]["validationAcceptance"] is False
        assert result["observation"]["features"] is None
        assert result["bootstrapFeatures"] is None


@pytest.mark.parametrize("stage", ["boundary", "reader"])
def test_eof_alone_never_fabricates_terminal_outcome(stage: str) -> None:
    fixture = _fixture(stage, actual=3 if stage == "boundary" else 5)
    del fixture.packet["resultEvidence"]
    result = _backend(fixture).step(36)
    assert result["status"] == "inconclusive"
    assert result["terminated"] is None
    assert result["truncated"] is None
    assert result["trainingEligible"] is False


def _session_method(name: str) -> Any:
    """Compile one production nested method; never invoke the Session constructor."""
    tree = ast.parse(Path(session.__file__).read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(methods) == 1
    scope = {
        "Any": Any,
        "E": env,
        "core": features,
        "base": host,
        "struct": struct,
        "QualifyReaderSpec": QualifyReaderSpec,
        "RequestSpec": RequestSpec,
        "ResponseSpec": ResponseSpec,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), session.__file__, "exec"), scope)
    return scope[name]


def _advance_method() -> Any:
    return _session_method("_advance")


def test_terminal_contract_matches_inert_backend_and_rx_metadata() -> None:
    root = Path.cwd() / "synthetic-runtime-not-created"
    settings = NativeRuntimeConfig(
        runtime_root=root,
        trace_dir=root / "trace",
        work_dir=root / "work",
        asset_manifest=root / "manifest.json",
    )
    assert RawNativeBackend(settings).capabilities["terminalContract"] == _CONTRACT
    tree = ast.parse(Path(session.__file__).read_text())
    expressions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "rx_active" for target in node.targets
        )
    ]
    assert len(expressions) == 1
    metadata = eval(
        compile(ast.Expression(expressions[0]), session.__file__, "eval"),
        {
            "E": env,
            "verified": SimpleNamespace(manifest_sha256="synthetic"),
            "settings": SimpleNamespace(deployment_manifest_sha256="synthetic"),
            "root": Path("/unused-native-fixture"),
        },
    )
    assert metadata["terminalContract"] == _CONTRACT


class AdvanceTransport:
    """Literal response/event sink for production _advance, without OS handles."""

    def __init__(self, fixture: ProtocolFixture, error: Exception | None = None) -> None:
        self.fixture, self.error = fixture, error
        self.pid, self.meta, self.epoch, self.seq = 42, _META.copy(), 7, 10
        self.reader = bytes.fromhex(fixture.before["readerHex"])
        self.boundary = _post(2230, gas=1.0)
        self.responses: list[bytes] = []
        self.requests: list[RequestSpec] = []
        self.audits: list[bytes] = []
        self.events: dict[int, bytes] = {}
        self.failed = False
        self.ended: list[bool] = []
        self.result_evidence = fixture.packet.get("resultEvidence")
        self.pending_provenance: dict[str, Any] = {}

    def poll_posts(self) -> None:
        if self.requests:
            self.events = {
                env.u(raw, 112): raw
                for raw in map(
                    bytes.fromhex, self.fixture.packet["authorization"]["nativeEventHex"]
                )
            }

    def read_mem(self, address: int, length: int) -> bytes:
        assert (address, length) == (300 - 4 + 156 * 32, 160)
        return (struct.pack("<I", 2) + _PENDING + bytes(16)) * 5

    def request(self, spec: RequestSpec) -> None:
        self.requests.append(spec)
        self.seq += 1
        if spec.op == 4:
            self.audits = list(
                map(bytes.fromhex, self.fixture.packet["authorization"]["auditEventHex"])
            )

    def response(self, spec: ResponseSpec) -> tuple[bytes, bytes]:
        wanted = {4: 0, 1: 1, 3: 2}[spec.kind]
        stage = self.fixture.packet["eofStage"]
        if (spec.kind == 1 and stage == "boundary") or (spec.kind == 3 and stage == "reader"):
            raise self.error or host.IncompleteReceiveError(
                {
                    "reason": "EOFError: synthetic EOF",
                    "currentResponseBytes": 0,
                    "currentResponseHex": "",
                }
            )
        raw = bytes.fromhex(self.fixture.packet["responseFramesHex"][wanted])
        self.responses.append(raw)
        self.epoch = spec.epoch
        return raw[32:288], raw[288:]

    def arm(self) -> None:
        self.request(RequestSpec(3, 0, 0, 0, (), ()))
        self.reader, _ = self.response(ResponseSpec(3, self.seq, 0, self.epoch + 1))

    def end(self, *, truncated: bool) -> None:
        self.ended.append(truncated)

    def packet(self) -> dict[str, Any]:
        return _session_method("_base_packet")(self)


@pytest.mark.parametrize("stage", [None, "boundary", "reader"])
def test_production_advance_catches_eof_during_ack_or_arm(stage: str | None) -> None:
    transport = AdvanceTransport(_fixture(stage, actual=3 if stage == "boundary" else 5))
    packet = _advance_method()(transport, 36, 5)
    assert packet["terminalContract"] == _CONTRACT
    assert packet["eofStage"] == stage
    assert packet["eof"] is (stage is not None)
    assert len(packet["responseFramesHex"]) == {None: 3, "boundary": 1, "reader": 2}[stage]
    assert transport.ended == ([] if stage is None else [False])
    fixture = ProtocolFixture(packet, transport.fixture.before)
    events, _ = fixture.qualify()
    assert len(events) == (3 if stage == "boundary" else 5)


@pytest.mark.parametrize("stage", ["boundary", "reader"])
@pytest.mark.parametrize(
    "case",
    [
        ("EOFError: partial", 4, "aabbccdd"),
        ("TimeoutError: timed out", 0, ""),
        ("EOFError: inconsistent tail", 0, "aa"),
        ("EOFError: bool is not count", False, ""),
    ],
)
def test_partial_or_timeout_is_not_clean_eof(stage: str, case: tuple[str, int, str]) -> None:
    reason, count, tail = case
    error = host.IncompleteReceiveError(
        {"reason": reason, "currentResponseBytes": count, "currentResponseHex": tail}
    )
    transport = AdvanceTransport(_fixture(stage), error)
    with pytest.raises(host.IncompleteReceiveError):
        _advance_method()(transport, 36, 5)
    assert transport.failed is True
    assert transport.ended == []


@pytest.mark.parametrize("stage", ["boundary", "reader"])
def test_storage_failure_never_becomes_native_eof(stage: str) -> None:
    error = RxPersistenceError("synthetic write failure")
    transport = AdvanceTransport(_fixture(stage), error)
    with pytest.raises(RxPersistenceError):
        _advance_method()(transport, 36, 5)
    assert transport.ended == []
