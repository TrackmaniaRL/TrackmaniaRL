from __future__ import annotations

import hashlib
import math
import struct
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v6, outcome
from trackmaniarl.trackmania.native_headless._runtime.artifacts import PINS as PINS
from trackmaniarl.trackmania.native_headless._runtime.artifacts import dependencies
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    FeaturesSpec,
    QualifyPlanSpec,
    QualifyReaderSpec,
    StateSpec,
    TransitionRewardSpec,
    VerifyCreditSpec,
)

FEATURE_PROFILE = "tmre.native16-pending-reference78/44/v2"
RUNTIME_PROFILE = "tmre.long4152-declared40000/v6"
TERMINAL_CONTRACT = "tmrl.native-headless.terminal/v2"


def u(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def q(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


def sha(b: Any) -> Any:
    return hashlib.sha256(b).hexdigest()


def require(value: Any, reason: Any) -> Any:
    if not value:
        raise ValueError(reason)


def qualify_post(raw: Any, pid: Any, meta: Any) -> Any:
    require(
        type(raw) is bytes and len(raw) == 256 and (raw[:8] == b"TMPOST03"), "own POST256 required"
    )
    require(
        (u(raw, 8), u(raw, 12), u(raw, 16), u(raw, 20), u(raw, 148), u(raw, 212))
        == (256, 3, pid, pid, 75, 10),
        "POST header/PID/TID/mask/role",
    )
    t = u(raw, 112)
    require(
        t == u(raw, 116) == u(raw, 144)
        and u(raw, 128) == 0
        and (u(raw, 136) == u(raw, 140) == 10)
        and (t % 10 == 0),
        "POST native grid",
    )
    require(q(raw, 40) == meta["controller"] and q(raw, 48) == meta["sim"], "POST owner chain")
    return (raw[152:212] + raw[88:104], t)


def qualify_reader(spec: QualifyReaderSpec) -> Any:
    raw, pid, meta, anchor = (spec.raw, spec.pid, spec.meta, spec.anchor)
    require(
        type(raw) is bytes and len(raw) == 256 and (raw[:8] == b"TMREAD01"),
        "own reader256 required",
    )
    require(
        (u(raw, 8), u(raw, 12), u(raw, 16), u(raw, 20), u(raw, 148), u(raw, 208), u(raw, 212))
        == (256, 1, pid, pid, 447, 1, 11),
        "reader header/PID/TID/mask/role",
    )
    require(
        q(raw, 40) == meta["controller"]
        and q(raw, 48) == meta["sim"]
        and (q(raw, 88) == meta["input_base"]),
        "reader owner chain",
    )
    require(
        u(raw, 116) == q(raw, 104) == u(raw, 188) == meta["input_count"] == 4152
        and u(raw, 112) < 4152
        and (u(raw, 120) == anchor + 10)
        and (u(raw, 124) == anchor),
        "reader index/count/anchor",
    )
    require(raw[156:160] == raw[184:188], "reader shadow")
    pending = list(struct.unpack_from("<3f", raw, 156))
    require(all(math.isfinite(v) for v in pending), "reader finite channels")
    return pending


def qualify_plan(spec: QualifyPlanSpec) -> Any:
    packet, before, action, n, core, codec = (
        spec.packet,
        spec.before,
        spec.action,
        spec.n,
        spec.core,
        spec.codec,
    )
    "Host-level joins around frozen pure per-interval credit helper."
    pid, meta = (packet["pid"], packet["meta"])
    require(packet["profile"] == RUNTIME_PROFILE, "runtime profile mismatch")
    reader = bytes.fromhex(before["readerHex"])
    anchor, p = (before["nativeT"], u(reader, 112))
    pending = qualify_reader(QualifyReaderSpec(raw=reader, pid=pid, meta=meta, anchor=anchor))
    a = packet["authorization"]
    require(
        (a["anchor"], a["ownP"], a["first"], a["epoch"]) == (anchor, p, p + 1, before["epoch"]),
        "plan anchor/P/epoch",
    )
    require(a["actionId"] == action and a["policyDecision"] is True, "policy action binding")
    plan = core.action_sgb(action)
    require(
        len(a["plannedSGB"]) == 5
        and all(
            (
                struct.pack("<3f", *v) == struct.pack("<3f", *w)
                for v, w in zip(a["plannedSGB"], plan, strict=False)
            )
        )
        and (struct.pack("<3f", *a["pendingSGB"]) == reader[156:168]),
        "desired/pending binding",
    )
    receipt = bytes.fromhex(a["receiptHex"])
    frames = [bytes.fromhex(v) for v in packet["responseFramesHex"]]
    eof = packet.get("eof")
    require(type(eof) is bool, "explicit EOF flag")
    contract = packet.get("terminalContract")
    stage = packet.get("eofStage")
    if contract is None:
        # Historical captures only had the EOF-before-ACK path.
        require(stage is None, "terminal stage requires versioned contract")
        stage = "boundary" if eof else None
    else:
        require(contract == TERMINAL_CONTRACT, "terminal contract version")
        require(
            stage in ("boundary", "reader") if eof else stage is None,
            "EOF transport stage",
        )
        if eof:
            receive_error: Any = packet.get("receiveError")
            require(isinstance(receive_error, dict), "literal EOF receive evidence")
            require(
                type(receive_error.get("reason")) is str
                and receive_error["reason"].startswith("EOFError:")
                and type(receive_error.get("currentResponseBytes")) is int
                and receive_error["currentResponseBytes"] == 0
                and receive_error.get("currentResponseHex") == "",
                "zero-tail socket EOF required",
            )
    has_ack = not eof or stage == "reader"
    require(
        len(frames) == (3 if not eof else 2 if has_ack else 1),
        "commit/boundary/ARM response frame count",
    )
    expected = [(4, a["sequence"], n, before["epoch"])]
    if has_ack:
        expected.append((1, a["sequence"], n, before["epoch"] + 1))
    if not eof:
        expected.append((3, a["sequence"] + 1, 0, before["epoch"] + 2))
    for frame, want in zip(frames, expected, strict=False):
        decoded = codec.decode_v6_response(frame)
        require(
            (decoded["kind"], decoded["sequence"], decoded["iterations"], decoded["epoch"]) == want,
            "response sequence/epoch/iterations",
        )
    require(
        frames[0][32:288] == reader and frames[0][288:] == receipt,
        "accepted commit frame/reader/receipt",
    )
    if has_ack:
        require(
            frames[1][32:288].hex() == packet["boundaryHex"]
            and u(frames[1], 32 + 112) == anchor + n * 10,
            "ACK raw event/target binding",
        )
    if not eof:
        require(frames[2][32:288].hex() == packet["readerHex"], "ARM raw event binding")
    if contract is not None:
        require(
            packet["epoch"] == before["epoch"] + (2 if not eof else 1 if has_ack else 0),
            "confirmed response epoch",
        )
    length = u(receipt, 8)
    require(
        1 <= length <= 5 and p + 1 + length <= 4152 and (length >= n - 1), "future array bounds"
    )
    require(
        (u(receipt, 0), u(receipt, 4), u(receipt, 12)) == (p, p + 1, 0), "accepted receipt metadata"
    )
    require(receipt[16 + 24 * length :] == bytes(144 - 24 * length), "receipt padding")
    audits = [bytes.fromhex(v) for v in a["auditEventHex"]]
    require(len(audits) == length, "one complete audit per write")
    seen = set()
    for j, audit in enumerate(audits):
        require(len(audit) == 256 and audit[:8] == b"TMAUD002", "write audit format")
        require(
            (
                u(audit, 8),
                u(audit, 12),
                u(audit, 16),
                u(audit, 20),
                u(audit, 176),
                u(audit, 180),
                u(audit, 184),
                u(audit, 188),
                u(audit, 200),
                u(audit, 204),
                u(audit, 208),
                u(audit, 212),
                u(audit, 248),
                u(audit, 252),
            )
            == (256, 2, pid, pid, p, p + 1 + j, n, anchor, a["sequence"], 1, p, 12, length, j),
            "write audit joins",
        )
        require(q(audit, 24) not in seen, "duplicate audit hit")
        seen.add(q(audit, 24))
        require(
            q(audit, 192) == before["epoch"]
            and q(audit, 216) == meta["input_base"]
            and (q(audit, 232) == meta["input_base"] - 4 + (p + 1 + j) * 32)
            and (q(audit, 240) == meta["input_base"] - 4 + p * 32),
            "audit epoch/address",
        )
        require(audit[104:136] == audit[136:168] == reader[152:184], "current reader unchanged")
        require(
            audit[40:44] == audit[72:76] and audit[56:72] == audit[88:104],
            "future nonchannel bytes unchanged",
        )
        require(
            audit[44:56] == receipt[16 + j * 24 : 28 + j * 24]
            and audit[76:88] == receipt[28 + j * 24 : 40 + j * 24] == struct.pack("<3f", *plan[j])
            and (audit[168:172] == audit[44:48])
            and (audit[172:176] == audit[76:80]),
            "receipt/readback/shadow join",
        )
    events = [bytes.fromhex(v) for v in a["nativeEventHex"]]
    require(0 <= len(events) <= n, "bounded actual interval count")
    if eof and has_ack:
        require(len(events) == n, "reader EOF requires every ACKed interval")
    hits = seen.copy()
    for event in events:
        qualify_post(event, pid, meta)
        require(q(event, 24) not in hits, "duplicate interval hit")
        hits.add(q(event, 24))
    credit = (
        core.verify_credit(
            VerifyCreditSpec(events=events, anchor=anchor, planned=plan, pending=pending)
        )
        if events
        else None
    )
    return (events, credit)


class Native16Env:
    closed: Any
    codec: Any
    core: Any
    current: Any
    duration_limit: Any
    elapsed: Any
    ended: Any
    factory: Any
    meta: Any
    outcomes: Any
    pid: Any
    reference: Any
    root: Any
    session: Any
    # Inconclusive flags remain None; no fabricated transition.

    def __init__(  # noqa: PLR0913 -- explicit reset profile beside pinned runtime inputs
        self,
        session_factory: Any,
        *,
        duration_limit: Any = 39950,
        root: Any,
        reset_profile: str = "legacy-gas-prelude/v1",
    ) -> None:
        require(callable(session_factory), "explicit session factory required")
        require(
            duration_limit is None
            or (type(duration_limit) is int and duration_limit > 0 and (duration_limit % 10 == 0)),
            "explicit duration in native units",
        )
        self.factory, self.duration_limit, self.root = (session_factory, duration_limit, Path(root))
        self.reset_profile = reset_profile
        self.session = None
        self.current = None
        self.ended = True
        self.closed = False

    @property
    def capabilities(self) -> Any:
        return {
            "profile": RUNTIME_PROFILE,
            "terminalContract": TERMINAL_CONTRACT,
            "featureProfile": FEATURE_PROFILE,
            "featureDimension": 44,
            "partiallyObserved": True,
            "actionCount": 78,
            "nativeIterations": [1, 2, 3, 4, 5],
            "nativeIntervalRaw": 10,
            "nativeTimeIsRaceTime": False,
            "arbitrarySnapshot": False,
            "restoreSnapshot": False,
            "reset": self.reset_profile,
            "SAC33": False,
            "clientParity": False,
            "nativeFactoryValidation": "source-only; runtime not executed for env002",
            "EOFv6PolicyRuntimeValidated": False,
            "terminalFlagsCanBeNone": True,
        }

    def _state(self, spec: StateSpec) -> Any:
        packet, history, index, terminal = (spec.packet, spec.history, spec.index, spec.terminal)
        require(packet["profile"] == RUNTIME_PROFILE, "runtime profile mismatch")
        raw = bytes.fromhex(packet["boundaryHex"])
        state76, clock = qualify_post(raw, packet["pid"], packet["meta"])
        native = self.core.native16(state76)
        projection = self.reference.locate(native["physicalPosition"], index)
        if terminal:
            reader = None
            pending = None
            x = None
        else:
            reader = bytes.fromhex(packet["readerHex"])
            pending = qualify_reader(
                QualifyReaderSpec(raw=reader, pid=packet["pid"], meta=packet["meta"], anchor=clock)
            )
            x = self.core.features(
                FeaturesSpec(
                    state=native,
                    pending=pending,
                    clock=clock,
                    reference=self.reference,
                    projection=projection,
                    history=history,
                )
            )
        require(x is None or len(x) == 44, "feature44 dimension")
        return {
            "schema": self.core.PROFILE,
            "resetProfile": self.reset_profile,
            "inputSchedule": "pending-then-future/v1",
            "featureProfile": FEATURE_PROFILE,
            "nativeT": clock,
            "native76Hex": state76.hex(),
            "readerHex": None if reader is None else reader.hex(),
            "pendingSGB": pending,
            "pendingOwnP": None if terminal else u(reader, 112),
            "epoch": packet.get("epoch"),
            "historyActionIds": history[:],
            "projection": projection,
            "features": x,
            "pendingProvenance": None if terminal else packet["pendingProvenance"],
        }

    def reset(self) -> Any:
        require(not self.closed, "env closed")
        if self.session is not None and (not self.ended):
            self.session.end(truncated=True)
        self.core, self.reference, data = dependencies(self.root)
        self.outcomes = outcome
        self.codec = codec_v6
        if self.session is None:
            self.session = self.factory()
        packet = self.session.reset()
        self.current = self._state(
            StateSpec(
                packet=packet, history=[], index=data["nativeTimes"].index(2230), terminal=False
            )
        )
        require(
            self.current["nativeT"] == 2230
            and self.current["pendingOwnP"] == 155
            and (
                self.current["pendingSGB"]
                == [0.0, float(self.reset_profile == "legacy-gas-prelude/v1"), 0.0]
            )
            and (packet["pendingProvenance"].get("actionId") is None),
            "uncredited administrative start required",
        )
        self.pid, self.meta = (packet["pid"], packet["meta"])
        self.elapsed = 0
        self.ended = False
        return self.read_state()

    def read_state(self) -> Any:
        require(self.current is not None and (not self.closed), "reset before read_state")
        return deepcopy(self.current)

    def step(self, action78: Any, native_iterations: Any = 5) -> Any:
        require(
            not self.closed and (not self.ended) and (self.current is not None),
            "active reset episode required",
        )
        self.core.action_sgb(action78)
        require(
            type(native_iterations) is int and 1 <= native_iterations <= 5, "native_iterations1..5"
        )
        n = native_iterations
        if self.duration_limit is not None:
            n = min(n, (self.duration_limit - self.elapsed) // 10)
        require(n > 0, "duration exhausted")
        before = self.current
        packet = None
        events, credit = (None, None)
        try:
            packet = self.session.advance(action78, n)
            require(
                packet["pid"] == self.pid and packet["meta"] == self.meta,
                "episode identity changed",
            )
            events, credit = qualify_plan(
                QualifyPlanSpec(
                    packet=packet,
                    before=before,
                    action=action78,
                    n=n,
                    core=self.core,
                    codec=self.codec,
                )
            )
            eof = packet.get("eof") is True
            require(eof or len(events) == n, "missing actual interval without EOF")
            require(events, "no completed interval")
            require(
                bytes.fromhex(packet["boundaryHex"]) == events[-1],
                "endpoint must be latest own POST",
            )
            history = (before["historyActionIds"] + [action78])[-2:]
            self.elapsed += len(events) * 10
            outcome = packet.get("outcome", {})
            if eof:
                outcome = self.outcomes.classify(packet, sys.modules[__name__], self.root)
                require(
                    outcome.get("status") == "terminated"
                    and outcome.get("terminated") is True
                    and (outcome.get("truncated") is False)
                    and (outcome.get("qualifiedBy") == "env002.full-original-validator+TMRESULT"),
                    "natural EOF outcome unqualified",
                )
            else:
                require(packet["epoch"] == before["epoch"] + 2, "ACK+ARM epoch")
                next_reader = bytes.fromhex(packet["readerHex"])
                require(
                    u(next_reader, 112) == before["pendingOwnP"] + n
                    and next_reader[156:168]
                    == struct.pack("<3f", *self.core.action_sgb(action78)[n - 1]),
                    "next pending consumed-plan index",
                )
            state = self._state(
                StateSpec(
                    packet=packet,
                    history=history,
                    index=before["projection"]["index"],
                    terminal=eof,
                )
            )
            truncated = (
                not eof
                and self.duration_limit is not None
                and (self.elapsed >= self.duration_limit)
            )
            if truncated:
                self.session.end(truncated=True)
            self.ended = eof or truncated
            self.current = state
            reward = self.core.transition_reward(
                TransitionRewardSpec(
                    before=before["projection"],
                    after=state["projection"],
                    delta_native=len(events) * 10,
                    finish=eof,
                )
            )
            return {
                "status": "terminated" if eof else "truncated" if truncated else "continue",
                "terminalContract": TERMINAL_CONTRACT,
                "eofStage": packet.get("eofStage") or ("boundary" if eof else None),
                "observation": self.read_state(),
                "reward": reward,
                "terminated": eof,
                "truncated": truncated,
                "actualIterations": len(events),
                "requestedIterations": native_iterations,
                "effectiveRequestedIterations": n,
                "deltaNative": len(events) * 10,
                "credit": credit,
                "authorization": packet["authorization"],
                "trainingEligible": True,
                "bootstrapFeatures": None if eof else state["features"][:],
                "truncationReason": "explicit native duration limit" if truncated else None,
                "outcome": outcome if eof else None,
                "completedRequestedBoundary": not eof or packet.get("eofStage") == "reader",
            }
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            EOFError,
            TimeoutError,
            AssertionError,
            struct.error,
        ) as error:
            self.ended = True
            endpoint = None
            if packet is not None:
                try:
                    require(packet["pid"] == self.pid and packet["meta"] == self.meta, "identity")
                    endpoint = self._state(
                        StateSpec(
                            packet=packet,
                            history=before["historyActionIds"],
                            index=before["projection"]["index"],
                            terminal=True,
                        )
                    )
                except (ValueError, KeyError, TypeError, struct.error):
                    pass
            self.current = endpoint or before
            return {
                "status": "inconclusive",
                "observation": endpoint,
                "reward": None,
                "terminated": None,
                "truncated": None,
                "actualIterations": len(events)
                if credit is not None and events is not None
                else None,
                "requestedIterations": native_iterations,
                "effectiveRequestedIterations": n,
                "deltaNative": credit["nativeDelta"] if credit is not None else None,
                "credit": credit,
                "trainingEligible": False,
                "bootstrapFeatures": None,
                "reason": type(error).__name__ + ": " + str(error),
                "evidence": packet,
            }

    def close(self) -> Any:
        if not self.closed:
            self.closed = True
            if self.session is not None:
                self.session.close()

    def snapshot(self) -> Any:
        raise NotImplementedError("arbitrary full-engine snapshots unsupported")

    def restore(self, snapshot: Any) -> Any:
        raise NotImplementedError("arbitrary full-engine snapshot restore unsupported")
