"""Explicit V7 current-input session; V6 launch and qualification stay unchanged."""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import time
from bisect import bisect_right, insort
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v7, env, features, outcome
from trackmaniarl.trackmania.native_headless._runtime.capture import (
    Capture,
    RxPersistenceError,
    json_exclusive,
)
from trackmaniarl.trackmania.native_headless._runtime.host import (
    IncompleteReceiveError,
    NativeSession,
)
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os, linux_socket
from trackmaniarl.trackmania.native_headless._runtime.quota import TraceUsage
from trackmaniarl.trackmania.native_headless._runtime.quota_monitor import TraceQuotaMonitor
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    CaptureSpec,
    QualifyReaderSpec,
    RequestSpec,
    ResponseSpec,
)
from trackmaniarl.trackmania.native_headless._runtime.supervisor import spawn_supervised
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROFILE,
    IMMEDIATE_PROTOCOL,
    IMMEDIATE_RESET_PROFILE,
    NativeRuntimeConfig,
)

SCHEDULE = "current-record-before-load/v1"
RESET_NATIVE_T = 2200
BOOTSTRAP_NATIVE_T = 2190
TERMINAL_CONTRACT = "tmrl.native-immediate.terminal/v1"


def physical_state(packet: dict[str, Any], *, terminal: bool = False) -> dict[str, Any]:
    raw = bytes.fromhex(packet["boundaryHex"])
    state, native_t = env.qualify_post(raw, packet["pid"], packet["meta"])
    actual = features.native16(state)
    reader = None if terminal else bytes.fromhex(packet["readerHex"])
    current = (
        None
        if reader is None
        else env.qualify_reader(
            QualifyReaderSpec(raw=reader, pid=packet["pid"], meta=packet["meta"], anchor=native_t)
        )
    )
    return {
        "schema": "tmrl.native-immediate-state/v1",
        "resetProfile": IMMEDIATE_RESET_PROFILE,
        "inputSchedule": SCHEDULE,
        "nativeT": native_t,
        "native76Hex": state.hex(),
        "physical": actual,
        "appliedSGB": [actual["steer"], actual["gas"], actual["brake"]],
        "readerHex": None if reader is None else reader.hex(),
        "currentRecordSGB": current,
        "currentRecordIndex": None if reader is None else env.u(reader, 112),
        "currentRecordProvenance": "unissued literal record; replaced before next input load",
        "pendingSGB": None,
        "featureProfile": None,
        "features": None,
        "epoch": packet["epoch"],
    }


def qualify_immediate(  # noqa: PLR0913
    packet: dict[str, Any], before: dict[str, Any], action: int, n: int
) -> list[bytes]:
    """Join exact receipt, channel-only audits, own POSTs and response epochs."""
    env.require(packet["profile"] == codec_v7.PROFILE_ID, "V7 runtime profile required")
    pid, meta = packet["pid"], packet["meta"]
    reader = bytes.fromhex(before["readerHex"])
    anchor, p = before["nativeT"], env.u(reader, 112)
    env.qualify_reader(QualifyReaderSpec(raw=reader, pid=pid, meta=meta, anchor=anchor))
    a = packet["authorization"]
    env.require(
        (a["anchor"], a["ownP"], a["first"], a["n"], a["actionId"], a["epoch"])
        == (anchor, p, p, n, action, before["epoch"]),
        "V7 action/anchor/current-index/epoch binding",
    )
    plan = features.action_sgb(action)[:n]
    afters = [struct.pack("<3f", *v) for v in plan]
    originals = [bytes.fromhex(v) for v in a["originalRecordHex"]]
    env.require(len(originals) == n and all(len(v) == 32 for v in originals), "V7 original records")
    befores = [v[4:16] for v in originals]
    receipt = bytes.fromhex(a["receiptHex"])
    env.require(
        receipt
        == struct.pack("<4I", p, p, n, 2)
        + b"".join(x + y for x, y in zip(befores, afters, strict=True))
        + bytes(144 - n * 24),
        "V7 exact accepted current-record receipt",
    )
    eof, stage = packet["eof"], packet["eofStage"]
    env.require(
        type(eof) is bool and (stage in ("boundary", "reader") if eof else stage is None),
        "V7 EOF stage",
    )
    if eof:
        error = packet["receiveError"]
        env.require(
            error["reason"].startswith("EOFError:")
            and error["currentResponseBytes"] == 0
            and error["currentResponseHex"] == "",
            "V7 only a zero-tail socket EOF may be terminal",
        )
    ack = not eof or stage == "reader"
    wanted = [(6, a["sequence"], n, before["epoch"])]
    if ack:
        wanted.append((1, a["sequence"], n, before["epoch"] + 1))
    if not eof:
        wanted.append((3, a["sequence"] + 1, 0, before["epoch"] + 2))
    frames = [bytes.fromhex(v) for v in packet["responseFramesHex"]]
    env.require(len(frames) == len(wanted), "V7 response count")
    for frame, want in zip(frames, wanted, strict=True):
        d = codec_v7.decode_v7_response(frame)
        env.require(
            (d["kind"], d["sequence"], d["iterations"], d["epoch"]) == want, "V7 response sequence"
        )
        env.require(d["event"]["pid"] == pid, "V7 response owner")
    env.require(frames[0][32:288] == reader and frames[0][288:] == receipt, "V7 commit frame join")
    env.require(
        packet["epoch"] == before["epoch"] + (2 if not eof else 1 if ack else 0), "V7 epoch advance"
    )
    if ack:
        env.require(frames[1][32:288].hex() == packet["boundaryHex"], "V7 ACK endpoint join")
    if not eof:
        env.require(frames[-1][32:288].hex() == packet["readerHex"], "V7 next reader join")
        next_reader = bytes.fromhex(packet["readerHex"])
        env.require(env.u(next_reader, 112) == p + n, "V7 exactly consumed current records")
    audits = [bytes.fromhex(v) for v in a["auditEventHex"]]
    env.require(len(audits) == n, "V7 one actual audit per write")
    hits = set()
    for j, audit in enumerate(audits):
        env.require(len(audit) == 256 and audit[:8] == b"TMAUD003", "V7 audit format")
        env.require(
            tuple(
                env.u(audit, o)
                for o in (8, 12, 16, 20, 176, 180, 184, 188, 200, 204, 208, 212, 248, 252)
            )
            == (256, 2, pid, pid, p, p + j, n, anchor, a["sequence"], 1, p, 12, n, j),
            "V7 own audit identity/sequence/indices",
        )
        env.require(
            env.q(audit, 192) == before["epoch"]
            and env.q(audit, 216) == meta["input_base"]
            and env.q(audit, 232) == meta["input_base"] - 4 + (p + j) * 32
            and env.q(audit, 240) == meta["input_base"] - 4 + p * 32,
            "V7 exact audit epoch/address joins",
        )
        old, new = audit[40:72], audit[72:104]
        env.require(
            old == originals[j]
            and new[:4] == old[:4]
            and new[16:] == old[16:]
            and new[4:16] == afters[j],
            "V7 channel-only record writes",
        )
        env.require(
            audit[104:136] == reader[152:184]
            and audit[136:140] == reader[152:156]
            and audit[140:152] == afters[0]
            and audit[152:168] == reader[168:184],
            "V7 current record before/after",
        )
        env.require(
            audit[168:172] == befores[j][:4] and audit[172:176] == afters[j][:4], "V7 shadow join"
        )
        env.require(env.q(audit, 24) not in hits, "V7 duplicate audit hit")
        hits.add(env.q(audit, 24))
    events = [bytes.fromhex(v) for v in a["nativeEventHex"]]
    env.require(
        0 < len(events) <= n and (eof or len(events) == n), "V7 completed actual interval count"
    )
    env.require(not ack or len(events) == n, "V7 every ACKed interval must have a POST")
    for j, post in enumerate(events):
        _, t = env.qualify_post(post, pid, meta)
        gas, brake, steer = struct.unpack_from("<3f", post, 88)
        env.require(
            t == anchor + (j + 1) * 10 and struct.pack("<3f", steer, gas, brake) == afters[j],
            "V7 actual applied POST controls/grid",
        )
        env.require(env.q(post, 24) not in hits, "V7 duplicate POST hit")
        hits.add(env.q(post, 24))
    env.require(bytes.fromhex(packet["boundaryHex"]) == events[-1], "V7 latest actual endpoint")
    return events


class ImmediateSession(NativeSession):
    proc: Any
    connection: Any
    listener: Any
    stderr: Any
    capture: Any
    episode: Any
    commands: list[Any]
    messages: list[Any]
    epoch: int
    seq: int
    requests: list[Any]
    responses: list[bytes]
    boundary: bytes
    post_offset: int
    hits: set[int]
    audits: list[bytes]
    events: dict[int, bytes]
    result_evidence: dict[str, Any]

    def __init__(self, settings: NativeRuntimeConfig, verified: VerifiedAssets) -> None:
        self._quota_monitor: TraceQuotaMonitor | None = None
        self.proc = self.connection = self.listener = self.stderr = self.capture = None
        self.closed, self.episode = False, None
        self.captures: list[Capture] = []
        self.commands, self.messages, self.buffer = [], [], b""
        self.out, self.work = settings.trace_dir, settings.work_dir
        self.max_trace_bytes = settings.max_trace_bytes or 64 * 1024 * 1024
        self.paths, self.settings = verified.paths, settings
        self.pins = {role: env.sha(path.read_bytes()) for role, path in self.paths.items()}
        env.require(
            settings.protocol_profile == IMMEDIATE_PROTOCOL
            and verified.profile == IMMEDIATE_PROFILE,
            "explicit V7 profile required",
        )
        env.require(
            sys.platform == "linux"
            and settings.runtime_root.resolve(strict=True) == settings.runtime_root,
            "V7 Linux canonical runtime required",
        )
        env.require(
            env.sha(settings.asset_manifest.read_bytes()) == verified.manifest_sha256,
            "V7 verified manifest changed",
        )
        manifest = json.loads(settings.asset_manifest.read_bytes())
        env.require(
            all(
                self.pins[role] == manifest["assets"][manifest["roles"][role]]["sha256"]
                for role in self.paths
            ),
            "V7 verified asset changed",
        )
        env.require(
            not self.out.exists() and not self.work.exists(), "V7 fresh output/work required"
        )
        try:
            self.out.mkdir(parents=True)
            self.work.mkdir(parents=True)
            self.tmp = Path(tempfile.mkdtemp(prefix="tmre-immediate-v7-", dir="/tmp"))
            self.local_posts = self.tmp / "posts"
            self.local_posts.mkdir()
            self.address = self.tmp / "control.sock"
            self.listener = socket.socket(linux_socket.AF_UNIX, socket.SOCK_STREAM)
            self.listener.bind(str(self.address))
            self.listener.listen(1)
            self.deadline = time.monotonic() + settings.timeout_s
            process_env = {
                k: v
                for k, v in os.environ.items()
                if k != "LD_PRELOAD" and not k.startswith(("FK_", "FKSHIM_"))
            }
            process_env.update(
                TMRE_RUNTIME_ROOT=str(settings.runtime_root),
                TMRE_TREE_DIR=str(self.tmp),
                FKSHIM_POST_DIR=str(self.local_posts),
                FKSHIM_INNER_SOCKET=str(self.address),
                FKSHIM_INNER_START=str(BOOTSTRAP_NATIVE_T),
                FKSHIM_READER="1",
                FKSHIM_RESULT="1",
                FK_FRAME_TIMEOUT_MS="180000",
            )
            args = [
                str(self.paths["driver"]),
                str(self.work / "work"),
                str(self.out),
                str(self.paths["tape"]),
                str(self.paths["shim"]),
            ]
            self.provenance = {
                "profile": codec_v7.PROFILE_ID,
                "inputSchedule": SCHEDULE,
                "assetManifestSha256": verified.manifest_sha256,
                "deploymentManifestSha256": settings.deployment_manifest_sha256,
                "pins": self.pins,
                "literalResetNativeT": RESET_NATIVE_T,
                "literalBootstrapNativeT": BOOTSTRAP_NATIVE_T,
            }
            json_exclusive(
                self.out / "launch-v7.json",
                {
                    **self.provenance,
                    "args": args,
                    "environment": {
                        k: v for k, v in process_env.items() if k.startswith(("FK", "TMRE"))
                    },
                },
            )
            self.stderr = (self.out / "driver.stderr.bin").open("xb")
            self.check_budget(448)
            self.proc = spawn_supervised(
                args,
                cwd=self.work,
                env=process_env,
                stderr=self.stderr,
                evidence=self.out / "native-supervisor.json",
                timeout=settings.timeout_s,
            )
            self.ready = self.line()
            env.require(
                self.ready["kind"] == "ready"
                and self.ready["metadata"]["input_count"] == 4152
                and self.ready["metadata"]["start_offset_ms"] == -1510,
                "V7 original driver bootstrap required",
            )
        except BaseException:
            self.close(error=True)
            raise

    def _quota_usage(self, *, force: bool = False) -> TraceUsage:
        monitor = getattr(self, "_quota_monitor", None)
        if monitor is None:
            paths = getattr(self, "paths", {})
            private = getattr(self, "tmp", None)
            allowed = (
                {paths["server"], paths["map"], paths["server"].parent / "Packs"}
                if "server" in paths and "map" in paths
                else set()
            )
            monitor = TraceQuotaMonitor(
                (self.out, self.work, private),
                work_dir=self.work,
                allowed_work_targets=allowed,
                local_posts=getattr(self, "local_posts", None),
                control_socket=getattr(self, "address", None),
                branch_socket=None if private is None else private / "branch.sock",
            )
            self._quota_monitor = monitor
        return monitor.usage(force=force)

    def _reconcile_quota(self) -> None:
        monitor = getattr(self, "_quota_monitor", None)
        if monitor is not None:
            total = monitor.usage(force=True).used_bytes
            if total > self.max_trace_bytes:
                raise RxPersistenceError(
                    f"V7 trace budget exceeded: {total}+0>{self.max_trace_bytes}"
                )

    def quota_status(self) -> dict[str, Any]:
        """Fresh boundary status with conservative current-session closure headroom.

        The metadata allowance includes exact current command/request JSON plus
        64KiB for the pinned supervisor/close summary and 4KiB per unpublished RX
        receipt, in addition to its actual provenance/path lengths. Future
        episodes and external native append growth need their own reservation.
        """
        usage = self._quota_usage()
        unpublished = [
            capture for capture in getattr(self, "captures", ()) if capture.published is None
        ]
        rx_copies = sum(2 * capture.persisted for capture in unpublished)

        def json_bytes(value: Any) -> int:
            return len(json.dumps(value, indent=2, allow_nan=False).encode("utf-8")) + 1

        metadata = 64 * 1024 + json_bytes(
            {"commands": getattr(self, "commands", []), "messages": getattr(self, "messages", [])}
        )
        for capture in unpublished:
            metadata += (
                json_bytes(capture.provenance)
                + json_bytes(str(capture.local))
                + 2 * json_bytes(str(capture.ep))
                + 4096
            )
        copies = usage.post_copy_bytes + rx_copies
        if getattr(self, "episode", None) is not None:
            copies += sum(len(raw) for raw in self.responses)
            metadata += json_bytes(self.requests) + 4096
        return {
            "used_bytes": usage.used_bytes,
            "max_bytes": self.max_trace_bytes,
            "close_duplicate_reserve_bytes": copies,
            "close_metadata_reserve_bytes": metadata,
            "close_reserve_bytes": copies + metadata,
            "accounting_mode": self._quota_monitor.mode if self._quota_monitor else "full-scan",
            "accounting_fallback_reason": (
                self._quota_monitor.fallback_reason if self._quota_monitor else None
            ),
        }

    def check_budget(self, additional: int = 0) -> None:
        if type(additional) is not int or additional < 0:
            raise RxPersistenceError("V7 trace reservation must be a nonnegative integer")
        total = self._quota_usage().used_bytes
        if total + additional > self.max_trace_bytes:
            raise RxPersistenceError(
                f"V7 trace budget exceeded: {total}+{additional}>{self.max_trace_bytes}"
            )

    def exact(self, size: int) -> bytes:
        env.require(size == 448 and self.capture is not None, "V7 persisted response448 required")
        result = bytearray()
        self.capture.exact_calls += 1
        phase = "socket"
        try:
            while len(result) < size:
                self.check_budget(size - len(result))
                phase = "socket"
                self.connection.settimeout(min(90, self.remaining()))
                chunk = self.connection.recv(size - len(result))
                if not chunk:
                    raise EOFError("native V7 EOF")
                result.extend(chunk)
                phase = "persistence"
                self.check_budget(len(chunk))
                self.capture.append(chunk)
            return bytes(result)
        except RxPersistenceError:
            raise
        except (OSError, EOFError, TimeoutError) as error:
            evidence = {
                "schema": "tmrl.v7-incomplete-receive/v1",
                "reason": type(error).__name__ + ": " + str(error),
                "currentResponseBytes": len(result),
                "currentResponseHex": result.hex(),
                "terminal": False,
                "persistedBytes": self.capture.persisted,
            }
            json_exclusive(self.ep / "receive-error-v7.json", evidence)
            if phase == "persistence":
                raise RxPersistenceError("V7 persist-before-decode failed", evidence) from error
            raise IncompleteReceiveError(evidence) from error

    def request(self, spec: RequestSpec) -> None:
        self.seq += 1
        raw = codec_v7.encode_request(
            op=spec.op,
            sequence=self.seq,
            n=spec.n,
            epoch=self.epoch,
            p=spec.p,
            k=spec.k,
            befores=tuple(spec.befores),
            afters=tuple(spec.afters),
        )
        self.requests.append({"hex": raw.hex(), "host_ns": time.perf_counter_ns()})
        self.connection.settimeout(min(90, self.remaining()))
        self.connection.sendall(raw)

    def response(self, spec: ResponseSpec) -> tuple[bytes, bytes]:
        raw = self.exact(448)
        decoded = codec_v7.decode_v7_response(raw)
        env.require(
            (decoded["kind"], decoded["sequence"], decoded["iterations"], decoded["epoch"])
            == (spec.kind, spec.seq, spec.n, spec.epoch),
            "V7 exact response sequence/epoch",
        )
        env.require(decoded["event"]["pid"] == self.pid, "V7 own response required")
        self.responses.append(raw)
        self.epoch = spec.epoch
        return raw[32:288], raw[288:]

    def poll_posts(self) -> None:
        path = self.local_posts / f"post-{self.pid}.bin"
        with path.open("rb") as stream:
            stream.seek(self.post_offset)
            raw = stream.read()
        env.require(len(raw) % 256 == 0, "V7 partial POST record")
        self.post_offset += len(raw)
        for offset in range(0, len(raw), 256):
            record = raw[offset : offset + 256]
            env.require(
                (env.u(record, 8), env.u(record, 16), env.u(record, 20))
                == (256, self.pid, self.pid),
                "V7 own POST stream",
            )
            hit = env.q(record, 24)
            env.require(hit not in self.hits, "V7 duplicate stream hit")
            self.hits.add(hit)
            if record[:8] == b"TMAUD003":
                self.audits.append(record)
            elif record[:8] == b"TMPOST03" and env.u(record, 148) == 75:
                _, native_t = env.qualify_post(record, self.pid, self.meta)
                env.require(native_t not in self.events, "V7 duplicate native boundary")
                self.events[native_t] = record
                insort(self._event_times, native_t)
            else:
                env.require(
                    record[:8] in (b"TMPOST03", b"TMREAD01", b"TMPOST02"),
                    "V7 unexpected stream record",
                )

    def packet(self) -> dict[str, Any]:
        return {
            "profile": codec_v7.PROFILE_ID,
            "terminalContract": TERMINAL_CONTRACT,
            "pid": self.pid,
            "meta": self.meta,
            "epoch": self.epoch,
            "boundaryHex": self.boundary.hex(),
            "readerHex": self.reader.hex(),
            "runtimeProvenance": self.provenance,
        }

    def reset(self, episode: Any = None) -> dict[str, Any]:
        if self.episode is not None:
            self.end(truncated=True)
        number = len(self.captures)
        spawned = self.command("spawn", episode=number)
        env.require(spawned["kind"] == "spawned", "V7 owned branch spawn")
        self.pid, self.meta, self.ep = (
            spawned["pid"],
            spawned["metadata"],
            Path(spawned["directory"]),
        )
        self.episode = number
        self.capture = Capture(CaptureSpec(self.ep, self.pid, self.tmp, self.provenance))
        self.captures.append(self.capture)
        self.capture.location()
        self.listener.settimeout(min(90, self.remaining()))
        self.connection, _ = self.listener.accept()
        env.require(
            struct.unpack(
                "3i", self.connection.getsockopt(socket.SOL_SOCKET, linux_socket.SO_PEERCRED, 12)
            )
            == (self.pid, linux_os.getuid(), linux_os.getgid()),
            "V7 owned socket peer",
        )
        self.seq, self.epoch, self.requests, self.responses = 0, 1, [], []
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=0, n=0, epoch=1))
        env.require(
            env.u(self.boundary, 112) == BOOTSTRAP_NATIVE_T, "V7 literal2190 bootstrap required"
        )
        self.arm()
        self.post_offset, self.events, self.audits, self.hits = 0, {}, [], set()
        self._event_times: list[int] = []
        self.poll_posts()
        state = physical_state(self.packet())
        env.require(state["appliedSGB"] == [0.0, 0.0, 0.0], "V7 neutral bootstrap POST required")
        reset_packet = self.advance(36, 1)
        events = qualify_immediate(reset_packet, state, 36, 1)
        env.require(
            not reset_packet["eof"] and len(events) == 1,
            "V7 one administrative neutral interval required",
        )
        json_exclusive(
            self.ep / "uncredited-neutral-reset-v7.json",
            {
                "policyDecision": False,
                "replayCredit": False,
                "bootstrapNativeT": BOOTSTRAP_NATIVE_T,
                "resetNativeT": RESET_NATIVE_T,
                "authorization": reset_packet["authorization"],
                "responseFramesHex": reset_packet["responseFramesHex"],
            },
        )
        state = physical_state(reset_packet)
        env.require(state["nativeT"] == RESET_NATIVE_T, "V7 neutral seek must end at2200")
        env.require(
            state["appliedSGB"] == [0.0, 0.0, 0.0], "V7 actually neutral reset POST required"
        )
        return self.packet()

    def advance(self, action: int, n: int) -> dict[str, Any]:
        reader, anchor = self.reader, env.u(self.boundary, 112)
        env.qualify_reader(
            QualifyReaderSpec(raw=reader, pid=self.pid, meta=self.meta, anchor=anchor)
        )
        p = env.u(reader, 112)
        # V7 ARM requires a subsequent decoded-record reader. At tape exhaustion
        # the original engine bypasses it and reaches another POST; the shim then
        # rejects ARMED!=0 (337). Reject the entire request before op6 or reads.
        env.require(
            p + n < self.meta["input_count"],
            "V7 unsupported-final-reader-boundary: request must leave a next input reader",
        )
        original = self.read_mem(env.q(reader, 88) - 4 + p * 32, n * 32)
        originals = [original[j * 32 : (j + 1) * 32] for j in range(n)]
        befores = [v[4:16] for v in originals]
        afters = [struct.pack("<3f", *v) for v in features.action_sgb(action)[:n]]
        epoch, sequence, response_start = self.epoch, self.seq + 1, len(self.responses)
        self.poll_posts()
        audit_start = len(self.audits)
        self.request(RequestSpec(op=6, n=n, p=p, k=p, befores=befores, afters=afters))
        committed, receipt = self.response(ResponseSpec(kind=6, seq=sequence, n=n, epoch=epoch))
        env.require(committed == reader, "V7 exact pre-write reader receipt")
        eof, stage, error = False, "boundary", None
        try:
            self.boundary, _ = self.response(
                ResponseSpec(kind=1, seq=sequence, n=n, epoch=epoch + 1)
            )
            env.require(
                env.u(self.boundary, 112) == anchor + n * 10, "V7 exact requested ACK boundary"
            )
            stage = "reader"
            self.arm()
        except IncompleteReceiveError as receive:
            error = receive.evidence
            env.require(
                error["reason"].startswith("EOFError:")
                and error["currentResponseBytes"] == 0
                and error["currentResponseHex"] == "",
                "V7 partial/error receive is not a terminal",
            )
            eof = True
            self.end(truncated=False)
        self.poll_posts()
        events = self._decision_events(anchor, anchor + n * 10)
        if eof and events:
            self.boundary = events[-1]
        packet = self.packet()
        packet.update(
            eof=eof,
            eofStage=stage if eof else None,
            receiveError=error,
            responseFramesHex=[v.hex() for v in self.responses[response_start:]],
            authorization={
                "anchor": anchor,
                "ownP": p,
                "first": p,
                "n": n,
                "epoch": epoch,
                "sequence": sequence,
                "actionId": action,
                "originalRecordHex": [v.hex() for v in originals],
                "receiptHex": receipt.hex(),
                "nativeEventHex": [v.hex() for v in events],
                "auditEventHex": [v.hex() for v in self.audits[audit_start:]],
                "inputSchedule": SCHEDULE,
            },
        )
        if eof:
            packet["resultEvidence"] = self.result_evidence
        return packet

    def _decision_events(self, anchor: int, end: int) -> list[bytes]:
        """Index retained evidence without sorting/scanning the full episode per tick."""
        clocks = self._event_times
        env.require(not clocks or clocks[-1] <= end, "V7 advanced beyond request")
        first, last = bisect_right(clocks, anchor), bisect_right(clocks, end)
        return [self.events[clock] for clock in clocks[first:last]]

    def _publish_bytes(self, path: Path, raw: bytes) -> None:
        self.check_budget(len(raw))
        with path.open("xb") as stream:
            if stream.write(raw) != len(raw):
                raise RxPersistenceError("V7 short end evidence write")

    def _publish_json(self, path: Path, value: Any) -> None:
        raw = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        self._publish_bytes(path, raw)

    def end(self, *, truncated: bool) -> None:
        if self.episode is None:
            return
        self._reconcile_quota()
        # Publication retains both its stage and its hard-linked final name;
        # logical-byte accounting charges both, independently of disk extents.
        self.check_budget(2 * self.capture.persisted)
        self.capture.finish()
        self.check_budget()
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        reaped = self.command("truncate" if truncated else "reap")
        env.require(
            reaped["kind"] == "reaped"
            and reaped["pid"] == self.pid
            and reaped["process_gone"] is True
            and reaped["parent_unchanged"] is True
            and reaped["exit_status"] is None
            and reaped["truncated_by_driver"] == truncated,
            "V7 original driver reap proof",
        )
        self.poll_posts()
        self._publish_bytes(self.ep / "responses-v7.bin", b"".join(self.responses))
        self._publish_json(self.ep / "requests-v7.json", self.requests)
        self._publish_json(
            self.ep / "host-summary-v7.json",
            {"pid": self.pid, "truncated": truncated, "reaped": reaped},
        )
        raw = (self.ep / "stdout-slice.bin").read_bytes()
        if raw.lstrip().startswith(b"["):
            self.opening = raw[: raw.index(b"[") + 1]
        if not truncated:
            collector = self.local_posts / f"result-{self.pid}.bin"
            try:
                collector_hex = collector.read_bytes().hex()
            except FileNotFoundError:
                # A crashed child can close the socket without ever publishing
                # a result. Preserve that absence for terminal classification.
                collector_hex = None
            self.result_evidence = {
                "profile": codec_v7.PROFILE_ID,
                "pins": self.pins,
                "reaped": reaped,
                "validatorArrayHex": (
                    raw if raw.lstrip().startswith(b"[") else getattr(self, "opening", b"") + raw
                ).hex(),
                "collectorHex": collector_hex,
            }
        self.episode = None

    def classify_terminal(self, packet: dict[str, Any]) -> dict[str, Any]:
        checks = SimpleNamespace(
            RUNTIME_PROFILE=codec_v7.PROFILE_ID,
            PINS=self.pins,
            require=env.require,
            qualify_post=env.qualify_post,
        )
        result: dict[str, Any] = outcome.classify(packet, checks, self.settings.runtime_root)
        return result

    def close(self, *, error: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._close_owned(error=error)
        finally:
            monitor = getattr(self, "_quota_monitor", None)
            if monitor is not None:
                monitor.close()

    def _close_owned(self, *, error: bool) -> None:
        failures = []
        try:
            if self.proc is not None and not error and self.proc.poll() is None:
                self.end(truncated=True)
                env.require(self.command("close")["kind"] == "closed", "V7 driver close ACK")
                self.proc.stdin.close()
                env.require(self.proc.wait(timeout=10) == 0, "V7 clean driver exit")
        except BaseException as exc:
            failures.append(repr(exc))
        finally:
            report = None
            try:
                if self.proc is not None:
                    report = self.proc.stop(timeout=5)
            except BaseException as exc:
                failures.append(repr(exc))
            finally:
                if self.proc is not None:
                    self.proc.close_control()
            try:
                self._reconcile_quota()
            except BaseException as exc:
                failures.append(repr(exc))
            for capture in self.captures:
                try:
                    self.check_budget(0 if capture.published else 2 * capture.persisted)
                    capture.finish()
                    self.check_budget()
                except BaseException as exc:
                    failures.append(repr(exc))
                    try:
                        capture.close_fd()
                    except OSError as close_error:
                        failures.append(repr(close_error))
            for obj in (self.connection, self.listener, self.stderr):
                if obj is not None:
                    try:
                        obj.close()
                    except OSError as close_error:
                        failures.append(repr(close_error))
            if self.out.is_dir():
                destination = self.out / "posts"
                destination.mkdir(exist_ok=True)
                for path in getattr(self, "local_posts", Path("/nonexistent-v7")).glob("*.bin"):
                    try:
                        self.check_budget(path.stat().st_size)
                    except RxPersistenceError as exc:
                        failures.append("retained local evidence: " + str(exc))
                        continue
                    shutil.copyfile(path, destination / path.name)
                    env.require(
                        env.sha((destination / path.name).read_bytes())
                        == env.sha(path.read_bytes()),
                        "V7 trace copy digest",
                    )
                accounting_monitor = getattr(self, "_quota_monitor", None)
                json_exclusive(
                    self.out / "close-v7.json",
                    {
                        "error": error,
                        "failures": failures,
                        "supervisor": report,
                        "commands": self.commands,
                        "messages": self.messages,
                        "retainedLocalDirectory": str(getattr(self, "tmp", "unallocated")),
                        "traceAccounting": (
                            {
                                "mode": accounting_monitor.mode,
                                "fallback_reason": accounting_monitor.fallback_reason,
                                "full_scans": accounting_monitor.full_scans,
                            }
                            if accounting_monitor is not None
                            else None
                        ),
                    },
                )
        if failures and not error:
            raise RuntimeError("; ".join(failures))
