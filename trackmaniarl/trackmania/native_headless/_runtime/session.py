from __future__ import annotations

import contextlib
import json
import shutil
import struct
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trackmaniarl.trackmania.native_headless._runtime import env as E
from trackmaniarl.trackmania.native_headless._runtime import features as core
from trackmaniarl.trackmania.native_headless._runtime import host as base
from trackmaniarl.trackmania.native_headless._runtime.artifacts import ROLE_PATHS, dependencies
from trackmaniarl.trackmania.native_headless._runtime.capture import (
    Capture,
    RxPersistenceError,
    json_exclusive,
)
from trackmaniarl.trackmania.native_headless._runtime.ownership import (
    owned_session_members,
    process_identity,
    signal_identity,
)
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os, linux_signal
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    CaptureSpec,
    QualifyReaderSpec,
    RequestSpec,
    ResponseSpec,
)

if TYPE_CHECKING:
    from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
    from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


def make_native_env(settings: NativeRuntimeConfig, verified: VerifiedAssets) -> E.Native16Env:
    root = Path(settings.runtime_root)
    output, workspace = (Path(settings.trace_dir), Path(settings.work_dir))
    timeout = settings.timeout_s

    def factory() -> Any:
        E.require(settings.allow_native is True, "explicit native execution required")
        E.require(
            sys.platform == "linux" and root.is_absolute() and root.resolve(strict=True) == root,
            "experimental fixed-layout Linux/WSL runtime required",
        )
        E.require(type(timeout) in (int, float) and 0 < timeout <= 240, "session deadline1..240s")
        manifest_bytes = settings.asset_manifest.read_bytes()
        E.require(
            E.sha(manifest_bytes) == verified.manifest_sha256,
            "asset manifest changed after verification",
        )
        declared = json.loads(manifest_bytes)["assets"]
        role_hashes = {role: declared[path]["sha256"] for role, path in ROLE_PATHS.items()}
        active_pins = {ROLE_PATHS[role]: value for role, value in role_hashes.items()}
        for name, expected in active_pins.items():
            E.require(
                E.sha((root / name).read_bytes()) == expected,
                "external runtime asset hash: " + name,
            )
        rx_active = {
            "schema": "tmrl.native-headless.runtime/v2",
            "terminalContract": E.TERMINAL_CONTRACT,
            "assetManifestSha256": verified.manifest_sha256,
            "deploymentManifestSha256": settings.deployment_manifest_sha256,
            "portableRuntimeRoot": str(root),
            "portablePreviewProfile": "tmre.preview200/root-configured/v1",
            "originalRXSourceManifestSha256": (
                "5dd254403171e82a48f971033b3ae763add75740937a289263e1b12afd384bae"
            ),
            "wireABI": "unchanged V6 request192/response448",
            "knownFactoryPopenSIGINTGap": False,
            "processLifecycle": "private-lease-subreaper-pidfd/v1",
        }
        E.require(not output.exists() and (not workspace.exists()), "fresh evidence/work required")
        receipt = output.with_name(output.name + ".native-headless-prelaunch.json")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        json_exclusive(
            receipt, {"assetPins": active_pins, "activeRuntime": rx_active, "seedIgnored": True}
        )
        _, _, reference_data = dependencies(root)
        cold_raw = (root / reference_data["source"]["file"]).read_bytes()
        E.require(E.sha(cold_raw) == reference_data["source"]["sha256"], "cold reference source")
        cold = {
            E.u(e, 112): base.state76(e)
            for i in range(0, len(cold_raw), 256)
            for e in [cold_raw[i : i + 256]]
            if e[:8] == b"TMPOST03" and E.u(e, 148) == 75
        }

        class Session(base.NativeSessionV6):
            _env002_closed: Any
            _rx009_records: Any
            audits: Any
            boundary: Any
            connection: Any
            driver_identity: Any
            episode: Any
            events: Any
            failed: Any
            hits: Any
            known_owned: Any
            number: Any
            opening: Any
            pending_provenance: Any
            post_offset: Any
            proc: Any
            result_evidence: Any

            def __init__(self) -> None:
                self.proc = None
                self.driver_identity = None
                self.known_owned = {}
                self.opening = None
                self.number = -1
                self.failed = False
                try:
                    super().__init__(
                        output,
                        workspace,
                        timeout,
                        runtime=base.RuntimeAssets(root, role_hashes["driver"]),
                    )
                except BaseException:
                    self.failed = True
                    self.close(error=True)
                    raise

            def line(self) -> Any:
                if self.driver_identity is None:
                    owner = process_identity(self.proc.pid)
                    E.require(
                        owner is not None
                        and owner["state"] not in ("Z", "X")
                        and (owner["uid"] == linux_os.getuid())
                        and (owner["sid"] == owner["pgid"] == self.proc.pid),
                        "live owned isolated driver identity required",
                    )
                    self.driver_identity = owner
                self.capture_owned()
                return super().line()

            def capture_owned(self) -> Any:
                for member in owned_session_members(self.driver_identity):
                    self.known_owned[member["pid"], member["birth"], member["uid"]] = member

            def poll_posts(self) -> Any:
                self.capture_owned()
                path = self.local_posts / f"post-{self.pid}.bin"
                with path.open("rb") as stream:
                    stream.seek(self.post_offset)
                    raw = stream.read()
                E.require(len(raw) % 256 == 0, "partial own post record; preserved, unqualified")
                self.post_offset += len(raw)
                for i in range(0, len(raw), 256):
                    b = raw[i : i + 256]
                    E.require(
                        (E.u(b, 8), E.u(b, 16), E.u(b, 20)) == (256, self.pid, self.pid),
                        "own post stream",
                    )
                    E.require(E.q(b, 24) not in self.hits, "duplicate own stream hit")
                    self.hits.add(E.q(b, 24))
                    if b[:8] == b"TMPOST03" and E.u(b, 148) == 75:
                        _, t = E.qualify_post(b, self.pid, self.meta)
                        E.require(t not in self.events, "duplicate native boundary")
                        self.events[t] = b
                    elif b[:8] == b"TMAUD002":
                        self.audits.append(b)
                    elif b[:8] == b"TMPOST03":
                        E.require(E.u(b, 148) == 8, "unknown POST mask")
                    else:
                        E.require(b[:8] in (b"TMREAD01", b"TMPOST02"), "unknown stream record")

            def _base_packet(self) -> Any:
                return {
                    "profile": E.RUNTIME_PROFILE,
                    "terminalContract": E.TERMINAL_CONTRACT,
                    "pid": self.pid,
                    "meta": self.meta,
                    "boundaryHex": self.boundary.hex(),
                    "readerHex": self.reader.hex(),
                    "epoch": self.epoch,
                    "pendingProvenance": self.pending_provenance,
                }

            def reset(self, episode: Any = None) -> Any:
                if self.episode is not None:
                    self.end(truncated=True)
                self.number += 1
                super().reset(self.number)
                self.post_offset = 0
                self.events, self.audits, self.hits = ({}, [], set())
                self.poll_posts()
                self.pending_provenance = {
                    "source": "administrative prelude in progress",
                    "actionId": None,
                }
                prelude_action = 39 if settings.reset_profile == "legacy-gas-prelude/v1" else 36
                packet = self._advance(prelude_action, 5, warmup=True)
                E.require(not packet["eof"], "administrative warmup EOF")
                self.pending_provenance = {
                    "source": settings.reset_profile + ": receipt then literal reader155",
                    "actionId": None,
                }
                return self.packet()

            def _advance(self, action: Any, n: Any, *, warmup: bool = False) -> Any:
                reader = self.reader
                anchor, p = (E.u(self.boundary, 112), E.u(reader, 112))
                E.qualify_reader(
                    QualifyReaderSpec(raw=reader, pid=self.pid, meta=self.meta, anchor=anchor)
                )
                k = p + 1
                E.require(
                    k + 5 <= 4152, "five future records unavailable; close/reset, no array overflow"
                )
                raw = self.read_mem(E.q(reader, 88) - 4 + k * 32, 160)
                befores = [raw[j * 32 + 4 : j * 32 + 16] for j in range(5)]
                planned = core.action_sgb(action)
                afters = [struct.pack("<3f", *v) for v in planned]
                epoch, sequence = (self.epoch, self.seq + 1)
                self.poll_posts()
                audit_start = len(self.audits)
                response_start = len(self.responses)
                self.request(RequestSpec(op=4, n=n, p=p, k=k, befores=befores, afters=afters))
                committed, receipt = self.response(
                    ResponseSpec(kind=4, seq=sequence, n=n, epoch=epoch)
                )
                E.require(
                    committed == reader
                    and receipt
                    == struct.pack("<IIII", p, k, 5, 0)
                    + b"".join((a + b for a, b in zip(befores, afters, strict=False)))
                    + bytes(24),
                    "exact accepted write receipt",
                )
                eof = False
                receive_error = None
                eof_stage = "boundary"
                try:
                    self.boundary, _ = self.response(
                        ResponseSpec(kind=1, seq=sequence, n=n, epoch=epoch + 1)
                    )
                    E.require(E.u(self.boundary, 112) == anchor + n * 10, "requested boundary ACK")
                    eof_stage = "reader"
                    self.arm()
                    self.pending_provenance = {
                        "source": "authorized future record then literal reader",
                        "actionId": action,
                        "receiptSequence": sequence,
                        "futureOrdinal": n - 1,
                    }
                except base.IncompleteReceiveError as error:
                    receive_error = error.evidence
                    eof = (
                        type(receive_error.get("reason")) is str
                        and receive_error["reason"].startswith("EOFError:")
                        and type(receive_error.get("currentResponseBytes")) is int
                        and receive_error["currentResponseBytes"] == 0
                        and receive_error["currentResponseHex"] == ""
                    )
                    if not eof:
                        self.failed = True
                        raise
                    self.end(truncated=False)
                self.poll_posts()
                events = [
                    self.events[t] for t in sorted(self.events) if anchor < t <= anchor + n * 10
                ]
                E.require(
                    not any(t > anchor + n * 10 for t in self.events),
                    "post advanced beyond request",
                )
                if eof and events:
                    self.boundary = events[-1]
                a = {
                    "anchor": anchor,
                    "end": anchor + len(events) * 10,
                    "ownP": p,
                    "first": k,
                    "epoch": epoch,
                    "sequence": sequence,
                    "pendingSGB": list(struct.unpack_from("<3f", reader, 156)),
                    "plannedSGB": planned,
                    "receiptHex": receipt.hex(),
                    "nativeEventHex": [e.hex() for e in events],
                    "auditEventHex": [e.hex() for e in self.audits[audit_start:]],
                    "policyDecision": not warmup,
                    "actionId": None if warmup else action,
                }
                if warmup:
                    E.require(
                        anchor == 2180
                        and p == 150
                        and (
                            action
                            == (39 if settings.reset_profile == "legacy-gas-prelude/v1" else 36)
                        )
                        and (n == 5)
                        and (len(events) == 5),
                        "exact administrative prelude domain",
                    )
                    if settings.reset_profile == "legacy-gas-prelude/v1":
                        E.require(
                            all(base.state76(e) == cold[E.u(e, 112)] for e in events),
                            "warmup matches cold original trajectory",
                        )
                    else:
                        # Check applied POST channels, including the inherited pending
                        # interval. No write to the already armed current record.
                        E.require(
                            all(
                                core.native16(base.state76(e))["gas"] == 0.0
                                and core.native16(base.state76(e))["brake"] == 0.0
                                and core.native16(base.state76(e))["steer"] == 0.0
                                for e in events
                            ),
                            "neutral reset requires five actually neutral POST intervals",
                        )
                    base.dump(self.ep / "uncredited-warmup.env002.json", a)
                packet = self.packet()
                packet.update(
                    authorization=a,
                    eof=eof,
                    eofStage=eof_stage if eof else None,
                    receiveError=receive_error,
                )
                packet["responseFramesHex"] = [v.hex() for v in self.responses[response_start:]]
                if eof:
                    packet["resultEvidence"] = self.result_evidence
                return packet

            def advance(self, action: Any, n: Any) -> Any:
                return self._advance(action, n)

            def _base_end(self, truncated: Any) -> Any:
                E.require(type(truncated) is bool, "explicit end flag")
                if self.episode is None:
                    return
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                v = self.command("truncate" if truncated else "reap")
                E.require(
                    v["kind"] == "reaped"
                    and v["pid"] == self.pid
                    and v["process_gone"]
                    and v["parent_unchanged"]
                    and (v["exit_status"] is None)
                    and (v["truncated_by_driver"] == truncated),
                    "driver reap proof",
                )
                self.poll_posts()
                (self.ep / "responses.bin").write_bytes(b"".join(self.responses))
                base.dump(self.ep / "requests.json", self.requests)
                base.dump(self.ep / "response-times.json", self.response_times)
                base.dump(
                    self.ep / "host-summary.json",
                    {"pid": self.pid, "truncated": truncated, "reaped": v},
                )
                raw_json = (self.ep / "stdout-slice.bin").read_bytes()
                if raw_json.lstrip().startswith(b"["):
                    self.opening = raw_json[: raw_json.index(b"[") + 1]
                if not truncated:
                    envelope = (
                        raw_json
                        if raw_json.lstrip().startswith(b"[")
                        else (self.opening or b"") + raw_json
                    )
                    self.result_evidence = {
                        "profile": E.RUNTIME_PROFILE,
                        "pins": E.PINS,
                        "reaped": v,
                        "validatorArrayHex": envelope.hex(),
                        "collectorHex": (self.local_posts / f"result-{self.pid}.bin")
                        .read_bytes()
                        .hex(),
                    }
                self.episode = None

            def _base_close(self, *, error: bool = False) -> Any:
                if getattr(self, "_env002_closed", False):
                    return
                self._env002_closed = True
                cleanup_error = None
                normal_closed = False
                try:
                    if (
                        self.proc is not None
                        and self.proc.poll() is None
                        and (not error)
                        and (not self.failed)
                    ):
                        if self.episode is not None:
                            self.end(truncated=True)
                        E.require(self.command("close")["kind"] == "closed", "driver close")
                        self.proc.stdin.close()
                        self.proc.wait(timeout=10)
                        E.require(self.proc.returncode == 0, "driver exit")
                        normal_closed = True
                except BaseException as exc:
                    cleanup_error = repr(exc)
                finally:
                    signals = []
                    supervisor_report = None
                    if self.proc is not None and (not normal_closed):
                        try:
                            supervisor_report = self.proc.stop(timeout=5)
                            signals = supervisor_report["cleanupPidfdSignals"]
                        except BaseException as exc:
                            cleanup_error = (cleanup_error or "") + repr(exc)
                            self.capture_owned()
                            rows = sorted(
                                self.known_owned.values(), key=lambda r: r["pid"] == self.proc.pid
                            )
                            for member in rows:
                                if signal_identity(member, linux_signal.SIGKILL):
                                    signals.append(member)
                    elif self.proc is not None:
                        supervisor_report = self.proc.report
                    if self.proc is not None:
                        self.proc.close_control()
                    for name in ("connection", "listener", "stderr"):
                        obj = getattr(self, name, None)
                        if obj is not None:
                            obj.close()
                    if output.exists():
                        for p in getattr(self, "local_posts", Path("/nonexistent-env002")).glob(
                            "*.bin"
                        ):
                            destination = (
                                output
                                / ("results" if p.name.startswith("result-") else "posts")
                                / p.name
                            )
                            destination.parent.mkdir(exist_ok=True)
                            E.require(not destination.exists(), "no evidence overwrite")
                            shutil.copyfile(p, destination)
                            E.require(
                                E.sha(destination.read_bytes()) == E.sha(p.read_bytes()),
                                "trace copy",
                            )
                        base.dump(
                            output / "env002-close.json",
                            {
                                "error": error or self.failed,
                                "cleanupError": cleanup_error,
                                "returncode": None if self.proc is None else self.proc.returncode,
                                "driverIdentity": self.driver_identity,
                                "cleanupPidfdSignals": signals,
                                "supervisor": supervisor_report,
                                "retainedLocalDirectory": str(getattr(self, "tmp", "unallocated")),
                                "commands": getattr(self, "commands", []),
                                "messages": getattr(self, "messages", []),
                            },
                        )
                if cleanup_error is not None and (not error):
                    raise RuntimeError(cleanup_error)

            def _base_response(self, *args: Any, **kwargs: Any) -> Any:
                return super().response(*args, **kwargs)

            def _rx009_capture(self) -> Any:
                if not hasattr(self, "_rx009_records"):
                    self._rx009_records = {}
                key = str(Path(self.ep).resolve())
                if key not in self._rx009_records:
                    cap = Capture(CaptureSpec(self.ep, self.pid, self.tmp, rx_active))
                    self._rx009_records[key] = cap
                    cap.location()
                cap = self._rx009_records[key]
                if cap.pid != self.pid or cap.closed:
                    raise RxPersistenceError("RX after finalization or child/episode reuse")
                return cap

            def _rx009_finish_all(self) -> Any:
                errors = []
                receipts = []
                for cap in getattr(self, "_rx009_records", {}).values():
                    try:
                        receipts.append(cap.finish())
                    except Exception as error:
                        errors.append(error)
                if errors:
                    combined_error = RxPersistenceError(
                        "RX finalization failed: " + "; ".join(repr(e) for e in errors)
                    )
                    combined_error.secondary_errors = errors
                    raise combined_error from errors[0]
                return receipts

            def _rx009_failure_close(self, evidence: Any) -> Any:
                try:
                    self.close(error=True)
                except BaseException as error:
                    evidence.setdefault("cleanupErrors", []).append(repr(error))

            def exact(self, n: Any) -> Any:
                if n != 448:
                    raise ValueError("V6 fixed response448 only")
                result = bytearray()
                cap = None
                phase = "persistence"
                try:
                    cap = self._rx009_capture()
                    cap.exact_calls += 1
                    while len(result) < n:
                        phase = "socket"
                        self.connection.settimeout(min(90, self.remaining()))
                        chunk = self.connection.recv(n - len(result))
                        if not chunk:
                            raise EOFError("native EOF")
                        result.extend(chunk)
                        phase = "persistence"
                        cap.append(chunk)
                    return bytes(result)
                except (OSError, EOFError, TimeoutError) as error:
                    evidence = {
                        "schema": "tmre.v6-incomplete-receive/v1",
                        "reason": type(error).__name__ + ": " + str(error),
                        "currentResponseBytes": len(result),
                        "currentResponseHex": bytes(result).hex(),
                        "rxFile": "rx.bin",
                        "lastConfirmedBoundaryNativeT": struct.unpack_from(
                            "<I", self.boundary, 112
                        )[0]
                        if hasattr(self, "boundary")
                        else None,
                        "terminal": False,
                        "finishClaim": False,
                        "localRxFile": None if cap is None else str(cap.local),
                        "persistedBytes": 0 if cap is None else cap.persisted,
                        "totalReceivedBytes": 0 if cap is None else cap.received,
                        "activeCaller009": deepcopy(rx_active),
                    }
                    storage_errors = []
                    try:
                        self._rx009_finish_all()
                    except Exception as storage:
                        storage_errors.append(repr(storage))
                    try:
                        json_exclusive(Path(self.ep) / "receive-error.json", evidence)
                    except Exception as storage:
                        storage_errors.append(repr(storage))
                    storage_failure = phase == "persistence"
                    if storage_errors or storage_failure:
                        failure = {
                            "schema": "tmre.v6-incomplete-receive/v1",
                            "reason": "RxPersistenceError: capture/publication failed",
                            "originalReceiveEvidence": evidence,
                            "storageErrors": storage_errors,
                            "currentResponseBytes": len(result),
                            "currentResponseHex": bytes(result).hex(),
                            "terminal": False,
                            "finishClaim": False,
                            "activeCaller009": deepcopy(rx_active),
                        }
                        self._rx009_failure_close(failure)
                        raise RxPersistenceError(failure["reason"], failure) from error
                    raise base.IncompleteReceiveError(evidence) from error

            def response(self, spec: ResponseSpec) -> Any:
                try:
                    return self._base_response(spec)
                except (base.IncompleteReceiveError, RxPersistenceError):
                    raise
                except Exception as error:
                    evidence = {
                        "reason": repr(error),
                        "terminal": False,
                        "finishClaim": False,
                        "decoderFailure": True,
                    }
                    try:
                        self._rx009_finish_all()
                    except Exception as storage:
                        evidence["publicationError"] = repr(storage)
                    self._rx009_failure_close(evidence)
                    with contextlib.suppress(Exception):
                        error.__dict__["rx009_evidence"] = evidence
                    raise

            def packet(self) -> Any:
                value = self._base_packet()
                value["activeCaller009"] = deepcopy(rx_active)
                return value

            def end(self, truncated: Any) -> Any:
                try:
                    self._rx009_finish_all()
                finally:
                    self._base_end(truncated)
                    if hasattr(self, "result_evidence"):
                        self.result_evidence["activeCaller009"] = deepcopy(rx_active)

            def close(self, *, error: bool = False) -> Any:
                try:
                    self._rx009_finish_all()
                finally:
                    try:
                        self._base_close(error=error)
                    finally:
                        self._rx009_finish_all()

        return Session()

    return E.Native16Env(
        factory,
        duration_limit=settings.duration_limit,
        root=root,
        reset_profile=settings.reset_profile,
    )
