"""Finite V7 source experiment: prove current-record actions from actual POSTs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import time
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime import artifacts, codec_v7, env, features
from trackmaniarl.trackmania.native_headless._runtime.host import NativeSession, u32, u64
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    QualifyReaderSpec,
    RequestSpec,
    ResponseSpec,
)
from trackmaniarl.trackmania.native_headless._runtime.supervisor import spawn_supervised


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class ImmediateSession(NativeSession):
    """Separate experiment launcher; no V6 hashes or runtime state are replaced."""

    boundary: bytes
    reader: bytes

    def __init__(  # noqa: PLR0913 -- explicit variant build/asset identity and clock anchor
        self, runtime: Path, receipt: dict, output: Path, start: int, driver_sha: str
    ) -> None:
        if not 2000 <= start <= 18000 or start % 10:
            raise ValueError("literal native start must match the shim's allowed domain")
        self.out, self.work = output, runtime / "build/server-probe" / output.name
        if self.out.exists() or self.work.exists():
            raise FileExistsError("fresh V7 output and driver work required")
        self.out.mkdir(parents=True)
        self.work.mkdir(parents=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="tmre-v7-", dir="/tmp"))
        self.address = self.tmp / "control.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.address))
        self.listener.listen(1)
        self.deadline = time.monotonic() + 120
        self.connection, self.episode, self.proc = None, None, None
        self.buffer, self.commands, self.messages = b"", [], []
        self.closed, self.start = False, start
        self.local_posts = self.tmp / "posts"
        self.local_posts.mkdir()
        self.pins = {}
        for role in ("server", "driver", "map", "tape"):
            relative = artifacts.ROLE_PATHS[role]
            path = runtime / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = driver_sha if role == "driver" else artifacts.PINS[relative]
            if digest != expected:
                raise ValueError("original asset identity differs: " + role)
            self.pins[str(path)] = digest
        shim = Path(receipt["artifact"])
        if not shim.is_relative_to(runtime / "build/forkoracle"):
            raise ValueError("new shim must remain in the driver's permitted tree")
        digest = hashlib.sha256(shim.read_bytes()).hexdigest()
        if digest != receipt["sha256"]:
            raise ValueError("new V7 build receipt mismatch")
        self.pins[str(shim)] = digest
        process_env = {
            k: v
            for k, v in os.environ.items()
            if k != "LD_PRELOAD" and not k.startswith(("FK_", "FKSHIM_"))
        }
        process_env.update(
            TMRE_RUNTIME_ROOT=str(runtime),
            TMRE_TREE_DIR=str(self.tmp),
            FKSHIM_POST_DIR=str(self.local_posts),
            FKSHIM_INNER_SOCKET=str(self.address),
            FKSHIM_INNER_START=str(start),
            FKSHIM_READER="1",
            FKSHIM_RESULT="1",
            FK_FRAME_TIMEOUT_MS="120000",
        )
        args = [
            str(runtime / artifacts.ROLE_PATHS["driver"]),
            str(self.work / "work"),
            str(self.out),
            str(runtime / artifacts.ROLE_PATHS["tape"]),
            str(shim),
        ]
        save(
            self.out / "launch-v7.json",
            {
                "schema": "tmrl.immediate-launch/v1",
                "args": args,
                "pins": self.pins,
                "literalStart": start,
                "inputSchedule": "current-record-before-load/v1",
                "buildReceipt": receipt,
            },
        )
        self.stderr = (self.out / "driver.stderr.bin").open("xb")
        try:
            self.proc = spawn_supervised(
                args,
                cwd=self.work,
                env=process_env,
                stderr=self.stderr,
                evidence=self.out / "supervisor.json",
                timeout=120,
            )
            self.ready = self.line()
            if (
                self.ready["kind"] != "ready"
                or self.ready["metadata"]["input_count"] != 4152
                or self.ready["metadata"]["start_offset_ms"] != -1510
            ):
                raise ValueError("original driver bootstrap identity required")
        except BaseException:
            self.close(error=True)
            raise

    def exact(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            self.connection.settimeout(min(30, self.remaining()))
            chunk = self.connection.recv(size - len(result))
            if not chunk:
                raise EOFError("literal V7 socket EOF")
            self.rx.write(chunk)
            self.rx.flush()
            os.fsync(self.rx.fileno())
            result.extend(chunk)
        return bytes(result)

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
        self.connection.sendall(raw)

    def response(self, spec: ResponseSpec) -> tuple[bytes, bytes]:
        raw = self.exact(448)
        decoded = codec_v7.decode_v7_response(raw)
        if (decoded["kind"], decoded["sequence"], decoded["iterations"], decoded["epoch"]) != (
            spec.kind,
            spec.seq,
            spec.n,
            spec.epoch,
        ):
            raise ValueError("V7 expected kind/sequence/intervals/epoch mismatch")
        if decoded["event"]["pid"] != self.pid:
            raise ValueError("V7 response from a foreign process")
        self.responses.append(raw)
        self.epoch = spec.epoch
        return raw[32:288], raw[288:]

    def reset(self) -> bytes:
        spawned = self.command("spawn", episode=1)
        if spawned["kind"] != "spawned":
            raise ValueError("driver failed to spawn the owned branch")
        self.pid, self.meta = spawned["pid"], spawned["metadata"]
        self.ep = Path(spawned["directory"])
        self.episode = 1
        self.rx = (self.ep / "rx-v7.bin").open("xb")
        self.listener.settimeout(min(30, self.remaining()))
        self.connection, _ = self.listener.accept()
        if struct.unpack(
            "3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        ) != (self.pid, os.getuid(), os.getgid()):
            raise ValueError("V7 owned socket peer required")
        self.seq, self.epoch, self.requests, self.responses = 0, 1, [], []
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=0, n=0, epoch=1))
        if u32(self.boundary, 112) != self.start:
            raise ValueError("V7 literal initial boundary differs")
        self.arm()
        return self.boundary

    def step(self, action: int, n: int) -> dict:
        reader, anchor = self.reader, u32(self.boundary, 112)
        env.qualify_reader(
            QualifyReaderSpec(raw=reader, pid=self.pid, meta=self.meta, anchor=anchor)
        )
        p = u32(reader, 112)
        original = self.read_mem(u64(reader, 88) - 4 + p * 32, n * 32)
        befores = [original[j * 32 + 4 : j * 32 + 16] for j in range(n)]
        desired = features.action_sgb(action)[:n]
        afters = [struct.pack("<3f", *v) for v in desired]
        epoch, sequence = self.epoch, self.seq + 1
        self.request(RequestSpec(op=6, n=n, p=p, k=p, befores=befores, afters=afters))
        committed, receipt = self.response(ResponseSpec(kind=6, seq=sequence, n=n, epoch=epoch))
        expected_receipt = (
            struct.pack("<4I", p, p, n, 2)
            + b"".join(a + b for a, b in zip(befores, afters, strict=True))
            + bytes(144 - n * 24)
        )
        if committed != reader or receipt != expected_receipt:
            raise ValueError("V7 exact current-record receipt mismatch")
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=sequence, n=n, epoch=epoch + 1))
        if u32(self.boundary, 112) != anchor + n * 10:
            raise ValueError("V7 exact physical endpoint required")
        self.arm()
        raw_posts = (self.local_posts / f"post-{self.pid}.bin").read_bytes()
        if len(raw_posts) % 256:
            raise ValueError("V7 partial POST log")
        records = [raw_posts[i : i + 256] for i in range(0, len(raw_posts), 256)]
        posts = [
            b
            for b in records
            if b[:8] == b"TMPOST03"
            and u32(b, 148) == 75
            and anchor < u32(b, 112) <= anchor + n * 10
        ]
        if [u32(b, 112) for b in posts] != [anchor + (j + 1) * 10 for j in range(n)]:
            raise ValueError("V7 complete consecutive POST evidence required")
        for actual, plan in zip(posts, desired, strict=True):
            env.qualify_post(actual, self.pid, self.meta)
            gas, brake, steer = struct.unpack_from("<3f", actual, 88)
            if struct.pack("<3f", steer, gas, brake) != struct.pack("<3f", *plan):
                raise ValueError("V7 actual applied input differs from current decision")
        audits = [b for b in records if b[:8] == b"TMAUD003" and u32(b, 200) == sequence]
        if len(audits) != n:
            raise ValueError("V7 exact audit count required")
        for j, audit in enumerate(audits):
            if (
                u32(audit, 176),
                u32(audit, 180),
                u32(audit, 184),
                u32(audit, 188),
                u32(audit, 204),
                u32(audit, 208),
                u32(audit, 248),
                u32(audit, 252),
            ) != (p, p + j, n, anchor, 1, p, n, j):
                raise ValueError("V7 audit identity/index/authorization mismatch")
            before, after = audit[40:72], audit[72:104]
            if (
                before != original[j * 32 : (j + 1) * 32]
                or after[:4] != before[:4]
                or after[16:] != before[16:]
                or after[4:16] != afters[j]
            ):
                raise ValueError("V7 audit channel-only write proof mismatch")
            if (
                audit[104:136] != reader[152:184]
                or audit[136:140] != reader[152:156]
                or audit[140:152] != afters[0]
                or audit[152:168] != reader[168:184]
            ):
                raise ValueError("V7 current-record exact before/after proof mismatch")
            if audit[168:172] != befores[j][:4] or audit[172:176] != afters[j][:4]:
                raise ValueError("V7 shadow write proof mismatch")
        return {
            "action": action,
            "n": n,
            "anchor": anchor,
            "end": anchor + n * 10,
            "p": p,
            "inputSchedule": "current-record-before-load/v1",
            "desiredSGB": desired,
            "appliedPostHex": [b.hex() for b in posts],
            "auditHex": [b.hex() for b in audits],
            "receiptHex": receipt.hex(),
            "nextReaderHex": self.reader.hex(),
        }

    def close(self, *, error: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        failure = None
        try:
            if self.proc is not None and not error and self.proc.poll() is None:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                if self.episode is not None:
                    reaped = self.command("truncate")
                    if not reaped["process_gone"] or not reaped["parent_unchanged"]:
                        raise ValueError("V7 driver failed to prove branch cleanup")
                    self.episode = None
                if self.command("close")["kind"] != "closed":
                    raise ValueError("V7 driver close ACK required")
                self.proc.stdin.close()
                if self.proc.wait(timeout=10) != 0:
                    raise ValueError("V7 driver did not exit cleanly")
        except BaseException as exc:
            failure = repr(exc)
        finally:
            if self.proc is not None:
                if self.proc.poll() is None:
                    self.proc.stop(timeout=10)
                self.cleanup = self.proc.report
                self.proc.close_control()
            for name in ("connection", "listener", "stderr", "rx"):
                obj = getattr(self, name, None)
                if obj is not None:
                    obj.close()
            posts_dir = self.out / "posts"
            posts_dir.mkdir()
            for path in self.local_posts.glob("*.bin"):
                shutil.copyfile(path, posts_dir / path.name)
            if hasattr(self, "ep"):
                save(self.ep / "requests-v7.json", self.requests)
                (self.ep / "responses-v7.bin").write_bytes(b"".join(self.responses))
            save(
                self.out / "close-v7.json",
                {
                    "error": error,
                    "closeFailure": failure,
                    "supervisor": getattr(self, "cleanup", None),
                    "commands": self.commands,
                    "messages": self.messages,
                    "retainedLocal": str(self.tmp),
                },
            )
        if failure:
            raise RuntimeError(failure)


def main() -> None:
    config_path, receipt_path, output = map(Path, sys.argv[1:4])
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 2190
    config, receipt = json.loads(config_path.read_bytes()), json.loads(receipt_path.read_bytes())
    deployment_raw = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(deployment_raw).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("pinned original deployment manifest differs")
    driver_sha = json.loads(deployment_raw)["driverBuildApproval"]["driverSha256"]
    output.mkdir(parents=True)
    report = {
        "schema": "tmrl.immediate-input-probe/v1",
        "passed": False,
        "literalStart": start,
        "cases": [],
        "learningExecuted": False,
        "clockMappingQualified": False,
    }
    try:
        for n in (1, 5):
            case = {"n": n, "rows": [], "passed": False}
            report["cases"].append(case)
            session = ImmediateSession(
                Path(config["runtime_root"]),
                receipt,
                output / f"{output.name}-n{n}",
                start,
                driver_sha,
            )
            try:
                case["initialPostHex"] = session.reset().hex()
                case["initialReaderHex"] = session.reader.hex()
                for action in range(78):
                    if (output / "STOP").exists():
                        raise InterruptedError("V7 cooperative STOP")
                    case["rows"].append(session.step(action, n))
                case["passed"] = len(case["rows"]) == 78
            finally:
                session.close(error=not case["passed"])
                case["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("V7 owned processes remain")
        report["passed"] = all(case["passed"] for case in report["cases"])
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save(output / "report.json", report)


if __name__ == "__main__":
    main()
