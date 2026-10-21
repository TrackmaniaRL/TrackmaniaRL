from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v6 as SDK
from trackmaniarl.trackmania.native_headless._runtime.shapes import RequestSpec, ResponseSpec


@dataclass(frozen=True)
class RuntimeAssets:
    root: Path
    driver_sha256: str


@dataclass(frozen=True, slots=True)
class SessionPaths:
    output: Path
    work: Path


def u32(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def u64(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


def sha(b: Any) -> Any:
    return hashlib.sha256(b).hexdigest()


def dump(p: Any, v: Any) -> Any:
    p.write_bytes((json.dumps(v, indent=2, allow_nan=False) + "\n").encode())


class NativeSession:
    address: Any
    boundary: Any
    buffer: Any
    closed: Any
    commands: Any
    connection: Any
    deadline: Any
    ep: Any
    episode: Any
    epoch: Any
    listener: Any
    local_posts: Any
    messages: Any
    meta: Any
    out: Any
    pid: Any
    proc: Any
    reader: Any
    ready: Any
    requests: Any
    reset_ms: Any
    response_times: Any
    responses: Any
    seq: Any
    stderr: Any
    tmp: Any
    work: Any

    def __init__(self, paths: SessionPaths, timeout: Any = 180, *, runtime: RuntimeAssets) -> None:
        root = runtime.root
        driver = (
            root / "build/tmrl-shortcut/long-horizon-001/driver/target/release/tmre-reset-driver"
        )
        shim = root / "build/forkoracle/native-long-v6-001/libforkshim.so"
        tape = root / "analysis/tmrl-shortcut/long-horizon-controls-001/n4152-decl40000.Ghost.Gbx"
        self.out = paths.output
        self.work = paths.work
        if self.out.exists():
            raise ValueError("native host invariant 1")
        if self.work.exists():
            raise ValueError("native host invariant 2")
        self.out.mkdir(parents=True)
        self.work.mkdir(parents=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="tmre-reset-", dir="/tmp"))
        self.address = self.tmp / "control.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.address))
        self.listener.listen(1)
        self.deadline = time.monotonic() + timeout
        self.connection = None
        self.episode = None
        self.buffer = b""
        self.commands = []
        self.messages = []
        self.closed = False
        pins = {
            str(tape): "1cb2dfe4749c9743c8aa7cc710c5442f82c1193157e2c15b8e973f5bb5d6979e",
            str(shim): "a14837a1c64595a697ddea89637e4f12a904df7ddaa84d814b4fc2cd47e97781",
            str(
                root / "build/server-probe/linux-validate-map2-001/TrackmaniaServer"
            ): "3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779",
            str(
                root / "analysis/tmrl-shortcut/assets-001/tmrl-test.Map.Gbx"
            ): "ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc",
        }
        for p, h in pins.items():
            if not sha(Path(p).read_bytes()) == h:
                raise ValueError("native host invariant 3")
        if not sha(driver.read_bytes()) == runtime.driver_sha256:
            raise ValueError("native host invariant 4")
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "LD_PRELOAD" and (not k.startswith(("FK_", "FKSHIM_")))
        }
        (self.out / "posts").mkdir()
        (self.out / "results").mkdir()
        self.local_posts = self.tmp / "posts"
        self.local_posts.mkdir()
        env.update(
            TMRE_RUNTIME_ROOT=str(root),
            TMRE_TREE_DIR=str(self.tmp),
            FKSHIM_POST_DIR=str(self.local_posts),
            FKSHIM_INNER_SOCKET=str(self.address),
            FKSHIM_INNER_START="2180",
            FKSHIM_READER="1",
            FKSHIM_RESULT="1",
            FK_FRAME_TIMEOUT_MS="180000",
        )
        self.stderr = (self.out / "driver.stderr.bin").open("wb")
        args = [str(driver), str(self.work / "work"), str(self.out), str(tape), str(shim)]
        dump(
            self.out / "launch.json",
            {
                "args": args,
                "pins": pins,
                "driver_sha256": sha(driver.read_bytes()),
                "environment": {
                    k: env[k]
                    for k in (
                        "TMRE_RUNTIME_ROOT",
                        "TMRE_TREE_DIR",
                        "FKSHIM_POST_DIR",
                        "FKSHIM_INNER_SOCKET",
                        "FKSHIM_INNER_START",
                        "FKSHIM_READER",
                    )
                },
                "timeout": timeout,
                "host_memory_writes": False,
            },
        )
        self.proc = subprocess.Popen(
            args,
            cwd=self.work,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=True,
        )
        try:
            self.ready = self.line()
            if not self.ready["kind"] == "ready":
                raise ValueError("native host invariant 5")
            if not self.ready["metadata"]["input_count"] == 4152:
                raise ValueError("native host invariant 6")
            if not self.ready["metadata"]["start_offset_ms"] == -1510:
                raise ValueError("native host invariant 7")
        except BaseException:
            self.close(error=True)
            raise

    def close(self, *, error: bool = False) -> None:
        raise NotImplementedError("owned Session close override required")

    def exact(self, size: int) -> bytes:
        raise NotImplementedError("persist-before-decode RX override required")

    def remaining(self) -> Any:
        n = self.deadline - time.monotonic()
        if n <= 0:
            raise TimeoutError("bounded native session deadline")
        return n

    def line(self) -> Any:
        while b"\n" not in self.buffer:
            if not select.select([self.proc.stdout], [], [], min(1, self.remaining()))[0]:
                continue
            b = os.read(self.proc.stdout.fileno(), 65536)
            if not b:
                raise EOFError(f"driver EOF return={self.proc.poll()}")
            self.buffer += b
        line, self.buffer = self.buffer.split(b"\n", 1)
        v = json.loads(line)
        self.messages.append(v)
        return v

    def command(self, op: Any, **fields: Any) -> Any:
        v = dict(op=op, **fields)
        self.commands.append(dict(**v, host_ns=time.perf_counter_ns()))
        self.proc.stdin.write((json.dumps(v) + "\n").encode())
        self.proc.stdin.flush()
        return self.line()

    def response(self, spec: ResponseSpec) -> Any:
        kind, seq, n, epoch = (spec.kind, spec.seq, spec.n, spec.epoch)
        b = self.exact(448)
        d = SDK.decode_v6_response(b)
        if not (d["kind"], d["sequence"], d["iterations"], d["epoch"]) == (kind, seq, n, epoch):
            raise ValueError("native host invariant 8")
        if not d["event"]["pid"] == self.pid:
            raise ValueError("native host invariant 9")
        if kind in (1, 2) and not d["event"]["readNativeSimBoundaryProfileReady"]:
            raise ValueError("native host invariant 10")
        self.responses.append(b)
        self.response_times.append(
            {"kind": kind, "sequence": seq, "host_ns": time.perf_counter_ns()}
        )
        self.epoch = epoch
        return (b[32:288], b[288:448])

    def request(self, spec: RequestSpec) -> Any:
        op, n, p, k, befores, afters = (spec.op, spec.n, spec.p, spec.k, spec.befores, spec.afters)
        self.seq += 1
        b = bytearray(192)
        b[:8] = b"TMA6\x06" + bytes([op]) + b"\x00\x00"
        struct.pack_into("<IIQIII", b, 8, self.seq, n, self.epoch, p, k, len(afters))
        if not len(befores) == len(afters) <= 5:
            raise ValueError("native host invariant 11")
        for j, (before, after) in enumerate(zip(befores, afters, strict=False)):
            if not len(before) == len(after) == 12:
                raise ValueError("native host invariant 12")
            b[40 + j * 24 : 52 + j * 24] = before
            b[52 + j * 24 : 64 + j * 24] = after
        self.requests.append(
            {
                "op": op,
                "sequence": self.seq,
                "n": n,
                "epoch": self.epoch,
                "hex": b.hex(),
                "host_ns": time.perf_counter_ns(),
            }
        )
        self.connection.settimeout(min(90, self.remaining()))
        self.connection.sendall(b)

    def read_mem(self, address: Any, n: Any) -> Any:
        fd = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY)
        try:
            b = os.pread(fd, n, address)
            if not len(b) == n:
                raise ValueError("native host invariant 13")
            return b
        finally:
            os.close(fd)

    def reset(self, episode: Any = None) -> Any:
        if self.episode is not None:
            raise ValueError("native host invariant 14")
        start = time.perf_counter_ns()
        v = self.command("spawn", episode=episode)
        if not v["kind"] == "spawned":
            raise ValueError("native host invariant 15")
        self.episode = episode
        self.ep = Path(v["directory"])
        self.pid = v["pid"]
        self.meta = v["metadata"]
        self.listener.settimeout(min(90, self.remaining()))
        self.connection, _ = self.listener.accept()
        cred = struct.unpack(
            "3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if not cred == (self.pid, os.getuid(), os.getgid()):
            raise ValueError("native host invariant 16")
        self.seq = 0
        self.epoch = 1
        self.requests = []
        self.responses = []
        self.response_times = []
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=0, n=0, epoch=1))
        if not u32(self.boundary, 112) == 2180:
            raise ValueError("native host invariant 17")
        self.arm()
        self.reset_ms = (time.perf_counter_ns() - start) / 1000000.0
        return self.boundary

    def arm(self) -> Any:
        previous = self.epoch
        self.request(RequestSpec(op=3, n=0, p=0, k=0, befores=(), afters=()))
        self.reader, _ = self.response(ResponseSpec(kind=3, seq=self.seq, n=0, epoch=previous + 1))
        if not u32(self.reader, 124) == u32(self.boundary, 112):
            raise ValueError("native host invariant 18")
        if not u32(self.reader, 120) == u32(self.boundary, 112) + 10:
            raise ValueError("native host invariant 19")
        if not u64(self.reader, 88) == self.meta["input_base"]:
            raise ValueError("native host invariant 20")
        if not u32(self.reader, 116) == self.meta["input_count"]:
            raise ValueError("native host invariant 21")
        if (
            not self.read_mem(u64(self.reader, 88) - 4 + u32(self.reader, 112) * 32, 32)
            == self.reader[152:184]
        ):
            raise ValueError("native host invariant 22")


def state76(event: Any) -> Any:
    return event[152:212] + event[88:104]


class IncompleteReceiveError(EOFError):
    evidence: Any

    def __init__(self, evidence: Any) -> None:
        self.evidence = evidence
        super().__init__(evidence["reason"])


class NativeSessionV6(NativeSession):
    def __init__(self, paths: SessionPaths, timeout: Any = 180, *, runtime: RuntimeAssets) -> None:
        if type(timeout) not in (int, float) or not 0 < timeout <= 240:
            raise ValueError("V6 host deadline1..240s")
        super().__init__(paths, timeout, runtime=runtime)
