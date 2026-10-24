"""Optional spawn-process ownership for independent, bounded native facades.

The parent owns policy/replay state. Workers only prepare, reset, step and close
their own facade. A failed command invalidates the entire round; peer results
are retained as acquired evidence and are never returned as accepted rows.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import multiprocessing as mp
import os
import select
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any
from uuid import uuid4

from trackmaniarl.trackmania.native_headless._runtime import ownership
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os, linux_signal
from trackmaniarl.trackmania.native_headless.config import IMMEDIATE_PROTOCOL, NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.training_storage import (
    atomic_json,
    verify_native_cleanup,
)

SCHEMA = "tmrl.native-actor-pool/v1"
MAX_IPC_BYTES = 4 * 1024 * 1024
CLEANUP_TIMEOUT_S = 10.0
EnvironmentFactory = Callable[[NativeRuntimeConfig, int], Any]


class NativeActorPoolError(RuntimeError):
    """A round failed, with acquired-but-unaccepted results and cleanup evidence."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _wire_ready(descriptor: int, deadline: float | None, *, writing: bool) -> None:
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("native actor framed IPC deadline")
    reads, writes, _ = select.select(
        [] if writing else [descriptor], [descriptor] if writing else [], [], remaining
    )
    if not (writes if writing else reads):
        raise TimeoutError("native actor framed IPC deadline")


def _send(
    connection: Connection, value: Mapping[str, Any], *, deadline: float | None = None
) -> None:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_IPC_BYTES:
        raise ValueError("bounded native actor IPC message required")
    descriptor = connection.fileno()
    os.set_blocking(descriptor, False)
    pending = memoryview(struct.pack("!I", len(raw)) + raw)
    while pending:
        _wire_ready(descriptor, deadline, writing=True)
        try:
            written = os.write(descriptor, pending)
        except BlockingIOError:
            continue
        if written <= 0:
            raise EOFError("native actor IPC write made no progress")
        pending = pending[written:]


def _receive(connection: Connection, *, deadline: float | None = None) -> dict[str, Any]:
    descriptor = connection.fileno()
    os.set_blocking(descriptor, False)

    def exact(size: int) -> bytes:
        raw = bytearray()
        while len(raw) < size:
            _wire_ready(descriptor, deadline, writing=False)
            try:
                chunk = os.read(descriptor, min(size - len(raw), 65536))
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("native actor IPC EOF")
            raw.extend(chunk)
        return bytes(raw)

    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_IPC_BYTES:
        raise ValueError("native actor IPC frame exceeds byte limit")
    value = json.loads(exact(size))
    if not isinstance(value, dict):
        raise ValueError("native actor IPC object required")
    return value


def _native_environment(config: NativeRuntimeConfig, seed: int) -> Any:
    from trackmaniarl.trackmania.native_headless.immediate_environment import (
        NativeImmediateEnvironmentFactory,
    )

    return NativeImmediateEnvironmentFactory(config).create(seed=seed)


def _parent_death_guard(parent: dict[str, Any]) -> None:
    """Arm before any engine can start, then close the parent-exit race."""
    if os.getppid() != parent["pid"]:
        raise RuntimeError("native actor parent changed before lifetime guard")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(1, linux_signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "native actor parent-death guard")
    if os.getppid() != parent["pid"] or not ownership.same_identity(
        parent, ownership.process_identity(parent["pid"])
    ):
        raise RuntimeError("native actor parent identity changed during lifetime guard")


def _metadata(  # noqa: PLR0913
    actor_id: int,
    identity: dict[str, Any],
    environment: Any,
    active: bool,  # noqa: FBT001
) -> dict[str, Any]:
    session = getattr(environment, "_session", None)
    process = getattr(session, "proc", None)
    return {
        "actor_id": actor_id,
        "worker_identity": identity,
        "prepared": environment is not None,
        "episode_active": active,
        "trace_dir": None if environment is None else str(environment.config.trace_dir),
        "work_dir": None if environment is None else str(environment.config.work_dir),
        "journal_path": (
            None
            if getattr(environment, "journal_path", None) is None
            else str(environment.journal_path)
        ),
        "driver_identity": getattr(process, "driver_identity", None),
        "supervisor_identity": getattr(process, "supervisor_identity", None),
        "session_deadline": getattr(session, "deadline", None),
    }


def _close_environment(environment: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    if environment is None:
        result: dict[str, Any] = {
            "verified": True,
            "owned_processes_gone": True,
            "close_errors": [],
            "native_failures": [],
            "retained_local_directory": None,
            "supervisor": None,
            "native_execution_started": False,
        }
    elif (
        getattr(environment, "_session", None) is None and not environment.config.trace_dir.exists()
    ):
        environment.close()
        result = {
            "verified": True,
            "owned_processes_gone": True,
            "close_errors": [],
            "native_failures": [],
            "retained_local_directory": None,
            "supervisor": None,
            "native_execution_started": False,
        }
    else:
        result = verify_native_cleanup(environment)
        result["native_execution_started"] = True
    return {
        **result,
        **{
            key: metadata.get(key)
            for key in (
                "actor_id",
                "worker_identity",
                "trace_dir",
                "work_dir",
                "journal_path",
            )
        },
    }


def _worker_main(  # noqa: PLR0913
    actor_id: int,
    connection: Connection,
    config: NativeRuntimeConfig,
    parent: dict[str, Any],
    factory: EnvironmentFactory,
) -> None:
    environment, active, request_id, operation = None, False, 0, "hello"
    identity = ownership.process_identity(os.getpid())
    if identity is None:
        connection.close()
        return
    try:
        _parent_death_guard(parent)
        _send(
            connection,
            {
                "schema": SCHEMA,
                "actor_id": actor_id,
                "request_id": 0,
                "op": "hello",
                "kind": "result",
                "payload": _metadata(actor_id, identity, None, False),
            },
        )
        last_request = 0
        while True:
            command = _receive(connection)
            next_request, next_operation = command.get("request_id"), command.get("op")
            if (
                command.get("schema") != SCHEMA
                or command.get("actor_id") != actor_id
                or type(next_request) is not int
                or next_request <= last_request
                or not isinstance(next_operation, str)
            ):
                raise ValueError("native actor request identity/sequence mismatch")
            request_id, operation = next_request, next_operation
            last_request = request_id
            fields = command["payload"]
            stop = False
            if operation == "prepare":
                if environment is not None:
                    raise RuntimeError("close_session is required before preparing a replacement")
                selected = replace(config, timeout_s=fields.get("timeout_s", config.timeout_s))
                environment = factory(selected, fields["seed"])
                payload = _metadata(actor_id, identity, environment, False)
            elif operation == "reset":
                if environment is None or active:
                    raise RuntimeError("prepared environment and ended episode required for reset")
                state, info = environment.reset(**fields)
                active = True
                payload = {
                    "state": state,
                    "info": info,
                    "metadata": _metadata(actor_id, identity, environment, active),
                }
            elif operation == "step":
                if environment is None or not active:
                    raise RuntimeError("active native actor episode required")
                payload = environment.step_native(**fields)
                active = not (payload["terminated"] or payload["truncated"])
            elif operation == "session_reuse_status":
                if environment is None:
                    raise RuntimeError("prepared native actor environment required")
                payload = environment.session_reuse_status(**fields)
            elif operation in ("close_session", "close"):
                metadata = _metadata(actor_id, identity, environment, active)
                payload = _close_environment(environment, metadata)
                if payload["verified"] is not True:
                    raise RuntimeError("native actor session cleanup was not verified")
                environment, active = None, False
                stop = operation == "close"
            else:
                raise ValueError("unknown native actor operation")
            _send(
                connection,
                {
                    "schema": SCHEMA,
                    "actor_id": actor_id,
                    "request_id": request_id,
                    "op": operation,
                    "kind": "result",
                    "payload": payload,
                    "metadata": _metadata(actor_id, identity, environment, active),
                },
            )
            if stop:
                break
    except BaseException as error:
        metadata = _metadata(actor_id, identity, environment, active)
        try:
            cleanup = _close_environment(environment, metadata)
        except BaseException as cleanup_error:
            cleanup = {
                "verified": False,
                "owned_processes_gone": False,
                "close_errors": [repr(cleanup_error)],
                **metadata,
            }
        environment = None
        with contextlib.suppress(Exception):
            _send(
                connection,
                {
                    "schema": SCHEMA,
                    "actor_id": actor_id,
                    "request_id": request_id,
                    "op": operation,
                    "kind": "error",
                    "payload": {"error": repr(error), "cleanup": cleanup, "metadata": metadata},
                },
            )
    finally:
        if environment is not None:
            with contextlib.suppress(BaseException):
                _close_environment(environment, _metadata(actor_id, identity, environment, active))
        connection.close()


@dataclass
class _Actor:
    process: BaseProcess
    connection: Connection
    identity: dict[str, Any] | None
    metadata: dict[str, Any]
    cleanup: dict[str, Any] | None = None
    closed: bool = False
    exitcode: int | None = None


def _regular_json(path: Path) -> dict[str, Any]:
    if path.parent.resolve() != path.parent:
        raise ValueError("canonical owned actor receipt parent required")
    directory = os.open(path.parent, os.O_RDONLY | linux_os.O_DIRECTORY | linux_os.O_NOFOLLOW)
    try:
        fd = os.open(
            path.name, os.O_RDONLY | linux_os.O_NOFOLLOW | linux_os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_IPC_BYTES:
                raise ValueError("bounded regular actor receipt required")
            raw = stream.read(MAX_IPC_BYTES + 1)
            if len(raw) > MAX_IPC_BYTES:
                raise ValueError("actor receipt grew beyond bound")
    finally:
        os.close(directory)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("actor receipt object required")
    return value


def _identity_gone(row: Any) -> bool:
    return not ownership.same_identity(row, ownership.process_identity(row["pid"]))


class NativeActorPool:
    """Synchronous rounds over explicit spawn workers; no learner imports.

    Roots must be fresh canonical ext4 directories. Call close_session before
    admitting a replacement session; prepare alone allocates no native process.
    The creator must be the main thread, matching Linux PDEATHSIG semantics.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: NativeRuntimeConfig,
        *,
        actor_count: int,
        trace_root: Path,
        work_root: Path,
        request_timeout_s: float | None = None,
        _environment_factory: EnvironmentFactory = _native_environment,
    ) -> None:
        if config.protocol_profile != IMMEDIATE_PROTOCOL:
            raise ValueError("explicit V7 actor configuration required")
        if type(actor_count) is not int or not 1 <= actor_count <= 8:
            raise ValueError("native actor count must be 1..8")
        timeout = config.timeout_s + 15 if request_timeout_s is None else request_timeout_s
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 300
        ):
            raise ValueError("finite native actor IPC timeout in (0,300] required")
        for path in (trace_root, work_root):
            if (
                not isinstance(path, Path)
                or not path.is_absolute()
                or str(path).startswith("/mnt/")
                or path.resolve() != path
            ):
                raise ValueError("canonical explicit ext4 actor root required")
        if (
            trace_root == work_root
            or trace_root in work_root.parents
            or work_root in trace_root.parents
            or not work_root.is_relative_to(config.runtime_root / "build/server-probe")
        ):
            raise ValueError("disjoint actor roots and bounded runtime work root required")
        self.config, self.actor_count = config, actor_count
        self.trace_root, self.work_root = trace_root, work_root
        self.request_timeout_s = float(timeout)
        self._factory = _environment_factory
        self._actors: dict[int, _Actor] = {}
        self._pending: dict[int, int] = {}
        self._round_operation: str | None = None
        self._round_deadline: float | None = None
        self._sequence = 0
        self._opened = self._closed = self._broken = False
        self._trace_owned = False
        self._owner: dict[str, Any] | None = None
        self._thread: int | None = None

    def __enter__(self) -> NativeActorPool:
        return self.open()

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        try:
            self.close_all()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"native actor pool cleanup also failed: {error!r}")

    def _owner_check(self) -> None:
        if (
            self._owner is None
            or self._owner["pid"] != os.getpid()
            or self._thread != threading.get_ident()
        ):
            raise RuntimeError("native actor pool is owned by its creating process/main thread")

    @property
    def in_flight(self) -> bool:
        return self._round_operation is not None

    def metadata(self) -> dict[int, dict[str, Any]]:
        self._owner_check()
        if self.in_flight:
            raise RuntimeError("native actor metadata requires a completed round")
        return {key: deepcopy(actor.metadata) for key, actor in sorted(self._actors.items())}

    def checkpoint_barrier(self) -> dict[int, dict[str, Any]]:
        self._owner_check()
        if self.in_flight or self._broken or self._closed:
            raise RuntimeError("healthy no-inflight actor boundary required")
        return self.metadata()

    def open(self) -> NativeActorPool:
        if self._opened or self._closed:
            raise RuntimeError("native actor pool may open only once")
        if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
            raise RuntimeError("native actor pool requires the Linux main thread")
        self._owner = ownership.process_identity(os.getpid())
        if self._owner is None:
            raise RuntimeError("native actor parent identity unavailable")
        self._thread, self._opened = threading.get_ident(), True
        try:
            self.trace_root.mkdir(parents=True)
            self._trace_owned = True
            self.work_root.mkdir(parents=True)
            context = mp.get_context("spawn")
            for actor_id in range(self.actor_count):
                parent, child = context.Pipe(duplex=True)
                selected = replace(
                    self.config,
                    trace_dir=self.trace_root / f"actor-{actor_id:04d}",
                    work_dir=self.work_root / f"actor-{actor_id:04d}",
                )
                process = context.Process(
                    target=_worker_main,
                    args=(actor_id, child, selected, self._owner, self._factory),
                    name=f"native-actor-{actor_id:04d}",
                    # A forgotten-close normal parent exit must not hang in
                    # multiprocessing's child join before PDEATHSIG can fire.
                    # Workers spawn their native owners through Popen only.
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._actors[actor_id] = _Actor(
                    process,
                    parent,
                    ownership.process_identity(process.pid),
                    {"actor_id": actor_id},
                )
            deadline = time.monotonic() + self.request_timeout_s
            for actor_id, actor in self._actors.items():
                if not actor.connection.poll(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError("native actor startup acknowledgement timeout")
                reply = _receive(actor.connection, deadline=deadline)
                self._match(reply, actor_id, 0, "hello")
                if reply.get("kind") != "result":
                    raise RuntimeError(f"native actor lifetime preflight failed: {reply}")
                identity = reply["payload"]["worker_identity"]
                if actor.identity is None or not ownership.same_identity(actor.identity, identity):
                    raise RuntimeError("native actor startup process identity mismatch")
                actor.metadata = reply["payload"]
            return self
        except BaseException as error:
            evidence = self._abort("open", {}, {"startup": repr(error)})
            raise NativeActorPoolError("native actor pool startup failed", evidence) from error

    @staticmethod
    def _match(  # noqa: PLR0913
        reply: Mapping[str, Any], actor_id: int, request_id: int, operation: str
    ) -> None:
        if (
            reply.get("schema") != SCHEMA
            or type(reply.get("actor_id")) is not int
            or type(reply.get("request_id")) is not int
            or reply.get("actor_id") != actor_id
            or reply.get("request_id") != request_id
            or reply.get("op") != operation
            or reply.get("kind") not in ("result", "error")
        ):
            raise ValueError("native actor response identity/sequence mismatch")

    def _validate(self, operation: str, requests: Mapping[int, Mapping[str, Any]]) -> None:
        self._owner_check()
        if self._closed or self._broken or self.in_flight:
            raise RuntimeError("healthy idle native actor pool required")
        for actor_id, fields in requests.items():
            if (
                type(actor_id) is not int
                or actor_id not in self._actors
                or self._actors[actor_id].closed
            ):
                raise ValueError("live native actor ID required")
            if not isinstance(fields, Mapping):
                raise ValueError("native actor request fields required")
            metadata = self._actors[actor_id].metadata
            if operation == "prepare":
                if (
                    set(fields) - {"seed", "timeout_s"}
                    or type(fields.get("seed")) is not int
                    or metadata.get("prepared")
                ):
                    raise ValueError("new environment and explicit seed required for prepare")
                timeout = fields.get("timeout_s", self.config.timeout_s)
                if (
                    type(timeout) not in (int, float)
                    or not math.isfinite(timeout)
                    or not 0 < timeout <= self.config.timeout_s
                ):
                    raise ValueError("new session timeout cannot extend configured deadline")
            elif operation == "reset":
                if (
                    set(fields) - {"seed", "duration_limit"}
                    or not metadata.get("prepared")
                    or metadata.get("episode_active")
                ):
                    raise ValueError("prepared environment with ended episode required for reset")
                if (
                    "seed" in fields
                    and fields["seed"] is not None
                    and type(fields["seed"]) is not int
                ):
                    raise ValueError("literal actor reset seed required")
                limit = fields.get("duration_limit")
                if limit is not None and (
                    type(limit) is not int
                    or not 10 <= limit <= 47770
                    or limit % 10
                    or (
                        self.config.duration_limit is not None
                        and limit > self.config.duration_limit
                    )
                ):
                    raise ValueError(
                        "actor reset duration outside configured native interval scope"
                    )
            elif operation == "step":
                if (
                    set(fields) != {"action", "native_iterations"}
                    or type(fields["action"]) is not int
                    or not 0 <= fields["action"] < 78
                    or type(fields["native_iterations"]) is not int
                    or not 1 <= fields["native_iterations"] <= 5
                    or not metadata.get("episode_active")
                ):
                    raise ValueError("active actor and literal bounded native step required")
            elif operation == "session_reuse_status":
                seconds: Any = fields.get("minimum_remaining_seconds")
                size: Any = fields.get("additional_trace_bytes")
                if (
                    set(fields) != {"minimum_remaining_seconds", "additional_trace_bytes"}
                    or not metadata.get("prepared")
                    or type(seconds) not in (int, float)
                    or not math.isfinite(seconds)
                    or seconds < 0
                    or type(size) is not int
                    or size < 0
                ):
                    raise ValueError("finite actor reuse reservations required")
            elif operation not in ("close_session", "close") or fields:
                raise ValueError("unknown actor operation or unexpected fields")

    def _begin_round(self, operation: str, requests: Mapping[int, Mapping[str, Any]]) -> None:
        self._validate(operation, requests)
        deadline = time.monotonic() + self.request_timeout_s
        self._round_operation, self._round_deadline = operation, deadline
        try:
            for actor_id in sorted(requests):
                self._sequence += 1
                self._pending[actor_id] = self._sequence
                _send(
                    self._actors[actor_id].connection,
                    {
                        "schema": SCHEMA,
                        "actor_id": actor_id,
                        "request_id": self._sequence,
                        "op": operation,
                        "payload": dict(requests[actor_id]),
                    },
                    deadline=deadline,
                )
        except BaseException as error:
            evidence = self._abort(operation, {}, {"round": repr(error)})
            raise NativeActorPoolError(
                "native actor round dispatch failed; no rows accepted", evidence
            ) from error

    def _finish_round(self, operation: str) -> dict[int, Any]:
        self._owner_check()
        if self._closed or self._broken or self._round_operation != operation:
            raise RuntimeError("matching in-flight native actor round required")
        deadline = self._round_deadline
        assert deadline is not None
        acquired: dict[int, Any] = {}
        failures: dict[str, Any] = {}
        try:
            while self._pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError("native actor round deadline")
                ready = wait(
                    [self._actors[key].connection for key in self._pending],
                    timeout=max(0.0, deadline - time.monotonic()),
                )
                if not ready:
                    raise TimeoutError("native actor round deadline")
                for actor_id in sorted(self._pending):
                    actor = self._actors[actor_id]
                    if actor.connection not in ready:
                        continue
                    reply = _receive(actor.connection, deadline=deadline)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native actor round deadline")
                    self._match(reply, actor_id, self._pending[actor_id], operation)
                    del self._pending[actor_id]
                    if reply["kind"] == "error":
                        failures[str(actor_id)] = reply["payload"]
                        actor.cleanup = reply["payload"].get("cleanup")
                        actor.metadata = reply["payload"]["metadata"]
                    else:
                        acquired[actor_id] = reply["payload"]
                        actor.metadata = reply["metadata"]
                        if operation == "prepare":
                            actor.cleanup = None
                        if operation in ("close_session", "close"):
                            actor.cleanup = reply["payload"]
                if failures:
                    raise RuntimeError("native actor peer rejected round")
            if operation in ("close_session", "close"):
                for key, result in acquired.items():
                    self._actors[key].cleanup = result
            self._round_operation = self._round_deadline = None
            return {key: acquired[key] for key in sorted(acquired)}
        except BaseException as error:
            failures.setdefault("round", repr(error))
            evidence = self._abort(operation, acquired, failures)
            raise NativeActorPoolError(
                "native actor round failed; no rows accepted", evidence
            ) from error

    def _round(self, operation: str, requests: Mapping[int, Mapping[str, Any]]) -> dict[int, Any]:
        self._begin_round(operation, requests)
        return self._finish_round(operation)

    def prepare_many(self, requests: Mapping[int, Mapping[str, Any]]) -> dict[int, Any]:
        return self._round("prepare", requests)

    def reset_many(self, requests: Mapping[int, Mapping[str, Any]]) -> dict[int, Any]:
        return self._round("reset", requests)

    def step_many(self, requests: Mapping[int, Mapping[str, Any]]) -> dict[int, Any]:
        self.begin_step_many(requests)
        return self.finish_step_many()

    def begin_step_many(self, requests: Mapping[int, Mapping[str, Any]]) -> None:
        """Dispatch one step round; its absolute deadline also covers parent work.

        Only the creating main thread may finish or cancel this round. Until
        finish_step_many returns, no current-round result has been returned for
        admission and the checkpoint barrier remains closed.
        """
        self._begin_round("step", requests)

    def finish_step_many(self) -> dict[int, Any]:
        """Collect the dispatched round without extending its original deadline."""
        return self._finish_round("step")

    def session_reuse_status_many(
        self, requests: Mapping[int, Mapping[str, Any]]
    ) -> dict[int, Any]:
        return self._round("session_reuse_status", requests)

    def reuse_status(
        self, actor_id: int, *, minimum_remaining_seconds: float, additional_trace_bytes: int
    ) -> dict[str, Any]:
        result: dict[str, Any] = self.session_reuse_status_many(
            {
                actor_id: {
                    "minimum_remaining_seconds": minimum_remaining_seconds,
                    "additional_trace_bytes": additional_trace_bytes,
                }
            }
        )[actor_id]
        return result

    def close_session(self, actor_id: int) -> dict[str, Any]:
        result: dict[str, Any] = self._round("close_session", {actor_id: {}})[actor_id]
        return result

    def _join(self, actor_id: int, deadline: float) -> None:
        actor = self._actors[actor_id]
        actor.process.join(timeout=max(0.0, deadline - time.monotonic()))
        if actor.process.is_alive():
            raise TimeoutError("native actor worker did not exit after close")
        actor.closed = True
        actor.connection.close()
        actor.exitcode = actor.process.exitcode
        actor.process.close()
        if actor.cleanup is not None:
            actor.cleanup.update(worker_reaped=True, worker_exitcode=actor.exitcode)
        if actor.exitcode != 0:
            raise RuntimeError("native actor worker exited abnormally after close")

    def close_actor(self, actor_id: int) -> dict[str, Any]:
        self._owner_check()
        if self.in_flight:
            raise RuntimeError("finish the native actor round or abort with close_all")
        actor = self._actors[actor_id]
        if not actor.closed:
            self._round("close", {actor_id: {}})
            try:
                self._join(actor_id, time.monotonic() + CLEANUP_TIMEOUT_S)
            except BaseException as error:
                evidence = self._abort("close", {}, {str(actor_id): repr(error)})
                raise NativeActorPoolError("native actor worker close failed", evidence) from error
        return deepcopy(actor.cleanup or {})

    def close_all(self) -> dict[int, dict[str, Any]]:
        self._owner_check()
        if self._closed:
            return {
                key: deepcopy(actor.cleanup or {}) for key, actor in sorted(self._actors.items())
            }
        if self.in_flight:
            evidence = self._abort(
                self._round_operation or "close",
                {},
                {"cancelled": "close_all cancelled an unfinished round"},
            )
            raise NativeActorPoolError(
                "native actor pending round cancelled; no rows accepted", evidence
            )
        live: dict[int, dict[str, Any]] = {
            key: {} for key, actor in self._actors.items() if not actor.closed
        }
        if live and not self._broken:
            self._round("close", live)
            try:
                deadline = time.monotonic() + CLEANUP_TIMEOUT_S
                for key in sorted(live):
                    self._join(key, deadline)
            except BaseException as error:
                evidence = self._abort("close", {}, {"join": repr(error)})
                raise NativeActorPoolError("native actor pool close failed", evidence) from error
        self._closed = True
        return {key: deepcopy(actor.cleanup or {}) for key, actor in sorted(self._actors.items())}

    def _recover(self, actor: _Actor, deadline: float) -> dict[str, Any]:
        metadata = actor.metadata
        trace = metadata.get("trace_dir")
        previous = actor.cleanup
        if (
            metadata.get("prepared")
            and previous is not None
            and any(previous.get(key) != metadata.get(key) for key in ("trace_dir", "work_dir"))
        ):
            previous = None
        if not metadata.get("prepared") and previous is not None:
            return deepcopy(previous)
        if trace is None or not Path(trace).exists():
            return {
                "verified": previous.get("verified", True) if previous else True,
                "owned_processes_gone": True,
                "native_execution_started": False,
                "close_errors": [],
                "retained_local_directory": None,
                **metadata,
            }
        errors: list[str] = []
        receipt: dict[str, Any] | None = None
        path = Path(trace) / "native-supervisor.json"
        while True:
            try:
                receipt = _regular_json(path)
                break
            except (FileNotFoundError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            except Exception as error:
                errors.append(repr(error))
                break
        gone = False
        if receipt is not None:
            try:
                owner = receipt["supervisorIdentity"]
                expected = metadata.get("supervisor_identity")
                if (
                    receipt.get("schema") != "tmrl.native-headless.supervisor/v1"
                    or actor.identity is None
                    or owner["ppid"] != actor.identity["pid"]
                    or owner["uid"] != actor.identity["uid"]
                    or owner["birth"] < actor.identity["birth"]
                    or (expected is not None and not ownership.same_identity(owner, expected))
                    or receipt.get("remainingOwned") != []
                    or receipt.get("cleanupErrors") != []
                    or receipt.get("supervisorError")
                    or receipt.get("publicationError")
                ):
                    raise ValueError("native actor recovery receipt identity/cleanup mismatch")
                rows = [owner, receipt["driverIdentity"], *receipt["capturedOwned"]]
                while not all(_identity_gone(row) for row in rows if row is not None):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("recorded native actor identity still exists")
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                gone = True
            except Exception as error:
                errors.append(repr(error))
        else:
            errors.append("native supervisor recovery receipt unavailable")
        retained = None
        try:
            launch = _regular_json(Path(trace) / "launch-v7.json")
            candidate = Path(launch["environment"]["TMRE_TREE_DIR"])
            if (
                candidate.parent != Path("/tmp")
                or not candidate.name.startswith("tmre-immediate-v7-")
                or candidate.resolve() != candidate
            ):
                raise ValueError("invalid recovered private native directory")
            retained = str(candidate)
        except Exception as error:
            errors.append(repr(error))
        return {
            **(previous or {}),
            "verified": bool(previous and previous.get("verified") and not errors),
            "owned_processes_gone": gone,
            "close_errors": [*(previous or {}).get("close_errors", []), *errors],
            "retained_local_directory": retained,
            "supervisor": receipt,
            "native_execution_started": True,
            **metadata,
        }

    def _abort(
        self, operation: str, acquired: dict[int, Any], failures: dict[str, Any]
    ) -> dict[str, Any]:
        self._broken = True
        cleanup_errors: dict[str, str] = {}
        for key, actor in self._actors.items():
            if actor.closed:
                continue
            try:
                if actor.identity is not None:
                    ownership.signal_identity(actor.identity, linux_signal.SIGKILL)
            except OSError as error:
                cleanup_errors[str(key)] = repr(error)
            actor.connection.close()
        self._pending.clear()
        self._round_operation = self._round_deadline = None
        deadline = time.monotonic() + CLEANUP_TIMEOUT_S
        for key, actor in self._actors.items():
            if actor.closed:
                continue
            actor.process.join(timeout=max(0.0, deadline - time.monotonic()))
            actor.closed = not actor.process.is_alive()
            actor.exitcode = actor.process.exitcode
            actor.cleanup = self._recover(actor, deadline)
            actor.cleanup.update(worker_reaped=actor.closed, worker_exitcode=actor.exitcode)
            if actor.closed:
                actor.process.close()
            if not actor.closed:
                cleanup_errors[str(key)] = "owned actor worker still alive"
                actor.cleanup.update(verified=False, owned_processes_gone=False)
        self._closed = True
        evidence = {
            "schema": SCHEMA,
            "operation": operation,
            "failures": failures,
            "acquired_unaccepted_results": acquired,
            "accepted_rows": 0,
            "cleanup": {key: actor.cleanup for key, actor in sorted(self._actors.items())},
            "cleanup_errors": cleanup_errors,
        }
        if self._trace_owned and self.trace_root.is_dir():
            path = self.trace_root / f"pool-failure-{uuid4().hex}.json"
            try:
                atomic_json(path, evidence, max_bytes=16 * 1024 * 1024)
                evidence["evidence_path"] = str(path)
            except Exception as error:
                evidence["evidence_publication_error"] = repr(error)
        return evidence
