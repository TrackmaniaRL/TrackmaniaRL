"""Socket-only rollback/client tests. No game, UI or production mutex calls."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from scripts.headless_input_profile import (
    MUTEX,
    PROFILE,
    PROTOCOL,
    InputProfileConfig,
    NoArtificialLagLease,
)


class FakeProfileServer:
    def __init__(self) -> None:
        self.baseline = {
            "available": True,
            "avg": 68,
            "var": struct.unpack("<f", struct.pack("<f", 0.15))[0],
            "MaxSampleRate": 166,
            "PollingEnabled": True,
            "IgnoreFocusForGamePads": False,
            "focused": True,
        }
        self.current = dict(self.baseline)
        self.active = False
        self.events: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(2)
        self.sock.settimeout(0.02)
        self.drop_after_acquire = False
        self.release_ack_actual: dict[str, Any] | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self) -> FakeProfileServer:
        self.thread.start()
        assert self.ready.wait(1)
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(2)
        self.sock.close()

    def send(self, connection: socket.socket, message: dict[str, Any]) -> None:
        connection.sendall(
            json.dumps({"protocol": PROTOCOL, "profile": PROFILE, **message}).encode() + b"\n"
        )

    def run(self) -> None:
        self.ready.set()
        while not self.stop.is_set():
            try:
                connection, _ = self.sock.accept()
            except TimeoutError:
                continue
            connection.settimeout(0.02)
            buffer = bytearray()
            deadline = 0.0
            sequence, nonce = 0, ""
            op = ""
            try:
                self.send(
                    connection,
                    {
                        "op": "hello",
                        "nonce": "",
                        "sequence": 0,
                        "baseline": self.current,
                        "actual_map_uid": "mock-map",
                    },
                )
                while not self.stop.is_set():
                    if self.active and time.monotonic() >= deadline:
                        self.current = dict(self.baseline)
                        self.active = False
                        self.events.append("timeout-restored")
                        self.send(
                            connection,
                            {
                                "op": "restored",
                                "nonce": nonce,
                                "sequence": sequence,
                                "restored_verified": True,
                                "actual": self.current,
                            },
                        )
                        break
                    try:
                        data = connection.recv(4096)
                    except TimeoutError:
                        continue
                    if not data:
                        break
                    buffer.extend(data)
                    while b"\n" in buffer:
                        line, _, remaining = buffer.partition(b"\n")
                        buffer = bytearray(remaining)
                        request = json.loads(line)
                        self.requests.append(request)
                        nonce, sequence = request["nonce"], request["sequence"]
                        op = request["op"]
                        if op == "acquire":
                            assert request["controller_mutex_held"] == MUTEX
                            assert request["expected_baseline"] == {
                                "avg": self.baseline["avg"],
                                "var": self.baseline["var"],
                            }
                            self.current = {**self.baseline, "avg": 0, "var": 0.0}
                            self.active = True
                            deadline = time.monotonic() + request["ttl_ms"] / 1000
                            self.events.append("applied")
                            self.send(
                                connection,
                                {
                                    "op": "applied",
                                    "nonce": nonce,
                                    "sequence": sequence,
                                    "baseline_avg": self.baseline["avg"],
                                    "baseline_var": self.baseline["var"],
                                    "actual": self.current,
                                },
                            )
                            if self.drop_after_acquire:
                                break
                        elif op == "status":
                            self.send(
                                connection,
                                {
                                    "op": "active",
                                    "nonce": nonce,
                                    "sequence": sequence,
                                    "actual_map_uid": "mock-map",
                                    "actual": self.current,
                                },
                            )
                        elif op == "release":
                            self.current = dict(self.baseline)
                            self.active = False
                            self.events.append("release-restored")
                            self.send(
                                connection,
                                {
                                    "op": "restored",
                                    "nonce": nonce,
                                    "sequence": sequence,
                                    "restored_verified": True,
                                    "actual": self.release_ack_actual or self.current,
                                },
                            )
                            break
                    if op == "release" or self.drop_after_acquire:
                        break
            except (ConnectionError, OSError):
                pass
            finally:
                if self.active:
                    self.current = dict(self.baseline)
                    self.active = False
                    self.events.append("eof-restored")
                connection.close()


@contextmanager
def _lock(events: list[str]) -> Iterator[None]:
    events.append("lock-acquired")
    try:
        yield
    finally:
        events.append("lock-released")


def _lease(
    server: FakeProfileServer, tmp_path: Path, **changes: Any
) -> tuple[NoArtificialLagLease, list[str]]:
    events: list[str] = []
    config = InputProfileConfig(
        expected_map_uid="mock-map",
        port=server.port,
        journal_path=tmp_path / "client.json",
        timeout_seconds=0.2,
        **changes,
    )
    return NoArtificialLagLease(
        config,
        controller_lease=lambda: _lock(events),
        verify_focus=lambda: events.append("focus-verified"),
    ), events


def test_profile_context_echoes_actual_baseline_and_restores_before_lock_release(
    tmp_path: Path,
) -> None:
    with FakeProfileServer() as server:
        lease, events = _lease(server, tmp_path)
        with lease:
            assert server.active
            count = len(server.requests)
            lease.check_local()
            assert len(server.requests) == count
            lease.check()
            assert server.current["avg"] == server.current["var"] == 0
        assert lease.restored_verified
        assert server.current == server.baseline
        assert events[0] == "lock-acquired"
        assert events[-1] == "lock-released"
        assert [request["sequence"] for request in server.requests] == [1, 2, 3]
        report = json.loads((tmp_path / "client.json").read_bytes())
        assert report["applied"]
        assert report["restored_verified"]


def test_body_exception_restores_original_values(tmp_path: Path) -> None:
    with FakeProfileServer() as server:
        lease, _ = _lease(server, tmp_path)
        with pytest.raises(RuntimeError, match="synthetic body failure"), lease:
            raise RuntimeError("synthetic body failure")
        assert lease.restored_verified
        assert server.current == server.baseline


def test_server_eof_restore_is_verified_by_fresh_readonly_hello(tmp_path: Path) -> None:
    with FakeProfileServer() as server:
        server.drop_after_acquire = True
        lease, _ = _lease(server, tmp_path)
        with lease:
            pass
        assert lease.restored_verified
        assert server.current == server.baseline
        assert "eof-restored" in server.events


def test_server_timeout_restore_accepts_asynchronous_ack(tmp_path: Path) -> None:
    with FakeProfileServer() as server:
        lease, _ = _lease(server, tmp_path, ttl_seconds=0.05)
        with lease:
            time.sleep(0.09)
        assert lease.restored_verified
        assert "timeout-restored" in server.events


def test_incorrect_restore_ack_requires_fresh_actual_baseline(tmp_path: Path) -> None:
    with FakeProfileServer() as server:
        server.release_ack_actual = {**server.baseline, "avg": 0, "var": 0}
        lease, events = _lease(server, tmp_path)
        with lease:
            pass
        assert lease.restored_verified
        assert server.current == server.baseline
        assert any(row.get("restored_via_fresh_hello") for row in lease.events)
        assert events[-1] == "lock-released"


def test_wrong_map_never_acquires_or_modifies_settings(tmp_path: Path) -> None:
    with FakeProfileServer() as server:
        config = InputProfileConfig(
            expected_map_uid="different-map", port=server.port, timeout_seconds=0.2
        )
        lease = NoArtificialLagLease(
            config, controller_lease=lambda: _lock([]), verify_focus=lambda: None
        )
        with pytest.raises(ValueError, match="local map differs"), lease:
            pytest.fail("wrong map entered")
        assert server.requests == []
        assert server.current == server.baseline


@pytest.mark.parametrize("ttl", [0, -1, 121, float("nan"), float("inf")])
def test_unbounded_or_invalid_ttl_is_rejected(ttl: float) -> None:
    with pytest.raises(ValueError, match="TTL"):
        InputProfileConfig(expected_map_uid="mock-map", ttl_seconds=ttl)
