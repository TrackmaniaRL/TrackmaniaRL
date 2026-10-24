"""A bounded rollback lease for the independent no-artificial-lag plugin.

The client acquires the existing comparison controller mutex before connecting.
It verifies foreground/map, echoes the live baseline, and requires verified
restoration before returning. It sends no driving controls and activates no UI.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import secrets
import socket
import struct
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

PROTOCOL = "tmrl.input-profile/v1"
PROFILE = "no-artificial-lag/v1"
MUTEX = "Global\\TrackmaniaRL.ComparisonController"


def existing_controller_lease(source: Path | None = None) -> AbstractContextManager[Any]:
    if sys.platform != "win32":
        raise RuntimeError("the existing Windows comparison controller mutex is required")
    if source is None:
        configured = os.environ.get("TRACKMANIARL_CONTROLLER_LEASE")
        if not configured:
            raise RuntimeError(
                "configure TRACKMANIARL_CONTROLLER_LEASE or explicitly inject the controller lease"
            )
        source = Path(configured)
    if not source.is_absolute() or source.resolve(strict=True) != source or not source.is_file():
        raise ValueError("an explicit canonical controller lease source file is required")
    spec = importlib.util.spec_from_file_location("input_profile_controller_lease", source)
    if spec is None or spec.loader is None:
        raise ImportError("existing controller lease helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(AbstractContextManager[Any], module.controller_lease())


def verify_trackmania_foreground() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Windows foreground verification required")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    target = user32.FindWindowW(None, "Trackmania")
    if not target or target != user32.GetForegroundWindow():
        raise RuntimeError("Trackmania must already be the foreground window")


def _settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("available") is not True:
        raise ValueError("actual input port settings unavailable")
    avg, variance = value.get("avg"), value.get("var")
    if type(avg) is not int or not 0 <= avg <= 500 or type(variance) not in (int, float):
        raise ValueError("actual integral lag average and finite variance required")
    assert isinstance(variance, (int, float))
    if not math.isfinite(variance) or not 0 <= variance <= 1:
        raise ValueError("actual lag variance outside its declared domain")
    return value


def _same_variance(first: Any, second: Any) -> bool:
    if type(first) not in (int, float) or type(second) not in (int, float):
        return False
    return struct.pack("<f", first) == struct.pack("<f", second)


@dataclass(frozen=True, kw_only=True)
class InputProfileConfig:
    expected_map_uid: str
    ttl_seconds: float = 60
    port: int = 19177
    journal_path: Path | None = None
    timeout_seconds: float = 2

    def __post_init__(self) -> None:
        if not self.expected_map_uid or len(self.expected_map_uid) > 128:
            raise ValueError("explicit expected local map UID required")
        if not 0 < self.ttl_seconds <= 120 or not math.isfinite(self.ttl_seconds):
            raise ValueError("a finite lease TTL of at most120seconds is required")
        if not 0 < self.timeout_seconds <= 5 or not math.isfinite(self.timeout_seconds):
            raise ValueError("bounded finite socket timeout required")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("valid explicit loopback port required")


class NoArtificialLagLease:
    """Own the controller lock, temporary input profile and verified rollback.

    An injected lock/focus verifier is intended for tests or an already guarded
    controller. Production defaults use the exact existing comparison mutex and
    a read-only foreground check. Confirm the remote profile with ``check`` at
    case boundaries, and use ``check_local`` before each measured action.
    """

    def __init__(
        self,
        config: InputProfileConfig,
        *,
        controller_lease: Callable[[], AbstractContextManager[Any]] = existing_controller_lease,
        verify_focus: Callable[[], None] = verify_trackmania_foreground,
    ) -> None:
        self.config, self._controller_lease, self._verify_focus = (
            config,
            controller_lease,
            verify_focus,
        )
        self._stack: ExitStack | None = None
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._sequence = 0
        self.nonce = secrets.token_hex(16)
        self.baseline: dict[str, Any] | None = None
        self.applied = False
        self.restored_verified = False
        self._started = 0.0
        self.events: list[dict[str, Any]] = []

    def _record(self, event: dict[str, Any]) -> None:
        self.events.append({"host_perf_counter_ns": time.perf_counter_ns(), **event})

    def _connect(self) -> None:
        self._socket = socket.create_connection(
            ("127.0.0.1", self.config.port), self.config.timeout_seconds
        )
        self._socket.settimeout(self.config.timeout_seconds)
        self._buffer.clear()

    def _receive(self) -> dict[str, Any]:
        assert self._socket is not None
        while b"\n" not in self._buffer:
            raw = self._socket.recv(4096)
            if not raw:
                raise ConnectionError("input profile socket EOF")
            self._buffer.extend(raw)
            if len(self._buffer) > 8192:
                raise ValueError("input profile response exceeds bounded JSON line")
        line, _, remaining = self._buffer.partition(b"\n")
        self._buffer = bytearray(remaining)
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or value.get("protocol") != PROTOCOL
            or value.get("profile") != PROFILE
        ):
            raise ValueError("input profile protocol or profile mismatch")
        self._record({"direction": "received", "message": value})
        return value

    def _request(self, op: str, **fields: Any) -> dict[str, Any]:
        assert self._socket is not None
        self._sequence += 1
        value = {
            "protocol": PROTOCOL,
            "op": op,
            "sequence": self._sequence,
            "nonce": self.nonce,
            **fields,
        }
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > 8192:
            raise ValueError("input profile request exceeds its bounded JSON line")
        self._record({"direction": "sent", "message": value})
        self._socket.sendall(encoded)
        response = self._receive()
        asynchronous_restore = (
            op == "release"
            and response.get("op") == "restored"
            and type(response.get("sequence")) is int
            and 0 <= response["sequence"] <= self._sequence
        )
        if response.get("nonce") != self.nonce or (
            response.get("sequence") != self._sequence and not asynchronous_restore
        ):
            raise ValueError("input profile nonce/sequence mismatch")
        return response

    def __enter__(self) -> NoArtificialLagLease:
        if self._stack is not None:
            raise RuntimeError("input profile lease is one-shot")
        if self.config.journal_path is not None and self.config.journal_path.exists():
            raise FileExistsError("fresh input profile client journal required")
        self._stack = ExitStack()
        try:
            self._stack.enter_context(self._controller_lease())
            self._verify_focus()
            self._connect()
            hello = self._receive()
            if (
                hello.get("op") != "hello"
                or hello.get("actual_map_uid") != self.config.expected_map_uid
            ):
                raise ValueError("actual local map differs before profile acquisition")
            self.baseline = dict(_settings(hello.get("baseline")))
            if self.baseline.get("focused") is not True:
                raise ValueError("input port must be focused before profile acquisition")
            self._verify_focus()
            self._started = time.monotonic()
            response = self._request(
                "acquire",
                profile=PROFILE,
                controller_mutex_held=MUTEX,
                expected_map_uid=self.config.expected_map_uid,
                ttl_ms=max(1, int(self.config.ttl_seconds * 1000)),
                expected_baseline={"avg": self.baseline["avg"], "var": self.baseline["var"]},
            )
            if response.get("op") != "applied":
                raise RuntimeError(
                    "input profile acquisition was not acknowledged: " + str(response.get("error"))
                )
            self.applied = True
            current = _settings(response.get("actual"))
            if current["avg"] != 0 or current["var"] != 0:
                raise ValueError("input profile did not read back zero artificial lag")
            if response.get("baseline_avg") != self.baseline["avg"] or not _same_variance(
                response.get("baseline_var"),
                self.baseline["var"],
            ):
                raise ValueError("input profile acknowledged a different baseline")
            self._check_unchanged(current)
            return self
        except BaseException as error:
            try:
                self._restore()
            except BaseException as rollback:
                error.add_note("Input profile restoration could not be verified: " + repr(rollback))
            self._finish()
            raise

    def _check_unchanged(self, current: dict[str, Any]) -> None:
        assert self.baseline is not None
        for key in ("MaxSampleRate", "PollingEnabled", "IgnoreFocusForGamePads"):
            if key not in self.baseline or current.get(key) != self.baseline[key]:
                raise ValueError("an unrelated input setting changed during the profile: " + key)

    def check_local(self) -> None:
        """Check focus/TTL locally; this does not confirm current remote settings."""
        self._verify_focus()
        if not self.applied or self.restored_verified or self._socket is None:
            raise RuntimeError("no locally active input profile lease")
        if time.monotonic() - self._started >= self.config.ttl_seconds:
            raise TimeoutError("input profile TTL expired")

    def check(self) -> None:
        """Confirm actual remote profile/map at a case boundary, without game input."""
        self.check_local()
        response = self._request("status")
        if (
            response.get("op") != "active"
            or response.get("actual_map_uid") != self.config.expected_map_uid
        ):
            raise RuntimeError("input profile lease is no longer active on the expected map")
        current = _settings(response.get("actual"))
        if current["avg"] != 0 or current["var"] != 0 or current.get("focused") is not True:
            raise RuntimeError("input profile or foreground changed")
        self._check_unchanged(current)

    def _verify_restored(self, current: Any) -> None:
        assert self.baseline is not None
        actual = _settings(current)
        if actual["avg"] != self.baseline["avg"] or not _same_variance(
            actual["var"], self.baseline["var"]
        ):
            raise RuntimeError("original artificial lag baseline was not restored")
        self._check_unchanged(actual)
        self.restored_verified = True

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _restore(self) -> None:
        if self.baseline is None:
            self._close_socket()
            return
        try:
            if self._socket is not None:
                response = self._request("release")
                if response.get("op") == "restored" and response.get("restored_verified") is True:
                    self._verify_restored(response.get("actual"))
        except (OSError, ValueError, RuntimeError):
            # Disable/timeout may already have sent an asynchronous restore ACK
            # before closing. Retain that evidence even when release send fails.
            try:
                response = self._receive()
                if (
                    response.get("nonce") == self.nonce
                    and response.get("op") == "restored"
                    and (response.get("restored_verified") is True)
                ):
                    self._verify_restored(response.get("actual"))
            except (OSError, ValueError, RuntimeError):
                pass
        finally:
            self._close_socket()  # EOF independently triggers plugin rollback.
        if self.restored_verified:
            return
        # A protocol failure can hide the release ACK. Reconnect read-only after
        # EOF and require the original baseline before releasing the controller.
        deadline = time.monotonic() + self.config.timeout_seconds + 1
        failure: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self._connect()
                hello = self._receive()
                if hello.get("op") != "hello":
                    raise ValueError("fresh read-only baseline hello required")
                self._verify_restored(hello.get("baseline"))
                self._record({"restored_via_fresh_hello": True})
                return
            except (OSError, ValueError, RuntimeError) as error:
                failure = error
                time.sleep(0.05)
            finally:
                self._close_socket()
        raise RuntimeError("input profile rollback could not be verified") from failure

    def _finish(self) -> None:
        self._close_socket()
        try:
            if self.config.journal_path is not None:
                report = {
                    "schema": "tmrl.input-profile-client/v1",
                    "nonce": self.nonce,
                    "profile": PROFILE,
                    "mutex": MUTEX,
                    "baseline": self.baseline,
                    "applied": self.applied,
                    "restored_verified": self.restored_verified,
                    "events": self.events,
                }
                with self.config.journal_path.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
        finally:
            if self._stack is not None:
                self._stack.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._restore()
        except BaseException as error:
            if exc is not None:
                exc.add_note("Input profile rollback error: " + repr(error))
            raise
        finally:
            self._finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-map-uid", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--controller-lease", type=Path, required=True)
    parser.add_argument("--ttl-seconds", type=float, default=30)
    parser.add_argument("--hold-seconds", type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.hold_seconds < args.ttl_seconds <= 120:
        raise ValueError("hold must be positive and shorter than the bounded TTL")
    config = InputProfileConfig(
        expected_map_uid=args.expected_map_uid,
        ttl_seconds=args.ttl_seconds,
        journal_path=args.journal,
    )
    with NoArtificialLagLease(
        config, controller_lease=lambda: existing_controller_lease(args.controller_lease)
    ) as lease:
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            lease.check()
            time.sleep(0.1)
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "applied": lease.applied,
                "restored_verified": lease.restored_verified,
                "nonce": lease.nonce,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
