"""One reset, bounded telemetry validation, and an auditable readiness receipt.

This entry point never drives the vehicle or sends an extra finish-confirmation. A future frozen
queue must explicitly import this module from its own runtime to use it.

The caller must hold the enclosing comparison launcher/queue mutex and perform
its idle-process checks. This child process deliberately does not acquire a
second lease. Standalone callers must preserve those same outer guards.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.control import GamepadController
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.keyboard_control import ResetGuard
from trackmaniarl.trackmania.session import OpenPlanetSessionClient, OpenPlanetSessionProtocolError
from trackmaniarl.trackmania.telemetry import (
    OpenPlanetClient,
    OpenPlanetClientConfig,
    TelemetryFrame,
)

_MAX_START_RACE_TIME_MS = 2500.0
_CANCELLATION_POLL_S = 0.25


class PreflightError(RuntimeError):
    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(f"Preflight {receipt['phase']}: {receipt['reason']}")


class _PreflightCancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Budget:
    deadline: float
    stop_files: tuple[Path, ...]

    def remaining(self) -> float:
        if any(path.exists() for path in self.stop_files):
            raise _PreflightCancelledError("Preflight cancelled by stop file")
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Race-start validation exceeded start_timeout_s")
        return remaining

    def pause(self, seconds: float) -> None:
        end = monotonic() + min(seconds, self.remaining())
        while monotonic() < end:
            sleep(max(0.0, min(_CANCELLATION_POLL_S, end - monotonic(), self.remaining())))
            self.remaining()


class _PreflightSessionClient(OpenPlanetSessionClient):
    def __init__(self, config: TrackmaniaEnvironmentConfig, budget: _Budget) -> None:
        super().__init__(config.host, config.session_port, timeout_s=config.timeout_s)
        self._budget = budget

    def _exchange(self, command: str, request: bytes) -> bytes:
        del command
        timeout = min(self.timeout_s, self._budget.remaining(), _CANCELLATION_POLL_S)
        with socket.create_connection((self.host, self.port), timeout=timeout) as connection:
            connection.settimeout(min(timeout, self._budget.remaining()))
            connection.sendall(request)
            response = bytearray()
            while not response.endswith(b"\n"):
                connection.settimeout(min(timeout, self._budget.remaining()))
                try:
                    chunk = connection.recv(4096)
                except TimeoutError:
                    self._budget.remaining()
                    continue
                self._budget.remaining()
                if not chunk:
                    raise ConnectionError("OpenPlanet session disconnected before response")
                response.extend(chunk)
                if len(response) > 16 * 1024:
                    raise OpenPlanetSessionProtocolError(
                        "OpenPlanet session response exceeds 16 KiB"
                    )
            return bytes(response)


class _PreflightTelemetryClient(OpenPlanetClient):
    """Use the existing packet decoder with a deadline for every receive.

    The regular transport retries disconnects and uses per-receive timeouts.
    Preflight instead fails on disconnect, and a trickling partial packet cannot
    extend its total deadline. No background thread retains a socket after exit.
    """

    def __init__(self, config: OpenPlanetClientConfig, budget: _Budget) -> None:
        super().__init__(config)
        self._budget = budget

    def read(self) -> TelemetryFrame:
        self._budget.remaining()
        frame = self._read_latest_connected()
        self._budget.remaining()
        return frame

    def _receive_packet(self) -> bytes:
        self.timeout_s = min(self.timeout_s, self._budget.remaining(), _CANCELLATION_POLL_S)
        self.connect()
        assert self._socket is not None
        while len(self._buffer) < self._packet.size:
            self._socket.settimeout(min(self.timeout_s, self._budget.remaining()))
            try:
                chunk = self._socket.recv(self._packet.size - len(self._buffer))
            except TimeoutError:
                self._budget.remaining()
                continue
            self._budget.remaining()
            if not chunk:
                raise ConnectionError("OpenPlanet closed the preflight telemetry connection")
            self._buffer.extend(chunk)
        return self._take_oldest_packet()

    def _receive_available(self) -> bool:
        assert self._socket is not None
        while True:
            self._budget.remaining()
            try:
                chunk = self._socket.recv(64 * 1024)
            except BlockingIOError:
                return False
            self._budget.remaining()
            if not chunk:
                raise ConnectionError("OpenPlanet closed the preflight telemetry connection")
            self._buffer.extend(chunk)


def _load_config(path: Path, *, require_new_run: bool) -> TrackmaniaEnvironmentConfig:
    spec = RunSpec.from_yaml(path)
    directory = (path.parent / spec.artifacts_dir / spec.run_id).resolve()
    if require_new_run and directory.exists():
        raise RuntimeError(f"Run already exists: {directory}. Resume its checkpoint explicitly.")
    component = spec.components.environment
    if (
        component is None
        or component.class_path
        != "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
    ):
        raise RuntimeError("Comparison preflight requires the first-party Trackmania environment")
    config = TrackmaniaEnvironmentConfig.model_validate(component.kwargs["config"])
    if config.control_backend != "gamepad" or config.restart_input != "editor_validation":
        raise RuntimeError("Preflight requires gamepad control with editor_validation restart")
    if not config.expected_map_uid:
        raise RuntimeError("Preflight requires an expected map UID")
    return config


def _wait_for_start(
    client: _PreflightTelemetryClient, config: TrackmaniaEnvironmentConfig, state: dict[str, Any]
) -> None:
    while True:
        client._budget.remaining()
        state["read_count"] += 1
        frame = client.read()
        state.update(
            race_time_ms=float(frame.values[3]),
            finished=bool(frame.values[2]),
            fields=len(frame.values),
        )
        client._budget.remaining()
        if (
            0 < state["race_time_ms"] <= _MAX_START_RACE_TIME_MS
            and state["race_time_ms"] >= config.start_race_time_ms
            and not state["finished"]
        ):
            return
        client._budget.pause(config.start_poll_s)


def _validate_start(
    config: TrackmaniaEnvironmentConfig, state: dict[str, Any], budget: _Budget
) -> None:
    uid = config.expected_map_uid
    assert uid is not None
    state.update(map_uid=uid, phase="verify_map")
    session = _PreflightSessionClient(config, budget)
    session.verify_loaded_map(uid)
    state["phase"] = "ready_before_reset"
    session.timeout_s = min(config.timeout_s, budget.remaining())
    session.confirm_ready(uid)
    budget.remaining()
    controller = GamepadController(restart_input=config.restart_input)
    client = None
    try:
        state["phase"] = "reset"
        budget.remaining()
        controller.reset(guard=ResetGuard(check=budget.remaining, wait=budget.pause))
        budget.remaining()
        state["phase"] = "telemetry"
        client = _PreflightTelemetryClient(
            OpenPlanetClientConfig(config.host, config.port, config.timeout_s), budget
        )
        _wait_for_start(client, config, state)
        state["phase"] = "ready_after_reset"
        session.timeout_s = min(config.timeout_s, budget.remaining())
        session.confirm_ready(uid)
        budget.remaining()
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            controller.close()


def run_preflight(  # noqa: PLR0913
    config_path: Path,
    *,
    stop_files: tuple[Path, ...] = (),
    receipt_path: Path | None = None,
    require_new_run: bool = False,
) -> dict[str, Any]:
    """Validate one reset under the caller's existing queue mutex and idle checks."""
    state: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "runtime": str(Path(__file__).resolve().parents[2]),
        "status": "failed",
        "phase": "configuration",
        "map_uid": None,
        "race_time_ms": None,
        "finished": None,
        "read_count": 0,
        "fields": None,
        "reason": None,
    }
    try:
        if any(path.exists() for path in stop_files):
            raise _PreflightCancelledError("Preflight cancelled by stop file")
        config = _load_config(config_path.resolve(), require_new_run=require_new_run)
        budget = _Budget(monotonic() + config.start_timeout_s, stop_files)
        _validate_start(config, state, budget)
        state.update(status="ready", phase="complete")
    except (Exception, KeyboardInterrupt) as error:
        state["reason"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (_PreflightCancelledError, KeyboardInterrupt)):
            state["status"] = "cancelled"
        raise PreflightError(state) from error
    finally:
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stop-file", type=Path, action="append", default=[])
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--require-new-run", action="store_true")
    args = parser.parse_args()
    try:
        receipt = run_preflight(
            args.config,
            stop_files=tuple(args.stop_file),
            receipt_path=args.receipt,
            require_new_run=args.require_new_run,
        )
    except PreflightError as error:
        print(json.dumps(error.receipt), file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
