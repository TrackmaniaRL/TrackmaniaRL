"""Bounded manual 005 capture with a short guarded Windows input diagnostic.

Importing this module performs no Windows or game operations. The live command
requires --live and uses the existing comparison controller lease and preflight.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import secrets
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

from scripts.headless_probe_paths import DiagnosticProbePaths

PROTOCOL = "tmrl.physical-probe/v5"
ORIGINAL_SHA256 = "7f7bd9aacd1ab110713106ffa656093b5c8eb2383744e9facbad58ed6844d76c"
EXECUTABLE_SHA256 = "3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda"
OPENPLANET_SHA256 = "f134f405979d2c2641f8f131491e3401fcdd4f4b5266cde9f1a7001a6d3b49ef"
MAX_RESPONSE_BYTES = 512 * 1024
MAX_PACKETS = 1024


def verify_restoration(report: dict[str, Any], nonce: str) -> None:
    """Require independent byte observations and actual unhook acknowledgement."""
    expected = {
        "schema": "tmrl.headless-physical-copy-probe/v5",
        "nonce": nonce,
        "target_rva": "0x7d0cb0",
        "expected_executable_sha256": EXECUTABLE_SHA256,
        "observed_exe_version": "2026-02-02_17_51",
        "openplanet_version": "1.29.14",
        "openplanet_git_commit": "6fd4200f",
        "hook_ever_installed": True,
        "code_hook_attempted": True,
        "unhooked": True,
        "original_64_bytes_restored": True,
        "exact_original_prefix_verified": True,
        "unhook_error": False,
    }
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError("005 restoration evidence mismatch: " + key)
    verify_code_evidence(report, ORIGINAL_SHA256, prefix_bytes=7, label="005")


def verify_code_evidence(  # noqa: PLR0913 - Two wire contracts have different prefixes.
    report: dict[str, Any], digest: str, *, prefix_bytes: int, label: str
) -> None:
    """Compare exact-length canonical observations using the pinned digest."""
    for key in (
        "expected_original_64_hex",
        "observed_before_64_hex",
        "observed_after_unhook_64_hex",
    ):
        value = report.get(key)
        if type(value) is not str or len(value) != 128:
            raise ValueError(f"{label} restoration evidence mismatch: {key}")
        try:
            raw = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(f"{label} restoration evidence mismatch: {key}") from error
        if raw.hex() != value or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{label} restoration evidence mismatch: {key}")
    key = f"observed_original_trampoline_{prefix_bytes}_hex"
    if (
        type(report.get(key)) is not str
        or report[key] != report["observed_before_64_hex"][: prefix_bytes * 2]
    ):
        raise ValueError(f"{label} restoration evidence mismatch: {key}")


def verify_capture(report: dict[str, Any]) -> None:
    """Validate bounded raw evidence without inferring a simulation clock or ACK."""
    count, rows = report.get("record_count"), report.get("records")
    if type(count) is not int or not 1 <= count <= 256 or not isinstance(rows, list):
        raise ValueError("005 requires 1..256 actual records")
    if len(rows) != count:
        raise ValueError("005 record count differs from retained rows")
    if report.get("stop_reason") not in ("explicit-stop", "record-cap", "duration-cap"):
        raise ValueError("005 capture stopped for an unexpected reason")
    if (
        report.get("callback_reentered") is not False
        or report.get("callback_pointer_identity_lost") is not False
    ):
        raise ValueError("005 callback identity/reentrancy failure")
    for key, expected in {
        "destination_samples_are_PRE_copy": True,
        "Time_Now_is_simulation_clock": False,
        "applied_input_acknowledgement": False,
        "game_data_or_controls_written": False,
    }.items():
        if report.get(key) is not expected:
            raise ValueError("005 sample interpretation mismatch: " + key)
    owner = int(report["expected_participant_address"], 0)
    previous_time = -1
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or type(row.get("sequence")) is not int
            or row["sequence"] != index
        ):
            raise ValueError("005 actual record sequence is incomplete")
        timestamp = row.get("Time_Now_raw")
        if type(timestamp) is not int or timestamp < previous_time:
            raise ValueError("005 host metadata clock is invalid or regressed")
        previous_time = timestamp
        if (
            int(row["pose_destination_address"], 0) != owner + 0xE14
            or int(row["velocity_destination_address"], 0) != owner + 0xE30
        ):
            raise ValueError("005 record belongs to another participant")
        for name in ("raw_body_bits_u32", "destination_PRE_copy_bits_u32"):
            words = row.get(name)
            if (
                not isinstance(words, list)
                or len(words) != 10
                or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words)
            ):
                raise ValueError("005 requires ten literal uint32 words per source")


class PhysicalProbeClient:
    """One arm connection, bounded background drain, verified restoration on exit."""

    def __init__(self, port: int = 19178, *, timeout_seconds: float = 1.0) -> None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("valid loopback port required")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 2:
            raise ValueError("bounded finite socket timeout required")
        self.port, self.timeout_seconds = port, timeout_seconds
        self.nonce = secrets.token_hex(16)
        self.messages: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.restored_verified = False
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._thread: threading.Thread | None = None
        self._reader_error: BaseException | None = None
        self._armed = False
        self._arm_requested = False
        self._closed = False

    def _receive(self, deadline: float) -> dict[str, Any]:
        assert self._socket is not None
        connection = self._socket
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("005 response deadline exceeded")
            connection.settimeout(min(self.timeout_seconds, remaining))
            try:
                data = connection.recv(4096)
            except TimeoutError:
                continue
            if not data:
                raise ConnectionError("005 socket EOF")
            self._buffer.extend(data)
            if len(self._buffer) > MAX_RESPONSE_BYTES:
                raise ValueError("005 response exceeds its byte budget")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
            raise ValueError("005 JSON-lines protocol mismatch")
        if len(self.messages) >= 16:
            raise ValueError("005 protocol message budget exhausted")
        self.messages.append({"host_perf_counter_ns": time.perf_counter_ns(), **value})
        return value

    def _send(self, operation: str) -> None:
        assert self._socket is not None
        value: dict[str, Any] = {"protocol": PROTOCOL, "op": operation, "nonce": self.nonce}
        if operation == "arm":
            value["acknowledge_temporary_code_hook"] = True
        self._socket.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")

    def _accept_result(self, value: dict[str, Any]) -> None:
        if value.get("nonce") != self.nonce or value.get("op") not in ("complete", "status"):
            raise ValueError("005 completion nonce/operation mismatch")
        report = value.get("report")
        if not isinstance(report, dict):
            raise ValueError("005 actual report missing")
        self.report = report
        verify_restoration(report, self.nonce)
        self.restored_verified = True

    def _drain(self) -> None:
        try:
            self._accept_result(self._receive(time.monotonic() + 7))
        except BaseException as error:
            self._reader_error = error

    def __enter__(self) -> Self:
        if self._socket is not None or self._closed:
            raise RuntimeError("005 client is one-shot")
        try:
            self._socket = socket.create_connection(("127.0.0.1", self.port), self.timeout_seconds)
            hello = self._receive(time.monotonic() + 2)
            if hello.get("op") != "hello" or hello.get("hook_installed") is not False:
                raise ValueError("005 must be idle before arm")
            if hello.get("max_duration_ms") != 5000 or hello.get("max_records") != 256:
                raise ValueError("005 unexpected capture limits")
            self._arm_requested = True
            self._send("arm")
            armed = self._receive(time.monotonic() + 2)
            if armed.get("nonce") != self.nonce or armed.get("op") != "armed":
                if isinstance(armed.get("report"), dict):
                    self.report = armed["report"]
                raise RuntimeError("005 failed to arm: " + str(armed.get("status")))
            self._armed = True
            self._thread = threading.Thread(target=self._drain, name="physical005-reader")
            self._thread.start()
            return self
        except BaseException as error:
            self._close_socket()
            # A lost/malformed arm ACK does not prove no hook was installed.
            if self._arm_requested:
                try:
                    if self.report is None or self.report.get("code_hook_attempted") is not False:
                        self._recover_status()
                except BaseException as cleanup:
                    error.add_note("005 restoration could not be verified: " + repr(cleanup))
            self._closed = True
            raise

    def _close_socket(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._socket = None

    def _recover_status(self) -> None:
        self._close_socket()  # EOF is an independent plugin unhook trigger.
        self._buffer.clear()
        deadline = time.monotonic() + 2
        failure: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self._socket = socket.create_connection(
                    ("127.0.0.1", self.port), self.timeout_seconds
                )
                hello = self._receive(deadline)
                if hello.get("op") != "hello":
                    raise ValueError("005 fresh status handshake missing")
                self._send("status")
                self._accept_result(self._receive(deadline))
                return
            except (OSError, ValueError, RuntimeError) as error:
                failure = error
                time.sleep(0.02)
            finally:
                self._close_socket()
        raise RuntimeError("005 original-code restoration could not be verified") from failure

    def finish(self) -> dict[str, Any]:
        if not self._armed:
            raise RuntimeError("005 was not armed")
        if self._thread is not None and self._thread.is_alive():
            try:
                self._send("stop")
            except OSError:
                self._close_socket()
            self._thread.join(8)
            if self._thread.is_alive():
                self._close_socket()
                self._thread.join(2)
            if self._thread.is_alive():
                raise RuntimeError("005 reader failed to stop")
        if not self.restored_verified:
            self._recover_status()
        assert self.report is not None
        return self.report

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.finish()
        except BaseException as error:
            if exc is not None:
                error.add_note("Original diagnostic failure: " + repr(exc))
            raise
        finally:
            self._close_socket()
            self._closed = True


def _raw33(value: Any) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != 33 or not all(math.isfinite(item) for item in result):
        raise ValueError("exact finite original Connect33 packet required")
    return result


def select_actions(table: Sequence[Sequence[float]]) -> tuple[int, int]:
    candidates = [
        (index, float(control[2]))
        for index, control in enumerate(table)
        if len(control) == 3 and float(control[0]) == 1 and float(control[1]) == 0
    ]
    gas = next((index for index, steer in candidates if steer == 0), None)
    left = [(index, steer) for index, steer in candidates if -0.25 <= steer < 0]
    if gas is None or not left:
        raise ValueError("actual action table lacks full gas and mild left without brake")
    return gas, max(left, key=lambda item: item[1])[0]


def drive_two_phases(env: Any, result: dict[str, Any], guard: Callable[[], None]) -> None:
    """Issue two actions; retain each actual original packet and observed duration."""
    raw = _raw33(result["initial_raw33"])
    result["phases"] = []
    result["packets"] = []
    gas, left = select_actions(env._action_table)
    try:
        for name, action in (("gas", gas), ("mild_left", left)):
            guard()
            phase: dict[str, Any] = {
                "name": name,
                "action": action,
                "before_raw33": raw,
                "issued_gbs": _raw33_control(env._action_table[action]),
                "apply_start_host_ns": time.perf_counter_ns(),
            }
            result["phases"].append(phase)
            env.controller.apply_discrete(env._action_table[action].copy())
            phase["apply_finished_host_ns"] = time.perf_counter_ns()
            target = raw[3] + 1000
            while raw[3] < target:
                guard()
                if len(result["packets"]) >= MAX_PACKETS:
                    raise RuntimeError("original telemetry packet budget exhausted")
                before = raw
                raw = _raw33(env.client.read_next().values)
                if raw[3] < before[3]:
                    raise RuntimeError("original race clock regressed during diagnostic")
                result["packets"].append(
                    {"raw33": raw, "host_perf_counter_ns": time.perf_counter_ns(), "phase": name}
                )
            phase["after_raw33"] = raw
            phase["observed_race_delta_ms"] = raw[3] - phase["before_raw33"][3]
            phase["completed"] = True
    finally:
        env.release_controls()


def _raw33_control(value: Sequence[float]) -> list[float]:
    control = [float(item) for item in value]
    if len(control) != 3 or control[0] != 1 or control[1] != 0 or not -0.25 <= control[2] <= 0:
        raise ValueError("bounded gas/mild-left diagnostic control required")
    return control


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("existing guarded helper unavailable: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True, kw_only=True)
class ProbeRunContract:
    """An explicit independent probe plugged into the existing guarded runner."""

    client_factory: Callable[[], PhysicalProbeClient] = PhysicalProbeClient
    verify_capture: Callable[[dict[str, Any]], None] = verify_capture
    drive: Callable[[Any, dict[str, Any], Callable[[], None]], None] = drive_two_phases
    report_name: str = "physical005.json"
    diagnostic_schema: str = "tmrl.windows-physical-diagnostic/v1"

    def __post_init__(self) -> None:
        if Path(self.report_name).name != self.report_name or not self.report_name.endswith(
            ".json"
        ):
            raise ValueError("probe report must use a plain JSON filename")
        if self.report_name.casefold() in {
            "diagnostic.json",
            "original-connect.json",
            "input-profile-client.json",
            "preflight.json",
        }:
            raise ValueError("probe report collides with a reserved diagnostic filename")


def require_game_directory(game_dir: Path | None) -> Path:
    """Reject ambiguous or absent installations before any game operation."""
    if not isinstance(game_dir, Path) or not game_dir.is_absolute() or not game_dir.is_dir():
        raise ValueError("an existing absolute game installation directory is required")
    return game_dir


def run_live(  # noqa: PLR0913 - Explicit diagnostic paths and probe options.
    output: Path,
    *,
    game_dir: Path | None = None,
    no_artificial_lag: bool = False,
    probe_contract: ProbeRunContract | None = None,
) -> int:
    if sys.platform != "win32":
        raise RuntimeError("live diagnostic requires Windows")
    game_dir = require_game_directory(game_dir)
    paths = DiagnosticProbePaths.from_environment()
    import psutil

    from scripts.headless_input_profile import InputProfileConfig, NoArtificialLagLease
    from trackmaniarl.core.runtime import _instantiate
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.trackmania.keyboard_control import (
        ResetGuard,
        _focus_trackmania,
        restart_trackmania_editor_validation,
    )
    from trackmaniarl.trackmania.session import (
        OpenPlanetSessionClient,
        OpenPlanetSessionProtocolError,
    )
    from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig

    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    pre = _load(
        "physical005_preflight",
        paths.comparison_root / "experiments/tmrl_test_comparison/preflight.py",
    )
    lease = _load(
        "physical005_lease",
        paths.comparison_root / "experiments/tmrl_test_comparison/release_20261021_v4/lease.py",
    )
    contract = probe_contract or ProbeRunContract()
    client = contract.client_factory()
    report: dict[str, Any] = {
        "schema": contract.diagnostic_schema,
        "passed": False,
        "model_executed": False,
        "optimizer_steps": 0,
        "max_host_seconds": 90,
        "simulation_clock_or_input_ack_claimed": False,
        "controller_closed": False,
        "artificial_lag_profile_requested": no_artificial_lag,
    }
    telemetry: dict[str, Any] = {"schema": "tmrl.original-connect-stream/v1", "packets": []}
    profile = None
    try:
        with lease.controller_lease(), contextlib.ExitStack() as stack:
            me = psutil.Process()
            allowed = {me.pid}
            parent = me.parent()
            if parent is not None and Path(parent.exe()) == Path(sys.executable):
                allowed.add(parent.pid)
            foreign = [
                p.pid
                for p in psutil.process_iter(["name"])
                if p.pid not in allowed
                and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe")
            ]
            if foreign:
                raise RuntimeError("competing Python actor; game untouched: " + str(foreign))
            identities = {}
            for filename, expected in (
                ("Trackmania.exe", EXECUTABLE_SHA256),
                ("Openplanet.dll", OPENPLANET_SHA256),
            ):
                path = game_dir / filename
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                identities[filename] = actual
                if actual != expected:
                    raise RuntimeError("static hook evidence differs from installed " + filename)
            report["actual_binary_sha256"] = identities
            spec = RunSpec.from_yaml(paths.config)
            if (
                spec.components.environment is None
                or spec.evaluation is None
                or not spec.evaluation.maps
            ):
                raise ValueError("existing diagnostic config lacks an environment or map")
            factory = _instantiate(spec.components.environment)
            cfg = factory.config
            session = OpenPlanetSessionClient(cfg.host, cfg.session_port, timeout_s=2)
            session.verify_loaded_map(cfg.expected_map_uid)
            budget = pre._Budget(begin + 90, (output / "STOP",))
            try:
                session.confirm_ready(cfg.expected_map_uid)
            except OpenPlanetSessionProtocolError as error:
                if "player_not_ready" not in str(error) or cfg.restart_input != "editor_validation":
                    raise
                window = _focus_trackmania()
                if window is None:
                    raise RuntimeError("Trackmania window unavailable") from error
                restart_trackmania_editor_validation(
                    guard=ResetGuard(check=budget.remaining, wait=budget.pause)
                )
            pre.run_preflight(
                paths.config, receipt_path=output / "preflight.json", stop_files=(output / "STOP",)
            )
            window = _focus_trackmania()
            if window is None:
                raise RuntimeError("Trackmania window unavailable")
            if no_artificial_lag:
                profile = NoArtificialLagLease(
                    InputProfileConfig(
                        expected_map_uid=cfg.expected_map_uid,
                        ttl_seconds=60,
                        journal_path=output / "input-profile-client.json",
                    ),
                    controller_lease=contextlib.nullcontext,
                    verify_focus=window.verify,
                )
                stack.enter_context(profile)
            env = factory.create(seed=1729)

            def close_environment() -> None:
                env.close()
                report["controller_closed"] = True

            stack.callback(close_environment)
            env.geometry.validate_map(spec.evaluation.maps[0].map_path)
            env.client.close()
            env.client = pre._PreflightTelemetryClient(OpenPlanetClientConfig(), budget)
            raw, reset_info = env.reset(seed=1729)
            telemetry["initial_raw33"] = _raw33(raw)
            report["actual_reset_info"] = reset_info
            report["expected_map_uid"] = cfg.expected_map_uid
            session.verify_loaded_map(cfg.expected_map_uid)
            window.verify()
            drive_budget = pre._Budget(min(begin + 90, time.monotonic() + 5), (output / "STOP",))
            env.client._budget = drive_budget

            def guard() -> None:
                drive_budget.remaining()
                window.verify()
                if profile is not None:
                    profile.check_local()

            with client:
                contract.drive(env, telemetry, guard)
                physical = client.finish()
                contract.verify_capture(physical)
            if profile is not None:
                profile.check()
            report["passed"] = True
        report["controller_closed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        report["error_notes"] = getattr(error, "__notes__", [])
    finally:
        report["full_main_host_seconds"] = time.monotonic() - begin
        report["original_code_restored_verified"] = client.restored_verified
        report["probe_protocol_messages"] = client.messages
        report["probe_reader_error"] = repr(client._reader_error) if client._reader_error else None
        if profile is not None:
            report["input_profile_restored_verified"] = profile.restored_verified
        if client.report is not None:
            _save(output / contract.report_name, client.report)
        _save(output / "original-connect.json", telemetry)
        _save(output / "diagnostic.json", report)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--no-artificial-lag", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; import and tests never operate the game")
    return run_live(
        args.output.absolute(), game_dir=args.game_dir, no_artificial_lag=args.no_artificial_lag
    )


if __name__ == "__main__":
    raise SystemExit(main())
