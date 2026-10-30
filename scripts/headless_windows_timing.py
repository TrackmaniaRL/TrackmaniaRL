"""Bounded Windows packet trace under the common controller mutex.

Records issued controls and observed API controls separately; no policy/training.
"""

import argparse
import contextlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from scripts.headless_input_profile import InputProfileConfig, NoArtificialLagLease
from scripts.headless_probe_paths import DiagnosticProbePaths
from scripts.headless_telemetry006 import (
    Telemetry006Client,
    Telemetry006Config,
    Telemetry006UnavailableError,
)
from trackmaniarl.core.runtime import _instantiate
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.keyboard_control import (
    ResetGuard,
    _focus_trackmania,
    restart_trackmania_editor_validation,
)
from trackmaniarl.trackmania.session import OpenPlanetSessionClient, OpenPlanetSessionProtocolError
from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig, TelemetryFrame


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local timing helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FreshProbeClient(Telemetry006Client):
    """Wait for real reset data within the probe budget, retaining unavailable replies."""

    def __init__(self, config: Telemetry006Config, budget: Any) -> None:
        super().__init__(config)
        self.budget = budget
        self.unavailable_samples: list[dict[str, Any]] = []

    def _remaining(self, deadline: float) -> float:
        return min(super()._remaining(deadline), float(self.budget.remaining()))

    def read(self) -> TelemetryFrame:
        while True:
            self.budget.remaining()
            try:
                frame = super().read()
                self.budget.remaining()
                return frame
            except Telemetry006UnavailableError:
                self.unavailable_samples.append(dict(self.last_metadata or {}))
                self.budget.pause(0.005)


def _packet(raw: np.ndarray, client: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"raw33": raw.tolist(), "host_ns": time.perf_counter_ns()}
    if isinstance(client, Telemetry006Client):
        if client.last_metadata is None or client.last_packet is None:
            raise ValueError("fresh telemetry evidence required for each returned sample")
        if raw.astype("<f4", copy=False).tobytes() != client.last_packet:
            raise ValueError("fresh telemetry metadata does not bind the returned sample")
        result["fresh_metadata"] = client.last_metadata
        result["raw33_le_hex"] = client.last_packet.hex()
    return result


def main() -> int:
    # Process enumeration is needed only by the opt-in live Windows probe.
    # Offline packet helpers and their tests do not require this extra package.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("tapes", type=Path)
    parser.add_argument("--no-artificial-lag", action="store_true")
    parser.add_argument("--fresh-telemetry-port", type=int)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("live diagnostic requires Windows")
    import psutil

    paths = DiagnosticProbePaths.from_environment()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    pre = load(
        "timing_preflight", paths.comparison_root / "experiments/tmrl_test_comparison/preflight.py"
    )
    lease = load(
        "timing_lease",
        paths.comparison_root / "experiments/tmrl_test_comparison/release_20261021_v4/lease.py",
    )
    tapes = json.loads(args.tapes.read_bytes())
    spec = RunSpec.from_yaml(paths.config)
    environment_component, evaluation = spec.components.environment, spec.evaluation
    if environment_component is None or evaluation is None or not evaluation.maps:
        raise ValueError("explicit live environment and evaluation map required")
    factory = _instantiate(environment_component)
    report = {
        "passed": False,
        "cases": {},
        "rows": [],
        "training": False,
        "controller_closed": False,
        "artificial_lag_profile_requested": args.no_artificial_lag,
        "host_timestamp_source": "perf_counter_ns",
        "telemetry_transport": (
            "request-response006" if args.fresh_telemetry_port is not None else "Connect-stream"
        ),
        "simulation_phase_qualified": False,
    }
    profile = None
    begin = time.monotonic()
    try:
        with lease.controller_lease(), contextlib.ExitStack() as stack:
            # Preserve the earlier no-competing-Python preflight guard in addition
            # to the global mutex. Never stop or bypass another actor.
            me = psutil.Process()
            allowed = {me.pid}
            parent = me.parent()
            if parent and Path(parent.exe()) == Path(sys.executable):
                allowed.add(parent.pid)
            foreign = [
                p.pid
                for p in psutil.process_iter(["name"])
                if p.pid not in allowed
                and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe")
            ]
            if foreign:
                raise RuntimeError(
                    "other Python processes present; probe left game untouched: " + str(foreign)
                )
            # The previous bounded validation can leave the same map open in
            # the editor with no vehicle. Re-enter its configured validation
            # mode once, through the standard guarded reset helper.
            cfg = factory.config
            session = OpenPlanetSessionClient(cfg.host, cfg.session_port, timeout_s=2)
            session.verify_loaded_map(cfg.expected_map_uid)
            try:
                session.confirm_ready(cfg.expected_map_uid)
            except OpenPlanetSessionProtocolError as error:
                if "player_not_ready" not in str(error) or cfg.restart_input != "editor_validation":
                    raise
                report["initial_player_not_ready"] = str(error)
                window = _focus_trackmania()
                if window is None:
                    raise RuntimeError(
                        "Trackmania window unavailable before validation reset"
                    ) from error
                budget = pre._Budget(begin + 30, (output / "STOP",))
                restart_trackmania_editor_validation(
                    guard=ResetGuard(check=budget.remaining, wait=budget.pause)
                )
            pre.run_preflight(
                paths.config, receipt_path=output / "preflight.json", stop_files=(output / "STOP",)
            )
            window = _focus_trackmania()
            if window is None:
                raise RuntimeError("Trackmania window unavailable")
            if args.no_artificial_lag:
                profile = NoArtificialLagLease(
                    InputProfileConfig(
                        expected_map_uid=cfg.expected_map_uid,
                        ttl_seconds=90,
                        journal_path=output / "input-profile-client.json",
                    ),
                    controller_lease=lease.controller_lease,
                    verify_focus=window.verify,
                )
                stack.enter_context(profile)
                report["input_profile_baseline"] = profile.baseline
            env = factory.create(seed=1729)
            stack.callback(env.close)
            env.geometry.validate_map(evaluation.maps[0].map_path)
            env.client.close()
            budget = pre._Budget(begin + 120, (output / "STOP",))
            if args.fresh_telemetry_port is None:
                env.client = pre._PreflightTelemetryClient(OpenPlanetClientConfig(), budget)
            else:
                env.client = _FreshProbeClient(
                    Telemetry006Config(port=args.fresh_telemetry_port), budget
                )
                report["fresh_unavailable_replies"] = env.client.unavailable_samples
            for name, actions in {"neutral_dense": [], **tapes}.items():
                window.verify()
                if profile is not None:
                    profile.check()  # Remote confirmation outside the measured decision loop.
                raw, reset_info = env.reset(seed=1729)
                initial = raw.tolist()
                case = {
                    "initial_raw33": initial,
                    "reset_info": reset_info,
                    "initial_packet": _packet(raw, env.client),
                    "rows": [],
                    "packets": [],
                }
                report["cases"][name] = case
                if not actions:
                    case["packets"].append(_packet(raw, env.client))
                    target = initial[3] + 650
                    while float(raw[3]) < target:
                        window.verify()
                        if profile is not None:
                            profile.check_local()
                        raw = env.client.read_next().values
                        case["packets"].append(_packet(raw, env.client))
                    report["rows"] = case["packets"]
                for action in actions:
                    window.verify()
                    if profile is not None:
                        profile.check_local()  # Foreground/TTL only; no RPC delay before input.
                    if (output / "STOP").exists() or time.monotonic() - begin > 120:
                        raise TimeoutError("STOP / 120-second timing budget")
                    before = raw.tolist()
                    issued = env._action_table[action].copy()
                    applied_host_ns = time.perf_counter_ns()
                    env.controller.apply_discrete(issued)
                    apply_finished_ns = time.perf_counter_ns()
                    target = before[3] + 50
                    packets = []
                    while float(raw[3]) < target:
                        window.verify()
                        raw = env.client.read_next().values
                        packets.append(_packet(raw, env.client))
                    case["rows"].append(
                        {
                            "action": action,
                            "before_raw33": before,
                            "after_raw33": raw.tolist(),
                            "issued_gbs": issued.tolist(),
                            "apply_start_host_ns": applied_host_ns,
                            "apply_finished_host_ns": apply_finished_ns,
                            "packets": packets,
                        }
                    )
                env.release_controls()
                if profile is not None:
                    profile.check()
                case["completed"] = True
                print(
                    json.dumps(
                        {
                            "case": name,
                            "decisions": len(case["rows"]),
                            "packets": len(case["packets"])
                            + sum(len(r["packets"]) for r in case["rows"]),
                        }
                    ),
                    flush=True,
                )
            report["passed"] = True
        report["controller_closed"] = True
    except BaseException as error:
        report.update(passed=False, error=repr(error))
        traceback.print_exc()
    finally:
        if profile is not None:
            report["input_profile_applied"] = profile.applied
            report["input_profile_restored_verified"] = profile.restored_verified
        report["full_main_host_seconds"] = time.monotonic() - begin
        (output / "windows-timing.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
