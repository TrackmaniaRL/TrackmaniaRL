"""Bounded input/telemetry probe. No policy, replay or learner."""

import argparse
import contextlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

from scripts.headless_probe_paths import DiagnosticProbePaths
from trackmaniarl.core.runtime import _instantiate
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.keyboard_control import _focus_trackmania
from trackmaniarl.trackmania.parity import from_connect33
from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig


def load(name: str, path: Path) -> ModuleType:
    s = importlib.util.spec_from_file_location(name, path)
    if s is None or s.loader is None:
        raise ImportError(f"Cannot load the explicitly configured probe helper: {path}")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("live diagnostic requires Windows")
    paths = DiagnosticProbePaths.from_environment()
    output = args.output.resolve()
    import psutil

    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    pre = load(
        "parity_preflight", paths.comparison_root / "experiments/tmrl_test_comparison/preflight.py"
    )
    lease = load(
        "parity_lease",
        paths.comparison_root / "experiments/tmrl_test_comparison/release_20261021_v4/lease.py",
    )
    tapes = json.loads((output / "tapes.json").read_text())
    spec = RunSpec.from_yaml(paths.config)
    factory = _instantiate(spec.components.environment)
    report = {
        "passed": False,
        "cases": {},
        "max_host_seconds": 150,
        "model_executed": False,
        "optimizer_steps": 0,
        "comparison_clock_hypothesis": (
            "50 native clock units per N5 compared provisionally to "
            "50 client race milliseconds; SI/parity not established."
        ),
    }
    begin = time.monotonic()
    try:
        with lease.controller_lease(), contextlib.ExitStack() as stack:
            me = psutil.Process()
            allowed = {me.pid}
            parent = me.parent()
            if parent is not None and Path(parent.exe()) == Path(sys.executable):
                assert Path(__file__).resolve() in [
                    Path(a).resolve() for a in parent.cmdline() if not a.startswith("-")
                ]
                allowed.add(parent.pid)
            assert not [
                p.pid
                for p in psutil.process_iter(["name"])
                if p.pid not in allowed
                and p.info["name"]
                and p.info["name"].lower() in ["python.exe", "pythonw.exe"]
            ]
            pre.run_preflight(
                paths.config,
                receipt_path=output / "windows-preflight.json",
                stop_files=(output / "STOP",),
            )
            window = _focus_trackmania()
            assert window is not None
            env = factory.create(seed=1729)
            stack.callback(env.close)
            env.geometry.validate_map(spec.evaluation.maps[0].map_path)
            env.client.close()
            env.client = pre._PreflightTelemetryClient(
                OpenPlanetClientConfig(), pre._Budget(begin + 150, (output / "STOP",))
            )
            for name, actions in tapes.items():
                window.verify()
                if (output / "STOP").exists() or time.monotonic() > begin + 150:
                    raise TimeoutError("STOP/150s probe budget")
                raw, _ = env.reset(seed=1729)
                initial = raw.tolist()
                assert 0 <= float(raw[3]) <= 2500
                case = {
                    "initial_raw33": initial,
                    "common8": from_connect33(initial).tolist(),
                    "rows": [],
                }
                report["cases"][name] = case
                for i, action in enumerate(actions):
                    window.verify()
                    if (output / "STOP").exists() or time.monotonic() > begin + 150:
                        raise TimeoutError("STOP/150s probe budget")
                    before = raw.tolist()
                    then = time.monotonic_ns()
                    raw, _reward, terminated, truncated, info = env.step(action)
                    case["rows"].append(
                        {
                            "step": i,
                            "action": action,
                            "before_raw33": before,
                            "after_raw33": raw.tolist(),
                            "elapsed_race_ms": float(raw[3]) - initial[3],
                            "host_step_ns": time.monotonic_ns() - then,
                            "info": info,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    if terminated or truncated:
                        raise RuntimeError("Unexpected terminal during finite tape: " + name)
                env.release_controls()
                case["completed"] = True
                print(
                    json.dumps(
                        {
                            "case": name,
                            "steps": len(case["rows"]),
                            "end_position": raw[4:7].tolist(),
                        }
                    ),
                    flush=True,
                )
            report.update(
                passed=True,
                map_uid=factory.config.expected_map_uid,
                steering_curve=str(factory.config.steering_curve_path),
            )
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        traceback.print_exc()
    finally:
        report["host_seconds"] = time.monotonic() - begin
        with (output / "windows-probe.json").open("x", encoding="utf8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
