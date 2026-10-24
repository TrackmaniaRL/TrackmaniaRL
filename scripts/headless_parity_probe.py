"""Finite real-engine probes; outputs stay on ext4, no optimizer or policy."""

import json
import resource
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime import ownership
from trackmaniarl.trackmania.native_headless.environment import NativeHeadlessEnvironmentFactory
from trackmaniarl.trackmania.native_headless.training_contract import qualify_raw_transition
from trackmaniarl.trackmania.parity import from_native, verify_applied


def main() -> int:
    config_file, output = map(Path, sys.argv[1:3])
    output = output.absolute()
    config = json.loads(config_file.read_bytes())
    output.mkdir(mode=0o700)
    start = time.monotonic()
    report = {
        "passed": False,
        "sdk": "1.2.14",
        "cases": [],
        "model_executed": False,
        "optimizer_steps": 0,
        "clock_mapping_qualified": False,
    }
    cases = [
        ("legacy-reset", "legacy-gas-prelude/v1", 5, [36] * 10),
        ("neutral-reset", "neutral-prelude/v1", 5, [36] * 10),
        ("all78-n1", "neutral-prelude/v1", 1, [*range(78), 36]),
        ("all78-n5", "neutral-prelude/v1", 5, [*range(78), 36]),
        ("neutral", "neutral-prelude/v1", 5, [36] * 10),
        ("straight_gas", "neutral-prelude/v1", 5, [39] * 40),
        ("mild_left", "neutral-prelude/v1", 5, [39] * 20 + [33] * 20),
        ("full_brake", "neutral-prelude/v1", 5, [39] * 20 + [40] * 20),
        ("brake_tap", "neutral-prelude/v1", 5, [39] * 20 + [41] * 20),
        ("actor-exception", "neutral-prelude/v1", 5, [39] * 3),
        ("transport-failure", "neutral-prelude/v1", 5, [39] * 3),
        ("cooperative-stop", "neutral-prelude/v1", 5, [39] * 3),
    ]
    cases.append(("repeat-reset", "neutral-prelude/v1", 5, [36] * 3))
    try:
        for name, profile, intervals, actions in cases:
            if (output / "STOP").exists() or time.monotonic() - start > 210:
                raise TimeoutError("STOP / probe soft wall budget")
            settings = {
                **config,
                "trace_dir": str(output / name),
                "reset_profile": profile,
                "duration_limit": 5000 if name.startswith("all78") else 2000,
            }
            env = NativeHeadlessEnvironmentFactory(settings).create(seed=1729)
            case = {
                "name": name,
                "reset_profile": profile,
                "intervals": intervals,
                "rows": [],
                "passed": False,
            }
            report["cases"].append(case)
            session = None
            then = time.monotonic()
            try:
                before, info = env.reset(seed=1729)
                case["reset_host_seconds"] = time.monotonic() - then
                case["initial"] = before
                case["reset_info"] = info
                case["initial_common8"] = from_native(before).tolist()
                session = env._backend._env.session
                if name == "repeat-reset":
                    before2, _ = env.reset(seed=1729)
                    assert before2["native76Hex"] == before["native76Hex"]
                    assert before2["pendingSGB"] == before["pendingSGB"]
                    assert before2["historyActionIds"] == []
                    before = before2
                    case["repeated_reset_equal"] = True
                if name == "transport-failure":
                    session.connection.close()
                for action in actions:
                    if (output / "STOP").exists() or time.monotonic() - start > 210:
                        raise TimeoutError("STOP / probe soft wall budget during tape")
                    step_start = time.monotonic()
                    result = env.step_native(action, native_iterations=intervals)
                    qualify_raw_transition(before, action, result)
                    schedule = verify_applied(before, action, result)
                    case["rows"].append(
                        {
                            "before": before,
                            "action": action,
                            "result": result,
                            "schedule": asdict(schedule),
                            "host_seconds": time.monotonic() - step_start,
                        }
                    )
                    before = result["observation"]
                    if result["terminated"] or result["truncated"]:
                        break
                if name == "actor-exception":
                    raise RuntimeError("injected actor failure after real transitions")
                if name == "cooperative-stop":
                    (output / name / "STOP").write_text("cooperative test stop\n")
                    if (output / name / "STOP").exists():
                        raise InterruptedError("cooperative stop without synthetic transition")
                if name.startswith("all78"):
                    assert len(case["rows"]) == 79, (
                        "all action IDs plus final pending flush required"
                    )
                case["passed"] = True
            except BaseException as error:
                case["error"] = repr(error)
                expected = (
                    (name == "actor-exception" and type(error) is RuntimeError)
                    or (name == "cooperative-stop" and type(error) is InterruptedError)
                    or (
                        name == "transport-failure"
                        and "NativeInconclusiveError" in type(error).__name__
                    )
                )
                if not expected:
                    traceback.print_exc()
                    raise
                case["expected_failure_observed"] = True
                case["passed"] = True
            finally:
                env.close()
                close_path = env.config.trace_dir / "env002-close.json"
                if session is not None:
                    assert close_path.is_file(), "owned close receipt is required"
                if close_path.exists():
                    case["close"] = json.loads(close_path.read_bytes())
                    assert case["close"]["cleanupError"] is None
                    assert case["close"]["returncode"] == 0
                    assert case["close"]["cleanupPidfdSignals"] == []
                if session is not None:
                    case["owned_identities"] = list(session.known_owned.values())
                    case["remaining_owned"] = [
                        p
                        for p in session.known_owned.values()
                        if ownership.same_identity(p, ownership.process_identity(p["pid"]))
                    ]
                    assert not case["remaining_owned"]
                case["full_case_host_seconds"] = time.monotonic() - then
                (output / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
            print(
                json.dumps({"case": name, "passed": case["passed"], "steps": len(case["rows"])}),
                flush=True,
            )
        report["passed"] = all(case["passed"] for case in report["cases"])
    except BaseException as error:
        report["error"] = repr(error)
    finally:
        report["full_host_seconds"] = time.monotonic() - start
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["disk_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
