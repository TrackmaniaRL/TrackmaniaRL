"""Bounded, CPU, full-start autonomous native qualification of a frozen policy."""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from tools.record_rl1168.closure import native_closure_clean
from tools.record_rl1168.provenance import (
    load_verified_backend,
    verify_backend,
    verify_preparation,
    verify_preparation_deployment,
    verify_source_pins,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linux_path(path):
    """Translate an absolute Windows drive path or accept an absolute Linux path."""
    raw = str(path)
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise ValueError("path contains an invalid character")
    drive = re.fullmatch(r"([a-zA-Z]):[\\/](.*)", raw)
    if drive:
        raw = "/mnt/" + drive.group(1).lower() + "/" + drive.group(2).replace("\\", "/")
    elif not raw.startswith("/"):
        raise ValueError("absolute Linux or Windows drive path required")
    if ".." in raw.split("/"):
        raise ValueError("parent traversal is not accepted")
    return raw


def close_qualification_episode(env, report, row, trial):
    """Record every closure failure and require a fully clean native supervisor."""
    errors = report["cleanupErrors"]
    before = len(errors)
    try:
        env.close()
    except Exception as error:
        errors.append({"trial": trial, "stage": "env.close", "error": repr(error)})
    try:
        closure = json.loads((env.settings.trace_dir / "native-supervisor.json").read_text())
        row["closure"] = closure
        if not native_closure_clean(closure):
            errors.append({"trial": trial, "stage": "supervisor", "closure": closure})
    except Exception as error:
        errors.append({"trial": trial, "stage": "supervisor", "error": repr(error)})
    if len(errors) != before:
        raise RuntimeError("qualification native closure failed")


def qualification_status(report):
    if report["cleanupErrors"] or not all(
        trial["status"] == "NATIVE_GOAL" and trial["actual_goal_ms"] == 17601
        for trial in report["trials"]
    ):
        return "BC_NOT_QUALIFIED"
    if len(report["trials"]) == 3:
        return "PASS_FULL_START_BC"
    if len(report["trials"]) == 1:
        return "PASS_ONE_FULL_START_BC"
    return "BC_NOT_QUALIFIED"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trials", type=int, default=3, choices=(1, 3))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    receipt = {}
    report = {
        "status": "FAILED",
        "policy_kind": "BC_ONLY",
        "RL_updates": 0,
        "trials": [],
        "cleanupErrors": [],
        "teacher_control_reads": False,
        "static_reference_geometry": True,
        "public_WR_claim": False,
    }
    started = time.monotonic()
    try:
        plan, receipt, pins, plan_sha, pins_sha, receipt_sha = verify_preparation(args.preparation)
        report.update(
            plan_sha256=plan_sha, source_pins_sha256=pins_sha, preparation_sha256=receipt_sha
        )
        checkpoint = args.preparation / "initialization.pt"
        if (
            sha(checkpoint) != receipt["checkpoint_sha256"]
            or sha(args.preparation / "route-geometry.npy") != receipt["route_sha256"]
        ):
            raise ValueError("initializer/route changed")
        verify_source_pins(pins, Path(__file__).parent)
        config = dict(plan["backend_config"])
        verified = verify_backend(plan)
        from tools.record_rl1168.deployment import activate_from_environment

        deployment = activate_from_environment()
        verify_preparation_deployment(receipt, deployment["deployment_manifest_sha256"])
        report["deployment_manifest_sha256"] = deployment["deployment_manifest_sha256"]
        from tools.record_rl1168.paths import private_base

        private = private_base() / (args.preparation.name + "-" + args.output.name)
        private.mkdir(mode=0o700, exist_ok=False)
        from tools.record_rl1168.runtime import isolate_runtime

        config = isolate_runtime(config, private)
        import numpy as np
        import torch

        fall_environment = load_verified_backend(verified)

        from tools.record_rl1168.features import RouteFeatures
        from tools.record_rl1168.model import ACTIONS, build_model
        from trackmaniarl.models.contracts import RiskSpec, ValuePhase

        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["RL_updates"] != 0:
            raise ValueError("this admission is for BC initialization only")
        model = build_model()
        model.load_state_dict(saved["state"])
        model.eval()
        geometry = np.load(args.preparation / "route-geometry.npy", allow_pickle=False)

        def no_teacher(event, parameters):
            if event != "open" or not isinstance(parameters[0], (str, bytes)):
                return
            mode = parameters[1] if len(parameters) > 1 else None
            if isinstance(mode, str) and mode.startswith(("w", "a", "x")):
                return
            if any(
                word in str(parameters[0]).lower()
                for word in ("training-only", "source-controls", "actual-rollout.npz")
            ):
                raise RuntimeError("teacher-control read in frozen evaluation")

        sys.addaudithook(no_teacher)
        for trial in range(args.trials):
            env = None
            row = {"trial": trial, "status": "FAILED", "actual_goal_ms": None}
            report["trials"].append(row)
            begin = time.monotonic()
            try:
                env = fall_environment(config, args.output / f"trial-{trial}")
                env.settings = replace(
                    env.settings, trace_dir=private / "traces" / f"trial-{trial}"
                )
                route = RouteFeatures(geometry)
                state, _, _ = route.observe(env.reset(), reset=True)
                for step in range(3000):
                    if time.monotonic() - started > 120:
                        raise TimeoutError("qualification total120s budget")
                    with torch.no_grad():
                        tensor = torch.from_numpy(state[None])
                        features = model.encode_frames(tensor)
                        values = model.expected_all_actions(
                            features, model.support(features, ValuePhase.EVALUATE), RiskSpec()
                        )
                        action = int(values.argmax(-1))
                    native, _, goal, truncated, info = env.step(ACTIONS[action])
                    state, progress, lateral = route.observe(native)
                    row.update(
                        steps=step + 1,
                        progress_m=progress,
                        lateral_m=lateral,
                        race_ms=info["race_ms"],
                    )
                    if goal or truncated:
                        row.update(
                            status="NATIVE_GOAL" if goal else "NO_GOAL",
                            actual_goal_ms=info["actual_goal_ms"],
                        )
                        break
            finally:
                try:
                    if env:
                        close_qualification_episode(env, report, row, trial)
                finally:
                    row["elapsed_seconds"] = time.monotonic() - begin
                    (args.output / "result.json").write_text(json.dumps(report, indent=2))
        if sha(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("checkpoint mutated")
        report["status"] = qualification_status(report)
    except Exception as error:
        report["error"] = repr(error)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["checkpoint_sha256"] = receipt.get("checkpoint_sha256")
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "trials"}))
        print(
            json.dumps(
                [
                    {
                        "trial": r["trial"],
                        "status": r["status"],
                        "goal": r.get("actual_goal_ms"),
                        "steps": r.get("steps"),
                        "elapsed": r.get("elapsed_seconds"),
                    }
                    for r in report["trials"]
                ]
            )
        )


if __name__ == "__main__":
    main()
