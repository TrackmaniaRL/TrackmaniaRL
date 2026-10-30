"""At most180s: validate real RL updates and post-update native full starts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tools.record_rl1168.paths import private_base
from tools.record_rl1168.provenance import (
    load_verified_backend,
    verify_backend,
    verify_preparation,
    verify_preparation_deployment,
    verify_qualification_binding,
    verify_source_pins,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "FAILED",
        "bounded_functional_test": True,
        "RL_updates": 0,
        "training_episodes": [],
        "evaluation": [],
        "public_WR_claim": False,
    }
    try:
        proof = json.loads((args.qualification / "result.json").read_text())
        plan, receipt, pins, plan_sha, pins_sha, receipt_sha = verify_preparation(args.preparation)
        if (
            proof["status"] != "PASS_FULL_START_BC"
            or proof["cleanupErrors"]
            or proof["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        ):
            raise ValueError("three clean full-start native initializer trials required")
        if len(proof["trials"]) != 3:
            raise ValueError("three initializer trials required")
        for name, key in (
            ("initialization.pt", "checkpoint_sha256"),
            ("route-geometry.npy", "route_sha256"),
            ("demonstrations-TRAINING-ONLY.npz", "dataset_sha256"),
        ):
            if hashlib.sha256((args.preparation / name).read_bytes()).hexdigest() != receipt[key]:
                raise ValueError("preparation changed: " + name)
        verify_source_pins(pins, Path(__file__).parent)
        config = dict(plan["backend_config"])
        verified = verify_backend(plan)
        from tools.record_rl1168.deployment import activate_from_environment

        deployment = activate_from_environment()
        verify_preparation_deployment(receipt, deployment["deployment_manifest_sha256"])
        verify_qualification_binding(
            proof, plan_sha, pins_sha, receipt_sha, deployment["deployment_manifest_sha256"]
        )
        import numpy as np
        import torch

        from tools.record_rl1168.episodes import episode
        from tools.record_rl1168.learner import Learner
        from tools.record_rl1168.runtime import isolate_runtime

        torch.set_num_threads(2)
        torch.manual_seed(1168)
        private = private_base() / (args.preparation.name + "-" + args.output.name)
        private.mkdir(mode=0o700, exist_ok=False)
        config = isolate_runtime(config, private)
        load_verified_backend(verified)
        initial_path = args.preparation / "initialization.pt"
        if hashlib.sha256(initial_path.read_bytes()).hexdigest() != receipt["checkpoint_sha256"]:
            raise ValueError("initializer changed")
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        learner = Learner(initial, device="cpu")
        learner.configure_demonstrations(args.preparation / "demonstrations-TRAINING-ONLY.npz")
        geometry = np.load(args.preparation / "route-geometry.npy", allow_pickle=False)
        for number, epsilon in enumerate((0.0, 0.005)):
            learner.begin_episode()
            row = episode(
                config,
                args.output / f"train-{number}",
                geometry,
                lambda state, epsilon=epsilon: learner.action(state, epsilon),
                deadline=started + 180,
                training=True,
                on_transition=learner.add,
            )
            report["training_episodes"].append(row)
        metrics = []
        for _update in range(16):
            if time.monotonic() - started >= 150:
                raise TimeoutError("functional update budget")
            metrics.append(learner.update())
        learner.sync_actor()
        saved = learner.checkpoint()
        torch.save(saved, args.output / "smoke-RL.pt")
        before = initial["state"]
        after = saved["state"]
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        if not any("encoder" in k for k in changed) or not any("head" in k for k in changed):
            raise RuntimeError("missing actual full-policy update")
        # Stop all learning; this separate evaluation uses only the frozen actor.
        checksum = hashlib.sha256((args.output / "smoke-RL.pt").read_bytes()).hexdigest()
        for number in range(2):
            report["evaluation"].append(
                episode(
                    config,
                    args.output / f"eval-{number}",
                    geometry,
                    lambda state: learner.action(state, 0.0),
                    deadline=started + 180,
                )
            )
        if hashlib.sha256((args.output / "smoke-RL.pt").read_bytes()).hexdigest() != checksum:
            raise ValueError("evaluation changed checkpoint")
        report.update(
            status="PASS_REAL_RL_FUNCTIONAL_TEST",
            RL_updates=saved["RL_updates"],
            auxiliary_imitation_updates=saved["auxiliary_imitation_updates"],
            new_real_transitions=saved["transitions"],
            changed_tensor_names=changed,
            checkpoint_sha256=checksum,
            learner_updates_during_evaluation=0,
            training_seconds_authorized=0,
        )
    except Exception as error:
        report["error"] = repr(error)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("training_episodes", "evaluation", "changed_tensor_names")
                }
            )
        )
        print(
            json.dumps(
                [
                    {"status": r["status"], "goal": r.get("actual_goal_ms")}
                    for r in report["evaluation"]
                ]
            )
        )


if __name__ == "__main__":
    main()
