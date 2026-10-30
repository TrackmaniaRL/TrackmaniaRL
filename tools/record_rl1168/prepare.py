"""Prepare a fresh bounded BC initializer and immutable pipeline contract."""

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path


def read(path):
    path = Path(path).absolute()
    if os.name == "nt":
        path = Path("\\\\?\\" + str(path))
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=60, choices=(30, 60, 120))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    report = {"status": "FAILED_PREPARATION", "RL_updates": 0}
    try:
        from tools.record_rl1168.deployment import activate_from_environment

        deployment = activate_from_environment()
        report["deployment_manifest_sha256"] = deployment["deployment_manifest_sha256"]
        import numpy as np
        import torch

        from tools.record_rl1168.features import RouteFeatures
        from tools.record_rl1168.model import ACTIONS, build_model
        from tools.vision_qr1167.quantile_initialization import initialize_point_mass
        from trackmaniarl.models.contracts import RiskSpec, ValuePhase

        proof = json.loads((args.previous / "qualification-source/result.json").read_text())
        closure = proof["supervisor"]
        if (
            proof["info"]["actual_goal_ms"] != 17601
            or closure["returncode"] != 0
            or closure["remainingOwned"]
            or closure["cleanupErrors"]
        ):
            raise ValueError("clean source NativeGoal required")
        for filename in ("qualify-supervision.json", "train-supervision.json"):
            record = json.loads((args.previous / filename).read_text())["closure"]
            if any(record[k] for k in ("remainingOwned", "cleanupErrors", "forced")):
                raise ValueError("previous processes did not close cleanly")
        if (
            json.loads((args.previous / "power-guard-state.json").read_text())["phase"]
            != "released"
        ):
            raise ValueError("previous power guard still active")
        source = args.previous / "qualification-source/actual-rollout.npz"
        data = np.load(io.BytesIO(read(source)), allow_pickle=False)
        native = data["observations"]
        controls = data["controls"]
        if (
            len(native) != len(controls) + 1
            or native.shape[1] != 22
            or not np.isfinite(native).all()
        ):
            raise ValueError("source state/control alignment")
        if not np.allclose(np.diff(native[:, 0]) * 30000, 10, atol=0.005):
            raise ValueError("retained source is not a contiguous 10ms rollout")
        geometry = native[:, 10:13].astype(np.float64) * 1000
        np.save(args.output / "route-geometry.npy", geometry, allow_pickle=False)
        route = RouteFeatures(geometry)
        observations = np.stack([route.observe(s)[0] for s in native[:-1]])
        labels = []
        for control in controls:
            matches = [
                i
                for i, a in enumerate(ACTIONS)
                if np.array_equal(np.asarray(a, dtype=np.float32), control)
            ]
            if len(matches) != 1:
                raise ValueError("demonstrated action not exactly in policy catalog")
            labels.append(matches[0])
        labels = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(
            args.output / "demonstrations-TRAINING-ONLY.npz",
            observations=observations,
            actions=labels,
        )
        backend = json.loads((args.previous / "environment.json").read_text())
        plan = {
            "schema": "full-control-native-QR1168/v1",
            "map_uid": backend["map_uid"],
            "backend_config": backend,
            "backend_source_manifest": str(args.previous / "source-manifest.json"),
            "backend_source_manifest_sha256": sha(args.previous / "source-manifest.json"),
            "source_rollout": str(source),
            "source_rollout_sha256": sha(source),
            "source_native_goal_ms": 17601,
            "comparison_threshold_ms": 17712,
            "RL_improvement_target_ms": 17600,
            "physics_tick_ms": 10,
            "decision_ms": 10,
            "action_catalog": ACTIONS,
            "quantiles": 64,
            "observations": "20 current physical sensors, one constant reserved slot, 24 body-frame reference preview coordinates, spatial progress and lateral distance",
            "route_disclosure": "static geometry derived from measured demonstration; available in training and evaluation; no reference clocks or controls",
            "policy_race_clock": False,
            "vision": False,
            "frozen_motor_base": False,
            "late_residual_gate": False,
            "curriculum": "50% full starts; 50% real training-only prefixes at 2..16s, followed by measured perturbations",
            "reward": "gamma=1 finite attempts; potential difference .05*arc_progress minus actual dt; terminal potential zero; Goal +100; failure -100",
            "RL_minibatch": 128,
            "replay_capacity": 20000,
            "n_step": 50,
            "updates_per_new_transition": 0.5,
            "evaluation": "frozen checkpoint; separate full starts; no teacher control files; first greedy then99 fixed seeded epsilon=.005 trials; disclose all outcomes",
            "gates": [
                "native_source_identity",
                "exact_command_and_10ms_clock",
                "BC_full_start_qualification",
                "real_reward_driven_parameter_updates",
                "full_start_native_goal",
                "independent_replay_validation",
                "Windows_TICK_model_validation",
            ],
            "training_authorized_until": None,
            "long_training_started": False,
            "public_WR_claim": False,
        }
        save(args.output / "plan.json", plan)
        code = Path(__file__).parent
        save(args.output / "source-pins.json", {p.name: sha(p) for p in code.glob("*.py")})
        report["plan_sha256"] = sha(args.output / "plan.json")
        report["source_pins_sha256"] = sha(args.output / "source-pins.json")
        torch.manual_seed(1168)
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model().to(device)
        x = torch.from_numpy(observations).to(device)
        y = torch.from_numpy(labels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        def logits():
            features = model.encode_frames(x)
            return model.expected_all_actions(
                features, model.support(features, ValuePhase.EVALUATE), RiskSpec()
            )

        started = time.monotonic()
        updates = 0
        errors = len(y)
        while time.monotonic() - started < args.seconds and updates < 6000:
            q = logits()
            loss = torch.nn.functional.cross_entropy(q, y) + 0.01 * q.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates % 100 == 0:
                with torch.no_grad():
                    q = logits()
                    errors = int((q.argmax(-1) != y).sum())
                    correct = q.gather(1, y[:, None]).squeeze(-1)
                    other = q.clone().scatter_(1, y[:, None], -torch.inf).max(-1).values
                    margin = float((correct - other).min())
                if errors == 0 and margin >= 2:
                    break
        with torch.no_grad():
            q = logits()
            errors = int((q.argmax(-1) != y).sum())
        initialize_point_mass(model.head)
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("nonfinite initializer")
        torch.save(
            {"schema": plan["schema"], "state": weights, "RL_updates": 0, "BC_updates": updates},
            args.output / "initialization.pt",
        )
        report.update(
            status="BC_INITIALIZER_READY_NATIVE_QUALIFICATION_PENDING",
            BC_updates=updates,
            source_rows=len(labels),
            training_action_errors=errors,
            training_action_accuracy=1 - errors / len(labels),
            held_out_driving_evaluation=False,
            elapsed_seconds=time.monotonic() - started,
            device=device,
            checkpoint_sha256=sha(args.output / "initialization.pt"),
            route_sha256=sha(args.output / "route-geometry.npy"),
            dataset_sha256=sha(args.output / "demonstrations-TRAINING-ONLY.npz"),
        )
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save(args.output / "preparation.json", report)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
