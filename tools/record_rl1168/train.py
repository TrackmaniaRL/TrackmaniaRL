"""Full-control QR training; a NEW explicit UTC budget is mandatory."""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tools.record_rl1168.paths import check_disk_reserves, private_base
from tools.record_rl1168.provenance import (
    load_verified_backend,
    verify_backend,
    verify_preparation,
    verify_preparation_deployment,
    verify_qualification_binding,
    verify_source_pins,
)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_recorder(target):
    """Collect five-field transitions until the episode passes all guards."""

    def record(*row):
        target.append(row)

    return record


def commit_complete_episode(result, transitions, learner):
    """Commit only validated, terminal episodes; never invent a terminal step."""
    if result.get("status") not in ("NATIVE_GOAL", "DRIVING_FAILURE"):
        return False
    if transitions and not transitions[-1][-1]:
        raise ValueError("complete episode is missing its terminal transition")
    for row in transitions:
        learner.add(*row)
    return True


def evaluation_complete(rows, expected):
    """A budget-stopped attempt cannot count as a completed evaluation."""
    return len(rows) == expected and all(
        row.get("status") != "STOPPED_BUDGET_OR_OPERATOR" for row in rows
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--until-utc", required=True)
    parser.add_argument("--seed", type=int, default=1168)
    parser.add_argument(
        "--retention-weight",
        type=float,
        default=2.0,
        help="Ablation only: 2 baseline, .2 weak retention, 0 pure TD",
    )
    parser.add_argument(
        "--functional-test",
        action="store_true",
        help="hard cap180s,2training episodes,32updates,2evals",
    )
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--evaluation-reserve", type=int, default=1200)
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("positive update cap required")
    if args.evaluation_reserve < 60:
        raise ValueError("evaluation reserve too short")
    end = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
    if end.tzinfo is None:
        raise ValueError("UTC offset required")
    budget = (end - datetime.now(UTC)).total_seconds()
    if args.functional_test:
        if not 60 <= budget <= 180:
            raise ValueError("functional test requires60..180s")
    elif not (600 if args.evaluation_only else 1800) <= budget <= 36000:
        raise ValueError("new explicit budget must be30min–10h; old budget is never extended")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + budget
    training_end = deadline - (60 if args.functional_test else args.evaluation_reserve)
    report = {
        "status": "preparing",
        "public_WR_claim": False,
        "deadline_utc": end.isoformat(),
        "budget_seconds": budget,
        "training_episodes": 0,
        "new_real_transitions": 0,
        "RL_updates": 0,
        "best_full_start_goal_ms": None,
        "initialization_goal_ms": 17601,
        "evaluation": [],
        "bounded_functional_test": args.functional_test,
        "retention_contract": {
            "optimizer_steps_per_update": 1,
            "learning_rate": 2e-6,
            "anchor_weight": args.retention_weight,
            "anchor_margin": 2.0,
            "anchor_rows": "all1761",
            "updates_per_transition": 0.05,
            "max_updates_per_episode": 64,
            "candidate_gate": "greedy full-start clean NativeGoal<=accepted time before actor replacement",
            "candidate_check_every_episodes": 2,
            "initial_actor_RL_updates": 0,
        },
    }
    learner = None
    private = None
    try:
        proof = json.loads((args.qualification / "result.json").read_text())
        plan, receipt, pins, plan_sha, pins_sha, receipt_sha = verify_preparation(args.preparation)
        if (
            proof["status"] != "PASS_FULL_START_BC"
            or proof["cleanupErrors"]
            or len(proof["trials"]) != 3
        ):
            raise ValueError("three clean initializer trials required")
        for name, key in [
            ("initialization.pt", "checkpoint_sha256"),
            ("route-geometry.npy", "route_sha256"),
            ("demonstrations-TRAINING-ONLY.npz", "dataset_sha256"),
        ]:
            if sha(args.preparation / name) != receipt[key]:
                raise ValueError("preparation changed: " + name)
        if proof["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
            raise ValueError("qualified checkpoint mismatch")
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

        from tools.record_rl1168.archive import archive_closed_trace
        from tools.record_rl1168.episodes import InvalidProgress, episode
        from tools.record_rl1168.learner import Learner
        from tools.record_rl1168.model import ACTIONS
        from tools.record_rl1168.runtime import isolate_runtime

        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        private = private_base() / (
            args.output.parent.name
            + "-"
            + args.output.name
            + "-"
            + hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:10]
        )
        private.mkdir(mode=0o700, exist_ok=False)
        config = isolate_runtime(config, private)
        load_verified_backend(verified)
        initial = torch.load(
            args.preparation / "initialization.pt", map_location="cpu", weights_only=True
        )
        learner = Learner(
            initial, device="cuda", seed=args.seed, retention_weight=args.retention_weight
        )
        learner.configure_demonstrations(args.preparation / "demonstrations-TRAINING-ONLY.npz")
        geometry = np.load(args.preparation / "route-geometry.npy", allow_pickle=False)
        with np.load(
            args.preparation / "demonstrations-TRAINING-ONLY.npz", allow_pickle=False
        ) as data:
            prefix_actions = np.array([ACTIONS[int(action)] for action in data["actions"]])
        rng = random.Random(args.seed)
        failures = 0
        write(
            args.output / "provenance.json",
            {
                "initialization": receipt,
                "qualification": proof,
                "plan": plan,
                "seed": args.seed,
                "teacher_prefixes_training_only": True,
                "static_reference_geometry": True,
                "code": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
                "native_config": config,
            },
        )
        while (
            not args.evaluation_only
            and time.monotonic() < training_end
            and not (args.output / "STOP").exists()
        ):
            if args.max_updates is not None and learner.learner.update_count >= args.max_updates:
                report["stop_reason"] = "matched_RL_update_cap"
                break
            if args.functional_test and report["training_episodes"] >= 2:
                break
            check_disk_reserves(args.output, private)
            number = report["training_episodes"]
            folder = args.output / f"episode-{number:06d}"
            # Real teacher prefix advances physics; policy receives actual state at handoff.
            # Alternate full starts with suffixes; random actions perturb the latter.
            prefix = prefix_actions[: rng.randrange(200, 1601)] if number % 2 else ()
            epsilon = 0.01 if number % 2 else (0.0 if number % 4 == 0 else 0.003)
            learner.begin_episode()
            actor_state = {
                k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()
            }
            handoff_steps = 0

            def training_action(state, epsilon=epsilon, prefix=prefix):
                nonlocal handoff_steps
                action = learner.action(state, epsilon)
                if len(prefix) and handoff_steps < 5:
                    steering, gas, brake = ACTIONS[action]
                    steering = max(-1.0, min(1.0, steering + rng.choice((-0.125, 0.125))))
                    action = ACTIONS.index((steering, gas, brake))
                handoff_steps += 1
                return action

            pending_transitions = []
            try:
                result = episode(
                    config,
                    folder,
                    geometry,
                    training_action,
                    deadline=training_end,
                    training=True,
                    prefix_controls=prefix,
                    on_transition=transition_recorder(pending_transitions),
                    stop_file=args.output / "STOP",
                )
            except InvalidProgress as error:
                # Invalid observations/progress never enter replay; retain the FAIL.
                # A new environment resets progress; repeated failures stop explicitly.
                failures += 1
                report["last_episode_error"] = repr(error)
                report["training_episodes"] += 1
                if failures >= 3:
                    raise
                continue
            # Archive integrity failures are fatal, not observation rejections.
            archive_closed_trace(private, result["trace_dir"], folder / "native-trace.tar.gz")
            failures = 0
            report["training_episodes"] += 1
            if not commit_complete_episode(result, pending_transitions, learner):
                report["last_incomplete_episode_status"] = result["status"]
                if result["status"] == "STOPPED_BUDGET_OR_OPERATOR":
                    break
                write(args.output / "state.json", report)
                continue
            goal = result.get("actual_goal_ms")
            rl_updates = learner.accepted_RL_updates
            if (
                not len(prefix)
                and goal is not None
                and rl_updates > 0
                and (
                    report["best_full_start_goal_ms"] is None
                    or goal <= report["best_full_start_goal_ms"]
                )
            ):
                best = folder / "driving-policy.pt"
                torch.save(
                    {
                        "state": actor_state,
                        "RL_updates": learner.accepted_RL_updates,
                        "auxiliary_imitation_updates": learner.imitation_updates,
                        "epsilon": epsilon,
                        "actual_goal_ms": goal,
                        "policy_kind": "BC_then_QR",
                    },
                    best,
                )
                report.update(
                    best_full_start_goal_ms=goal,
                    best_driving_policy=str(best),
                    best_policy_sha256=sha(best),
                )
            for _ in range(
                min(16 if args.functional_test else 64, max(1, int(result["policy_steps"] * 0.05)))
            ):
                if time.monotonic() >= training_end or (args.output / "STOP").exists():
                    break
                if (
                    args.max_updates is not None
                    and learner.learner.update_count >= args.max_updates
                ):
                    break
                metrics = learner.update()
                if metrics:
                    report["retention_diagnostics"] = {
                        k: metrics[k]
                        for k in (
                            "auxiliary_margin",
                            "anchor_errors_before_update",
                            "optimizer_steps",
                        )
                    }
            # Keep the proven actor serving collection while the candidate learns.
            # Only a real, greedy, full-start finish can replace that actor.
            if report["training_episodes"] % 2 == 0 and time.monotonic() < training_end - 15:
                champion = {
                    k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()
                }
                candidate = {
                    k: v.detach().cpu().clone()
                    for k, v in learner.learner.model.state_dict().items()
                }
                learner.actor.load_state_dict(candidate)
                candidate_folder = args.output / f"candidate-{number:06d}"
                try:
                    admitted = episode(
                        config,
                        candidate_folder,
                        geometry,
                        lambda state: learner.action(state, 0.0),
                        deadline=training_end,
                    )
                    archive_closed_trace(
                        private, admitted["trace_dir"], candidate_folder / "native-trace.tar.gz"
                    )
                finally:
                    learner.actor.load_state_dict(champion)
                accepted = learner.consider_candidate(admitted, candidate)
                torch.save(
                    {"state": candidate, "RL_updates": learner.learner.update_count},
                    candidate_folder / "candidate-policy.pt",
                )
                write(
                    candidate_folder / "admission.json",
                    {
                        "accepted": accepted,
                        "goal_ms": admitted.get("actual_goal_ms"),
                        "candidate_RL_updates": learner.learner.update_count,
                        "accepted_RL_updates": learner.accepted_RL_updates,
                        "accepted_goal_ms": learner.accepted_goal_ms,
                        "policy_sha256": sha(candidate_folder / "candidate-policy.pt"),
                    },
                )
                report.update(
                    accepted_goal_ms=learner.accepted_goal_ms,
                    accepted_RL_updates=learner.accepted_RL_updates,
                    rejected_candidates=learner.rejected_candidates,
                    last_candidate_goal_ms=admitted.get("actual_goal_ms"),
                )
            saved = learner.checkpoint()
            temporary = args.output / "latest.tmp"
            torch.save(saved, temporary)
            temporary.replace(args.output / "latest-full-checkpoint.pt")
            report.update(
                status="training",
                new_real_transitions=learner.transitions,
                RL_updates=learner.learner.update_count,
                auxiliary_imitation_updates=learner.imitation_updates,
                elapsed_seconds=time.monotonic() - started,
            )
            write(args.output / "state.json", report)
        if args.evaluation_only:
            report["stop_reason"] = "BC_control_evaluation_only"
        report.setdefault(
            "stop_reason",
            "operator_STOP"
            if (args.output / "STOP").exists()
            else "functional_test_episode_cap"
            if args.functional_test
            else "training_budget_reserve",
        )
        # Final trials are frozen and include ALL failures in the denominator.
        # Select the actual driving snapshot, never the weights updated afterwards.
        report["evaluated_policy_RL_updates"] = learner.accepted_RL_updates
        torch.save(
            {
                "state": {
                    k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()
                },
                "RL_updates": report["evaluated_policy_RL_updates"],
                "route_sha256": receipt["route_sha256"],
                "static_reference_geometry": True,
                "policy_kind": "BC_then_QR" if learner.accepted_RL_updates else "BC_fallback",
            },
            args.output / "evaluated-policy.pt",
        )
        report["evaluated_policy_sha256"] = sha(args.output / "evaluated-policy.pt")
        learner.demo = None
        prefix_actions = None

        def deny_teacher(event, parameters):
            if event != "open" or not isinstance(parameters[0], (str, bytes)):
                return
            mode = parameters[1] if len(parameters) > 1 else None
            if isinstance(mode, str) and mode.startswith(("w", "a", "x")):
                return
            if any(
                token in str(parameters[0]).lower()
                for token in ("training-only", "source-controls", "actual-rollout.npz")
            ):
                raise RuntimeError("teacher-control read during evaluation")

        sys.addaudithook(deny_teacher)
        report["status"] = "evaluating"
        write(args.output / "state.json", report)
        expected_trials = 2 if args.functional_test else 100
        for trial in range(expected_trials):
            if time.monotonic() >= deadline - 10 or (args.output / "STOP").exists():
                break
            check_disk_reserves(args.output, private)
            eval_rng = random.Random(args.seed + 100000 + trial)
            epsilon = 0.0 if trial == 0 else 0.005
            row = episode(
                config,
                args.output / f"evaluation-{trial:03d}",
                geometry,
                lambda state, epsilon=epsilon, eval_rng=eval_rng: learner.action(
                    state, epsilon, eval_rng
                ),
                deadline=deadline - 10,
                stop_file=args.output / "STOP",
            )
            archive_closed_trace(
                private,
                row["trace_dir"],
                args.output / f"evaluation-{trial:03d}" / "native-trace.tar.gz",
            )
            row.update(trial=trial, seed=args.seed + 100000 + trial, epsilon=epsilon)
            report["evaluation"].append(row)
            write(args.output / "state.json", report)
        goals = [r["actual_goal_ms"] for r in report["evaluation"] if r["status"] == "NATIVE_GOAL"]
        report.update(
            status="completed"
            if evaluation_complete(report["evaluation"], expected_trials)
            else "incomplete",
            evaluation_trials=len(report["evaluation"]),
            evaluation_finishes=len(goals),
            best_evaluation_goal_ms=min(goals) if goals else None,
            wins_vs_17712=sum(g < 17712 for g in goals),
            improvements_over_BC=sum(g < 17601 for g in goals),
            teacher_control_reads_during_evaluation=False,
            learner_updates_during_evaluation=0,
            Windows_TICK_model_verification_pending=True,
            visual_ghost_pending=True,
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
    finally:
        if learner:
            import torch

            torch.save(learner.checkpoint(), args.output / "final-full-checkpoint.pt")
        report["elapsed_seconds"] = time.monotonic() - started
        write(args.output / "completion.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "evaluation"}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
