"""Bounded real V7/common6 rollouts with immutable clock and qualification gates.

Collection uses an explicit existing action tape and never creates a model or
optimizer. Training requires a complete separately selected paired evidence
gate. This module does not control the Windows game.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_PROCESS_STARTED = time.monotonic()

import torch  # noqa: E402

from trackmaniarl.core.contracts import PolicyMode  # noqa: E402
from trackmaniarl.core.data import BatchRequest, Transition  # noqa: E402
from trackmaniarl.trackmania.common_training import (  # noqa: E402
    CommonPolylineReference,
    CommonReplayStore,
    CommonSampler,
    CommonTrainingContract,
    DurationUpdateBudget,
    make_learner,
)
from trackmaniarl.trackmania.common_training.adapters import (  # noqa: E402
    KNOWN_REFERENCE_SHA256,
    NativeCommonAdapter,
    PhysicalClockProof,
    collect_contract,
    model_weights_sha256,
    qualified_native_contract,
)
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig  # noqa: E402
from trackmaniarl.trackmania.native_headless.immediate_environment import (  # noqa: E402
    NativeImmediateEnvironmentFactory,
)

TAPES = ("neutral", "straight-gas", "mild-left", "full-brake", "brake-tap")


@dataclass(frozen=True, kw_only=True)
class LiveOptions:
    output: Path
    stop_file: Path
    collect_only: bool
    infrastructure_smoke: bool = False
    tape: str | None = None
    episodes: int = 2
    decisions: int = 10
    native_iterations: int = 5
    batch_size: int = 8
    seed: int = 1729
    max_wall_seconds: float = 240.0
    host_started_at: float | None = None

    def __post_init__(self) -> None:
        for name, maximum in (
            ("episodes", 2),
            ("decisions", 64),
            ("native_iterations", 5),
            ("batch_size", 128),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"bounded positive {name} <= {maximum} required")
        if self.collect_only and self.tape not in TAPES:
            raise ValueError("collection requires an explicit existing action tape")
        if not self.collect_only and self.tape is not None:
            raise ValueError("scripted action tapes are available only for collection")
        if self.collect_only and self.infrastructure_smoke:
            raise ValueError("choose collection or native infrastructure smoke")
        if type(self.seed) is not int or not 0 < self.max_wall_seconds <= 240:
            raise ValueError("integer seed and at most240second host budget required")


@dataclass(kw_only=True)
class LiveComponents:
    config: NativeRuntimeConfig
    proof: PhysicalClockProof
    reference: CommonPolylineReference
    contract: CommonTrainingContract
    gate_sha256: str | None = None
    gate_path: Path | None = None
    learner: Any = None


def _tape_action(name: str, index: int, total: int) -> int:
    if name == "neutral":
        return 36
    if name == "straight-gas" or index < total // 2:
        return 39
    return {"mild-left": 33, "full-brake": 40, "brake-tap": 41}[name]


def _cleanup(environment: Any) -> dict[str, Any]:
    path = environment.config.trace_dir / "close-v7.json"
    close_error = None
    try:
        environment.close()
    except (Exception, KeyboardInterrupt) as error:
        close_error = repr(error)
    if not path.is_file():
        return {
            "path": str(path),
            "verified": False,
            "error": close_error or "missing cleanup evidence",
        }
    try:
        report = json.loads(path.read_bytes())
        supervisor = report["supervisor"]
    except (OSError, ValueError, KeyError) as error:
        return {"path": str(path), "verified": False, "error": repr(error)}
    verified = (
        close_error is None
        and report["failures"] == []
        and isinstance(supervisor, Mapping)
        and supervisor["remainingOwned"] == []
        and supervisor["cleanupErrors"] == []
    )
    return {
        "path": str(path),
        "verified": verified,
        "supervisor": supervisor,
        "close_error": close_error,
    }


def _row_record(row: Transition) -> dict[str, Any]:
    return {
        "episode_id": row.episode_id,
        "step": row.step,
        "observation": row.observation.tolist(),
        "action": row.action,
        "reward": row.reward,
        "next_observation": None if row.next_observation is None else row.next_observation.tolist(),
        "terminated": row.terminated,
        "truncated": row.truncated,
        "info": row.info,
    }


def run_rollout(
    components: LiveComponents,
    options: LiveOptions,
    factory: Any = None,
) -> dict[str, Any]:
    """Execute with an injectable factory; tests never instantiate native code."""
    start = options.host_started_at or time.monotonic()
    contract, learner = components.contract, components.learner
    if options.collect_only and (
        learner is not None
        or components.gate_sha256 is not None
        or components.gate_path is not None
    ):
        raise ValueError("collection cannot accept a learner or a paired training gate")
    paired_training = not options.collect_only and not options.infrastructure_smoke
    if options.infrastructure_smoke and (
        learner is None
        or contract.purpose != "offline-plumbing"
        or components.gate_sha256 is not None
        or components.gate_path is not None
    ):
        raise ValueError("native infrastructure smoke requires an explicit offline plumbing model")
    if paired_training and (
        components.gate_sha256 is None
        or components.gate_path is None
        or learner is None
        or contract.purpose != "qualified-training"
    ):
        raise ValueError("optimization requires a complete verified paired evidence gate")
    if paired_training:
        assert components.gate_path is not None
        assert components.gate_sha256 is not None
        verified = qualified_native_contract(
            components.gate_path,
            expected_sha256=components.gate_sha256,
            bindings={
                "initial_model_sha256": model_weights_sha256(learner.model),
                "clock_proof_sha256": components.proof.sha256,
                "native_iterations": options.native_iterations,
                "decisions": options.decisions,
                "seed": options.seed,
            },
        )
        contract.require_checkpoint(verified.envelope())
    if learner is not None:
        contract.require_checkpoint(learner.common_contract.envelope())
    adapter = NativeCommonAdapter(contract, proof=components.proof, reference=components.reference)
    requested_duration = options.decisions * options.native_iterations * 10
    duration = min(components.config.duration_limit or requested_duration, requested_duration)
    components.proof.require_interval(2200, 2200 + duration)
    config = replace(components.config, duration_limit=duration)
    replay, sampler = CommonReplayStore(contract, 128), CommonSampler(contract, seed=options.seed)
    budget, policy = DurationUpdateBudget(contract), None if learner is None else learner.policy()
    report: dict[str, Any] = {
        "schema": "tmrl.common-live-bounded/v1",
        "passed": False,
        "collect_only": options.collect_only,
        "infrastructure_smoke": options.infrastructure_smoke,
        "learning_comparison_run": False,
        "model_executed": False,
        "optimizer_steps": 0,
        "transitions": 0,
        "episodes": [],
        "cleanup": [],
        "stopped": False,
        "clock_proof_sha256": components.proof.sha256,
        "clock_proof_scope": components.proof.scope,
        "clock_native_bounds": [components.proof.minimum_native, components.proof.maximum_native],
        "proof_input_scope": (
            "observed supplied trace; other action equivalence requires paired gate"
        ),
        "paired_gate_sha256": components.gate_sha256,
        "contract_sha256": contract.digest,
        "duration_limit_native_units": duration,
        "scripted_tape": options.tape,
    }

    def stop_reason() -> str | None:
        if options.stop_file.exists():
            return "STOP file"
        if time.monotonic() - start >= options.max_wall_seconds:
            return "full host rollout deadline"
        return None

    def drain_credit() -> None:
        if learner is None:
            return
        request = BatchRequest(options.batch_size, gamma=contract.gamma)
        while budget.integral_available and len(replay) >= options.batch_size:
            reason = stop_reason()
            if reason:
                report.update(stopped=True, stop_reason=reason)
                return
            metrics, _ = learner.update(sampler.sample(replay, request))
            if any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()):
                raise ValueError("nonfinite shared learner metrics")
            budget.consume()
            report["optimizer_steps"] = learner.update_count
        if policy is not None:
            policy.load_state(learner.model.state_dict())

    try:
        with (options.output / "common-transitions.jsonl").open("x", encoding="utf-8") as journal:
            for episode in range(options.episodes):
                reason = stop_reason()
                if reason:
                    report.update(stopped=True, stop_reason=reason)
                    break
                remaining = options.max_wall_seconds - (time.monotonic() - start)
                active_factory = factory or NativeImmediateEnvironmentFactory(
                    replace(config, timeout_s=min(config.timeout_s, remaining)),
                )
                environment = active_factory.create(seed=options.seed + episode)
                case: dict[str, Any] = {
                    "episode": episode,
                    "decisions": 0,
                    "terminated": False,
                    "truncated": False,
                }
                report["episodes"].append(case)
                try:
                    reset_start = time.monotonic()
                    before, info = environment.reset(seed=options.seed + episode)
                    observation = adapter.bind_reset(
                        before, info, episode_id=f"common-v7/{episode}"
                    )
                    case["reset_host_seconds"] = time.monotonic() - reset_start
                    case["native_reset"] = before["nativeT"]
                    case["raw_journal"] = str(environment.journal_path)
                    if policy is not None:
                        policy.reset_episode()
                    for decision in range(options.decisions):
                        reason = stop_reason()
                        if reason:
                            report.update(stopped=True, stop_reason=reason)
                            break
                        components.proof.require_interval(
                            before["nativeT"],
                            min(
                                before["nativeT"] + options.native_iterations * 10, 2200 + duration
                            ),
                        )
                        if options.collect_only:
                            action = _tape_action(str(options.tape), decision, options.decisions)
                        else:
                            assert policy is not None
                            action = policy.act(observation, PolicyMode.ONLINE)
                            report["model_executed"] = True
                        reason = stop_reason()
                        if reason:
                            report.update(stopped=True, stop_reason=reason)
                            break
                        result = environment.step_native(
                            action, native_iterations=options.native_iterations
                        )
                        row = adapter.transition(action, result)
                        journal.write(json.dumps(_row_record(row), allow_nan=False) + "\n")
                        journal.flush()
                        replay.append(row)
                        budget.credit(row)
                        report["transitions"] += 1
                        case["decisions"] += 1
                        before = result["observation"]
                        case["last_nativeT"] = before["nativeT"]
                        case["last_race_ms"] = components.proof.race_milliseconds(before["nativeT"])
                        drain_credit()
                        if row.terminated or row.truncated:
                            case.update(terminated=row.terminated, truncated=row.truncated)
                            break
                        observation = row.next_observation
                    if not report["stopped"] and not (case["terminated"] or case["truncated"]):
                        raise ValueError("decision cap requires an actual environment endpoint")
                finally:
                    cleanup = _cleanup(environment)
                    report["cleanup"].append(cleanup)
                    if not cleanup["verified"]:
                        raise RuntimeError("owned native cleanup could not be verified")
                if report["stopped"]:
                    break
        if not report["stopped"]:
            drain_credit()
        report["passed"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
    finally:
        checkpoint = {
            "common_contract": contract.envelope(),
            "replay": replay.state_dict(),
            "sampler": sampler.state_dict(),
            "budget": budget.state_dict(),
            "learner": None if learner is None else learner.state_dict(),
            "paired_gate_sha256": components.gate_sha256,
        }
        torch.save(checkpoint, options.output / "checkpoint.pt")
        if learner is not None and report["passed"]:
            loaded = torch.load(
                options.output / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            restored = make_learner(contract, seed=options.seed)
            restored.load_state_dict(loaded["learner"])
            restored_replay, restored_sampler = (
                CommonReplayStore(contract, 128),
                CommonSampler(contract),
            )
            restored_replay.load_state_dict(loaded["replay"])
            restored_sampler.load_state_dict(loaded["sampler"])
            restored_budget = DurationUpdateBudget(contract)
            restored_budget.load_state_dict(loaded["budget"])
            same_weights = all(
                torch.equal(first, second)
                for first, second in zip(
                    learner.model.parameters(),
                    restored.model.parameters(),
                    strict=True,
                )
            )
            if (
                not same_weights
                or restored.update_count != learner.update_count
                or (restored_budget.state_dict() != budget.state_dict())
            ):
                raise ValueError("real replay/learner/budget checkpoint roundtrip failed")
            if len(replay):
                request = BatchRequest(min(options.batch_size, len(replay)), gamma=contract.gamma)
                actual, reloaded = (
                    sampler.sample(replay, request),
                    restored_sampler.sample(restored_replay, request),
                )
                if actual.transition_ids != reloaded.transition_ids or not torch.equal(
                    actual.bootstrap_discounts,
                    reloaded.bootstrap_discounts,
                ):
                    raise ValueError("real replay/sampler duration checkpoint roundtrip failed")
            report["checkpoint_roundtrip"] = True
        report.update(
            retained_rows=len(replay),
            credited_transitions=budget.transitions,
            credited_simulated_seconds=budget.elapsed_seconds,
            remaining_update_credit=budget.credit_available,
            remaining_integral_update_credit=budget.integral_available,
            credit_drained=learner is not None and budget.integral_available == 0,
            owned_cleanup_verified=all(item["verified"] for item in report["cleanup"]),
            checkpoint_written=True,
            full_host_rollout_seconds=time.monotonic() - start,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--clock-proof", type=Path, required=True)
    parser.add_argument("--clock-proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--infrastructure-smoke", action="store_true")
    parser.add_argument("--tape", choices=TAPES)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--gate-sha256")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--decisions", type=int, default=10)
    parser.add_argument("--native-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-wall-seconds", type=float, default=240)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"passed": False, "learning_comparison_run": False}
    try:
        options = LiveOptions(
            output=args.output,
            stop_file=args.stop_file or args.output / "STOP",
            collect_only=args.collect_only,
            infrastructure_smoke=args.infrastructure_smoke,
            tape=args.tape,
            episodes=args.episodes,
            decisions=args.decisions,
            native_iterations=args.native_iterations,
            batch_size=args.batch_size,
            seed=args.seed,
            max_wall_seconds=args.max_wall_seconds,
            host_started_at=_PROCESS_STARTED,
        )
        config = NativeRuntimeConfig.from_mapping(json.loads(args.config.read_bytes()))
        proof = PhysicalClockProof.from_json(
            args.clock_proof, expected_sha256=args.clock_proof_sha256
        )
        reference = CommonPolylineReference.from_json(
            args.reference, expected_sha256=KNOWN_REFERENCE_SHA256
        )
        contract = collect_contract(proof, reference)
        learner = None
        if args.infrastructure_smoke:
            learner = make_learner(contract, seed=args.seed)
        elif not args.collect_only:
            if args.gate is None or args.gate_sha256 is None:
                raise ValueError(
                    "training requires --gate and its independently selected --gate-sha256"
                )
            initial = make_learner(contract, seed=args.seed)
            contract = qualified_native_contract(
                args.gate,
                expected_sha256=args.gate_sha256,
                bindings={
                    "initial_model_sha256": model_weights_sha256(initial.model),
                    "clock_proof_sha256": proof.sha256,
                    "native_iterations": args.native_iterations,
                    "decisions": args.decisions,
                    "seed": args.seed,
                },
            )
            learner = make_learner(contract, seed=args.seed)
        components = LiveComponents(
            config=config,
            proof=proof,
            reference=reference,
            contract=contract,
            learner=learner,
            gate_sha256=args.gate_sha256,
            gate_path=args.gate,
        )
        (args.output / "contract.json").write_text(json.dumps(contract.envelope(), indent=2) + "\n")
        report = run_rollout(components, options)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
    report["full_host_seconds"] = time.monotonic() - _PROCESS_STARTED
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
