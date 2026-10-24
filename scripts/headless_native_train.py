"""Native-only V7 training with finite sessions and resumable learner state.

Uses measured POST inputs/endpoints and a separately selected native clock
contract. No Windows controller or paired-comparison gate is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

_PROCESS_STARTED = time.monotonic()

import torch  # noqa: E402

from trackmaniarl.core.contracts import PolicyMode  # noqa: E402
from trackmaniarl.core.data import BatchRequest  # noqa: E402
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
    model_weights_sha256,
)
from trackmaniarl.trackmania.common_training.native_only import (  # noqa: E402
    NativeOnlyAdapter,
    NativeOnlyClockProof,
    native_only_contract,
)
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig  # noqa: E402
from trackmaniarl.trackmania.native_headless.immediate_environment import (  # noqa: E402
    NativeImmediateEnvironmentFactory,
)
from trackmaniarl.trackmania.native_headless.training_storage import (  # noqa: E402
    atomic_checkpoint,
    atomic_json,
    directory_bytes,
    verify_native_cleanup,
)

SCHEMA = "tmrl.native-only-trainer/v1"
CHECKPOINT_SCHEMA = "tmrl.native-only-trainer-checkpoint/v1"
MIB = 1024 * 1024
JOURNAL_ROW_BYTES = 65536
JOURNAL_GROUP_ROWS = 64


@dataclass(frozen=True, kw_only=True)
class TrainOptions:
    output: Path
    stop_file: Path
    transitions: int = 4096
    episode_native_units: int = 20000
    native_iterations: int = 5
    replay_capacity: int = 10000
    batch_size: int = 32
    checkpoint_every: int = 256
    seed: int = 1729
    max_wall_seconds: float = 240
    max_run_bytes: int = 512 * MIB
    max_checkpoint_bytes: int = 128 * MIB
    reuse_sessions: bool = False
    resume: Path | None = None
    resume_sha256: str | None = None
    host_started_at: float | None = None

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("transitions", 1, 100_000_000),
            ("episode_native_units", 10, 47770),
            ("native_iterations", 1, 5),
            ("replay_capacity", 1, 1_000_000),
            ("batch_size", 1, 4096),
            ("checkpoint_every", 1, 1_000_000),
            ("max_run_bytes", 1, 1024**4),
            ("max_checkpoint_bytes", 1, 1024**3),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"literal {name} in {minimum}..{maximum} required")
        if self.episode_native_units % 10 or self.batch_size > self.replay_capacity:
            raise ValueError("whole native intervals and replay capacity >= batch size required")
        if (
            type(self.seed) is not int
            or type(self.max_wall_seconds) not in (int, float)
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("integer seed and finite positive run deadline required")
        if (self.resume is None) != (self.resume_sha256 is None):
            raise ValueError("resume requires its explicitly selected SHA-256")
        if type(self.reuse_sessions) is not bool:
            raise ValueError("reuse_sessions must be an explicit boolean")
        if self.resume_sha256 is not None and (
            len(self.resume_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.resume_sha256)
        ):
            raise ValueError("literal lowercase resume SHA-256 required")

    def schedule(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "episode_native_units",
                "native_iterations",
                "replay_capacity",
                "batch_size",
                "seed",
            )
        }


@dataclass(kw_only=True)
class TrainComponents:
    config: NativeRuntimeConfig
    proof: NativeOnlyClockProof
    reference: CommonPolylineReference
    contract: CommonTrainingContract


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_binding(config: NativeRuntimeConfig) -> dict[str, Any]:
    return {
        "asset_manifest_sha256": _file_sha(config.asset_manifest),
        "deployment_manifest_sha256": config.deployment_manifest_sha256,
        "protocol_profile": config.protocol_profile,
        "reset_profile": config.reset_profile,
    }


def run_training(
    components: TrainComponents, options: TrainOptions, factory: Any = None
) -> dict[str, Any]:
    """Train on actual endpoints; a test may supply a non-native factory."""
    start = options.host_started_at or time.monotonic()
    if any(
        (options.output / name).exists() or (options.output / name).is_symlink()
        for name in ("latest.pt", "report.json", "transitions.jsonl", "native", "work")
    ):
        raise ValueError("training requires fresh owned output/checkpoint paths")
    contract, config = components.contract, components.config
    expected = native_only_contract(components.proof, components.reference)
    expected.require_checkpoint(contract.envelope())
    components.proof.require_interval(2200, 2200 + options.episode_native_units)
    if config.duration_limit is not None and config.duration_limit < options.episode_native_units:
        raise ValueError("provisioned duration limit is shorter than requested training episode")
    session_limit = config.max_trace_bytes or 64 * MIB
    reserve = (
        session_limit + options.max_checkpoint_bytes + MIB + JOURNAL_ROW_BYTES * JOURNAL_GROUP_ROWS
    )
    if options.max_run_bytes <= reserve:
        raise ValueError("run storage budget must reserve one full session and atomic checkpoint")
    binding = _runtime_binding(config)
    learner = make_learner(contract, seed=options.seed)
    replay = CommonReplayStore(contract, options.replay_capacity)
    sampler = CommonSampler(contract, seed=options.seed)
    budget = DurationUpdateBudget(contract)
    run_id, next_episode = uuid4().hex, 0
    # The pinned driver accepts work only under runtime/build/server-probe.
    # A unique owned child preserves that boundary across independent resumes.
    run_work = config.work_dir / ("train-" + uuid4().hex)
    extra_roots: list[Path] = []
    active_session = False
    environment: Any = None
    last_episode_seconds = 0.0
    phase_seconds = dict.fromkeys(
        ("reset", "environment_step", "learner", "checkpoint", "cleanup"), 0.0
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": False,
        "native_only_training": True,
        "learning_comparison_run": False,
        "paired_training_qualified": False,
        "campaign_qualified": False,
        "contract_sha256": contract.digest,
        "clock_proof_sha256": components.proof.sha256,
        "runtime_binding": binding,
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(options).items()
        },
        "run_id": run_id,
        "work_root": str(run_work),
        "transitions": 0,
        "native_intervals": 0,
        "model_executed": False,
        "episodes": [],
        "sessions": [],
        "session_rollovers": [],
        "phase_host_seconds": phase_seconds,
        "cleanup": [],
        "stopped": False,
        "resume_loaded": False,
    }

    def usage() -> int:
        return directory_bytes((options.output, run_work, *extra_roots))

    def stop_reason() -> str | None:
        if options.stop_file.exists():
            return "STOP file"
        if time.monotonic() - start >= options.max_wall_seconds:
            return "full host training deadline"
        return None

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        before = time.monotonic()
        try:
            yield
        finally:
            phase_seconds[name] += time.monotonic() - before

    def close_session() -> None:
        nonlocal environment, active_session
        if environment is None:
            return
        closing, environment = environment, None
        with phase("cleanup"):
            cleanup = verify_native_cleanup(closing)
        report["cleanup"].append(cleanup)
        session = report["sessions"][-1]
        session["cleanup_index"] = len(report["cleanup"]) - 1
        session["trace_accounting"] = cleanup.get("trace_accounting")
        supervisor = cleanup.get("supervisor")
        if isinstance(supervisor, dict):
            session["driver_identity"] = supervisor.get("driverIdentity")
            session["supervisor_identity"] = supervisor.get("supervisorIdentity")
        retained = cleanup.get("retained_local_directory")
        if retained:
            extra_roots.append(Path(retained))
        active_session = False
        if not cleanup["verified"]:
            raise RuntimeError("owned native cleanup could not be verified")

    def checkpoint() -> None:
        with phase("checkpoint"):
            publish_checkpoint()

    def publish_checkpoint() -> None:
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "common_contract": contract.envelope(),
            "schedule": options.schedule(),
            "runtime_binding": binding,
            "run_id": run_id,
            "next_episode": next_episode,
            "resume_requires_fresh_episode": True,
            "learner": learner.state_dict(),
            "replay": replay.state_dict(),
            "sampler": sampler.state_dict(),
            "budget": budget.state_dict(),
        }
        active_reserve = (
            session_limit + JOURNAL_ROW_BYTES * JOURNAL_GROUP_ROWS if active_session else 0
        )
        allowance = min(
            options.max_checkpoint_bytes, options.max_run_bytes - usage() - MIB - active_reserve
        )
        report["checkpoint"] = atomic_checkpoint(
            options.output / "latest.pt", state, max_bytes=max(0, allowance)
        )

    def drain_credit() -> None:
        request = BatchRequest(options.batch_size, gamma=contract.gamma)
        previous_updates = learner.update_count
        while budget.integral_available and len(replay) >= options.batch_size:
            if stop_reason():
                return
            with phase("learner"):
                metrics, _ = learner.update(sampler.sample(replay, request))
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("nonfinite native learner metrics")
            budget.consume()
            report["last_metrics"] = dict(metrics)
        if learner.update_count != previous_updates:
            policy.load_state(learner.model.state_dict())

    try:
        if options.resume is not None:
            if _file_sha(options.resume) != options.resume_sha256:
                raise ValueError("resume checkpoint SHA-256 mismatch")
            saved = torch.load(options.resume, map_location="cpu", weights_only=False)
            if (
                saved.get("schema") != CHECKPOINT_SCHEMA
                or saved.get("schedule") != options.schedule()
                or saved.get("runtime_binding") != binding
                or saved.get("resume_requires_fresh_episode") is not True
                or type(saved.get("next_episode")) is not int
                or saved["next_episode"] < 0
                or not isinstance(saved.get("run_id"), str)
            ):
                raise ValueError("native checkpoint schema/schedule/runtime mismatch")
            contract.require_checkpoint(saved["common_contract"])
            replay.load_state_dict(saved["replay"])
            sampler.load_state_dict(saved["sampler"])
            budget.load_state_dict(saved["budget"])
            learner.load_state_dict(saved["learner"])
            if learner.update_count != budget.updates:
                raise ValueError("checkpoint learner updates and duration credit disagree")
            run_id, next_episode = saved["run_id"], saved["next_episode"]
            report.update(
                resume_loaded=True,
                run_id=run_id,
                resumed_transitions=budget.transitions,
                resumed_optimizer_steps=learner.update_count,
            )
        report["initial_model_sha256"] = model_weights_sha256(learner.model)
        policy = learner.policy()
        initial_updates = learner.update_count
        drain_credit()
        checkpoint()
        journal_path = options.output / "transitions.jsonl"
        with journal_path.open("xb") as journal:
            while report["transitions"] < options.transitions:
                reason = stop_reason()
                if reason:
                    report.update(stopped=True, stop_reason=reason)
                    break
                # Session quota includes its private /tmp captures. Reserve the
                # whole amount until close records expose their retained roots.
                if usage() + reserve > options.max_run_bytes:
                    report.update(stopped=True, stop_reason="run evidence budget")
                    break
                remaining = options.transitions - report["transitions"]
                duration = min(
                    options.episode_native_units, remaining * options.native_iterations * 10
                )
                reused = environment is not None
                if environment is not None:
                    # Provision a full next episode, plus known close copies. The
                    # writer's exact quota checks remain authoritative on every
                    # append; this boundary reserve only decides driver rollover.
                    intervals = duration // 10
                    decisions = math.ceil(intervals / options.native_iterations)
                    status = environment.session_reuse_status(
                        minimum_remaining_seconds=max(30.0, 2 * last_episode_seconds + 5.0),
                        additional_trace_bytes=2 * MIB + decisions * 65536 + intervals * 4096,
                    )
                    if not status["eligible"]:
                        report["session_rollovers"].append(
                            {"before_episode": next_episode, **status}
                        )
                        close_session()
                        reused = False
                if environment is None:
                    # Closing exposes retained /tmp evidence and may duplicate
                    # POST files into output. Admit a new session only after
                    # charging those bytes as well as its own full reserve.
                    if usage() + reserve > options.max_run_bytes:
                        report.update(stopped=True, stop_reason="run evidence budget")
                        break
                    remaining_wall = options.max_wall_seconds - (time.monotonic() - start)
                    if reason := stop_reason():
                        report.update(stopped=True, stop_reason=reason)
                        break
                    environment_config = replace(
                        config,
                        timeout_s=min(config.timeout_s, remaining_wall),
                        trace_dir=options.output / "native",
                        work_dir=run_work,
                    )
                    active_factory = (
                        NativeImmediateEnvironmentFactory(environment_config)
                        if factory is None
                        else factory(environment_config)
                    )
                    environment = active_factory.create(seed=options.seed + next_episode)
                    active_session = True
                    report["sessions"].append(
                        {
                            "index": len(report["sessions"]),
                            "trace_dir": str(environment.config.trace_dir),
                            "episodes": 0,
                        }
                    )
                case: dict[str, Any] = {
                    "episode": next_episode,
                    "session_index": len(report["sessions"]) - 1,
                    "reused_session": reused,
                    "duration_limit": duration,
                    "transitions": 0,
                    "terminated": False,
                    "truncated": False,
                }
                episode_id = f"native-only/{run_id}/{next_episode}"
                next_episode += 1
                report["episodes"].append(case)
                report["sessions"][-1]["episodes"] += 1
                episode_error: BaseException | None = None
                episode_start = time.monotonic()
                try:
                    if reason := stop_reason():
                        report.update(stopped=True, stop_reason=reason)
                        break
                    reset_start = time.monotonic()
                    with phase("reset"):
                        state, info = environment.reset(
                            seed=options.seed + case["episode"], duration_limit=duration
                        )
                    adapter = NativeOnlyAdapter(
                        contract, proof=components.proof, reference=components.reference
                    )
                    observation = adapter.bind_reset(state, info, episode_id=episode_id)
                    case.update(
                        reset_host_seconds=time.monotonic() - reset_start,
                        reset_nativeT=state["nativeT"],
                        branch_pid=info.get("episodeProcessId"),
                        raw_journal=str(environment.journal_path),
                    )
                    policy.reset_episode()
                    while True:
                        reason = stop_reason()
                        if reason:
                            report.update(stopped=True, stop_reason=reason)
                            break
                        if (
                            report["transitions"] % JOURNAL_GROUP_ROWS == 0
                            and usage() + reserve > options.max_run_bytes
                        ):
                            report.update(stopped=True, stop_reason="run evidence budget")
                            break
                        action = policy.act(observation, PolicyMode.ONLINE)
                        report["model_executed"] = True
                        reason = stop_reason()
                        if reason:
                            report.update(stopped=True, stop_reason=reason)
                            break
                        with phase("environment_step"):
                            result = environment.step_native(
                                action, native_iterations=options.native_iterations
                            )
                        row = adapter.transition(action, result)
                        journal_record = (
                            json.dumps(
                                {
                                    "episode_id": row.episode_id,
                                    "step": row.step,
                                    "action": row.action,
                                    "reward": row.reward,
                                    "terminated": row.terminated,
                                    "truncated": row.truncated,
                                    "info": row.info,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        if len(journal_record) > JOURNAL_ROW_BYTES:
                            raise ValueError("outer transition journal row exceeds reserved size")
                        if journal.write(journal_record) != len(journal_record):
                            raise OSError("short outer transition journal write")
                        journal.flush()
                        replay.append(row)
                        budget.credit(row)
                        report["transitions"] += 1
                        report["native_intervals"] += result["actualIterations"]
                        case["transitions"] += 1
                        case["last_nativeT"] = result["observation"]["nativeT"]
                        case["reward"] = case.get("reward", 0.0) + row.reward
                        drain_credit()
                        if report["transitions"] % options.checkpoint_every == 0:
                            checkpoint()
                        if row.terminated or row.truncated:
                            case.update(terminated=row.terminated, truncated=row.truncated)
                            break
                        if report["transitions"] >= options.transitions:
                            raise ValueError(
                                "native environment missed its explicit budget endpoint"
                            )
                        observation = row.next_observation
                except BaseException as error:
                    episode_error = error
                    raise
                finally:
                    last_episode_seconds = time.monotonic() - episode_start
                    case["episode_host_seconds"] = last_episode_seconds
                    if not options.reuse_sessions or episode_error is not None:
                        try:
                            close_session()
                        except BaseException as error:
                            if episode_error is None:
                                raise
                            episode_error.add_note(f"owned native cleanup also failed: {error!r}")
                checkpoint()
                if report["stopped"]:
                    break
        drain_credit()
        report.update(passed=True, optimizer_steps_this_run=learner.update_count - initial_updates)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
    finally:
        try:
            close_session()
        except (Exception, KeyboardInterrupt) as error:
            report.update(passed=False, cleanup_error=repr(error))
            report.setdefault("error", repr(error))
        report.update(
            cumulative_transitions=budget.transitions,
            optimizer_steps=learner.update_count,
            retained_rows=len(replay),
            credited_simulated_seconds=budget.elapsed_seconds,
            remaining_update_credit=budget.credit_available,
            remaining_integral_update_credit=budget.integral_available,
            next_episode=next_episode,
            final_model_sha256=model_weights_sha256(learner.model),
            native_environment_instances=len(report["sessions"]),
            native_sessions_started=sum(
                item.get("driver_identity") is not None for item in report["sessions"]
            ),
            native_episode_processes=sum(
                type(item.get("branch_pid")) is int for item in report["episodes"]
            ),
            owned_cleanup_verified=(
                all(item["verified"] for item in report["cleanup"]) if report["cleanup"] else None
            ),
            retained_local_directories=[str(path) for path in extra_roots],
        )
        # Never replace a valid prior checkpoint with partially loaded state.
        if "error" not in report and (options.resume is None or report["resume_loaded"]):
            try:
                checkpoint()
            except (Exception, KeyboardInterrupt) as error:
                report.update(passed=False, checkpoint_error=repr(error))
        report["full_host_seconds"] = time.monotonic() - start
        try:
            report["evidence_bytes"] = usage()
            atomic_json(
                options.output / "report.json",
                report,
                max_bytes=max(0, options.max_run_bytes - usage()),
            )
        except (Exception, KeyboardInterrupt) as error:
            report.update(passed=False, report_error=repr(error))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "clock-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--clock-proof-sha256", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--stop-file", type=Path)
    for name, default in (
        ("transitions", 4096),
        ("episode-native-units", 20000),
        ("native-iterations", 5),
        ("replay-capacity", 10000),
        ("batch-size", 32),
        ("checkpoint-every", 256),
        ("seed", 1729),
        ("max-run-bytes", 512 * MIB),
        ("max-checkpoint-bytes", 128 * MIB),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--max-wall-seconds", type=float, default=240)
    parser.add_argument("--reuse-sessions", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any] = {"schema": SCHEMA, "passed": False, "learning_comparison_run": False}
    try:
        output = args.output
        if (
            sys.platform != "linux"
            or not output.is_absolute()
            or output.exists()
            or output.resolve() != output
            or str(output).startswith("/mnt/")
            or any(parent.is_symlink() for parent in output.parents)
        ):
            raise ValueError("fresh canonical Linux/ext4 output required")
        values = vars(args).copy()
        for key in ("config", "reference", "clock_proof", "clock_proof_sha256"):
            values.pop(key)
        values["stop_file"] = args.stop_file or output / "STOP"
        values["host_started_at"] = _PROCESS_STARTED
        options = TrainOptions(**values)
        proof = NativeOnlyClockProof.from_artifact(
            args.clock_proof, expected_sha256=args.clock_proof_sha256
        )
        reference = CommonPolylineReference.from_json(
            args.reference, expected_sha256=KNOWN_REFERENCE_SHA256
        )
        config = NativeRuntimeConfig.from_mapping(json.loads(args.config.read_bytes()))
        contract = native_only_contract(proof, reference)
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        atomic_json(output / "contract.json", contract.envelope())
        report = run_training(
            TrainComponents(config=config, proof=proof, reference=reference, contract=contract),
            options,
        )
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
        if (
            locals().get("output_created") is True
            and output.is_dir()
            and "options" in locals()
            and not (output / "report.json").exists()
        ):
            atomic_json(output / "report.json", report, max_bytes=MIB)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "transitions",
                    "cumulative_transitions",
                    "optimizer_steps",
                    "stopped",
                    "stop_reason",
                    "resume_loaded",
                    "owned_cleanup_verified",
                    "full_host_seconds",
                    "error",
                    "checkpoint_error",
                    "report_error",
                )
                if key in report
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
