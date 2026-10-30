"""Experimental native actors with one fixed common6 learner and barrier rounds.

Torch imports occur inside run/main: spawned environment workers must not build
or import the learner through Python's __mp_main__ startup path. Only the pinned
stateless IdentityTemporalCore policy without an action selector is supported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from scripts.headless_native_train import TrainComponents, TrainOptions

_PROCESS_STARTED = time.monotonic()
SCHEMA = "tmrl.native-multi-trainer/v1"
CHECKPOINT_SCHEMA = "tmrl.native-multi-trainer-checkpoint/v1"
SCHEDULER = "fixed-actor-id/barrier-round/v1"
POLICY_REFRESH = "all-actions-before-ingest/refresh-after-round/v1"
RNG_SCHEDULE = "one-main-torch/serial-actor-id/v1"
MIB = 1024 * 1024


def multi_schedule(  # noqa: PLR0913 -- two independent, versioned performance modes
    options: TrainOptions,
    actors: int,
    *,
    batch_actions: bool = False,
    overlap_learning: bool = False,
    learner_device: str = "cpu",
) -> dict[str, Any]:
    schedule = {
        **options.schedule(),
        "actor_count": actors,
        "scheduler": SCHEDULER,
        "policy_refresh": POLICY_REFRESH,
        "policy_lag_bound_rounds": 1,
        "rng": RNG_SCHEDULE,
    }
    if batch_actions:
        from trackmaniarl.trackmania.common_training.batched_policy import BATCH_NUMERICS

        schedule["inference_batch"] = BATCH_NUMERICS
    if overlap_learning:
        schedule.update(
            learner_overlap="one-issued-round/precollated-row-updates/checkpoint-drain/v1",
            policy_refresh="refresh-after-prior-round-updates/v1",
            policy_lag_bound_rounds=2,
            overlap_checkpoint_every=options.checkpoint_every,
        )
    if learner_device == "cuda":
        import torch

        schedule["learner_execution"] = {
            "profile": "cuda-float32-ieee/deterministic/cpu-behavior/v1",
            "torch_build": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "actor_device": "cpu",
            "precision": "float32",
            "tf32": False,
            "cublas_workspace": ":4096:8",
            "cuda_cache_max_bytes": 64 * MIB,
        }
    return schedule


def run_training(  # noqa: PLR0913 -- explicit actor count and test transport are independent
    components: TrainComponents,
    options: TrainOptions,
    *,
    actors: int = 1,
    pool_factory: Any = None,
    batch_actions: bool = False,
    overlap_learning: bool = False,
    learner_device: str = "cpu",
) -> dict[str, Any]:
    """No result of an incomplete or invalid round reaches replay or credit."""
    if type(actors) is not int or not 1 <= actors <= 8:
        raise ValueError("literal actor count in 1..8 required")
    if type(batch_actions) is not bool or type(overlap_learning) is not bool:
        raise ValueError("literal performance mode flags required")
    if overlap_learning and options.batch_size != 32:
        raise ValueError("overlap v1 is qualified only for the fixed batch32 profile")
    if learner_device not in {"cpu", "cuda"}:
        raise ValueError("explicit CPU or CUDA learner device required")
    import torch

    from scripts import headless_native_train as single
    from trackmaniarl.core.contracts import PolicyMode
    from trackmaniarl.core.data import BatchRequest
    from trackmaniarl.trackmania.common_training import (
        CommonReplayStore,
        CommonSampler,
        DurationUpdateBudget,
        make_learner,
    )
    from trackmaniarl.trackmania.common_training.adapters import model_weights_sha256
    from trackmaniarl.trackmania.common_training.native_only import (
        NativeOnlyAdapter,
        native_only_contract,
    )
    from trackmaniarl.trackmania.native_headless.training_storage import (
        atomic_checkpoint,
        atomic_json,
        directory_bytes,
    )

    start = options.host_started_at or time.monotonic()
    contract, config = components.contract, components.config
    native_only_contract(components.proof, components.reference).require_checkpoint(
        contract.envelope()
    )
    components.proof.require_interval(2200, 2200 + options.episode_native_units)
    if config.duration_limit is not None and config.duration_limit < options.episode_native_units:
        raise ValueError("provisioned duration limit is shorter than requested training episode")
    reserved_paths = ["latest.pt", "report.json", "transitions.jsonl", "native", "work"]
    if learner_device == "cuda":
        reserved_paths.append("gpu-cache")
    if any(
        (options.output / name).exists() or (options.output / name).is_symlink()
        for name in reserved_paths
    ):
        raise ValueError("multi training requires fresh owned output/checkpoint paths")
    session_limit = config.max_trace_bytes or 64 * MIB
    accelerator_cache_reserve = 64 * MIB if learner_device == "cuda" else 0
    reserve = (
        actors * session_limit
        + accelerator_cache_reserve
        + options.max_checkpoint_bytes
        + MIB
        + single.JOURNAL_ROW_BYTES * single.JOURNAL_GROUP_ROWS
    )
    if options.max_run_bytes <= reserve:
        raise ValueError("run storage must reserve every actor session and an atomic checkpoint")
    schedule = multi_schedule(
        options,
        actors,
        batch_actions=batch_actions,
        overlap_learning=overlap_learning,
        learner_device=learner_device,
    )
    binding = single._runtime_binding(config)
    if learner_device == "cuda":
        from trackmaniarl.algorithms.execution import TorchExecutionConfig

        if torch.version.cuda is None or not torch.cuda.is_available():
            raise ValueError("CUDA float32 learner requested but unavailable")
        workspace = ":4096:8"
        initialized = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        if initialized and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
            raise ValueError("CUDA context predates the required deterministic workspace")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        cache = options.output / "gpu-cache"
        cache.mkdir(parents=True, exist_ok=False)
        for name in ("cuda", "triton"):
            (cache / name).mkdir(exist_ok=False)
        os.environ["CUDA_CACHE_PATH"] = str(cache / "cuda")
        os.environ["CUDA_CACHE_MAXSIZE"] = str(accelerator_cache_reserve)
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        learner = make_learner(
            contract,
            seed=options.seed,
            execution=TorchExecutionConfig(
                device="cuda", precision="float32", deterministic=True, torch_threads=2
            ),
        )
    else:
        learner = make_learner(contract, seed=options.seed)
    replay = CommonReplayStore(contract, options.replay_capacity)
    sampler = CommonSampler(contract, seed=options.seed)
    budget = DurationUpdateBudget(contract)
    ids = list(range(actors))
    next_episode, historical_rows = dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
    new_rows = dict.fromkeys(ids, 0)
    quotient, remainder = divmod(options.transitions, actors)
    assigned = {actor: quotient + (actor < remainder) for actor in ids}
    run_id, completed_rounds = uuid4().hex, 0
    run_work = config.work_dir / ("multi-" + uuid4().hex)
    extra_roots: set[Path] = set()
    sessions: dict[int, int] = {}
    adapters: dict[int, Any] = {}
    observations: dict[int, Any] = {}
    cases: dict[int, dict[str, Any]] = {}
    last_episode_seconds = dict.fromkeys(ids, 0.0)
    phase = dict.fromkeys(
        (
            "pool_startup",
            "reset",
            "environment_dispatch",
            "environment_step",
            "policy",
            "sample",
            "learner",
            "checkpoint",
            "cleanup",
        ),
        0.0,
    )
    pool: Any = None
    pool_closed = False
    unchecked_outer_rows = 0
    storage_admission_required = True
    prepared_updates: deque[Any] = deque()
    maximum_prepared_updates = 16
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "experimental": True,
        "passed": False,
        "native_only_training": True,
        "learning_comparison_run": False,
        "paired_training_qualified": False,
        "campaign_qualified": False,
        "actor_count": actors,
        "scheduler": schedule,
        "contract_sha256": contract.digest,
        "clock_proof_sha256": components.proof.sha256,
        "runtime_binding": binding,
        "transitions": 0,
        "native_intervals": 0,
        "model_executed": False,
        "episodes": [],
        "sessions": [],
        "cleanup": [],
        "rounds": [],
        "session_rollovers": [],
        "phase_host_seconds": phase,
        "work_root": str(run_work),
        "stopped": False,
        "resume_loaded": False,
        "maximum_observed_policy_update_lag": 0,
        "storage_admission_checks": 0,
        "maximum_unchecked_outer_rows": 0,
        "batch_actions": batch_actions,
        "overlap_learning": overlap_learning,
        "learner_device": learner_device,
        "maximum_prepared_updates": 0,
    }

    def usage() -> int:
        return directory_bytes((options.output, run_work, *extra_roots))

    def admit_storage() -> bool:
        # Every admission reads fresh path metadata. Between admissions the
        # reserved journal group permits at most 64 newly written outer rows;
        # all actor-private native evidence remains covered by full sessions.
        nonlocal unchecked_outer_rows, storage_admission_required
        used = usage()
        report["storage_admission_checks"] += 1
        unchecked_outer_rows, storage_admission_required = 0, False
        return used + reserve <= options.max_run_bytes

    def stop_reason() -> str | None:
        if options.stop_file.exists():
            return "STOP file"
        if time.monotonic() - start >= options.max_wall_seconds:
            return "full host training deadline"
        return None

    def timed(name: str, operation: Any, *args: Any, **kwargs: Any) -> Any:
        before = time.monotonic()
        try:
            return operation(*args, **kwargs)
        finally:
            phase[name] += time.monotonic() - before

    def record_cleanup(actor: int, cleanup: dict[str, Any]) -> None:
        retained = cleanup.get("retained_local_directory")
        if retained:
            extra_roots.add(Path(retained))
        index = sessions.pop(actor, None)
        if index is not None:
            report["cleanup"].append({"actor_id": actor, **cleanup})
            session = report["sessions"][index]
            session["cleanup_index"] = len(report["cleanup"]) - 1
            session["trace_accounting"] = cleanup.get("trace_accounting")
            supervisor = cleanup.get("supervisor")
            if isinstance(supervisor, dict):
                session["driver_identity"] = supervisor.get("driverIdentity")
                session["supervisor_identity"] = supervisor.get("supervisorIdentity")
        if cleanup.get("verified") is not True:
            raise RuntimeError(f"actor{actor}: owned native cleanup could not be verified")

    def close_session(actor: int) -> None:
        nonlocal storage_admission_required
        cleanup = timed("cleanup", pool.close_session, actor)
        record_cleanup(actor, cleanup)
        storage_admission_required = True

    def checkpoint() -> None:
        if pool is not None:
            if pool.in_flight:
                raise RuntimeError("checkpoint requires a complete round without in-flight work")
            if not pool_closed:
                pool.checkpoint_barrier()
        if overlap_learning:
            # Prepared batches already advanced the sampler. Finish this
            # bounded work even after STOP before serializing its RNG state.
            apply_prepared(force=True)
            policy.load_state(learner.model.state_dict())
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "common_contract": contract.envelope(),
            "schedule": schedule,
            "runtime_binding": binding,
            "run_id": run_id,
            "completed_rounds": completed_rounds,
            "actor_next_episode": next_episode,
            "actor_cumulative_transitions": {
                actor: historical_rows[actor] + new_rows[actor] for actor in ids
            },
            "resume_requires_fresh_episodes": True,
            "learner": learner.state_dict(),
            "replay": replay.state_dict(),
            "sampler": sampler.state_dict(),
            "budget": budget.state_dict(),
        }
        active_reserve = (
            actors * session_limit + single.JOURNAL_ROW_BYTES * single.JOURNAL_GROUP_ROWS
            if sessions
            else 0
        )
        allowance = min(
            options.max_checkpoint_bytes,
            options.max_run_bytes - usage() - MIB - active_reserve - accelerator_cache_reserve,
        )
        report["checkpoint"] = timed(
            "checkpoint",
            atomic_checkpoint,
            options.output / "latest.pt",
            state,
            max_bytes=max(0, allowance),
        )

    def drain_credit() -> None:
        request = BatchRequest(options.batch_size, gamma=contract.gamma)
        while budget.integral_available and len(replay) >= options.batch_size:
            if stop_reason():
                return
            metrics, _ = timed("learner", lambda: learner.update(sampler.sample(replay, request)))
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("nonfinite multi learner metrics")
            budget.consume()
            report["last_metrics"] = dict(metrics)

    def prepare_credit() -> None:
        # Collate at the same per-row replay boundary as synchronous training.
        # Only execution is delayed; a later peer cannot enter these batches.
        request = BatchRequest(options.batch_size, gamma=contract.gamma)
        while budget.integral_available > len(prepared_updates) and len(replay) >= 32:
            if len(prepared_updates) >= maximum_prepared_updates:
                raise ValueError("bounded prepared-update queue exhausted")
            prepared_updates.append(timed("sample", sampler.sample, replay, request))
            report["maximum_prepared_updates"] = max(
                report["maximum_prepared_updates"], len(prepared_updates)
            )

    def apply_prepared(*, force: bool = False) -> None:
        while prepared_updates:
            if not force and stop_reason():
                return
            metrics, _ = timed("learner", learner.update, prepared_updates[0])
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("nonfinite prepared learner metrics")
            budget.consume()
            prepared_updates.popleft()
            report["last_metrics"] = dict(metrics)

    try:
        if options.resume is not None:
            if single._file_sha(options.resume) != options.resume_sha256:
                raise ValueError("resume checkpoint SHA-256 mismatch")
            saved = torch.load(options.resume, map_location="cpu", weights_only=False)
            if (
                saved.get("schema") != CHECKPOINT_SCHEMA
                or saved.get("schedule") != schedule
                or saved.get("runtime_binding") != binding
                or saved.get("resume_requires_fresh_episodes") is not True
                or type(saved.get("completed_rounds")) is not int
                or saved["completed_rounds"] < 0
                or not isinstance(saved.get("run_id"), str)
            ):
                raise ValueError("multi checkpoint schema/actor/scheduler/runtime mismatch")
            for name in ("actor_next_episode", "actor_cumulative_transitions"):
                value = saved.get(name)
                if (
                    not isinstance(value, dict)
                    or set(value) != set(ids)
                    or any(type(key) is not int for key in value)
                    or any(type(count) is not int or count < 0 for count in value.values())
                ):
                    raise ValueError("multi checkpoint requires exact per-actor counters")
            contract.require_checkpoint(saved["common_contract"])
            replay.load_state_dict(saved["replay"])
            sampler.load_state_dict(saved["sampler"])
            budget.load_state_dict(saved["budget"])
            learner.load_state_dict(saved["learner"])
            total_rows = sum(saved["actor_cumulative_transitions"].values())
            rounds = saved["completed_rounds"]
            if (
                learner.update_count != budget.updates
                or total_rows != budget.transitions
                or replay.available_id_range().stop != budget.transitions
                or (total_rows == 0 and rounds != 0)
                or (
                    total_rows > 0
                    and not max(saved["actor_cumulative_transitions"].values())
                    <= rounds
                    <= total_rows
                )
            ):
                raise ValueError(
                    "multi checkpoint duration/learner/replay/actor/round counters disagree"
                )
            next_episode = dict(saved["actor_next_episode"])
            historical_rows = dict(saved["actor_cumulative_transitions"])
            run_id, completed_rounds = saved["run_id"], saved["completed_rounds"]
            report.update(
                resume_loaded=True,
                resumed_transitions=budget.transitions,
                resumed_optimizer_steps=learner.update_count,
            )
        report["run_id"] = run_id
        report["initial_model_sha256"] = model_weights_sha256(learner.model)
        if learner_device == "cuda":
            from trackmaniarl.algorithms.value_based.policy import (
                DiscreteValuePolicy,
                DiscreteValuePolicyConfig,
            )

            # Exploration stays on the main CPU RNG. Environment workers never
            # import Torch; only learner numerical execution moves to CUDA.
            policy = DiscreteValuePolicy(
                learner.model,
                torch.device("cpu"),
                DiscreteValuePolicyConfig(
                    learner.exploration_epsilon,
                    learner.policy_action_ids,
                    learner.online_risk,
                    learner.evaluation_risk,
                    learner.action_selector,
                ),
            )
            report["learner_execution"] = {
                **schedule["learner_execution"],
                "device_name": torch.cuda.get_device_name(learner.device),
                "device_capability": list(torch.cuda.get_device_capability(learner.device)),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
            if (
                report["learner_execution"]["matmul_precision"] != "highest"
                or report["learner_execution"]["matmul_allow_tf32"]
                or report["learner_execution"]["cudnn_allow_tf32"]
                or not report["learner_execution"]["deterministic_algorithms"]
            ):
                raise ValueError("CUDA training requires deterministic IEEE float32 execution")
        else:
            policy = learner.policy()
        initial_updates = learner.update_count
        drain_credit()
        if learner.update_count != initial_updates:
            policy.load_state(learner.model.state_dict())
        batched_policy = None
        if batch_actions:
            from trackmaniarl.trackmania.common_training.batched_policy import BatchedCommonPolicy

            batched_policy = BatchedCommonPolicy(policy)
        if pool_factory is None:
            from trackmaniarl.trackmania.native_headless.concurrent import NativeActorPool

            pool_factory = NativeActorPool
        pool = pool_factory(
            config,
            actor_count=actors,
            trace_root=options.output / "native",
            work_root=run_work,
            request_timeout_s=min(config.timeout_s, options.max_wall_seconds),
        )
        timed("pool_startup", pool.open)
        checkpoint()
        with (options.output / "transitions.jsonl").open("xb") as journal:
            while report["transitions"] < options.transitions:
                if reason := stop_reason():
                    report.update(stopped=True, stop_reason=reason)
                    break
                reset_ids = [
                    actor
                    for actor in ids
                    if new_rows[actor] < assigned[actor] and actor not in observations
                ]
                round_rows = len(observations) + len(reset_ids)
                if (
                    reset_ids
                    or storage_admission_required
                    or unchecked_outer_rows + round_rows > single.JOURNAL_GROUP_ROWS
                ) and not admit_storage():
                    report.update(stopped=True, stop_reason="run evidence budget")
                    break
                durations = {
                    actor: min(
                        options.episode_native_units,
                        (assigned[actor] - new_rows[actor]) * options.native_iterations * 10,
                    )
                    for actor in reset_ids
                }
                eligible: dict[int, bool] = dict.fromkeys(reset_ids, False)
                if options.reuse_sessions:
                    requests = {
                        actor: {
                            "minimum_remaining_seconds": max(
                                30.0, 2 * last_episode_seconds[actor] + 5.0
                            ),
                            "additional_trace_bytes": 2 * MIB
                            + math.ceil(durations[actor] / (10 * options.native_iterations)) * 65536
                            + (durations[actor] // 10) * 4096,
                        }
                        for actor in reset_ids
                        if actor in sessions
                    }
                    if requests:
                        status = pool.session_reuse_status_many(requests)
                        for actor in sorted(requests):
                            eligible[actor] = status[actor]["eligible"] is True
                            if not eligible[actor]:
                                report["session_rollovers"].append(
                                    {"actor_id": actor, **status[actor]}
                                )
                fresh = [actor for actor in reset_ids if not eligible[actor]]
                for actor in fresh:
                    if actor in sessions:
                        close_session(actor)
                if fresh and not admit_storage():
                    report.update(stopped=True, stop_reason="run evidence budget")
                    break
                if reason := stop_reason():
                    report.update(stopped=True, stop_reason=reason)
                    break
                if fresh:
                    remaining = options.max_wall_seconds - (time.monotonic() - start)
                    prepared = pool.prepare_many(
                        {
                            actor: {
                                "seed": options.seed + actor + actors * next_episode[actor],
                                "timeout_s": min(config.timeout_s, remaining),
                            }
                            for actor in fresh
                        }
                    )
                    for actor in sorted(fresh):
                        sessions[actor] = len(report["sessions"])
                        report["sessions"].append(
                            {
                                "actor_id": actor,
                                "trace_dir": prepared[actor]["trace_dir"],
                                "work_dir": prepared[actor]["work_dir"],
                                "episodes": 0,
                            }
                        )
                # Lazy facades cannot launch native until after this last
                # admission accounts for their worker/lease paths as well.
                if (reset_ids and not admit_storage()) or stop_reason():
                    report.update(stopped=True, stop_reason=stop_reason() or "run evidence budget")
                    break
                if reset_ids:
                    reset_start = time.monotonic()
                    reset = timed(
                        "reset",
                        pool.reset_many,
                        {
                            actor: {
                                "seed": options.seed + actor + actors * next_episode[actor],
                                "duration_limit": durations[actor],
                            }
                            for actor in reset_ids
                        },
                    )
                    for actor in sorted(reset_ids):
                        item = reset[actor]
                        episode = next_episode[actor]
                        next_episode[actor] += 1
                        adapter = NativeOnlyAdapter(
                            contract, proof=components.proof, reference=components.reference
                        )
                        observations[actor] = adapter.bind_reset(
                            item["state"],
                            item["info"],
                            episode_id=f"native-multi/{run_id}/{actor}/{episode}",
                        )
                        adapters[actor] = adapter
                        policy.reset_episode()
                        case = {
                            "actor_id": actor,
                            "episode": episode,
                            "session_index": sessions[actor],
                            "raw_journal": item["metadata"]["journal_path"],
                            "duration_limit": durations[actor],
                            "reused_session": eligible[actor],
                            "transitions": 0,
                            "terminated": False,
                            "truncated": False,
                            "reset_nativeT": item["state"]["nativeT"],
                            "branch_pid": item["info"].get("episodeProcessId"),
                            "reset_host_seconds": time.monotonic() - reset_start,
                            "episode_host_started": time.monotonic(),
                        }
                        cases[actor] = case
                        report["episodes"].append(case)
                        report["sessions"][sessions[actor]]["episodes"] += 1
                if reason := stop_reason():
                    report.update(stopped=True, stop_reason=reason)
                    break
                active = sorted(observations)
                version = learner.update_count
                actions = timed(
                    "policy",
                    lambda active=active: (
                        batched_policy.act_many(observations, PolicyMode.ONLINE)
                        if batched_policy is not None
                        else {
                            actor: policy.act(observations[actor], PolicyMode.ONLINE)
                            for actor in active
                        }
                    ),
                )
                report["model_executed"] = True
                if reason := stop_reason():
                    report.update(stopped=True, stop_reason=reason)
                    break
                step_requests = {
                    actor: {
                        "action": actions[actor],
                        "native_iterations": options.native_iterations,
                    }
                    for actor in active
                }
                if overlap_learning:
                    timed("environment_dispatch", pool.begin_step_many, step_requests)
                    # Workers own only environments. The main thread owns all
                    # model, sampler and replay mutations while they advance.
                    apply_prepared()
                    results = timed("environment_step", pool.finish_step_many)
                else:
                    results = timed("environment_step", pool.step_many, step_requests)
                if set(results) != set(active):
                    raise ValueError("complete actor-indexed round results required")
                # Validate EVERY result before journal/replay/credit/update.
                staged = {
                    actor: adapters[actor].transition(actions[actor], results[actor])
                    for actor in active
                }
                # A final actor CLOSE is part of accepting its last round:
                # verify it before any peer row obtains journal/replay credit.
                for actor in active:
                    row = staged[actor]
                    if (row.terminated or row.truncated) and new_rows[actor] + 1 == assigned[actor]:
                        close_session(actor)
                round_id = completed_rounds
                records = []
                for actor in active:
                    row = staged[actor]
                    raw = (
                        json.dumps(
                            {
                                "actor_id": actor,
                                "round_id": round_id,
                                "policy_update_version": version,
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
                    ).encode()
                    if len(raw) > single.JOURNAL_ROW_BYTES:
                        raise ValueError("outer transition journal row exceeds reserved size")
                    records.append(raw)
                block = b"".join(records)
                if journal.write(block) != len(block):
                    raise OSError("short complete-round journal write")
                journal.flush()
                unchecked_outer_rows += len(active)
                report["maximum_unchecked_outer_rows"] = max(
                    report["maximum_unchecked_outer_rows"], unchecked_outer_rows
                )
                ended = False
                admissions = []
                for actor in active:
                    row = staged[actor]
                    lag = learner.update_count - version
                    admissions.append(
                        {
                            "actor_id": actor,
                            "issued_policy_update_version": version,
                            "learner_update_version_at_admission": learner.update_count,
                            "update_lag": lag,
                        }
                    )
                    report["maximum_observed_policy_update_lag"] = max(
                        report["maximum_observed_policy_update_lag"], lag
                    )
                    replay.append(row)
                    budget.credit(row)
                    new_rows[actor] += 1
                    report["transitions"] += 1
                    report["native_intervals"] += results[actor]["actualIterations"]
                    case = cases[actor]
                    case["transitions"] += 1
                    case["last_nativeT"] = results[actor]["observation"]["nativeT"]
                    case["reward"] = case.get("reward", 0.0) + row.reward
                    if overlap_learning:
                        prepare_credit()
                    else:
                        drain_credit()
                    if row.terminated or row.truncated:
                        ended = True
                        case.update(terminated=row.terminated, truncated=row.truncated)
                        last_episode_seconds[actor] = (
                            time.monotonic() - case["episode_host_started"]
                        )
                        case["episode_host_seconds"] = last_episode_seconds[actor]
                        del observations[actor]
                    else:
                        if new_rows[actor] >= assigned[actor]:
                            raise ValueError(
                                "actor environment missed its explicit final duration endpoint"
                            )
                        observations[actor] = row.next_observation
                completed_rounds += 1
                if learner.update_count != version:
                    policy.load_state(learner.model.state_dict())
                report["rounds"].append(
                    {
                        "round_id": round_id,
                        "actor_ids": active,
                        "policy_update_version": version,
                        "learner_update_version_after_round": learner.update_count,
                        "admissions": admissions,
                    }
                )
                if (
                    ended
                    or report["transitions"] // options.checkpoint_every
                    != (report["transitions"] - len(active)) // options.checkpoint_every
                ):
                    checkpoint()
        if overlap_learning:
            apply_prepared(force=True)
        drain_credit()
        if batched_policy is not None:
            report["batched_policy"] = {
                "batch_count": batched_policy.batch_count,
                "serial_recheck_count": batched_policy.serial_recheck_count,
            }
        report.update(passed=True, optimizer_steps_this_run=learner.update_count - initial_updates)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
        evidence = getattr(error, "evidence", None)
        if evidence is not None:
            report["pool_failure_evidence"] = evidence
    finally:
        if pool is not None:
            try:
                cleanup = timed("cleanup", pool.close_all)
                pool_closed = True
                cleanup_errors = []
                for actor in sorted(cleanup):
                    try:
                        record_cleanup(actor, cleanup[actor])
                    except (Exception, KeyboardInterrupt) as error:
                        cleanup_errors.append(repr(error))
                report["worker_cleanup"] = cleanup
                if cleanup_errors:
                    raise RuntimeError(f"actor cleanup verification failures: {cleanup_errors}")
                if any(item.get("worker_reaped") is not True for item in cleanup.values()):
                    raise RuntimeError("all actor workers must be reaped")
                if any(
                    type(item.get("worker_exitcode")) is not int or item["worker_exitcode"] != 0
                    for item in cleanup.values()
                ):
                    raise RuntimeError("successful training requires zero actor worker exit codes")
            except (Exception, KeyboardInterrupt) as error:
                report.update(passed=False, cleanup_error=repr(error))
                report.setdefault("error", repr(error))
                evidence = getattr(error, "evidence", None)
                if evidence is not None:
                    report["cleanup_abort_evidence"] = evidence
                    try:
                        recovered = pool.close_all()
                        pool_closed = True
                        report["worker_cleanup"] = recovered
                        for actor in sorted(recovered):
                            record_cleanup(actor, recovered[actor])
                    except (Exception, KeyboardInterrupt) as recovery_error:
                        report["cleanup_recovery_error"] = repr(recovery_error)
        report.update(
            cumulative_transitions=budget.transitions,
            optimizer_steps=learner.update_count,
            retained_rows=len(replay),
            credited_simulated_seconds=budget.elapsed_seconds,
            remaining_update_credit=budget.credit_available,
            remaining_integral_update_credit=budget.integral_available,
            completed_rounds=completed_rounds,
            actor_new_transitions=new_rows,
            actor_next_episode=next_episode,
            final_model_sha256=model_weights_sha256(learner.model),
            native_environment_instances=len(report["sessions"]),
            native_sessions_started=sum(
                item.get("driver_identity") is not None for item in report["sessions"]
            ),
            native_episode_processes=sum(
                type(item.get("branch_pid")) is int for item in report["episodes"]
            ),
            owned_cleanup_verified=all(item["verified"] for item in report["cleanup"])
            if report["cleanup"]
            else None,
            retained_local_directories=sorted(str(path) for path in extra_roots),
        )
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
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--reuse-sessions", action="store_true")
    parser.add_argument("--batch-actions", action="store_true")
    parser.add_argument("--overlap-learning", action="store_true")
    parser.add_argument("--learner-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-wall-seconds", type=float, default=240)
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
    args = parser.parse_args()
    report: dict[str, Any] = {"schema": SCHEMA, "passed": False, "experimental": True}
    created = False
    try:
        from scripts.headless_native_train import TrainComponents, TrainOptions
        from trackmaniarl.trackmania.common_training import CommonPolylineReference
        from trackmaniarl.trackmania.common_training.adapters import KNOWN_REFERENCE_SHA256
        from trackmaniarl.trackmania.common_training.native_only import (
            NativeOnlyClockProof,
            native_only_contract,
        )
        from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
        from trackmaniarl.trackmania.native_headless.training_storage import atomic_json

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
        if type(args.actors) is not int or not 1 <= args.actors <= 8:
            raise ValueError("literal actor count in 1..8 required")
        values = vars(args).copy()
        for key in (
            "config",
            "reference",
            "clock_proof",
            "clock_proof_sha256",
            "actors",
            "batch_actions",
            "overlap_learning",
            "learner_device",
        ):
            values.pop(key)
        values["stop_file"], values["host_started_at"] = (
            args.stop_file or output / "STOP",
            _PROCESS_STARTED,
        )
        options = TrainOptions(**values)
        proof = NativeOnlyClockProof.from_artifact(
            args.clock_proof, expected_sha256=args.clock_proof_sha256
        )
        reference = CommonPolylineReference.from_json(
            args.reference, expected_sha256=KNOWN_REFERENCE_SHA256
        )
        config = NativeRuntimeConfig.from_mapping(json.loads(args.config.read_bytes()))
        output.mkdir(parents=True, exist_ok=False)
        created = True
        contract = native_only_contract(proof, reference)
        atomic_json(output / "contract.json", contract.envelope())
        report = run_training(
            TrainComponents(config=config, proof=proof, reference=reference, contract=contract),
            options,
            actors=args.actors,
            batch_actions=args.batch_actions,
            overlap_learning=args.overlap_learning,
            learner_device=args.learner_device,
        )
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = repr(error)
        if created and not (output / "report.json").exists():
            atomic_json(output / "report.json", report, max_bytes=MIB)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "actor_count",
                    "transitions",
                    "cumulative_transitions",
                    "optimizer_steps",
                    "stopped",
                    "stop_reason",
                    "resume_loaded",
                    "owned_cleanup_verified",
                    "full_host_seconds",
                    "error",
                    "cleanup_error",
                    "checkpoint_error",
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
