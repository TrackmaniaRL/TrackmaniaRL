"""Prepare and optionally run an isolated, FULL-state recovery attempt.

Never edits the source run. A failed/interrupted source must have no surviving
native/training processes or unacknowledged actor spool files. Queue-level
continuation and final evaluation remain separate, explicit steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from tools.headless_campaign_queue.__main__ import atomic_json


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a, b, path: str = "root") -> int:
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or not torch.equal(a.detach().cpu(), b.detach().cpu())
        ):
            raise ValueError(f"{path}: tensor mismatch")
        return 1
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"{path}: array mismatch")
        return 1
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            raise ValueError(f"{path}: mapping keys differ")
        return sum(equal(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"{path}: sequence differs")
        return sum(equal(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if type(a) is not type(b) or a != b:
        raise ValueError(f"{path}: value differs")
    return 1


def source_signatures(journal: Path) -> dict[str, str | None]:
    result = {}
    for suffix in ("", "-wal"):
        path = Path(str(journal) + suffix)
        if path.is_symlink():
            raise ValueError(f"unsafe journal symlink: {path}")
        result[suffix or "db"] = sha(path) if path.exists() else None
    if result["db"] is None:
        raise ValueError("source journal is missing")
    return result


def snapshot_journal(source: Path, destination: Path) -> dict:
    """SQLite backup includes committed rows still present in the WAL."""
    before = source_signatures(source)
    if destination.exists() or destination.is_symlink():
        raise ValueError("journal destination must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # SQLite may create or checkpoint WAL/SHM even for some read-only opens.
    # Copy the original database and WAL first; open only those private copies.
    with tempfile.TemporaryDirectory(prefix="journal-copy-", dir=destination.parent) as tmp:
        local = Path(tmp) / source.name
        shutil.copyfile(source, local)
        wal = Path(str(source) + "-wal")
        if before["-wal"] is not None:
            shutil.copyfile(wal, Path(str(local) + "-wal"))
        if source_signatures(source) != before:
            raise ValueError("source journal changed during snapshot")
        with closing(sqlite3.connect(local.resolve().as_uri() + "?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(destination)) as dst:
                src.backup(dst)
    if source_signatures(source) != before:
        raise ValueError("source journal changed during snapshot")
    return {"source": before, "snapshot_sha256": sha(destination)}


def inspect_journal(path: Path, journal_id: str, frontier: int) -> dict:
    if (
        type(frontier) is not int
        or frontier < 0
        or not isinstance(journal_id, str)
        or not journal_id
    ):
        raise ValueError("invalid checkpoint journal identity or frontier")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("journal integrity check failed")
        meta = dict(db.execute("SELECT key,value FROM metadata"))
        if meta.get("journal_id") != journal_id or int(meta.get("pruned_through", -1)) > frontier:
            raise ValueError("journal identity or pruned frontier mismatch")
        rows = db.execute(
            "SELECT c.id,c.session_id,c.sequence,c.payload,r.row_id,r.payload_sha256 FROM chunks c LEFT JOIN receipts r ON r.session_id=c.session_id AND r.sequence=c.sequence WHERE c.id>? ORDER BY c.id",
            (frontier,),
        ).fetchall()
        for row_id, session, sequence, payload, receipt_id, digest in rows:
            if receipt_id != row_id or bytes(digest or b"") != hashlib.sha256(payload).digest():
                raise ValueError(f"journal receipt mismatch for row {row_id}")
    return {
        "journal_id": journal_id,
        "frontier": frontier,
        "pending_rows": len(rows),
        "pending_ids": [row[0] for row in rows],
    }


def reject_pending_spool(run_dir: Path) -> None:
    actors = run_dir / "actors"
    if actors.is_symlink():
        raise ValueError("unsafe actor spool directory")
    if actors.exists() and any(
        p.is_file() for p in actors.rglob("*") if p.suffix in (".rollout", ".tmp")
    ):
        raise ValueError(
            "unacknowledged actor spool requires separate recovery; original data retained"
        )


def reject_live_competitors() -> None:
    """Conservatively require a quiet Linux host before touching a recovery copy."""
    if sys.platform != "linux" or os.geteuid() != 0:
        raise ValueError("recovery requires Linux root for process visibility")
    own = {os.getpid(), os.getppid()}
    needles = (
        "TrackmaniaServer",
        "multiprocessing.spawn",
        "trackmaniarl\x00train",
        "trackmaniarl\x00resume",
        "headless_campaign_queue.job",
        "native_headless/_runtime/supervisor.py",
        "trackmaniarl.distributed.actor",
        "trackmaniarl.distributed.coordinator",
    )
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in own:
            continue
        try:
            command = (path / "cmdline").read_bytes().decode("utf-8", "replace")
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError("cannot inspect process ownership") from exc
        if any(token in command for token in needles):
            raise ValueError(
                f"live potentially competing process {path.name}; inspect before recovery"
            )


def process_identity(pid: int) -> dict:
    stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(stat[19]),
        "process_group": int(stat[2]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def reject_surviving_group(group: int) -> None:
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(stat[2]) == group and stat[0] != "Z":
            raise ValueError(f"owned process group still has live PID {path.name}")


def wait_for_group_shutdown(group: int, boot_id: str, timeout: float = 20.0) -> None:
    """Allow descendants to finish closing; fail closed if they remain alive."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != boot_id:
            raise ValueError("boot changed while waiting for owned process group")
        try:
            reject_surviving_group(group)
            return
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def validate_state(
    state: Mapping,
    fingerprint: str,
    target: int,
    on_policy: bool,
    *,
    stage: str = "training",
    max_overshoot: int = 0,
) -> dict:
    if not isinstance(state, Mapping) or state.get("run_fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    learner = state.get("learner")
    counters = state.get("counters" if on_policy else "distributed")
    if not isinstance(learner, Mapping) or not learner or not isinstance(counters, Mapping):
        raise ValueError("FULL learner state and counters required")
    rng = learner.get("rng") or learner.get("training", {}).get("rng")
    if not isinstance(rng, Mapping) or not {"python", "numpy", "torch"}.issubset(rng):
        raise ValueError("checkpoint learner RNG state missing")
    if not any("optimizer" in str(key) for key in learner):
        raise ValueError("checkpoint optimizer state missing")
    transitions, updates = counters.get("transitions"), counters.get("updates")
    if type(transitions) is not int or type(updates) is not int or updates < 0:
        raise ValueError("checkpoint must have valid counters")
    if stage == "training":
        if not 0 <= transitions < target:
            raise ValueError("checkpoint must be an incomplete run with valid counters")
    elif stage == "evaluation":
        if (
            type(max_overshoot) is not int
            or max_overshoot < 0
            or not target <= transitions <= target + max_overshoot
        ):
            raise ValueError("evaluation checkpoint is outside the accepted budget")
        from tools.headless_campaign_queue.verify_artifacts import validate_checkpoint

        validate_checkpoint(
            state, fingerprint, target, on_policy=on_policy, max_overshoot=max_overshoot
        )
    else:
        raise ValueError("unsupported recovery stage")
    if not on_policy and (
        not isinstance(state.get("replay_store"), Mapping)
        or not isinstance(state.get("sampler"), Mapping)
    ):
        raise ValueError("FULL replay and sampler state required; reset replay is forbidden")
    return {"transitions": transitions, "updates": updates}


def prove_restore(
    spec,
    base: Path,
    checkpoint: Path,
    state: Mapping,
    fingerprint: str,
    journal_path: Path | None,
    expected_frontier: int | None,
) -> dict:
    """Restore and replay a private copy; never mutate the prepared continuation."""
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.runtime import prepare_run, resolve_run
    from trackmaniarl.core.training import Trainer
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig, ReplayRestoreMode

    with tempfile.TemporaryDirectory(
        prefix="recovery-audit-", dir=checkpoint.parent.parent.parent
    ) as tmp:
        output = Path(tmp)
        components = spec.components.model_copy(update={"additional_loggers": ()})
        updates = {"artifacts_dir": output / "artifacts", "components": components}
        if "checkpoint_root" in type(spec).model_fields:
            updates["checkpoint_root"] = None
        copied = spec.model_copy(update=updates)
        if run_fingerprint(copied, base) != fingerprint:
            raise ValueError("audit configuration changes the fingerprint")
        run = resolve_run(copied, base_dir=base)
        coordinator = None
        replay_leaves = sampler_leaves = None
        try:
            on_policy = bool(getattr(run.learner, "on_policy", False))
            if on_policy:
                prepare_run(run)
                run.learner.setup(
                    {
                        "seed": copied.seed,
                        "run_dir": run.run_dir,
                        "model_factory": run.model_factory,
                        "total_transitions": copied.training.total_transitions,
                        "restoring_checkpoint": True,
                    }
                )
                counters = Trainer(run, resume_checkpoint=checkpoint)._restore_checkpoint(state)
                actual_transitions = counters["transitions"]
            else:
                if journal_path is None:
                    raise ValueError("FULL distributed restore needs a journal")
                target = run.run_dir / "distributed" / "rollouts.sqlite3"
                target.parent.mkdir(parents=True, exist_ok=True)
                snapshot_journal(journal_path, target)
                coordinator = Coordinator(
                    run,
                    CoordinatorConfig(
                        bind="127.0.0.1:41987",
                        token="offline-recovery-audit-token-123456789",
                        fingerprint=fingerprint,
                        resume_checkpoint=checkpoint,
                        restore_mode=ReplayRestoreMode.FULL,
                    ),
                )
                coordinator._prepare_training()
                coordinator.restore_checkpoint(checkpoint, ReplayRestoreMode.FULL)
                counters = coordinator.counters
                if counters.journal_applied_frontier != expected_frontier:
                    raise ValueError("journal replay frontier mismatch")
                if coordinator.journal.has_rows_after(counters.journal_applied_frontier):
                    raise ValueError("journal replay left pending rows")
                actual_transitions = counters.transitions
                if expected_frontier == state["distributed"]["journal_applied_frontier"]:
                    replay_leaves = equal(
                        state["replay_store"], run.replay_store.state_dict(), "replay"
                    )
                    sampler_leaves = equal(state["sampler"], run.sampler.state_dict(), "sampler")
            leaves = equal(state["learner"], run.learner.state_dict(), "learner")
            expected = state["counters" if on_policy else "distributed"]["transitions"]
            if actual_transitions < expected:
                raise ValueError("restored transition counter decreased")
            return {
                "status": "PASS",
                "learner_equal_leaves": leaves,
                "replay_equal_leaves": replay_leaves,
                "sampler_equal_leaves": sampler_leaves,
                "transitions_after_wal": actual_transitions,
                "journal_frontier_after_wal": expected_frontier,
                "note": "replay and sampler equality is checked only when no pending WAL rows are applied",
            }
        finally:
            if coordinator is not None:
                coordinator._close_runtime()
            run.logger.close()


def prepare(
    config: Path,
    checkpoint: Path,
    destination: Path,
    expected_config_sha: str,
    *,
    stage: str = "training",
    max_overshoot: int = 0,
) -> dict:
    from tools.headless_campaign_queue.verify_artifacts import final_checkpoint
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.runtime import import_symbol
    from trackmaniarl.core.spec import RunSpec

    if not all(p.is_absolute() for p in (config, checkpoint, destination)):
        raise ValueError("absolute paths required")
    if any(p.is_symlink() for p in (config, checkpoint, destination)) or destination.exists():
        raise ValueError("fresh non-symlink destination and non-symlink inputs required")
    if sha(config) != expected_config_sha:
        raise ValueError("source config hash differs from campaign manifest")
    spec = RunSpec.from_yaml(config)
    base = config.parent
    run_dir = (base / spec.artifacts_dir / spec.run_id).resolve(strict=True)
    checkpoint_dir = (
        base / (getattr(spec, "checkpoint_root", None) or spec.artifacts_dir) / spec.run_id / "checkpoints"
    ).resolve(strict=True)
    if destination.resolve().is_relative_to(run_dir) or run_dir.is_relative_to(
        destination.resolve()
    ):
        raise ValueError("recovery must be isolated from original run")
    on_policy = bool(getattr(import_symbol(spec.components.learner.class_path), "on_policy", False))
    recorded, _ = final_checkpoint(run_dir, checkpoint_dir, on_policy=on_policy)
    if recorded != checkpoint.resolve(strict=True):
        raise ValueError("checkpoint is not the latest durably recorded checkpoint")
    reject_pending_spool(run_dir)
    fingerprint = run_fingerprint(spec, base)
    original_checkpoint_sha = sha(checkpoint)
    updates = {"artifacts_dir": destination / "artifacts"}
    if "checkpoint_root" in type(spec).model_fields:
        updates["checkpoint_root"] = None
    cloned = spec.model_copy(update=updates)
    destination.mkdir(parents=True)
    clone_config = destination / "recovery.yaml"
    clone_config.write_text(cloned.to_yaml(), encoding="utf-8")
    if run_fingerprint(cloned, destination) != fingerprint:
        raise ValueError("isolated configuration changes training fingerprint")
    cloned_run = destination / "artifacts" / spec.run_id
    copied_checkpoint = cloned_run / "checkpoints" / checkpoint.name
    copied_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint, copied_checkpoint)
    if (
        sha(copied_checkpoint) != original_checkpoint_sha
        or sha(checkpoint) != original_checkpoint_sha
    ):
        raise ValueError("checkpoint changed during copy")
    state = TorchCheckpointCodec().load(copied_checkpoint)
    counters = validate_state(
        state,
        fingerprint,
        spec.training.total_transitions,
        on_policy,
        stage=stage,
        max_overshoot=max_overshoot,
    )
    journal = None
    target_journal = None
    if not on_policy:
        source_journal = run_dir / "distributed" / "rollouts.sqlite3"
        target_journal = cloned_run / "distributed" / "rollouts.sqlite3"
        snapshot = snapshot_journal(source_journal, target_journal)
        journal = {
            **snapshot,
            **inspect_journal(
                target_journal,
                state.get("journal_id"),
                state["distributed"].get("journal_applied_frontier"),
            ),
        }
    frontier = (
        None
        if journal is None
        else (journal["pending_ids"][-1] if journal["pending_ids"] else journal["frontier"])
    )
    proof = prove_restore(
        cloned, destination, copied_checkpoint, state, fingerprint, target_journal, frontier
    )
    if stage == "training" and proof["transitions_after_wal"] >= spec.training.total_transitions:
        if on_policy or proof["transitions_after_wal"] > spec.training.total_transitions + max_overshoot:
            raise ValueError("recovered WAL exceeds the accepted budget")
        stage = "wal_finalization"
    if stage == "evaluation" and not on_policy:
        from tools.headless_campaign_queue.verify_artifacts import validate_drained_journal

        if journal["pending_rows"]:
            raise ValueError("evaluation requires a fully drained final journal")
        validate_drained_journal(cloned_run, state)
    reject_pending_spool(run_dir)
    if sha(checkpoint) != original_checkpoint_sha:
        raise ValueError("source checkpoint changed during restore audit")
    if journal is not None and source_signatures(source_journal) != journal["source"]:
        raise ValueError("source journal changed during restore audit")
    if sha(config) != expected_config_sha:
        raise ValueError("source config changed during preparation")
    report = {
        "kind": "isolated-full-recovery-v1",
        "status": "prepared_not_started",
        "stage": stage,
        "config": str(config),
        "config_sha256": expected_config_sha,
        "source_run": str(run_dir),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": original_checkpoint_sha,
        "fingerprint": fingerprint,
        "counters": counters,
        "journal": journal,
        "offline_restore_proof": proof,
        "recovery_config": str(clone_config),
        "recovery_config_sha256": sha(clone_config),
        "checkpoint": str(copied_checkpoint),
        "on_policy": on_policy,
        "limitations": [
            "pending actor spool is not recovered",
            "in-memory data not durably accepted may be lost",
            "queue continuation is not automatic",
            "evaluation recovery starts a new full suite and retains previous partial results",
        ],
    }
    atomic_json(destination / "recovery.json", report)
    return report


def finish_replayed_coordinator(coordinator, target: int, max_overshoot: int) -> Path:
    """Drain earned updates without an actor, then durably save the full state.

    Only use on a private restored copy after producer shutdown and spool audit.
    This path performs no collection and does not grant an update-credit exemption.
    """
    import math
    from trackmaniarl.distributed.coordinator_learning import _minimum_replay_size

    counters = coordinator.counters
    if not target <= counters.transitions <= target + max_overshoot:
        raise ValueError("replayed transitions outside accepted budget")
    if coordinator.journal.has_rows_after(counters.journal_applied_frontier):
        raise ValueError("journal replay left pending rows")
    credit = counters.update_credit
    if isinstance(credit, bool) or not isinstance(credit, (int, float)) or not math.isfinite(credit) or credit < 0:
        raise ValueError("invalid replayed update credit")
    if credit >= 1 and len(coordinator.run.replay_store) < _minimum_replay_size(coordinator):
        raise ValueError("insufficient replay to drain update credit")
    if getattr(coordinator.external_stop, "wait_for_actor_drain", False):
        raise ValueError("offline finalization cannot wait for an actor")
    if coordinator._has_active_actor() or coordinator._server is not None:
        raise ValueError("offline finalization cannot run with an actor or server")
    if coordinator._external_stop_requested():
        raise ValueError("offline finalization stopped")
    coordinator._learn()
    if coordinator._external_stop_requested() or counters.transitions < target or counters.update_credit >= 1 or coordinator.journal.has_rows_after(counters.journal_applied_frontier):
        raise ValueError("replayed budget finalization did not drain updates and journal")
    checkpoint = coordinator._checkpoint()
    coordinator._checkpoint_writer.wait()
    if not checkpoint.is_file():
        raise ValueError("final checkpoint was not written")
    return checkpoint


def finalize_replayed_budget(spec, base: Path, checkpoint: Path, fingerprint: str, report: dict, max_overshoot: int, stop: Path, original_stop: Path) -> Path:
    """Restore the prepared private copy and finalize replayed budget offline."""
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.runtime import resolve_run
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig, ReplayRestoreMode
    from tools.headless_campaign_queue.verify_artifacts import validate_drained_journal

    if report.get("stage") != "wal_finalization" or run_fingerprint(spec, base) != fingerprint:
        raise ValueError("replayed finalization provenance mismatch")
    source_hash = sha(checkpoint)
    journal = base / "artifacts" / spec.run_id / "distributed" / "rollouts.sqlite3"
    if (source_hash != report.get("source_checkpoint_sha256") or
            sha(Path(report["recovery_config"])) != report.get("recovery_config_sha256") or
            journal.is_symlink() or not journal.is_file() or
            sha(journal) != report.get("journal", {}).get("snapshot_sha256")):
        raise ValueError("prepared recovery inputs changed after the restore audit")
    # Keep the cloned starting checkpoint even if the final update number is unchanged.
    preserved = checkpoint.with_name("source-" + checkpoint.name)
    if preserved.exists():
        raise ValueError("preserved source checkpoint already exists")
    shutil.copyfile(checkpoint, preserved)
    if sha(preserved) != source_hash or sha(checkpoint) != source_hash:
        raise ValueError("checkpoint changed during preservation")
    components = spec.components.model_copy(update={"additional_loggers": ()})
    offline_spec = spec.model_copy(update={"components": components})
    if run_fingerprint(offline_spec, base) != fingerprint:
        raise ValueError("offline finalization changes training fingerprint")
    run = resolve_run(offline_spec, base_dir=base)
    coordinator = None
    try:
        class FileStop:
            def is_set(self):
                return stop.exists() or original_stop.exists()

        coordinator = Coordinator(run, CoordinatorConfig(
            bind="127.0.0.1:0", token="offline-recovery-audit-token-123456789",
            fingerprint=fingerprint, resume_checkpoint=checkpoint, restore_mode=ReplayRestoreMode.FULL,
            external_stop=FileStop(),
        ))
        coordinator._prepare_training()
        coordinator.restore_checkpoint(checkpoint, ReplayRestoreMode.FULL)
        expected = report["offline_restore_proof"]
        if (coordinator.counters.transitions != expected["transitions_after_wal"] or
                coordinator.counters.journal_applied_frontier != expected["journal_frontier_after_wal"]):
            raise ValueError("replayed state differs from independent restore proof")
        final = finish_replayed_coordinator(coordinator, spec.training.total_transitions, max_overshoot)
        final_state = TorchCheckpointCodec().load(final)
        validate_state(final_state, fingerprint, spec.training.total_transitions, False,
                       stage="evaluation", max_overshoot=max_overshoot)
        validate_drained_journal(run.run_dir, final_state)
        report.update(checkpoint=str(final), preserved_checkpoint=str(preserved),
                      preserved_checkpoint_sha256=source_hash,
                      status="training_verified_evaluation_pending")
        atomic_json(base / "recovery.json", report)
        return final
    finally:
        if coordinator is not None:
            coordinator._close_runtime()
        run.logger.close()


def evaluate_copy(spec, base: Path, checkpoint: Path, stop: Path, report: dict) -> int:
    """Run a fresh evaluation suite; never overwrite a partial prior evaluation."""
    from tools.headless_campaign_queue.job import monitor_disk
    from tools.headless_campaign_queue.verify_artifacts import validate_evaluation
    from trackmaniarl.commands.evaluation import (
        _BenchmarkArtifact,
        _configure_evaluation_stop,
        _evaluate_checkpoint,
        _has_runtime_errors,
        _validate_benchmark_artifact,
    )
    from trackmaniarl.core.runtime import resolve_run

    evaluation = spec.evaluation
    if evaluation is None or len(evaluation.maps) != 1 or evaluation.trials_per_map != 30:
        raise ValueError("a pinned single-map, 30-trial evaluation suite is required")
    eval_id = spec.run_id + "-recovery-evaluation"
    eval_spec = spec.model_copy(update={"run_id": eval_id})
    eval_dir = (base / eval_spec.artifacts_dir / eval_id).resolve()
    if eval_dir.exists():
        raise ValueError("evaluation output must be fresh")
    run = resolve_run(eval_spec, base_dir=base)
    if run.run_dir.resolve() != eval_dir:
        run.logger.close()
        raise ValueError("resolved evaluation directory differs from preflight")
    report["status"] = "evaluating"
    atomic_json(base / "recovery.json", report)
    checkpoint_sha = sha(checkpoint)
    try:
        _configure_evaluation_stop(run, stop)
        with monitor_disk(spec, base, stop):
            trials, _, evaluated_checkpoint = _evaluate_checkpoint(run, checkpoint)
        _validate_benchmark_artifact(
            _BenchmarkArtifact(trials, evaluated_checkpoint, evaluation, checkpoint)
        )
        if _has_runtime_errors(trials) or stop.exists() or sha(checkpoint) != checkpoint_sha:
            raise ValueError("evaluation interrupted or its input changed")
        artifact = run.run_dir / "evaluation.json"
        validate_evaluation(
            json.loads(artifact.read_text(encoding="utf-8")),
            checkpoint,
            evaluation.maps[0].id,
            evaluation.maps[0].expected_map_uid,
            30,
        )
        report.update(
            status="evaluation_verified_queue_pending",
            evaluation=str(artifact),
            evaluation_sha256=sha(artifact),
            checkpoint_sha256=checkpoint_sha,
        )
        atomic_json(base / "recovery.json", report)
        return 3  # queue continuation and receipt linkage are still required
    finally:
        run.logger.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--stage", choices=("training", "evaluation"), default="training")
    parser.add_argument("--max-overshoot", type=int, default=0)
    parser.add_argument(
        "--execute", action="store_true", help="resume the isolated FULL-state copy"
    )
    args = parser.parse_args()
    os.environ["WANDB_MODE"] = "offline"
    lock = Path.home() / ".trackmaniarl-headless-campaign.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("Recovery blocked: campaign lock exists; inspect ownership first.", file=sys.stderr)
        return 2
    release_lock = True
    try:
        reject_live_competitors()
        report = prepare(
            args.config,
            args.checkpoint,
            args.destination,
            args.config_sha256,
            stage=args.stage,
            max_overshoot=args.max_overshoot,
        )
        if not args.execute:
            print(json.dumps(report))
            return 0
        # prepare may discover that replayed WAL already reached the target.
        # This standalone CLI cannot safely finalize and audit that path yet;
        # never start the normal actor-backed resume in its place.
        if report.get("stage") != args.stage:
            raise ValueError("replayed WAL requires audited budget finalization")
        reject_live_competitors()
        stop = args.destination / "STOP"
        if (
            sha(Path(report["recovery_config"])) != report["recovery_config_sha256"]
            or sha(Path(report["checkpoint"])) != report["source_checkpoint_sha256"]
        ):
            raise ValueError("recovery inputs changed after the restore audit")
        if (
            report["journal"] is not None
            and sha(
                args.destination
                / "artifacts"
                / Path(report["source_run"]).name
                / "distributed"
                / "rollouts.sqlite3"
            )
            != report["journal"]["snapshot_sha256"]
        ):
            raise ValueError("recovery journal changed after the restore audit")
        command = [
            sys.executable,
            "-m",
            "trackmaniarl",
            "resume",
            report["recovery_config"],
            report["checkpoint"],
            "--stop-file",
            str(stop),
        ]
        from tools.headless_campaign_queue.disk_guard import check_disk_space
        from tools.headless_campaign_queue.job import monitor_disk
        from trackmaniarl.core.spec import RunSpec

        cloned_spec = RunSpec.from_yaml(Path(report["recovery_config"]))
        check_disk_space(cloned_spec, args.destination)
        if args.stage == "evaluation":
            release_lock = False
            result = evaluate_copy(
                cloned_spec, args.destination, Path(report["checkpoint"]), stop, report
            )
            reject_live_competitors()
            release_lock = True
            return result
        report["status"] = "starting"
        atomic_json(args.destination / "recovery.json", report)
        release_lock = False
        with (
            (args.destination / "stdout.log").open("xb") as out,
            (args.destination / "stderr.log").open("xb") as err,
        ):
            with monitor_disk(cloned_spec, args.destination, stop):
                env = os.environ.copy()
                env["WANDB_MODE"] = "offline"
                process = subprocess.Popen(
                    command, stdout=out, stderr=err, start_new_session=True, env=env
                )
                report.update(status="running", process=process_identity(process.pid))
                atomic_json(args.destination / "recovery.json", report)
                code = process.wait()
        wait_for_group_shutdown(report["process"]["process_group"], report["process"]["boot_id"])
        reject_live_competitors()
        release_lock = True
        report.update(
            status="training_exited_unverified" if code == 0 else "failed", exit_code=code
        )
        atomic_json(args.destination / "recovery.json", report)
        # Exit zero is not a verified final checkpoint or evaluation; 3 signals
        # that explicit finalization is still needed, never campaign success.
        return 3 if code == 0 else 1
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Recovery blocked: {exc}", file=sys.stderr)
        return 2
    finally:
        if release_lock:
            lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
