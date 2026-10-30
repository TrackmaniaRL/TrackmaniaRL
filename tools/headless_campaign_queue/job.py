"""Experimental single-job wrapper. Never launch without native qualification."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
from contextlib import contextmanager

from tools.headless_campaign_queue.__main__ import atomic_json
from tools.headless_campaign_queue.disk_guard import check_disk_space
from tools.headless_campaign_queue.verify_artifacts import final_checkpoint, validate_checkpoint, validate_drained_journal, validate_evaluation


@contextmanager
def monitor_disk(spec, base: Path, stop: Path, *, interval: float = 10):
    """Request cooperative stop if the actual volume reserve is breached."""
    done = threading.Event()
    errors = []

    def watch():
        while not done.wait(interval):
            try:
                check_disk_space(spec, base)
            except (OSError, ValueError) as exc:
                errors.append(exc)
                try:
                    stop.touch(exist_ok=True)
                except OSError as stop_error:
                    errors.append(stop_error)
                return

    thread = threading.Thread(target=watch, name="campaign-disk-monitor", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
    if errors:
        raise RuntimeError(f"disk reserve check failed: {errors[0]}")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(config: Path, receipt: Path, budget: int, stop: Path, *, max_overshoot: int, qualification_trials: int | None = None, expected_config_sha: str | None = None, expected_fingerprint: str | None = None) -> None:
    # Imports are deferred so merely inspecting the wrapper starts no environment.
    from trackmaniarl.commands.evaluation import _BenchmarkArtifact, _configure_evaluation_stop, _evaluate_checkpoint, _has_runtime_errors, _validate_benchmark_artifact
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.runtime import import_symbol, resolve_run
    from trackmaniarl.core.spec import RunSpec

    job_id = os.environ.get("TRACKMANIARL_QUEUE_JOB_ID")
    manifest = os.environ.get("TRACKMANIARL_QUEUE_MANIFEST_SHA256")
    if not job_id or not manifest or len(manifest) != 64:
        raise ValueError("job must be launched by the campaign queue")
    if not all(p.is_absolute() for p in (config, receipt, stop)):
        raise ValueError("config, receipt and stop paths must be absolute")
    if receipt.exists() or stop.exists():
        raise ValueError("receipt and stop paths must be fresh")
    original_config_sha = file_sha(config)
    if qualification_trials is None and (not expected_config_sha or not expected_fingerprint):
        raise ValueError("campaign must pin its configuration and run fingerprint")
    if expected_config_sha is not None and expected_config_sha != original_config_sha:
        raise ValueError("configuration changed since campaign preflight")
    spec = RunSpec.from_yaml(config)
    check_disk_space(spec, config.parent)
    if spec.training.total_transitions != budget or budget <= 0:
        raise ValueError("configuration budget mismatch")
    if type(max_overshoot) is not int or max_overshoot < 0 or not spec.training.save_final_checkpoint:
        raise ValueError("explicit nonnegative acceptance limit and final checkpoint required")
    evaluation = spec.evaluation
    expected_trials = 30 if qualification_trials is None else qualification_trials
    if type(expected_trials) is not int or not 1 <= expected_trials <= 30:
        raise ValueError("qualification trials must be between 1 and 30")
    if evaluation is None or len(evaluation.maps) != 1 or evaluation.trials_per_map != expected_trials:
        raise ValueError("evaluation map or trial count mismatch")
    if budget not in spec.training.checkpoint_milestones:
        raise ValueError("final milestone is not configured")
    base = config.parent
    run_dir = (base / spec.artifacts_dir / spec.run_id).resolve()
    checkpoint_root = (base / (spec.checkpoint_root or spec.artifacts_dir) / spec.run_id / "checkpoints").resolve()
    evaluation_id = f"{spec.run_id}-final-evaluation"
    evaluation_dir = (base / spec.artifacts_dir / evaluation_id).resolve()
    if run_dir.exists() or checkpoint_root.exists() or evaluation_dir.exists():
        raise ValueError("run and final evaluation artifact directories must be fresh")
    fingerprint = run_fingerprint(spec, base)
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError("source or configuration changed since campaign preflight")
    with monitor_disk(spec, base, stop):
        code = subprocess.call([sys.executable, "-m", "trackmaniarl", "train", str(config), "--stop-file", str(stop)])
    if code != 0 or stop.exists():
        raise RuntimeError("training interrupted or failed")
    if file_sha(config) != original_config_sha or run_fingerprint(spec, base) != fingerprint:
        raise RuntimeError("source or configuration changed during training")
    on_policy = bool(getattr(import_symbol(spec.components.learner.class_path), "on_policy", False))
    if not on_policy and not spec.distributed.strict_update_budget:
        raise ValueError("off-policy campaign requires strict update budget")
    checkpoint, logged_update = final_checkpoint(run_dir, checkpoint_root, on_policy=on_policy)
    state = TorchCheckpointCodec().load(checkpoint)
    actual = validate_checkpoint(state, fingerprint, budget, on_policy=on_policy, max_overshoot=max_overshoot)
    counters = state["counters" if on_policy else "distributed"]
    if counters["updates"] != logged_update:
        raise ValueError("final checkpoint update counter differs from completion log")
    if not on_policy:
        validate_drained_journal(run_dir, state)
    checkpoint_sha = file_sha(checkpoint)
    check_disk_space(spec, base)
    eval_spec = spec.model_copy(update={"run_id": evaluation_id})
    run = resolve_run(eval_spec, base_dir=base)
    try:
        _configure_evaluation_stop(run, stop)
        with monitor_disk(spec, base, stop):
            trials, _, evaluated_checkpoint = _evaluate_checkpoint(run, checkpoint)
        _validate_benchmark_artifact(_BenchmarkArtifact(trials, evaluated_checkpoint, evaluation, checkpoint))
        if _has_runtime_errors(trials):
            raise RuntimeError("final evaluation had runtime errors")
    finally:
        run.logger.close()
    if stop.exists() or file_sha(config) != original_config_sha or run_fingerprint(spec, base) != fingerprint or file_sha(checkpoint) != checkpoint_sha:
        raise RuntimeError("stop requested or source/configuration/checkpoint changed during evaluation")
    # Check exact identities again against the serialized artifact, not the gate.
    import json
    artifact_path = evaluation_dir / "evaluation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_evaluation(artifact, checkpoint, evaluation.maps[0].id, evaluation.maps[0].expected_map_uid, expected_trials)
    atomic_json(receipt, {"job_id": job_id, "status": "completed", "requested_transitions": budget, "transitions": actual, "max_overshoot": max_overshoot,
                          "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                          "evaluation": str(artifact_path), "evaluation_sha256": file_sha(artifact_path),
                          "run_fingerprint": fingerprint, "manifest_sha256": manifest,
                          "process_pid": os.getpid(), "qualification_only": qualification_trials is not None})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-overshoot", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--run-fingerprint")
    parser.add_argument("--qualification-trials", type=int, help="explicit qualification-only trial count; cannot satisfy the campaign queue")
    args = parser.parse_args()
    try:
        execute(args.config, args.receipt, args.budget, args.stop_file, max_overshoot=args.max_overshoot, qualification_trials=args.qualification_trials, expected_config_sha=args.config_sha256, expected_fingerprint=args.run_fingerprint)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Job failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
