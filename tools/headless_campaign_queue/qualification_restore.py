"""Restore one short-run checkpoint in isolation; never start an environment.

The caller must coordinate exclusive use of the configured accelerator. This is
an offline state-equality check, not proof that a live continuation will work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a: Any, b: Any, path: str = "root") -> int:
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
        raise ValueError(f"{path}: scalar mismatch")
    return 1


def audit(
    config: Path, config_sha: str, receipt: Path, receipt_sha: str, deployment: Path, output: Path
) -> dict:
    import trackmaniarl
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.runtime import prepare_run, resolve_run
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.core.training import Trainer
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig, ReplayRestoreMode

    deployment = deployment.resolve(strict=True)
    if not Path(trackmaniarl.__file__).resolve().is_relative_to(deployment):
        raise ValueError("imports must come from the pinned deployment")
    if (
        not all(p.is_absolute() for p in (config, receipt, output))
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("absolute input paths and a fresh output directory are required")
    if sha(config) != config_sha or sha(receipt) != receipt_sha:
        raise ValueError("input hash mismatch")
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    if doc.get("status") != "completed" or doc.get("qualification_only") is not True:
        raise ValueError("a completed qualification-only receipt is required")
    checkpoint = Path(doc["checkpoint"])
    if (
        not checkpoint.is_absolute()
        or checkpoint.is_symlink()
        or sha(checkpoint) != doc["checkpoint_sha256"]
    ):
        raise ValueError("qualification checkpoint hash mismatch")
    evaluation = Path(doc["evaluation"])
    if (
        not evaluation.is_absolute()
        or evaluation.is_symlink()
        or sha(evaluation) != doc["evaluation_sha256"]
    ):
        raise ValueError("qualification evaluation hash mismatch")
    spec = RunSpec.from_yaml(config)
    expected_run = (config.parent / spec.artifacts_dir / spec.run_id).resolve()
    if checkpoint.resolve().parent != expected_run / "checkpoints":
        raise ValueError("checkpoint does not belong to qualification config")
    if output.parent.resolve(strict=True).is_relative_to(
        expected_run
    ) or output.parent.resolve().is_relative_to(deployment):
        raise ValueError("restore output must be outside the original run and source deployment")
    fingerprint = run_fingerprint(spec, config.parent)
    if fingerprint != doc.get("run_fingerprint"):
        raise ValueError("short-run source/configuration fingerprint mismatch")
    components = spec.components.model_copy(update={"additional_loggers": ()})
    updates = {"artifacts_dir": output / "artifacts", "components": components}
    if "checkpoint_root" in type(spec).model_fields:
        updates["checkpoint_root"] = None
    cloned = spec.model_copy(update=updates)
    if run_fingerprint(cloned, config.parent) != fingerprint:
        raise ValueError("isolated restore changed run fingerprint")
    output.mkdir(parents=True)
    copied_checkpoint = output / "checkpoint.pt"
    shutil.copyfile(checkpoint, copied_checkpoint)
    if sha(copied_checkpoint) != doc["checkpoint_sha256"]:
        raise ValueError("checkpoint copy differs")
    run = coordinator = None
    originals = {
        "config": sha(config),
        "receipt": sha(receipt),
        "checkpoint": sha(checkpoint),
        "evaluation": sha(evaluation),
    }
    try:
        run = resolve_run(cloned, base_dir=config.parent)
        state = run.checkpoint_codec.load(copied_checkpoint)
        if state.get("run_fingerprint") != fingerprint:
            raise ValueError("checkpoint fingerprint mismatch")
        on_policy = bool(getattr(run.learner, "on_policy", False))
        if on_policy:
            prepare_run(run)
            run.learner.setup(
                {
                    "seed": cloned.seed,
                    "run_dir": run.run_dir,
                    "model_factory": run.model_factory,
                    "total_transitions": cloned.training.total_transitions,
                    "restoring_checkpoint": True,
                }
            )
            trainer = Trainer(run, resume_checkpoint=copied_checkpoint)
            counters = trainer._restore_checkpoint(state)
        else:
            source_journal = expected_run / "distributed" / "rollouts.sqlite3"
            source_wal = Path(str(source_journal) + "-wal")
            if (
                source_journal.is_symlink()
                or not source_journal.is_file()
                or (source_wal.exists() and source_wal.stat().st_size)
            ):
                raise ValueError("source journal is unsafe or has a non-empty WAL")
            originals["journal"] = sha(source_journal)
            copydir = run.run_dir / "distributed"
            copydir.mkdir(parents=True, exist_ok=True)
            copy = copydir / "rollouts.sqlite3"
            shutil.copyfile(source_journal, copy)
            if sha(copy) != originals["journal"]:
                raise ValueError("journal copy differs")
            coordinator = Coordinator(
                run,
                CoordinatorConfig(
                    bind="127.0.0.1:41987",
                    token="offline-restore-audit-token-123456789",
                    fingerprint=fingerprint,
                    resume_checkpoint=copied_checkpoint,
                    restore_mode=ReplayRestoreMode.FULL,
                ),
            )
            coordinator._prepare_training()
            coordinator.restore_checkpoint(copied_checkpoint, ReplayRestoreMode.FULL)
            counters = coordinator.counters
            if coordinator.journal.has_rows_after(counters.journal_applied_frontier):
                raise ValueError("pending journal rows were not expected")
        restored = run.learner.state_dict()
        learner_leaves = equal(state["learner"], restored, "learner")
        replay_leaves = sampler_leaves = None
        if not on_policy:
            replay_leaves = equal(state["replay_store"], run.replay_store.state_dict(), "replay")
            sampler_leaves = equal(state["sampler"], run.sampler.state_dict(), "sampler")
        expected_counters = state["counters" if on_policy else "distributed"]
        for key in ("transitions", "updates", "episodes"):
            actual = counters[key] if on_policy else getattr(counters, key)
            if actual != expected_counters[key]:
                raise ValueError(f"{key} counter mismatch")
        learner = state["learner"]
        rng = learner.get("rng") or learner.get("training", {}).get("rng")
        if not isinstance(rng, Mapping) or not {"python", "numpy", "torch"}.issubset(rng):
            raise ValueError("checkpoint learner RNG missing")
        if not any("optimizer" in key for key in learner):
            raise ValueError("checkpoint optimizer state missing")
        after = {
            "config": sha(config),
            "receipt": sha(receipt),
            "checkpoint": sha(checkpoint),
            "evaluation": sha(evaluation),
        }
        if not on_policy:
            if source_wal.exists() and source_wal.stat().st_size:
                raise ValueError("source WAL changed during audit")
            after["journal"] = sha(source_journal)
        if after != originals:
            raise ValueError("original input changed during audit")
        return {
            "kind": "headless-qualification-offline-restore-v1",
            "status": "PASS",
            "job_id": doc["job_id"],
            "source_sha256": originals,
            "run_fingerprint": fingerprint,
            "accepted_transitions": expected_counters["transitions"],
            "updates": expected_counters["updates"],
            "on_policy": on_policy,
            "learner_leaves_equal": learner_leaves,
            "replay_leaves_equal": replay_leaves,
            "sampler_leaves_equal": sampler_leaves,
            "learner_rng_keys": sorted(rng),
            "environment_started": False,
            "learner_update_executed": False,
            "limitations": [
                "no live continuation or physics parity was verified",
                "pending WAL replay was not exercised",
                "short config differs from full campaign",
            ],
        }
    finally:
        if coordinator is not None:
            coordinator._close_runtime()
        if run is not None:
            run.logger.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "config-sha256", "receipt", "receipt-sha256", "deployment", "output"):
        parser.add_argument(
            "--" + key,
            required=True,
            type=Path if key in ("config", "receipt", "deployment", "output") else str,
        )
    parser.add_argument(
        "--assert-exclusive-accelerator",
        action="store_true",
        help="caller asserts no other training is using the accelerator",
    )
    args = parser.parse_args()
    if not args.assert_exclusive_accelerator:
        parser.error("coordinate GPU access, then pass --assert-exclusive-accelerator")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new directory")
    try:
        result = audit(
            args.config,
            args.config_sha256,
            args.receipt,
            args.receipt_sha256,
            args.deployment,
            args.output,
        )
    except Exception as exc:
        result = {
            "kind": "headless-qualification-offline-restore-v1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        if not args.output.is_dir():
            print(json.dumps(result), file=sys.stderr)
            return 1
    report = args.output / "report.json"
    with report.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
