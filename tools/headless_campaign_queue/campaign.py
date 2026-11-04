"""Strict launcher for a separately qualified five-algorithm campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.headless_campaign_queue.__main__ import run
from tools.headless_campaign_queue.job import file_sha


def prepare(path: Path) -> tuple[dict, str]:
    from tools.headless_campaign_queue.disk_guard import check_disk_space
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.spec import RunSpec

    raw = path.read_bytes()
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema_version") != 3
        or doc.get("status") != "qualified"
    ):
        raise ValueError("campaign is not qualified; no job can start")
    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or len(jobs) < 5:
        raise ValueError("campaign must contain at least five jobs")
    expected = {"qr", "iqn", "sac", "tqc", "ppo"}
    supported = expected | {"q", "fqf", "discrete-sac", "redq"}
    ids = [j.get("id") if isinstance(j, dict) else None for j in jobs]
    if any(
        not isinstance(n, str)
        or not n.isascii()
        or not n.replace("-", "").replace("_", "").isalnum()
        for n in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("job identities must be unique alphanumeric names")
    if not expected.issubset(set(ids)):
        raise ValueError("campaign must include QR, IQN, SAC, TQC and PPO")
    if not set(ids).issubset(supported):
        raise ValueError("campaign includes an algorithm without a validated recipe")
    qualification_path = doc.get("qualification_report")
    if not isinstance(qualification_path, str) or not Path(qualification_path).is_absolute():
        raise ValueError("a pinned qualification report is required")
    qualification_file = Path(qualification_path)
    if not qualification_file.is_file() or file_sha(qualification_file) != doc.get(
        "qualification_sha256"
    ):
        raise ValueError("qualification report missing or hash mismatch")
    qualification = json.loads(qualification_file.read_text(encoding="utf-8"))
    if (
        not isinstance(qualification, dict)
        or qualification.get("kind") != "headless-five-qualification-v1"
        or not isinstance(qualification.get("jobs"), dict)
    ):
        raise ValueError("invalid qualification report")
    seed, budget, uid, limit = (
        doc.get("seed"),
        doc.get("transitions"),
        doc.get("map_uid"),
        doc.get("max_overshoot"),
    )
    if (
        type(seed) is not int
        or type(budget) is not int
        or budget <= 0
        or type(limit) is not int
        or limit < 0
        or not isinstance(uid, str)
        or not uid
    ):
        raise ValueError("invalid common seed, budget or map UID")
    repository = Path(__file__).resolve().parent.parent.parent
    compiled, paths = [], set()
    value_learner = "trackmaniarl.algorithms.value_based:DiscreteValueLearner"
    learners = {
        "q": value_learner,
        "qr": value_learner,
        "iqn": value_learner,
        "fqf": value_learner,
        "discrete-sac": "trackmaniarl.algorithms:StableDiscreteSoftActorCritic",
        "sac": "trackmaniarl.algorithms:SoftActorCritic",
        "tqc": "trackmaniarl.algorithms:TruncatedQuantileCritic",
        "redq": "trackmaniarl.algorithms:RandomizedEnsembleSAC",
        "ppo": "trackmaniarl.algorithms:ProximalPolicyOptimization",
    }
    value_heads = {
        "q": ("ScalarQHead", "ScalarValueStrategy"),
        "qr": ("FixedQuantileHead", "FixedQuantileStrategy"),
        "iqn": ("DuelingImplicitQuantileHead", "RandomQuantileStrategy"),
        "fqf": ("DuelingImplicitQuantileHead", "LearnedFractionStrategy"),
    }
    for job in jobs:
        name = job["id"]
        config, receipt, stop = (Path(job[k]) for k in ("config", "receipt", "stop_file"))
        if any(not p.is_absolute() for p in (config, receipt, stop)):
            raise ValueError(f"{name}: paths must be absolute")
        if not config.is_file() or file_sha(config) != job.get("config_sha256"):
            raise ValueError(f"{name}: configuration file or hash mismatch")
        evidence = qualification["jobs"].get(name)
        if (
            not isinstance(evidence, dict)
            or evidence.get("config_sha256") != job.get("config_sha256")
            or evidence.get("run_fingerprint") != job.get("run_fingerprint")
            or any(
                evidence.get(key) != "pass"
                for key in (
                    "native_action_ack",
                    "checkpoint_restore",
                    "evaluation",
                    "clean_process_shutdown",
                )
            )
        ):
            raise ValueError(f"{name}: qualification evidence missing or mismatched")
        if (
            receipt.exists()
            or stop.exists()
            or not receipt.parent.is_dir()
            or not stop.parent.is_dir()
        ):
            raise ValueError(f"{name}: receipt/stop paths must be fresh with existing parents")
        spec = RunSpec.from_yaml(config)
        check_disk_space(spec, config.parent)
        if name in learners and spec.components.learner.class_path != learners[name]:
            raise ValueError(f"{name}: learner does not match algorithm")
        if name in value_heads:
            head = spec.components.model_factory.kwargs.get("head", {}).get("class_path", "")
            strategy = spec.components.model_factory.kwargs.get("strategy", {}).get(
                "class_path", ""
            )
            expected_head, expected_strategy = value_heads[name]
            if not head.endswith(":" + expected_head) or not strategy.endswith(
                ":" + expected_strategy
            ):
                raise ValueError(f"{name}: value head or strategy does not match algorithm")
        if (
            spec.seed != seed
            or spec.training.total_transitions != budget
            or budget not in spec.training.checkpoint_milestones
        ):
            raise ValueError(f"{name}: mismatched seed, budget or final milestone")
        if not spec.training.save_final_checkpoint:
            raise ValueError(f"{name}: final checkpoint must be enabled")
        if name != "ppo" and not spec.distributed.strict_update_budget:
            raise ValueError(f"{name}: strict update budget must be enabled")
        evaluation = spec.evaluation
        if (
            evaluation is None
            or len(evaluation.maps) != 1
            or evaluation.maps[0].expected_map_uid != uid
            or evaluation.trials_per_map != 30
            or spec.components.evaluator is None
        ):
            raise ValueError(f"{name}: evaluation map or trials mismatch")
        env = spec.components.environment.class_path
        if not env.startswith("trackmaniarl.trackmania.native_headless"):
            raise ValueError(f"{name}: environment is not native headless")
        if run_fingerprint(spec, config.parent) != job.get("run_fingerprint"):
            raise ValueError(f"{name}: source/configuration fingerprint mismatch")
        artifact = (config.parent / spec.artifacts_dir / spec.run_id).resolve()
        checkpoint = (
            config.parent
            / (getattr(spec, "checkpoint_root", None) or spec.artifacts_dir)
            / spec.run_id
            / "checkpoints"
        ).resolve()
        evaluation_path = (
            config.parent / spec.artifacts_dir / (spec.run_id + "-final-evaluation")
        ).resolve()
        if any(p.exists() for p in (artifact, checkpoint, evaluation_path)):
            raise ValueError(f"{name}: output path already exists")
        current_paths = (artifact, checkpoint, evaluation_path, receipt.resolve(), stop.resolve())
        for p in current_paths:
            if any(
                p == prior or p.is_relative_to(prior) or prior.is_relative_to(p) for prior in paths
            ):
                raise ValueError(f"{name}: output paths overlap another job")
        paths.update(current_paths)
        compiled.append(
            {
                "id": name,
                "cwd": str(repository),
                "argv": [
                    sys.executable,
                    "-m",
                    "tools.headless_campaign_queue.job",
                    "--config",
                    str(config),
                    "--receipt",
                    str(receipt),
                    "--budget",
                    str(budget),
                    "--max-overshoot",
                    str(limit),
                    "--stop-file",
                    str(stop),
                    "--config-sha256",
                    job["config_sha256"],
                    "--run-fingerprint",
                    job["run_fingerprint"],
                ],
                "transitions": budget,
                "max_overshoot": limit,
                "receipt": str(receipt),
                "run_fingerprint": job["run_fingerprint"],
                "map_id": evaluation.maps[0].id,
                "map_uid": uid,
                "on_policy": name == "ppo",
                "run_dir": str(artifact),
                "stop_file": str(stop),
            }
        )
    return {"schema_version": 2, "jobs": compiled}, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        data, digest = prepare(args.manifest)
        if not args.execute:
            print(f"Preflight OK for {len(data['jobs'])} jobs; nothing started")
            return 0
        return run(data, digest, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
