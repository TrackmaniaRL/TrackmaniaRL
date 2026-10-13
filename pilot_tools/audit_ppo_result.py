"""Combine saved PPO evidence without driving, model inference or optimizer steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def window_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("control_gas", "control_brake", "projected_velocity_mps", "nearest_distance_m")
    return {key: mean(float(row[key]) for row in rows) for key in keys}


def rollout_evidence(run: Path) -> dict[str, Any]:
    ends = []
    blocks = []
    total = 0
    paths = sorted((run / "episodes").glob("episode-rollout-*.json.gz"))
    for index, path in enumerate(paths, start=1):
        artifact = json.loads(gzip.decompress(path.read_bytes()))
        rows = artifact["telemetry"]
        if len(rows) != 2048 or len(artifact["actions"]) != 2048:
            raise ValueError("Incomplete saved rollout")
        total += len(rows)
        for position, row in enumerate(rows):
            reason = row.get("termination_reason")
            if not reason:
                continue
            start = max(0, position - min(100, int(row["episode_step"]) + 1) + 1)
            ends.append({
                "rollout": index,
                "reason": reason,
                "progress_pct": row["progress_pct"],
                "race_time_s": row["race_time_ms"] / 1000,
                "episode_steps": row["episode_step"] + 1,
                "position": row["position"],
                "tail_steps": position - start + 1,
                "tail": window_metrics(rows[start:position + 1]),
            })
        blocks.append({"rollout": index, **window_metrics(rows)})
    counts = Counter(row["reason"] for row in ends)
    late = [row for row in ends if row["rollout"] >= 61]
    return {
        "rollouts": len(paths), "transitions": total,
        "end_counts": dict(counts), "late_61_71_end_counts": dict(Counter(
            row["reason"] for row in late
        )),
        "first_finish_rollout": next(row["rollout"] for row in ends
                                     if row["reason"] == "finished"),
        "late_finish_times_s": [row["race_time_s"] for row in late
                                if row["reason"] == "finished"],
        "endpoints": ends, "rollout_controls": blocks,
        "limitation": "Endpoint telemetry is post-action, not causal current model input",
    }


def update_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [event["payload"] for event in events if event["event"] == "train/update"]
    keys = ("state/approx_kl", "state/clip_fraction", "state/early_stop",
            "state/entropy", "loss/value", "state/learning_rate")
    groups = []
    for start in range(0, len(rows), 10):
        selected = rows[start:start + 10]
        groups.append({"first_update": start + 1, "last_update": start + len(selected),
                       **{key: mean(row[key] for row in selected) for key in keys}})
    return {
        "updates": len(rows), "early_stop_count": sum(row["state/early_stop"] for row in rows),
        "first": rows[0], "last": rows[-1], "blocks": groups,
        "mean_kl_is_epoch_average_not_final_full_batch_kl": True,
    }


def checkpoint_evidence(run: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((run / "checkpoints").glob("update-*.pt")):
        before = digest(path)
        state = TorchCheckpointCodec().load(path)
        learner = state["learner"]
        log_std = learner["model"]["actor.log_std"]
        results.append({
            "name": path.name, "sha256": before,
            "counters": state["counters"],
            "log_std": log_std.tolist(), "latent_std": log_std.exp().tolist(),
            "optimizer_lr": [group["lr"] for group in learner["optimizer"]["param_groups"]],
            "observation_normalization": learner["observation_normalizer"]["enabled"],
            "reward_normalization": learner["reward_normalizer"]["enabled"],
            "saved_replay_present": state["replay_store"] is not None,
        })
        if before != digest(path):
            raise RuntimeError("Checkpoint changed during read-only audit")
    return results


def audit(run: Path) -> dict[str, Any]:
    evaluation = json.loads((run / "evaluation.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    rollouts = rollout_evidence(run)
    updates = update_evidence(events)
    logged = [event["payload"] for event in events if event["event"] == "train/rollout"]
    if (rollouts["transitions"] != 145408 or updates["updates"] != 71
            or sum(row["finished_episodes"] for row in logged) != rollouts["end_counts"]["finished"]
            or sum(row["rollout_boundaries"] for row in logged) != 71):
        raise ValueError("Saved artifacts and training counters disagree")
    trials = [{key: trial[key] for key in (
        "trial_index", "finished", "progress_pct", "steps", "termination_reason", "finish_time_s"
    )} for trial in evaluation["trials"]]
    if len(trials) != 10 or sum(row["finished"] for row in trials) != 3:
        raise ValueError("Audit requires the original completed 3/10 evaluation")
    failure_clusters = Counter(
        "0-10%" if row["progress_pct"] < 10 else
        "35-45%" if row["progress_pct"] < 45 else "80-85%"
        for row in trials if not row["finished"]
    )
    return {
        "run_directory": str(run), "optimizer_steps_taken": 0, "game_control_used": False,
        "training": rollouts, "updates": updates, "checkpoints": checkpoint_evidence(run),
        "evaluation_trials": trials, "evaluation_failure_clusters": dict(failure_clusters),
        "discount_at_55s": 0.995 ** (55 / 0.05),
        "discount_e_folding_seconds": -0.05 / math.log(0.995),
        "gae_direct_residual_e_folding_seconds": -0.05 / math.log(0.995 * 0.95),
        "prefix_fraction_of_campaign": 145408 / 8192000,
        "limitations": [
            "No saved per-step evaluation observations or action traces",
            "No evidence that optimizer or entropy changes would repair driving",
            "Training samples actions; evaluation uses squashed latent means",
            "Rollout resets censor ongoing races; natural-termination rates are conditional",
            "One training seed and ten trials do not establish multi-seed reliability",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: result[key] for key in (
        "evaluation_failure_clusters", "discount_at_55s", "prefix_fraction_of_campaign"
    )}))


if __name__ == "__main__":
    main()
