"""Read-only CP50/CP100 trace and matched-observation comparison; no game or optimizer."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.pytree import tree_collate


def component(spec: dict[str, Any]) -> Any:
    module, name = spec["class_path"].split(":")
    return getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "steps": len(rows),
        "time_s": [rows[0]["after_action"]["race_time_ms"] / 1000,
                   rows[-1]["after_action"]["race_time_ms"] / 1000],
        "progress_pct": [rows[0]["after_action"]["progress_pct"],
                         rows[-1]["after_action"]["progress_pct"]],
        "projected_speed_mean_mps": mean(r["after_action"]["projected_velocity_mps"] for r in rows),
        "physical_speed_mean_mps": mean(math.sqrt(sum(v*v for v in r["raw_next"][7:10]))
                                        for r in rows),
        "center_distance_mean_m": mean(r["after_action"]["nearest_distance_m"] for r in rows),
        "issued_action_mean_gas_brake_steer": [mean(r["issued_action"][i] for r in rows)
                                               for i in range(3)],
        "position_start": rows[0]["after_action"]["position"],
        "position_end": rows[-1]["after_action"]["position"],
        "end_progress_counter": rows[-1]["after_action"]["steps_since_progress"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    paths = [args.run / "evaluation-checkpoints/update-00000050.pt",
             args.run / "checkpoints/update-00000100.pt"]
    from pilot_tools.ppo_pilot import sha

    hashes = [sha(p) for p in paths]
    states = [TorchCheckpointCodec().load(p) for p in paths]
    models = []
    for state in states:
        assert state["learner"]["observation_normalizer"]["enabled"] is False
        model = component(config["components"]["model_factory"]).build().cpu().eval()
        model.load_state_dict(state["learner"]["model"], strict=True)
        models.append(model)
    pipeline = component(config["components"]["feature_pipeline"])
    prepared, records, trials = [], [], []
    for phase in (50, 100):
        directory = args.run / f"evaluations/update-{phase:08d}"
        evaluation = json.loads((directory / "evaluation.json").read_text())
        for outcome in evaluation["trials"]:
            index = outcome["trial_index"]
            trace_path = directory / f"trial-{index:02d}.json.gz"
            trace = json.loads(gzip.decompress(trace_path.read_bytes()))
            rows = trace["rows"][1:]
            pipeline.reset_episode()
            features = pipeline.transform_observation(trace["rows"][0]["raw_observation"])
            for row in rows:
                progress = row["after_action"]["progress_pct"]
                if 76 <= progress <= 83 and row["step"] % 5 == 0:
                    prepared.append(features)
                    records.append({"phase": phase, "trial": index, "step": row["step"],
                                    "progress_pct_after_action": progress,
                                    "recorded_action": row["issued_action"]})
                features = pipeline.transform_observation(row["raw_next"])
            windows = {f"{lo}_{hi}": summary([r for r in rows
                       if lo <= r["after_action"]["progress_pct"] < hi])
                       for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 75),
                                      (75, 78), (78, 80), (80, 82), (82, 85), (85, 100))}
            trials.append({"phase": phase, "trial": index, "finished": outcome["finished"],
                           "termination_reason": outcome["termination_reason"],
                           "finish_time_s": outcome["finish_time_s"], "windows": windows,
                           "tail100": summary(rows[-100:]),
                           "entry_times_s": {str(p): next((r["after_action"]["race_time_ms"] / 1000
                               for r in rows if r["after_action"]["progress_pct"] >= p), None)
                               for p in (20, 40, 60, 75, 78, 80, 81, 82, 85)},
                           "timing_violations": [{"step": r["step"],
                               "progress_pct": r["after_action"]["progress_pct"]}
                               for r in rows if r["after_action"]["step_race_time_ms"] > 100]})
    outputs: list[list[list[float]]] = [[], []]
    with torch.inference_mode():
        for start in range(0, len(prepared), 64):
            batch = tree_collate(prepared[start:start+64])
            for i, model in enumerate(models):
                action, _, _ = model.actor.sample_with_latent(batch, mode=PolicyMode.EVALUATION)
                outputs[i].extend(action.tolist())
    own_errors = []
    for record, first, second in zip(records, *outputs, strict=True):
        record["cp50_action"] = first
        record["cp100_action"] = second
        own = first if record["phase"] == 50 else second
        differences = zip(own, record["recorded_action"], strict=True)
        own_errors.append(max(abs(a-b) for a, b in differences))
    assert max(own_errors) < 1e-4, "Saved actions not reproduced; no matched-state inference claim"
    events = [json.loads(line) for line in (args.run / "events.jsonl").read_text().splitlines()]
    updates = [e["payload"] for e in events if e["event"] == "train/update"]
    training = []
    for lo in range(1, 101, 10):
        block = [u for u in updates if lo <= u["counters"]["updates"] < lo+10]
        training.append({"updates": [lo, lo+9], "metrics_mean": {
            key: mean(u[key] for u in block) for key in ("loss/value", "state/final_kl",
                "state/epochs", "state/explained_variance", "state/entropy",
                "state/return_mean", "state/value_mean", "state/gradient_norm_actor",
                "state/gradient_norm_value")}})
    assert hashes == [sha(p) for p in paths]
    result = {"optimizer_steps": 0, "game_controls": 0, "device": "cpu", "torch_threads": 1,
              "checkpoint_sha256": hashes, "checkpoints_unchanged": True, "trials": trials,
              "matched_state_count": len(records), "saved_action_max_abs_error": max(own_errors),
              "matched_actions": records, "training_blocks": training,
              "latent_std": [s["learner"]["model"]["actor.log_std"].exp().tolist() for s in states],
              "limitations": ["Offline actions do not simulate counterfactual driving",
                              "Separate evaluations are not a paired randomized experiment",
                              "No causal attribution of drift to an optimizer hyperparameter"]}
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("matched_state_count", "saved_action_max_abs_error",
                                           "latent_std", "training_blocks")}))


if __name__ == "__main__":
    main()
