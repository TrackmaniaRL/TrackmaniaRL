"""Fresh CPU QR-DQN example for the pinned experimental native44/78 profile.

Inspect --help without starting a process or building a model. Game assets,
traces and checkpoints remain outside the source checkout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True, slots=True)
class Run:
    root: Path
    manifest: Path
    output: Path
    work: Path
    episodes: int
    seed: int
    deployment_manifest: Path
    deployment_manifest_sha256: str


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _arguments() -> Run:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-native-model", action="store_true")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--deployment-manifest-sha256")
    args = parser.parse_args()
    if not args.execute_native_model:
        parser.error("execution requires --execute-native-model")
    if not 1 <= args.train_episodes <= 3:
        parser.error("this bounded example supports one to three training episodes")
    paths = (args.runtime_root, args.asset_manifest, args.out, args.work)
    if not all(path.is_absolute() for path in paths):
        parser.error("all paths must be absolute")
    checkout = Path(__file__).resolve().parents[2]
    for path in (args.out, args.work):
        if path.resolve().is_relative_to(checkout) or path.exists():
            parser.error("use fresh output/work directories outside the checkout")
    if args.out.resolve() == args.work.resolve():
        parser.error("output and work must be separate directories")
    if args.deployment_manifest is None or not args.deployment_manifest.is_absolute():
        parser.error("an absolute --deployment-manifest is required for the portable profile")
    digest = args.deployment_manifest_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        parser.error("an explicit lowercase SHA-256 is required for the deployment manifest")
    return Run(*paths, args.train_episodes, args.seed, args.deployment_manifest, digest)


def train(run: Run) -> dict[str, Any]:
    import torch

    from examples.native_headless.learning import create_learner, make_policy
    from examples.native_headless.replay import Replay
    from trackmaniarl.core.contracts import PolicyMode
    from trackmaniarl.trackmania.native_headless import (
        NativeFeaturePipeline,
        NativeHeadlessEnvironment,
        NativeRuntimeConfig,
    )

    torch.set_num_threads(1)
    run.output.mkdir(parents=True)
    learner = create_learner(run.seed)
    replay = Replay(capacity=100_000, seed=run.seed)
    features = NativeFeaturePipeline()
    report: dict[str, Any] = {
        "schema": "tmrl.native-headless.example-training/v1",
        "passed": False,
        "seed": run.seed,
        "packageVersion": version("TrackmaniaRL"),
        "torch": torch.__version__,
        "device": "cpu",
        "referenceAssistance": True,
        "actorRefresh": "episode_boundary",
        "updateSchedule": "one per four accepted training transitions; carry; after episode",
        "syntheticReplayRows": 0,
        "demonstrationRows": 0,
        "clientParity": False,
        "learnedFinishClaim": False,
        "rawClockIsRaceTime": False,
        "trainTransitions": 0,
        "evaluationTransitions": 0,
        "updates": 0,
        "phases": [],
    }
    started = perf_counter()
    try:
        phases = ["evaluation", *(["train"] * run.episodes), "evaluation"]
        with (run.output / "updates.jsonl").open("x", encoding="utf-8") as updates:
            for serial, role in enumerate(phases):
                phase: dict[str, Any] = {"id": serial, "role": role, "steps": 0, "return": 0.0}
                report["phases"].append(phase)
                config = NativeRuntimeConfig(
                    runtime_root=run.root,
                    asset_manifest=run.manifest,
                    trace_dir=run.output / f"phase-{serial:03d}" / "trace",
                    work_dir=run.work / f"phase-{serial:03d}",
                    allow_native=True,
                    duration_limit=39950,
                    deployment_manifest=run.deployment_manifest,
                    deployment_manifest_sha256=run.deployment_manifest_sha256,
                )
                environment = NativeHeadlessEnvironment(config)
                policy = make_policy(learner)
                policy.reset_episode()
                phase_start = perf_counter()
                try:
                    before, info = environment.reset(seed=run.seed)
                    phase["assetManifestSha256"] = info["assetManifestSha256"]
                    phase["resetHostSeconds"] = perf_counter() - phase_start
                    with (run.output / f"phase-{serial:03d}" / "decisions.jsonl").open(
                        "x", encoding="utf-8"
                    ) as decisions:
                        while True:
                            if phase["steps"] >= 799:
                                raise RuntimeError(
                                    "backend endpoint missing; do not fabricate a truncation"
                                )
                            credit = report["trainTransitions"]
                            epsilon = 0.5 - 0.45 * min(credit / 100_000, 1.0)
                            policy.set_exploration_epsilon(epsilon if role == "train" else 0.0)
                            mode = PolicyMode.ONLINE if role == "train" else PolicyMode.EVALUATION
                            action = policy.act(
                                torch.tensor(
                                    features.transform_observation(before), dtype=torch.float32
                                ),
                                mode,
                            )
                            result = environment.step_native(action, native_iterations=5)
                            if role == "train":
                                replay.append(before, action, result)
                                report["trainTransitions"] += 1
                            else:
                                report["evaluationTransitions"] += 1
                            phase["steps"] += 1
                            phase["return"] += result["reward"]
                            decisions.write(
                                json.dumps(
                                    {"action": action, "before": before, "result": result},
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            before = result["observation"]
                            if result["terminated"] or result["truncated"]:
                                phase.update(
                                    terminated=result["terminated"],
                                    truncated=result["truncated"],
                                    finalNativeT=before["nativeT"],
                                    finalProjection=before["projection"],
                                )
                                break
                finally:
                    environment.close()
                    phase["hostSeconds"] = perf_counter() - phase_start
                    _save(run.output / "report.json", report)
                # Redeem only accepted training credit, after closing the native
                # session. Short episodes keep their pending credit until 64
                # real rows exist. Evaluation never changes this budget.
                if role == "train":
                    while (
                        len(replay) >= 64 and learner.update_count < report["trainTransitions"] // 4
                    ):
                        metrics, _ = learner.update(replay.sample(64))
                        report["updates"] = learner.update_count
                        updates.write(
                            json.dumps(
                                {"update": learner.update_count, "metrics": dict(metrics)},
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    _save(run.output / "report.json", report)
        # Fresh-only example artifact: no resume/migration contract is implied.
        torch.save(
            {"schema": report["schema"], "learner": learner.state_dict(), "seed": run.seed},
            run.output / "final.pt",
        )
        report["passed"] = True
    finally:
        report["fullHostSeconds"] = perf_counter() - started
        _save(run.output / "report.json", report)
    return report


def main() -> None:
    report = train(_arguments())
    print(json.dumps({key: report[key] for key in ("passed", "trainTransitions", "updates")}))


if __name__ == "__main__":
    main()
