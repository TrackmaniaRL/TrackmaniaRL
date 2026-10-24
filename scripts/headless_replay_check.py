"""Bounded SDK 1.2.14 replay/learner/checkpoint integration on actual probe rows.

This is an offline plumbing check, not a learning quality or Windows speed test.
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay.store import InMemoryReplayStore
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator_ingest import replay_info_for_transition
from trackmaniarl.distributed.coordinator_validation import _validate_wire_transition
from trackmaniarl.distributed.protocol import transition_from_wire, transition_to_wire
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import FixedQuantileHead
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.trackmania.native_headless.training_contract import (
    NATIVE_INTERVAL_KEY,
    NativeTensorFeaturePipeline,
    qualify_raw_transition,
)
from trackmaniarl.trackmania.native_headless.training_learning import NativeRaggedLearner
from trackmaniarl.trackmania.native_headless.training_sampler import NativeRaggedSampler


def learner() -> NativeRaggedLearner:
    model = CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(44, 256),
            IdentityTemporalCore(256),
            FixedQuantileHead({"feature_dim": 256, "action_count": 78, "quantile_count": 32}),
            FixedQuantileStrategy(32),
        )
    )
    result = NativeRaggedLearner(
        model,
        learning_rate=1e-4,
        seed=1729,
        execution=TorchExecutionConfig(device="cpu", precision="float32", torch_threads=2),
    )
    result.setup({"seed": 1729})
    return result


def main() -> int:
    torch.set_num_threads(2)
    start = time.monotonic()
    source, output = map(Path, sys.argv[1:3])
    output.mkdir(mode=0o700)
    report = {
        "passed": False,
        "comparison_training": False,
        "model": "QR-DQN/MLP256/native44",
        "transitions": 0,
        "updates": 0,
        "retained_capacity": 128,
        "reset_credit": 0,
    }
    try:
        probe = json.loads(source.read_bytes())
        assert probe["passed"]
        store = InMemoryReplayStore(128)
        pipeline = NativeTensorFeaturePipeline()
        codec = WireCodec(16 * 1024 * 1024)
        sampler = NativeRaggedSampler(seed=1729)
        model = learner()
        for case in probe["cases"]:
            if (
                case["reset_profile"] != "neutral-prelude/v1"
                or "failure" in case["name"]
                or "exception" in case["name"]
                or "stop" in case["name"]
            ):
                continue
            for step, item in enumerate(case["rows"]):
                result = item["result"]
                compact = qualify_raw_transition(item["before"], item["action"], result)
                row = Transition(
                    pipeline.transform_observation(item["before"]),
                    item["action"],
                    result["reward"],
                    None
                    if result["terminated"]
                    else pipeline.transform_observation(result["observation"]),
                    result["terminated"],
                    result["truncated"],
                    {NATIVE_INTERVAL_KEY: compact},
                    case["name"],
                    step,
                )
                wire = codec.decode(codec.encode(transition_to_wire(row)))
                _validate_wire_transition(wire, pipeline=pipeline)
                decoded = transition_from_wire(wire)
                store.append(
                    Transition(
                        decoded.observation,
                        decoded.action,
                        decoded.reward,
                        decoded.next_observation,
                        decoded.terminated,
                        decoded.truncated,
                        replay_info_for_transition(decoded.info, pipeline=pipeline),
                        decoded.episode_id,
                        decoded.step,
                    )
                )
                report["transitions"] += 1
                while len(store) >= 64 and model.update_count < report["transitions"] // 4:
                    metrics, _ = model.update(sampler.sample(store, BatchRequest(64, gamma=0.995)))
                    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
                    report["updates"] = model.update_count
        report["remaining_integral_update_credit"] = report["transitions"] // 4 - model.update_count
        report["fractional_credit"] = (report["transitions"] % 4) / 4
        assert report["remaining_integral_update_credit"] == 0
        state = deepcopy(model.state_dict())
        torch.save(
            {"learner": state, "replay": store.state_dict(), "sampler": sampler.state_dict()},
            output / "checkpoint.pt",
        )
        loaded = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        restored = learner()
        restored.load_state_dict(loaded["learner"])
        assert restored.update_count == model.update_count
        for a, b in zip(model.model.parameters(), restored.model.parameters(), strict=True):
            assert torch.equal(a, b)
        replay2 = InMemoryReplayStore(128)
        replay2.load_state_dict(loaded["replay"])
        sampler2 = NativeRaggedSampler()
        sampler2.load_state_dict(loaded["sampler"])
        before = sampler.sample(store, BatchRequest(64, gamma=0.995))
        after = sampler2.sample(replay2, BatchRequest(64, gamma=0.995))
        assert before.transition_ids == after.transition_ids
        assert torch.equal(before.bootstrap_discounts, after.bootstrap_discounts)
        restored.update(after)
        report.update(
            passed=True,
            checkpoint_roundtrip=True,
            update_after_load=True,
            retained_rows=len(store),
            checkpoint_bytes=(output / "checkpoint.pt").stat().st_size,
        )
    except BaseException as error:
        import traceback

        traceback.print_exc()
        report["error"] = repr(error)
    finally:
        report["full_host_seconds"] = time.monotonic() - start
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
