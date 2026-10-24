"""Bounded offline shared-training/replay/checkpoint check; never starts engines.

Self-test: ``python -m scripts.headless_common_check --self-test --output DIR``.
Actual data: ``--records FILE --output DIR`` with JSON containing a complete
``contract`` envelope and ``records`` serialized CommonTransitionInput objects.
Actual clocks, raw journal hashes and projection measurements must be supplied
by the producer; this script never infers them from engine iteration counts.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.trackmania.common_training import (
    COMMON6_PROFILE,
    ClockSpec,
    CommonReplayStore,
    CommonSampler,
    CommonTrainingContract,
    CommonTransitionInput,
    DurationUpdateBudget,
    RewardMeasurement,
    RewardSpec,
    make_learner,
    make_transition,
)
from trackmaniarl.trackmania.common_training.contract import sha256


def _endpoint(backend: str, position: float) -> dict[str, Any]:
    if backend == "windows-connect33":
        raw: Any = [0.0] * 33
        raw[4:10], raw[17:19] = [position, 10, 20, 1, 2, 3], [200, 1]
    else:
        data = bytearray(76)
        struct.pack_into("<10f", data, 0, 1, 0, 0, 0, position, 10, 20, 1, 2, 3)
        struct.pack_into("<fI", data, 40, 200, 1)
        raw = {"native76Hex": data.hex()}
    return {"backend": backend, "raw": raw}


def _synthetic(profile: str) -> tuple[CommonTrainingContract, list[Transition]]:
    contract = CommonTrainingContract(
        clock=ClockSpec(
            clock_id="synthetic-offline-clock",
            unit="milliseconds",
            seconds_per_tick=0.001,
            semantics="synthetic-simulation/v1",
            evidence_sha256=sha256("synthetic-clock"),
            synthetic=True,
        ),
        reward=RewardSpec(
            reference_sha256=sha256("synthetic-geometry"),
            projection_profile="synthetic/v1",
        ),
        reset_profile="synthetic-neutral/v1",
        input_schedule="synthetic-applied/v1",
        observation_semantics="synthetic-common-kinematics/v1",
    )
    if profile == "common6":
        contract = replace(contract, observation_profile=COMMON6_PROFILE)
    rows = []
    for index in range(64):
        backend = "windows-connect33" if index % 2 == 0 else "native-headless"
        terminal, truncated = index % 8 == 0, index % 8 == 1
        duration = 10 * (index % 5 + 1)
        source = CommonTransitionInput(
            observation=_endpoint(backend, 1),
            next_observation=None if terminal else _endpoint(backend, 2),
            action=index % 78,
            before_timestamp=1000,
            after_timestamp=1000 + duration,
            reward_measurement=RewardMeasurement(
                progress_before_units=1,
                progress_after_units=2,
                deviation_after_units=3,
                finished=terminal,
            ),
            terminated=terminal,
            truncated=truncated,
            raw_transition_sha256=sha256(["synthetic-row", index]),
        )
        rows.append(make_transition(source, contract))
    return contract, rows


def _load(path: Path) -> tuple[CommonTrainingContract, list[Transition]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    contract = CommonTrainingContract.from_envelope(data["contract"])
    records = data["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 4096:
        raise ValueError("bounded one through 4096 explicit actual records required")
    rows = []
    for record in records:
        payload = dict(record)
        payload["reward_measurement"] = RewardMeasurement(**payload["reward_measurement"])
        rows.append(make_transition(CommonTransitionInput(**payload), contract))
    return contract, rows


def _check(
    contract: CommonTrainingContract,
    rows: list[Transition],
    output: Path,
) -> dict[str, Any]:
    replay = CommonReplayStore(contract, capacity=128)
    sampler, budget = CommonSampler(contract, seed=1729), DurationUpdateBudget(contract)
    learner = make_learner(contract, seed=1729)
    for row in rows:
        replay.append(row)
        budget.credit(row)
        while len(replay) >= 16 and budget.integral_available > 1 and learner.update_count < 128:
            metrics, _ = learner.update(
                sampler.sample(replay, BatchRequest(16, gamma=contract.gamma))
            )
            if any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()):
                raise ValueError("nonfinite common learner metrics")
            budget.consume()
    if len(replay) < 2 or learner.update_count == 0:
        raise ValueError("at least one duration-earned update and two retained rows required")
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "learner": learner.state_dict(),
            "replay": replay.state_dict(),
            "sampler": sampler.state_dict(),
            "budget": budget.state_dict(),
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = make_learner(contract, seed=1729)
    restored.load_state_dict(saved["learner"])
    replay2, sampler2 = CommonReplayStore(contract, 128), CommonSampler(contract)
    budget2 = DurationUpdateBudget(contract)
    replay2.load_state_dict(saved["replay"])
    sampler2.load_state_dict(saved["sampler"])
    budget2.load_state_dict(saved["budget"])
    request = BatchRequest(min(16, len(replay)), gamma=contract.gamma)
    original_batch, restored_batch = (
        sampler.sample(replay, request),
        sampler2.sample(replay2, request),
    )
    if original_batch.transition_ids != restored_batch.transition_ids:
        raise ValueError("sampler RNG checkpoint changed sampled replay identities")
    if not torch.equal(original_batch.bootstrap_discounts, restored_batch.bootstrap_discounts):
        raise ValueError("checkpoint changed actual duration bootstrap discounts")
    for first, second in zip(learner.model.parameters(), restored.model.parameters(), strict=True):
        if not torch.equal(first, second):
            raise ValueError("learner checkpoint changed model weights")
    if budget.state_dict() != budget2.state_dict():
        raise ValueError("checkpoint changed the duration update budget")
    # Resume with the same batch only when the real-duration budget permits it.
    # A new genuine row may earn the next update; the check never fabricates one.
    resumed_update = False
    if budget.integral_available:
        learner.update(original_batch)
        restored.update(restored_batch)
        budget.consume()
        budget2.consume()
        resumed_update = True
        for first, second in zip(
            learner.model.parameters(), restored.model.parameters(), strict=True
        ):
            if not torch.equal(first, second):
                raise ValueError("checkpoint did not restore optimizer/target update behavior")
    return {
        "passed": True,
        "learning_comparison_run": False,
        "physical_equivalence_qualified": False,
        "synthetic": contract.clock.synthetic,
        "purpose": contract.purpose,
        "profile": contract.observation_profile,
        "contract_sha256": contract.digest,
        "transitions": len(rows),
        "backends": dict(
            Counter(row.info["_trackmaniarl_common_interval"]["backend"] for row in rows)
        ),
        "terminated": sum(row.terminated for row in rows),
        "truncated": sum(row.truncated for row in rows),
        "elapsed_simulated_seconds": budget.elapsed_seconds,
        "updates": learner.update_count,
        "retained_rows": len(replay),
        "remaining_update_credit": budget.credit_available,
        "checkpoint_roundtrip": True,
        "resume_update_checked": resumed_update,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "qualification_note": (
            "Offline pipeline check only. Raw reset, applied input, clock mapping, observation "
            "and reward geometry evidence still require independent physical qualification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--self-test", action="store_true")
    source.add_argument("--records", type=Path)
    parser.add_argument("--profile", choices=("common6", "common8"), default="common6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        contract, rows = _synthetic(args.profile) if args.self_test else _load(args.records)
        (args.output / "contract.json").write_text(
            json.dumps(contract.envelope(), indent=2) + "\n",
            encoding="utf-8",
        )
        report = _check(contract, rows, args.output)
    except (Exception, KeyboardInterrupt) as error:
        report = {"passed": False, "learning_comparison_run": False, "error": repr(error)}
    report["host_seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
