"""Materialize a search queue and macro vocabulary without importing the runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .plans import candidates, select, validate

JIMMY = "bcfe2be9-e460-45b4-a6b9-fa48c3609b30"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1105)
    args = parser.parse_args()
    study = Path(args.study).resolve()
    output = study / "local-search1105"
    if output.exists():
        raise FileExistsError("refusing to overwrite prepared search")
    if not 1 <= args.budget <= 4096:
        raise ValueError("preparation budget must be 1..4096")
    inputs = [study / f"{JIMMY}.controls.npz", study / f"{JIMMY}.controls.csv",
              study / "macro-catalog.npz", study / "native" / JIMMY / "receipt.json",
              study / "native" / JIMMY / "factory-config.json"]
    hashes = {str(path): sha(path) for path in inputs}
    receipt = json.loads(inputs[3].read_text(encoding="utf-8-sig"))
    if receipt["status"] != "PASS_NATIVE_DEMONSTRATION" or receipt["actual_goal_ms"] != 17715:
        raise ValueError("qualified original Jimmy receipt required")
    with np.load(inputs[0], allow_pickle=False) as saved:
        array = saved["programs"]
        if array.shape != (355, 5) or array.dtype.kind not in "iu":
            raise ValueError("expected qualified 355 five-tick Jimmy programs")
        ticks = tuple(int(a) for a in array.reshape(-1))
    with inputs[1].open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["race_ms"]) for r in rows] != list(range(0, len(rows) * 10, 10)):
        raise ValueError("continuous 10 ms source clock required")
    active = min(len(ticks), len(rows) - 1)
    # Preserve independently qualified +1 source offset, not a fitted alignment.
    for tick, row in zip(ticks[:active], rows[1:], strict=False):
        steering, variant = divmod(tick, 6)
        gas, brake = divmod(variant, 3)
        if (steering - 6) * 127 != int(row["steer"]) * 6:
            raise ValueError("baseline steering is not exact digital source input")
        if (gas, brake) != (int(row["accel"]), int(row["brake"])):
            raise ValueError("baseline gas/brake mismatch")
    if any(a != 36 for a in ticks[active:]):
        raise ValueError("unexpected qualified trailing padding")
    validate(ticks, active)
    pool = candidates(ticks, active)
    selected = select(pool, args.budget, args.seed)
    with np.load(inputs[2], allow_pickle=False) as saved:
        old = {tuple(int(v) for v in p) for p in saved["programs"]}
    plans = [ticks, *(candidate.ticks for candidate in selected)]
    vocabulary = sorted(old | {p[i:i + 5] for p in plans for i in range(0, len(p), 5)})
    if len(vocabulary) > 1024:
        raise ValueError("prepared macro vocabulary exceeds qualified catalog bound")
    lookup = {p: i for i, p in enumerate(vocabulary)}
    output.mkdir()
    np.savez_compressed(output / "macro-catalog.npz",
                        schema=np.asarray("tmrl.autonomous-macro-action50ms-native10ms/v1"),
                        programs=np.asarray(vocabulary, dtype=np.int64))
    np.savez_compressed(output / "candidate-plans.npz",
                        motor_ticks=np.asarray(plans, dtype=np.int64),
                        macro_actions=np.asarray([[lookup[p[i:i + 5]]
                                                   for i in range(0, len(p), 5)]
                                                  for p in plans], dtype=np.int64))
    manifest = {
        "status": "PREPARED_NOT_EXECUTED", "native_starts": 0, "learner_updates": 0,
        "baseline_goal_ms": 17715, "source_offset_ticks": 1, "active_ticks": active,
        "tick_ms": 10, "decision_ms": 50, "seed": args.seed,
        "candidate_budget": args.budget, "available_single_edits": len(pool),
        "selected_families": dict(Counter(c.kind for c in selected)),
        "catalog_count": len(vocabulary), "input_sha256": hashes,
        "output_sha256": {p.name: sha(p) for p in sorted(output.glob("*.npz"))},
        "baseline_row": 0,
        "candidates": [{"row": i, "kind": c.kind, "start_tick": c.start,
                        "stop_tick_exclusive": c.stop, "motor_sha256": c.digest,
                        "changed_ticks": sum(a != b for a, b in zip(ticks, c.ticks, strict=True))}
                       for i, c in enumerate(selected, 1)],
        "execution_gate": "Separate explicit launch authorization and verified run1096 closure",
        "execution_available": "runner requires explicit --execute-after-1096 flag",
        "note": "Prepared only; runner has not been native-qualified or executed",
    }
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("source changed during preparation")
    (output / "plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "candidates"}))


if __name__ == "__main__":
    main()
