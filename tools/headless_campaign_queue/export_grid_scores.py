"""Export existing IQN grid-probe tensors into paired JSON for offline analysis.

Only use trusted torch files. This utility checks shapes, not whether the two
arrays actually came from the declared checkpoints or identical state order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_scores(before: torch.Tensor, after: torch.Tensor) -> dict:
    if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
        raise ValueError("scores must be tensors")
    if before.ndim != 2 or before.shape != after.shape or before.shape[0] < 1:
        raise ValueError("scores must be equally shaped nonempty two-dimensional tensors")
    if before.shape[1] < 2 or not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("scores require at least two actions and all values must be finite")
    left = before.detach().cpu().tolist()
    right = after.detach().cpu().tolist()
    return {
        "actions": [f"action-{index}" for index in range(before.shape[1])],
        "observations": [
            {"id": f"state-{index}", "before": a, "after": b}
            for index, (a, b) in enumerate(zip(left, right, strict=True))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="trusted grid-values BEFORE checkpoint")
    parser.add_argument("after", type=Path, help="trusted grid-values AFTER checkpoint")
    parser.add_argument("output", type=Path, help="new paired JSON; never overwritten")
    parser.add_argument("--key", required=True, help="shared quantile grid key, e.g. 32")
    parser.add_argument("--state-set-sha256", required=True, help="user-recorded state-set hash")
    parser.add_argument("--before-checkpoint-sha256", required=True)
    parser.add_argument("--after-checkpoint-sha256", required=True)
    args = parser.parse_args()
    before = torch.load(args.before, map_location="cpu", weights_only=True)
    after = torch.load(args.after, map_location="cpu", weights_only=True)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("grid files must be dictionaries")
    if args.key not in before or args.key not in after:
        raise ValueError("quantile key not present in both grids")
    result = pair_scores(before[args.key], after[args.key])
    result["provenance"] = {
        "grid_key": args.key,
        "before_grid_sha256": file_sha256(args.before),
        "after_grid_sha256": file_sha256(args.after),
        "claimed_state_set_sha256": args.state_set_sha256,
        "claimed_before_checkpoint_sha256": args.before_checkpoint_sha256,
        "claimed_after_checkpoint_sha256": args.after_checkpoint_sha256,
        "verification": "caller claims not independently verified; action labels are indices",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
