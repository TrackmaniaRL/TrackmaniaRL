"""Analyze pre-collected, paired action scores; never changes a training run.

The input schema and the experimental controls are documented in
docs/headless-policy-diagnostics.md. Scores are expected values of the same
action set on the same observations, not samples of quantiles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def analyze(data: dict) -> dict:
    """Summarize matching observations before and after an update/checkpoint.

    Q-vs-return is only reported when an observed discounted return is supplied
    for the selected action under each respective policy.
    """
    actions = data.get("actions")
    observations = data.get("observations")
    if (
        not isinstance(actions, list)
        or len(actions) < 2
        or any(not isinstance(a, str) or not a for a in actions)
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("actions must contain at least two unique nonempty strings")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a nonempty list")
    seen: set[str] = set()
    differences: list[float] = []
    gaps: dict[str, list[float]] = {"before": [], "after": []}
    errors: dict[str, list[float]] = {"before": [], "after": []}
    changes = 0
    ties = {"before": 0, "after": 0}
    for row in observations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("each observation requires a nonempty string id")
        if row["id"] in seen:
            raise ValueError(f"duplicate observation id: {row['id']}")
        seen.add(row["id"])
        scores: dict[str, list[float]] = {}
        winners: dict[str, int] = {}
        for phase in ("before", "after"):
            raw = row.get(phase)
            if not isinstance(raw, list) or len(raw) != len(actions):
                raise ValueError(f"{row['id']}: {phase} scores must match actions")
            scores[phase] = [_number(x, f"{row['id']} {phase}") for x in raw]
            ordered = sorted(scores[phase], reverse=True)
            gap = ordered[0] - ordered[1]
            gaps[phase].append(gap)
            ties[phase] += gap == 0
            winners[phase] = scores[phase].index(ordered[0])
            if f"{phase}_return" in row:
                target = _number(row[f"{phase}_return"], f"{row['id']} {phase}_return")
                errors[phase].append(ordered[0] - target)
        changes += winners["before"] != winners["after"]
        differences.append(
            max(abs(x - y) for x, y in zip(scores["before"], scores["after"], strict=True))
        )

    def summary(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": _quantile(values, 0.5),
            "p90": _quantile(values, 0.9),
            "min": min(values),
            "max": max(values),
        }

    return {
        "schema_version": 1,
        "observation_count": len(seen),
        "changed_greedy_action_count": changes,
        "changed_greedy_action_fraction": changes / len(seen),
        "tie_break": "first action in input order",
        "ties": ties,
        "action_gap": {phase: summary(gaps[phase]) for phase in gaps},
        "max_abs_score_change_per_observation": summary(differences),
        "selected_q_minus_observed_return": {phase: summary(errors[phase]) for phase in errors},
        "limitations": [
            "Scores and paired observations must use the same action ordering and preprocessing.",
            "Q versus return requires matched policy, discount and rollout conditions.",
            "This is a diagnostic, not a stability guarantee or a final evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="paired observation JSON")
    parser.add_argument("output", type=Path, help="new report JSON (never overwritten)")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        result = analyze(json.load(handle))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
