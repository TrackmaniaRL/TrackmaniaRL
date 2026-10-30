"""Read-only checks for the campaign's saved training and evaluation artifacts.

These checks are necessary, not a substitute for exercising the native runtime
or a restoration. Call only on files produced by trusted, qualified code.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_checkpoint(state: Mapping[str, Any], fingerprint: str, transitions: int, *, on_policy: bool) -> None:
    if not isinstance(state, Mapping) or state.get("run_fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    if type(transitions) is not int or transitions <= 0:
        raise ValueError("expected transition budget must be positive")
    if not isinstance(state.get("learner"), Mapping) or not state["learner"]:
        raise ValueError("checkpoint has no learner state")
    counters = state.get("counters" if on_policy else "distributed")
    if not isinstance(counters, Mapping) or type(counters.get("transitions")) is not int:
        raise ValueError("checkpoint has no valid transition counter")
    if counters["transitions"] != transitions:
        raise ValueError("checkpoint transition budget mismatch")
    if type(counters.get("updates")) is not int or counters["updates"] <= 0:
        raise ValueError("checkpoint contains no learner updates")
    if not on_policy and (not isinstance(state.get("replay_store"), Mapping) or not isinstance(state.get("sampler"), Mapping)):
        raise ValueError("checkpoint is missing replay or sampler state")


def validate_evaluation(artifact: Mapping[str, Any], checkpoint: Path, map_id: str, map_uid: str, trials: int) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("checkpoint") != str(checkpoint):
        raise ValueError("evaluation did not identify the final checkpoint")
    rows = artifact.get("trials")
    if type(trials) is not int or trials <= 0 or not isinstance(rows, list) or len(rows) != trials:
        raise ValueError("evaluation trial count mismatch")
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("map_id") != map_id or row.get("map_uid") != map_uid:
            raise ValueError("evaluation map identity mismatch")
        index = row.get("trial_index")
        if type(index) is not int or index in seen:
            raise ValueError("duplicate or invalid evaluation trial index")
        seen.add(index)
    if seen != set(range(trials)):
        raise ValueError("evaluation trial indices incomplete")
