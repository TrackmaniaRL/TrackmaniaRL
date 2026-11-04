"""Read-only checks for the campaign's saved training and evaluation artifacts.

These checks are necessary, not a substitute for exercising the native runtime
or a restoration. Call only on files produced by trusted, qualified code.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any


def validate_checkpoint(state: Mapping[str, Any], fingerprint: str, transitions: int, *, on_policy: bool, max_overshoot: int = 0) -> int:
    if not isinstance(state, Mapping) or state.get("run_fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    if type(transitions) is not int or transitions <= 0:
        raise ValueError("expected transition budget must be positive")
    if not isinstance(state.get("learner"), Mapping) or not state["learner"]:
        raise ValueError("checkpoint has no learner state")
    counters = state.get("counters" if on_policy else "distributed")
    if not isinstance(counters, Mapping) or type(counters.get("transitions")) is not int:
        raise ValueError("checkpoint has no valid transition counter")
    if type(max_overshoot) is not int or max_overshoot < 0:
        raise ValueError("maximum accepted overshoot must be a non-negative integer")
    actual = counters["transitions"]
    if not transitions <= actual <= transitions + max_overshoot:
        raise ValueError("checkpoint transition budget mismatch")
    if type(counters.get("updates")) is not int or counters["updates"] <= 0:
        raise ValueError("checkpoint contains no learner updates")
    if not on_policy and (not isinstance(state.get("replay_store"), Mapping) or not isinstance(state.get("sampler"), Mapping)):
        raise ValueError("checkpoint is missing replay or sampler state")
    if not on_policy:
        credit = counters.get("update_credit")
        if isinstance(credit, bool) or not isinstance(credit, (float, int)) or not math.isfinite(credit) or not 0 <= credit < 1:
            raise ValueError("final checkpoint has undrained update credit")
    return actual


def validate_drained_journal(run_dir: Path, state: Mapping[str, Any]) -> None:
    """Verify no durable submitted chunk remains past the final checkpoint."""
    import sqlite3
    from contextlib import closing
    journal = run_dir / "distributed" / "rollouts.sqlite3"
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("distributed journal missing or unsafe")
    distributed = state.get("distributed")
    if not isinstance(distributed, Mapping) or type(distributed.get("journal_applied_frontier")) is not int:
        raise ValueError("checkpoint journal frontier missing")
    frontier = distributed["journal_applied_frontier"]
    if frontier < 0 or not isinstance(state.get("journal_id"), str):
        raise ValueError("checkpoint journal identity invalid")
    with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("journal_id") != state["journal_id"] or int(metadata.get("pruned_through", -1)) != frontier:
            raise ValueError("journal identity or checkpoint frontier mismatch")
        pending = connection.execute("SELECT COUNT(*) FROM chunks WHERE id > ?", (frontier,)).fetchone()[0]
        if pending:
            raise ValueError("submitted rollout journal is not drained")
    actors = run_dir / "actors"
    if actors.exists() and any(p.is_file() for p in actors.rglob("*") if p.suffix in (".rollout", ".tmp")):
        raise ValueError("actor spool contains pending rollout data")


def final_checkpoint(run_dir: Path, checkpoint_dir: Path, *, on_policy: bool) -> tuple[Path, int]:
    """Select the last durably logged regular checkpoint after process exit."""
    import json
    import re
    events_path = run_dir / "events.jsonl"
    if events_path.is_symlink() or not events_path.is_file():
        raise ValueError("run event log missing or unsafe")
    matches = []
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("event") == "train/checkpoint_completed":
                matches.append(item)
    if not matches:
        raise ValueError("no completed checkpoint recorded")
    last = matches[-1]
    payload = last.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str) or type(last.get("step")) is not int:
        raise ValueError("invalid checkpoint completion event")
    path = Path(payload["path"])
    pattern = r"update-(\d+)\.pt" if on_policy else r"distributed-update-(\d+)\.pt"
    match = re.fullmatch(pattern, path.name)
    if not match or path.is_symlink() or not path.is_file() or path.resolve().parent != checkpoint_dir.resolve() or int(match.group(1)) != last["step"]:
        raise ValueError("last completed checkpoint is not the expected regular final checkpoint")
    return path.resolve(), last["step"]


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
