"""Metric projection for the optional W&B adapter."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from trackmaniarl.observability._wandb_metric_catalog import (
    _DEBUG_METRICS,
    _EPISODE_METRICS,
    _EVALUATION_ALIASES,
    _EVALUATION_METRICS,
    _EXPERT_METRICS,
    _IGNORED_REMOTE_EVENTS,
    _IMITATION_METRICS,
    _LEADER_METRICS,
    _RECOVERY_METRICS,
    _REPLAY_METRICS,
    _TIMING_METRICS,
    _UPDATE_METRICS,
)

__all__ = ["_IGNORED_REMOTE_EVENTS"]


def _flat_name(namespace: str, key: str) -> str:
    return f"{namespace}/{key.replace('/', '_')}"


def _selected(payload: Mapping[str, Any], keys: set[str], namespace: str) -> dict[str, Any]:
    return {_flat_name(namespace, key): payload[key] for key in keys if key in payload}


def _normalized_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.update(
        {
            key.removeprefix("eval/"): value
            for key, value in payload.items()
            if key.startswith("eval/")
        }
    )
    return normalized


def _evaluation_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_evaluation(payload)
    normalized.update(
        {
            target: normalized[source]
            for source, target in _EVALUATION_ALIASES.items()
            if source in normalized and target not in normalized
        }
    )
    return _selected(normalized, _EVALUATION_METRICS, "evaluation")


def _recovery_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        key.removeprefix("recovery/"): value
        for key, value in payload.items()
        if key.startswith("recovery/")
    }
    return _selected(normalized, _RECOVERY_METRICS, "recovery")


def _update_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = {name: payload[key] for key, name in _UPDATE_METRICS.items() if key in payload}
    for key, value in payload.items():
        prefix, separator, suffix = key.partition("/")
        if prefix in {"loss", "gradients"} and separator:
            values[_flat_name("learner", key)] = value
        elif prefix == "debug" and suffix in _DEBUG_METRICS:
            values[_flat_name("learner", suffix)] = value
        elif prefix == "replay" and suffix in _REPLAY_METRICS:
            values[_flat_name("replay", suffix)] = value
        elif prefix == "timing" and suffix in _TIMING_METRICS:
            values[_flat_name("performance", suffix)] = value
    return values


def _event_metrics(event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if event in {"train/update", "train/offline_pretrain", "train/offline_pretrain_progress"}:
        return _update_metrics(payload)
    if event == "train/episode":
        return _episode_metrics(payload)
    if event == "train/rollout":
        return _selected(
            payload,
            _EPISODE_METRICS
            | {
                "finished_episodes",
                "completed_episodes",
                "rollout_boundaries",
                "step_race_time_ms_max",
                "reward",
                "transitions",
            },
            "rollout",
        )
    if event in {"eval/summary", "eval/suite"}:
        return _evaluation_metrics(payload)
    if event in {"bc/train", "bc/validation"}:
        return _selected(payload, _IMITATION_METRICS, f"imitation_{event[3:]}")
    if event in {"recovery/update", "train/checkpoint"}:
        return _single_event_metrics(event, payload)
    if event in {"diagnose/expert", "diagnose/expert_progress"}:
        return _selected(payload, _EXPERT_METRICS, "expert")
    if event in {"eval/best_checkpoint", "eval/fastest_checkpoint"}:
        return _leader_metrics(event, payload)
    if event == "distributed/policy_published":
        return _policy_publish_metrics(payload)
    return {}


def _episode_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = _selected(payload, _EPISODE_METRICS, "episode")
    best = payload.get("best_finish_time_s")
    if (
        not isinstance(best, (int, float))
        or isinstance(best, bool)
        or not isfinite(best)
        or best <= 0
    ):
        values.pop("episode/best_finish_time_s", None)
    return values


def _single_event_metrics(event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if event == "recovery/update":
        return _recovery_metrics(payload)
    return _checkpoint_metrics(payload)


def _leader_metrics(event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    kind = event.removeprefix("eval/").removesuffix("_checkpoint")
    return _selected(payload, _LEADER_METRICS, f"checkpoint_{kind}")


def _policy_publish_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pipeline/policy_version": payload.get("policy_version", 0),
        "performance/policy_publish_s": payload.get("timing/policy_publish_s", 0.0),
    }


def _checkpoint_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "performance/checkpoint_snapshot_s": payload.get("timing/checkpoint_snapshot_s", 0.0),
    }
