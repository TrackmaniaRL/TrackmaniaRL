from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from typing import Any, cast

import numpy as np
import torch

from trackmaniarl.core.replay.store_pace import validated_sampling_pace
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator_observability_validation import (
    _validate_binary_flag,
    _validate_finite_number,
    _validate_observability_summary,
)
from trackmaniarl.distributed.protocol import transition_from_wire


def _validate_submit_payload(value: Mapping[str, Any], codec: WireCodec) -> None:
    _validate_fields(value, _SUBMIT_FIELDS, "rollout")
    _validate_submit_identity(value)
    transitions, episodes, evaluations = _submit_collections(value)
    _validate_submit_collections(transitions, episodes, evaluations)
    _validate_episode_owners(value, transitions, episodes)
    _validate_evaluation_snapshot(value, evaluations, codec)


def _validate_submit_identity(value: Mapping[str, Any]) -> None:
    _required_nonempty_string(value, "actor_id")
    _required_nonempty_string(value, "session_id")
    _required_integer(value, "sequence", minimum=0)
    _required_integer(value, "policy_version", minimum=-1)


def _submit_collections(
    value: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    transitions = _required_list(value, "transitions")
    episodes = _required_list(value, "episodes")
    evaluations = _required_list(value, "evaluations")
    return transitions, episodes, evaluations


def _validate_submit_collections(
    transitions: list[Any], episodes: list[Any], evaluations: list[Any]
) -> None:
    for item in transitions:
        _validate_wire_transition(item)
    for summary in episodes:
        _validate_episode_summary(summary)
    for summary in evaluations:
        _validate_evaluation_summary(summary)


def _validate_episode_owners(
    value: Mapping[str, Any], transitions: list[Any], episodes: list[Any]
) -> None:
    prefix = f"{value['actor_id']}/{value['session_id']}/"
    for transition in transitions:
        _validate_episode_owner(transition["episode_id"], prefix)
    for summary in episodes:
        _validate_episode_owner(summary["episode_id"], prefix)


def _validate_episode_owner(episode_id: Any, prefix: str) -> None:
    if not isinstance(episode_id, str) or not episode_id.startswith(prefix):
        raise ValueError("episode_id must belong to the submitting actor session")
    if len(episode_id) == len(prefix):
        raise ValueError("episode_id must have a non-empty episode suffix")


def _validate_evaluation_snapshot(
    value: Mapping[str, Any], evaluations: list[Any], codec: WireCodec
) -> None:
    snapshot = value["evaluation_snapshot"]
    if not isinstance(snapshot, bytes):
        raise TypeError("evaluation_snapshot must be bytes")
    if not evaluations:
        if snapshot:
            raise ValueError("evaluation_snapshot requires evaluations")
        return
    if not snapshot:
        raise ValueError("evaluations require an evaluation_snapshot")
    policy_state = codec.decode(snapshot)
    if not isinstance(policy_state, Mapping):
        raise TypeError("evaluation_snapshot must decode to a mapping")
    versions = {_required_integer(item, "policy_version", minimum=0) for item in evaluations}
    if len(versions) != 1:
        raise ValueError("evaluation_snapshot cannot cover mixed policy versions")
    _validate_finite_tree(policy_state, "evaluation_snapshot")


def _validate_wire_transition(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("transitions must contain mappings")
    _validate_transition_fields(value)
    _validate_transition_identity(value)
    _validate_transition_values(value)
    transition_from_wire(value)


def _validate_transition_fields(value: Mapping[str, Any]) -> None:
    _validate_fields(value, _TRANSITION_FIELDS, "transition")
    if not isinstance(value["terminated"], bool) or not isinstance(value["truncated"], bool):
        raise TypeError("transition terminal flags must be booleans")
    if not isinstance(value["info"], Mapping):
        raise TypeError("transition info must be a mapping")


def _validate_transition_identity(value: Mapping[str, Any]) -> None:
    episode_id = value["episode_id"]
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise TypeError("transition episode_id must be a non-empty string or null")
    step = value["step"]
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise TypeError("transition step must be a non-negative integer or null")


def _validate_transition_values(value: Mapping[str, Any]) -> None:
    _validate_numeric_tree(value["observation"], "transition observation")
    _validate_numeric_tree(value["action"], "transition action")
    _validate_numeric_tree(value["next_observation"], "transition next_observation")
    _validate_finite_number(value["reward"], "transition reward")
    projected = value["info"].get("sampling/projected_lap_time_s")
    if projected is not None:
        _validate_finite_number(projected, "projected lap time")
    entropy_key = "_trackmaniarl_behavior_entropy"
    if entropy_key in value["info"]:
        entropy = value["info"][entropy_key]
        _validate_finite_number(entropy, "behavior entropy")
        if entropy < 0:
            raise ValueError("behavior entropy must be non-negative")


def _validate_episode_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("episodes must contain mappings")
    missing = ({"episode_id", "finished", "termination"} | _EPISODE_NUMERIC_FIELDS) - value.keys()
    if missing:
        raise ValueError(f"episode summary is missing {sorted(missing)}")
    _required_nonempty_string(value, "episode_id")
    _validate_episode_fields(value)
    steps = _required_integer(value, "steps", minimum=1)
    _validate_observability_summary(value, "episode", steps)
    _validate_finite_tree(value, "episode summary")


def _validate_episode_fields(value: Mapping[str, Any]) -> None:
    _validate_binary_flag(value["finished"], "episode finished")
    if not isinstance(value["termination"], str):
        raise TypeError("episode termination must be a string")
    for key in _EPISODE_NUMERIC_FIELDS:
        if key == "exploration_epsilon" and value[key] is None:
            continue
        _validate_finite_number(value[key], f"episode {key}")
    _validate_exploration_availability(value)
    _validate_episode_outcome(value)


def _validate_exploration_availability(value: Mapping[str, Any]) -> None:
    used = value.get("exploration_epsilon_used")
    if used is not None and not isinstance(used, bool):
        raise TypeError("exploration_epsilon_used must be a boolean or null")
    epsilon = value["exploration_epsilon"]
    if used is True and epsilon is None:
        raise ValueError("epsilon-controlled policy requires a measured exploration_epsilon")
    if used is False and epsilon is not None:
        raise ValueError("unused exploration_epsilon must be null")


def _validate_episode_outcome(value: Mapping[str, Any]) -> None:
    finished = bool(value["finished"])
    if finished != (value["termination"] == "finished"):
        raise ValueError("episode finished flag and termination disagree")
    finish_time_s = float(value["finish_time_s"])
    if not finished:
        if finish_time_s != 0.0:
            raise ValueError("unfinished episode finish_time_s must be zero")
        return
    if finish_time_s <= 0.0:
        raise ValueError("finished episode finish_time_s must be positive")
    validated_sampling_pace(finish_time_s)
    if not isclose(finish_time_s, float(value["race_time_s"])):
        raise ValueError("finished episode finish_time_s must match race_time_s")


def _validate_evaluation_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("evaluations must contain mappings")
    _validate_binary_flag(value["finished"], "evaluation finished")
    _validate_finite_number(value["finish_time_s"], "evaluation finish_time_s")
    _validate_evaluation_outcome(value)
    steps = _required_integer(value, "steps", minimum=0)
    _required_integer(value, "policy_version", minimum=0)
    _validate_observability_summary(value, "evaluation", steps)
    _validate_finite_tree(value, "evaluation summary")


def _validate_evaluation_outcome(value: Mapping[str, Any]) -> None:
    finished = bool(value["finished"])
    finish_time_s = float(value["finish_time_s"])
    if finished and finish_time_s <= 0.0:
        raise ValueError("finished evaluation finish_time_s must be positive")
    if not finished and finish_time_s != 0.0:
        raise ValueError("unfinished evaluation finish_time_s must be zero")


def _required_list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value[key]
    if not isinstance(result, list):
        raise TypeError(f"{key} must be a list")
    return result


def _required_nonempty_string(value: Mapping[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _required_integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise TypeError(f"{key} must be an integer >= {minimum}")
    return cast(int, result)


def _validate_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _validate_numeric_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        _validate_numeric_children(value.values(), name)
        return
    if isinstance(value, (tuple, list)):
        _validate_numeric_children(value, name)
        return
    if isinstance(value, (torch.Tensor, np.ndarray)):
        _validate_numeric_array(value, name)
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, float, np.number)):
        _validate_finite_number(value, name)
        return
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _validate_numeric_children(values: Any, name: str) -> None:
    for item in values:
        _validate_numeric_tree(item, name)


def _validate_numeric_array(value: torch.Tensor | np.ndarray[Any, Any], name: str) -> None:
    if value.dtype in {torch.bool, np.bool_}:
        return
    finite = (
        torch.isfinite(value).all() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    )
    if not bool(finite):
        raise ValueError(f"{name} contains non-finite values")


def _validate_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (torch.Tensor, np.ndarray, bool, int, float, np.number)):
        _validate_numeric_tree(value, name)
        return
    if value is not None and not isinstance(value, (str, bytes)):
        raise TypeError(f"{name} contains unsupported {type(value).__name__}")


_SUBMIT_FIELDS = frozenset(
    {
        "protocol_version",
        "fingerprint",
        "actor_id",
        "session_id",
        "sequence",
        "policy_version",
        "transitions",
        "episodes",
        "evaluations",
        "evaluation_snapshot",
    }
)

_TRANSITION_FIELDS = frozenset(
    {
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "info",
        "episode_id",
        "step",
    }
)

_EPISODE_NUMERIC_FIELDS = {
    "finish_time_s",
    "progress_pct",
    "return",
    "reward/time",
    "reward/pace",
    "reward/pbrs",
    "reward/progress",
    "reward/projected_velocity",
    "reward/projected_speed",
    "reward/steering_delta",
    "reward/collision",
    "collision/count",
    "collision/detected_count",
    "reward/terminal",
    "reward/time_attack_terminal",
    "velocity/ratio_mean",
    "velocity/ratio_max",
    "race_time_s",
    "exploration_epsilon",
}
