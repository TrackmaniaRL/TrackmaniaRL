"""Pinned native44 interval qualification for explicit tensor/ragged training.

Raw qualification derives from the native example replay contract. No missing
endpoint is encoded and a compact provenance hash does not replace raw evidence.
Import this optional training module explicitly; package initialization stays inert.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import isfinite
from typing import Any, cast

import torch

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.native_headless.environment import NativeFeaturePipeline

NATIVE_INTERVAL_KEY = "_trackmaniarl_native_interval"
NATIVE_INTERVAL_SCHEMA = "tmrl.native-interval/v2"
PROFILE = "tmre.native16-pending-reference78/44/v2"
STATE_SCHEMA = "tmre.native16-pending-reference78/v1"
REFERENCE = "e256a74ca7ee82b3ea25287f49459a9d6e7773719d5eaafc996a9446f95907ae"
NATIVE_INTERVAL_FIELDS = frozenset(
    {
        "schema",
        "resetProfile",
        "inputSchedule",
        "profile",
        "referenceFingerprint",
        "beforeNativeT",
        "afterNativeT",
        "actualIterations",
        "deltaNative",
        "requestedIterations",
        "effectiveRequestedIterations",
        "pendingNotCredited",
        "status",
        "terminated",
        "truncated",
        "action",
        "rawTransitionSha256",
    }
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native metadata mapping required")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("explicit integer native metadata required")
    return value


def _flag(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("explicit bool native flags required")
    return value


def _action(value: Any) -> int:
    action = _integer(value)
    _require(0 <= action < 78, "action78 required")
    return action


def canonical_raw_transition(
    before: Mapping[str, Any], action: int, result: Mapping[str, Any]
) -> bytes:
    """The exact portable journal hash carrier; tuple/list share JSON semantics."""
    return json.dumps(
        {"before": before, "action": action, "result": result},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _tensor(observation: Any) -> torch.Tensor:
    values = NativeFeaturePipeline().transform_observation(observation)
    tensor = torch.tensor(values, dtype=torch.float32, device="cpu")
    _require(bool(torch.isfinite(tensor).all()), "actual44 must be finite float32")
    return tensor


def _validate_tensor(value: Any) -> None:
    _require(
        isinstance(value, torch.Tensor)
        and value.shape == (44,)
        and value.dtype == torch.float32
        and value.device.type == "cpu"
        and value.layout == torch.strided,
        "actual CPU float32 tensor[44] required",
    )
    _require(bool(torch.isfinite(value).all()), "finite actual44 required")


def validate_native_interval(record: Mapping[str, Any]) -> None:
    """Validate compact metadata without reading journals, tensors or processes."""
    record = _mapping(record)
    _require(set(record) == NATIVE_INTERVAL_FIELDS, "exact compact native interval fields required")
    _require(record.get("schema") == NATIVE_INTERVAL_SCHEMA, "native interval schema mismatch")
    _require(
        record.get("resetProfile") in ("neutral-prelude/v1", "legacy-gas-prelude/v1"),
        "explicit reset profile required",
    )
    _require(record.get("inputSchedule") == "pending-then-future/v1", "input schedule required")
    _require(record.get("profile") == PROFILE, "native feature profile mismatch")
    _require(record.get("referenceFingerprint") == REFERENCE, "native reference mismatch")
    terminal, truncated = _flag(record.get("terminated")), _flag(record.get("truncated"))
    _require(not (terminal and truncated), "contradictory native endpoint flags")
    status = "terminated" if terminal else "truncated" if truncated else "continue"
    _require(record.get("status") == status, "native status/flags mismatch")
    actual = _integer(record.get("actualIterations"))
    effective = _integer(record.get("effectiveRequestedIterations"))
    requested = _integer(record.get("requestedIterations"))
    _require(1 <= actual <= effective <= requested <= 5, "bounded native iterations required")
    _require(terminal or actual == effective, "partial nonterminal endpoint is inconclusive")
    delta = _integer(record.get("deltaNative"))
    _require(delta == 10 * actual, "native duration must equal 10*N")
    before = _integer(record.get("beforeNativeT"))
    after = _integer(record.get("afterNativeT"))
    _require(
        before >= 2230 and before % 10 == 0 and after == before + delta,
        "native clock grid/continuity mismatch",
    )
    _require(record.get("pendingNotCredited") is True, "pending input must not be credited")
    _action(record.get("action"))
    digest = record.get("rawTransitionSha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "canonical raw transition SHA256 required",
    )


def qualify_raw_transition(
    before: Mapping[str, Any], action: int, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Port the example's real before/action/result checks before actor admission."""
    before, result = _mapping(before), _mapping(result)
    _tensor(before)
    _require(before.get("schema") == STATE_SCHEMA, "current native schema required")
    _require(
        _mapping(before.get("projection")).get("referenceFingerprint") == REFERENCE,
        "current native reference mismatch",
    )
    action = _action(action)
    _require(result.get("trainingEligible") is True, "unqualified native result")
    terminal, truncated = _flag(result.get("terminated")), _flag(result.get("truncated"))
    after = _mapping(result.get("observation"))
    _require(
        after.get("schema") == STATE_SCHEMA and after.get("featureProfile") == PROFILE,
        "qualified endpoint schema/profile required",
    )
    raw = after.get("native76Hex")
    try:
        valid_raw = isinstance(raw, str) and len(bytes.fromhex(raw)) == 76
    except ValueError:
        valid_raw = False
    _require(valid_raw, "actual native76 endpoint required")
    history = before.get("historyActionIds")
    if not isinstance(history, list):
        raise ValueError("actual action78 input history required")
    _require(
        len(history) <= 2 and all(type(item) is int and 0 <= item < 78 for item in history),
        "actual action78 input history required",
    )
    _require(after.get("historyActionIds") == [*history, action][-2:], "native history mismatch")
    projection = _mapping(after.get("projection"))
    _require(projection.get("referenceFingerprint") == REFERENCE, "endpoint reference mismatch")
    reward = result.get("reward")
    if type(reward) not in (float, int):
        raise ValueError("finite actual reward required")
    _require(isfinite(cast(float, reward)), "finite actual reward required")
    credit = _mapping(result.get("credit"))
    record = {
        "schema": NATIVE_INTERVAL_SCHEMA,
        "resetProfile": before.get("resetProfile"),
        "inputSchedule": before.get("inputSchedule"),
        "profile": PROFILE,
        "referenceFingerprint": REFERENCE,
        "beforeNativeT": before.get("nativeT"),
        "afterNativeT": after.get("nativeT"),
        "actualIterations": result.get("actualIterations"),
        "deltaNative": result.get("deltaNative"),
        "requestedIterations": result.get("requestedIterations"),
        "effectiveRequestedIterations": result.get("effectiveRequestedIterations"),
        "pendingNotCredited": credit.get("pendingNotCredited"),
        "status": result.get("status"),
        "terminated": terminal,
        "truncated": truncated,
        "action": action,
        "rawTransitionSha256": hashlib.sha256(
            canonical_raw_transition(before, action, result)
        ).hexdigest(),
    }
    validate_native_interval(record)
    _require(
        after.get("resetProfile") == record["resetProfile"]
        and after.get("inputSchedule") == record["inputSchedule"],
        "endpoint reset/input contract mismatch",
    )
    _require(
        _integer(credit.get("intervals")) == record["actualIterations"]
        and _integer(credit.get("nativeDelta")) == record["deltaNative"],
        "native interval credit mismatch",
    )
    if terminal:
        _require(
            all(
                key in after and after[key] is None
                for key in ("features", "pendingSGB", "readerHex", "pendingOwnP")
            ),
            "terminal features/pending must remain absent",
        )
        _require(result.get("bootstrapFeatures") is None, "terminal bootstrap must be absent")
        outcome = _mapping(result.get("outcome"))
        _require(
            outcome.get("status") == "terminated"
            and outcome.get("terminated") is True
            and outcome.get("truncated") is False
            and outcome.get("qualifiedBy") == "env002.full-original-validator+TMRESULT",
            "qualified original validator/TMRESULT outcome required",
        )
        computed = _mapping(outcome.get("computedResult"))
        time = _integer(computed.get("Time"))
        _require(
            0 <= time < 0xFFFFFFFF and _integer(computed.get("NbCheckpoints")) == 1,
            "computed finish required; IsValid and nativeT are not finish",
        )
    else:
        next_tensor = _tensor(after)
        bootstrap = {"features": result.get("bootstrapFeatures"), "featureProfile": PROFILE}
        _require(torch.equal(_tensor(bootstrap), next_tensor), "actual final44 bootstrap required")
    return record


def validate_native_transition(transition: Transition) -> None:
    """Validate a sampled/stored row using its qualified compact actor contract.

    The hash format is checked here; resolving its original bytes is the actor
    journal audit's responsibility, not a sampler filesystem side effect.
    """
    record = _mapping(transition.info.get(NATIVE_INTERVAL_KEY))
    validate_native_interval(record)
    _require(
        record["resetProfile"] == "neutral-prelude/v1",
        "new training contract requires neutral reset; legacy is a raw reference only",
    )
    _validate_tensor(transition.observation)
    _require(_action(transition.action) == record["action"], "transition action mismatch")
    _require(
        _flag(transition.terminated) == record["terminated"]
        and _flag(transition.truncated) == record["truncated"],
        "transition flags mismatch",
    )
    _require(
        type(transition.reward) in (float, int) and isfinite(transition.reward),
        "finite transition reward required",
    )
    if transition.terminated:
        _require(transition.next_observation is None, "terminal NEXT must be absent")
    else:
        _validate_tensor(transition.next_observation)


class NativeTensorFeaturePipeline:
    """Explicit actual44 tensor and optional-terminal endpoint hooks, not a full stack."""

    terminal_next_observation_optional = True
    requires_environment_endpoint = True

    def transform_observation(self, observation: Any) -> torch.Tensor:
        return _tensor(observation)

    def validate_replay_transition(self, transition: Transition) -> None:
        validate_native_transition(transition)

    def replay_info(self, info: Mapping[str, Any]) -> Mapping[str, Any]:
        record = dict(_mapping(info.get(NATIVE_INTERVAL_KEY)))
        validate_native_interval(record)
        return {NATIVE_INTERVAL_KEY: record}

    def collate(self, transitions: list[Transition]) -> Any:
        raise ValueError("NativeTensorFeaturePipeline requires the explicit native ragged sampler")
