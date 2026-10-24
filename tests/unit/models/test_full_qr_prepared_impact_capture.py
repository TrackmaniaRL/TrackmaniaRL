"""Actual fast-hook records, distinct from full reward-source qualification."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact import project_prepared_impact
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
    decode_prepared_impact,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/prepared_impact_hook_v1"


def evidence() -> tuple[bytes, PreparedImpactContext, list[dict]]:
    raw = (FIXTURE / "contacts.bin").read_bytes()
    proof = json.loads((FIXTURE / "context.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == proof["fixture_sha256"]
    context = proof["context"]
    for field in ("normalization_header", "normalization_thresholds"):
        context[field] = bytes.fromhex(context[field])
    context["wheel_models"] = tuple(context["wheel_models"])
    context["wheel_headers"] = tuple(bytes.fromhex(x) for x in context["wheel_headers"])
    return raw, PreparedImpactContext(**context), proof["expected"]


def test_fast_actual_delivery_preserves_vectors_future_times_and_projection() -> None:
    raw, context, expected = evidence()
    decoded = [decode_prepared_impact(raw[i : i + 512], context) for i in range(0, len(raw), 512)]
    assert json.loads(json.dumps([asdict(x) for x in decoded])) == expected
    assert {x.wheel_index for x in decoded} == {-1, 0, 1, 2, 3}
    assert all(x.event_time > x.native_time for x in decoded)
    body = decoded[0]
    assert body.wheel_index == -1
    assert body.wheel_category is None
    peak = max((project_prepared_impact(x) for x in decoded), key=lambda x: x.raw_strength)
    assert peak.raw_strength == 77.80525970458984
    assert peak.normalized_strength == 1.0
    assert not hasattr(body, "contact_flags")
    assert not hasattr(body, "collision_detected")


def test_unknown_call_path_cannot_silently_enter_the_known_delivery() -> None:
    raw, context, _ = evidence()
    changed = bytearray(raw[:512])
    pack_into("<Q", changed, 64, context.module_base + 0xAC0374)
    with pytest.raises(ValueError, match="caller"):
        decode_prepared_impact(bytes(changed), context)


def test_scratch_argument_is_not_collision_flags() -> None:
    raw, context, _ = evidence()
    original = decode_prepared_impact(raw[:512], context)
    changed = bytearray(raw[:512])
    pack_into("<Q", changed, 312, 0xFFFFFFFFFFFFFFFF)
    assert decode_prepared_impact(bytes(changed), context) == original


def test_independent_actor_and_threshold_binding_is_required() -> None:
    raw, context, _ = evidence()
    for wrong in (
        replace(context, vehicle=context.vehicle + 8),
        replace(context, normalization_thresholds=bytes(16)),
        replace(context, pid=context.pid + 1),
    ):
        with pytest.raises(ValueError, match="provenance"):
            decode_prepared_impact(raw[:512], wrong)


def test_nonwheel_contact_rejects_an_invented_wheel_category() -> None:
    raw, context, _ = evidence()
    changed = bytearray(raw[:512])
    pack_into("<I", changed, 212, 1)
    with pytest.raises(ValueError, match="fabricated"):
        decode_prepared_impact(bytes(changed), context)


def test_partial_record_is_never_an_empty_negative_contact_batch() -> None:
    raw, context, _ = evidence()
    with pytest.raises(ValueError, match="complete native delivery record"):
        decode_prepared_impact(raw[:511], context)
