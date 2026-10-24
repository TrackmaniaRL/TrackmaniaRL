"""Actual end-of-physics material/model snapshot; no post-exit cached defaults."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from struct import pack_into, unpack_from

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    CommittedGoalEventWindow,
)
from trackmaniarl.trackmania.native_headless.full_qr.final_capture import (
    FinalCollectorBoundary,
    decode_final_collector_boundary,
)
from trackmaniarl.trackmania.native_headless.full_qr.final_rumble import (
    FinalRumbleContext,
    decode_final_rumble_input,
)
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import (
    NativeGoalContext,
    decode_goal_boundary,
    decode_goal_facts,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
)
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import CommittedGoalCapture
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    VehicleCapture,
    VehicleCaptureContext,
    decode_vehicle_capture,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/final_material584"


def _sources() -> tuple[bytes, VehicleCapture, FinalCollectorBoundary, FinalRumbleContext]:
    stored = json.loads((FIXTURE / "context.json").read_text())
    for name, digest in stored["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    pid, meta = stored["pid"], stored["meta"]
    raw = (FIXTURE / "material.bin").read_bytes()
    proof = decode_final_collector_boundary(
        (FIXTURE / "collector.bin").read_bytes(),
        (FIXTURE / "goals.bin").read_bytes(),
        context=NativeGoalContext(pid, meta),
    )
    capture = decode_vehicle_capture(
        (FIXTURE / "vehicle.bin").read_bytes(),
        VehicleCaptureContext(
            pid, meta["participant"], meta["vehicle"], meta["module_slide"], proof.native_t
        ),
    )
    models = tuple(raw[208 + i * 32 : 240 + i * 32] for i in range(4))
    # Unit fixture supplies its measured source binding. Production additionally
    # compares against the independently held reset binding in HeldRumbleBindings.
    contacts = PreparedImpactContext(
        pid=pid,
        tid=pid,
        module_base=meta["module_slide"],
        vehicle=meta["vehicle"],
        participant=meta["participant"],
        sim=meta["sim"],
        controller=meta["controller"],
        normalization_owner=unpack_from("<Q", raw, 80)[0],
        normalization_header=raw[128:160],
        normalization_thresholds=raw[160:176],
        wheel_models=unpack_from("<4Q", raw, 176),
        wheel_headers=tuple(model[:16] for model in models),
    )
    context = FinalRumbleContext(contacts, unpack_from("<Q", raw, 72)[0], models, False)
    return raw, capture, proof, context


def test_actual_surface_source_is_preserved_at_final_native_tick() -> None:
    raw, capture, proof, context = _sources()
    source = decode_final_rumble_input(raw, capture, proof, context=context)
    assert proof.native_t == 55000
    assert set(source.palette) == {16}
    assert source.palette[16] == raw[344:488]
    assert source.visual == capture.visual_state
    assert source.wheel_models == context.models


@pytest.mark.parametrize(("offset", "value"), [(32, 55010), (104, 13), (176, 0), (336, 0)])
def test_wrong_clock_selection_or_model_pointer_is_rejected(offset: int, value: int) -> None:
    raw, capture, proof, context = _sources()
    changed = bytearray(raw)
    pack_into("<Q" if offset in (176, 336) else "<I", changed, offset, value)
    with pytest.raises(ValueError, match=r"mismatch|disagree|required"):
        decode_final_rumble_input(bytes(changed), capture, proof, context=context)


def test_foreign_binding_and_missing_record_cannot_supply_zero_motor() -> None:
    raw, capture, proof, context = _sources()
    foreign = replace(context, contacts=replace(context.contacts, participant=4096))
    with pytest.raises(ValueError, match="binding mismatch"):
        decode_final_rumble_input(raw, capture, proof, context=foreign)
    with pytest.raises(ValueError, match="complete actual"):
        decode_final_rumble_input(b"", capture, proof, context=context)


def test_real_completed50ms_sample_requires_its_own_collector_proof() -> None:
    _, vehicle, proof, _ = _sources()
    stored = json.loads((FIXTURE / "context.json").read_text())
    context = NativeGoalContext(stored["pid"], stored["meta"])
    facts = decode_goal_facts((FIXTURE / "goals.bin").read_bytes(), context)
    goal = decode_goal_boundary((FIXTURE / "goal-receipt.bin").read_bytes(), facts, context)
    with pytest.raises(ValueError, match="actual collector proof"):
        CommittedGoalCapture(54950, vehicle, goal, facts[0])
    frame = CommittedGoalCapture(54950, vehicle, goal, facts[0], proof)
    assert frame.vehicle.native_t == 55000
    assert frame.fact.event_native_t == 55007
    with pytest.raises(ValueError, match="actual collector proof"):
        replace(frame, final_boundary=replace(proof, goal_native_t=55008))


def test_completed_sample_does_not_relax_the_existing_partial_window() -> None:
    with pytest.raises(ValueError, match="separately timestamped"):
        CommittedGoalEventWindow(0, 50, 57, False)
    event = CommittedGoalEventWindow(0, 50, 57, False, final_sample_completed=True)
    assert event.sampled_ms == 50
    assert event.score_ms == 57
    with pytest.raises(ValueError, match="separately timestamped"):
        CommittedGoalEventWindow(0, 40, 47, False, final_sample_completed=True)
