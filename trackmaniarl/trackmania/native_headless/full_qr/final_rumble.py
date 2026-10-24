"""Actual collector material/model snapshot; never cached post-exit defaults."""

from dataclasses import dataclass
from struct import unpack_from

from trackmaniarl.trackmania.native_headless.full_qr.final_capture import FinalCollectorBoundary
from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import PhysicalRumbleInput
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_material import select_rumble_material
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCapture


@dataclass(frozen=True, slots=True)
class FinalRumbleContext:
    contacts: PreparedImpactContext
    descriptor: int
    models: tuple[bytes, ...]
    material14_uses_default: bool


def decode_final_rumble_input(  # noqa: PLR0913 - independent raw, physical, collector and binding sources.
    raw: bytes,
    capture: VehicleCapture,
    witness: FinalCollectorBoundary,
    *,
    context: FinalRumbleContext,
) -> PhysicalRumbleInput:
    if type(raw) is not bytes or len(raw) != 1536 or raw[:8] != b"TMFM0001":
        raise ValueError("complete actual final material source required")
    binding = context.contacts
    if (
        unpack_from("<4I", raw, 8) != (1536, 1, binding.pid, binding.tid)
        or unpack_from("<Q", raw, 24)[0] != witness.sequence
        or unpack_from("<II", raw, 32) != (witness.native_t, 0)
        or capture.native_t != witness.native_t
        or unpack_from("<7Q", raw, 40)
        != (
            binding.sim,
            binding.controller,
            binding.participant,
            binding.vehicle,
            context.descriptor,
            binding.normalization_owner,
            binding.module_base,
        )
        or unpack_from("<I", raw, 100)[0] != 4
        or unpack_from("<I", raw, 108)[0] != 0x11834F2
        or raw[112:128] != bytes(16)
        or raw[1504:] != bytes(32)
        or raw[128:160] != binding.normalization_header
        or raw[160:176] != binding.normalization_thresholds
        or unpack_from("<4Q", raw, 176) != binding.wheel_models
        or tuple(raw[208 + i * 32 : 240 + i * 32] for i in range(4)) != context.models
        or raw[640:1504] != capture.visual_state
        or type(context.material14_uses_default) is not bool
    ):
        raise ValueError("actual final material/model/visual binding mismatch")
    material = select_rumble_material(capture.visual_state, context.models).material
    if unpack_from("<I", raw, 104)[0] != material:
        raise ValueError("native and qualified final material selection disagree")
    selected = unpack_from("<Q", raw, 336)[0]
    default = unpack_from("<Q", raw, 488)[0]
    if material == 80:
        if raw[336:488] != bytes(152):
            raise ValueError("sentinel final material has unexpected palette")
        palette = {}
    else:
        if not 4096 <= selected < 1 << 47:
            raise ValueError("actual selected final palette pointer required")
        palette = {material: raw[344:488]}
    if material == 14:
        if not 4096 <= default < 1 << 47:
            raise ValueError("actual material14 default source required")
        if context.material14_uses_default:
            palette[material] = raw[496:640]
    elif raw[488:640] != bytes(152):
        raise ValueError("unselected default palette must remain absent")
    return PhysicalRumbleInput(
        capture.visual_state,
        context.models,
        palette,
        unpack_from("<I", raw, 96)[0] != 0,
    )
