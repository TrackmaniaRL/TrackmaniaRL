"""Join actual per-tick contact deliveries and bound surface inputs.

This staged producer has no collision authority or reward watermark. Its
physical output and scheduling must be qualified against the client before
the full environment can use it as a notification source.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from struct import unpack_from

from trackmaniarl.trackmania.native_headless.full_qr.contact_delivery import (
    DeliveredContactBoundary,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact import project_prepared_impact
from trackmaniarl.trackmania.native_headless.full_qr.rumble_material import select_rumble_material
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import (
    RumbleState,
    SurfaceRumbleInput,
    _round_float32,
    combine_contact_rumble,
    produce_rumble,
)


@dataclass(frozen=True, slots=True)
class PhysicalRumbleInput:
    visual: bytes
    wheel_models: tuple[bytes, ...]
    palette: Mapping[int, bytes]
    surface_enabled: bool


@dataclass(frozen=True, slots=True)
class PhysicalRumbleStep:
    native_t: int
    state: RumbleState
    normalized_impacts: tuple[float, float, float]
    material: int
    longitudinal_speed: float
    new_contacts: int


def longitudinal_speed(visual: bytes) -> float:
    """Port 1407cdbf2..1407cdc22 on an actual matching held physical visual.

    Products are added in y,x,z order. The owner must prove that the visual
    belongs to the held physical tick; an interpolated client visual differs.
    """
    if type(visual) is not bytes or len(visual) != 864:
        raise ValueError("complete actual physical visual source required")
    matrix = unpack_from("<9f", visual, 0x2C)
    velocity = unpack_from("<3f", visual, 0x5C)
    if not all(isfinite(value) for value in (*matrix, *velocity)):
        raise ValueError("finite actual physical transform and velocity required")
    f32 = _round_float32
    y = f32(matrix[5] * velocity[1])
    x = f32(matrix[2] * velocity[0])
    z = f32(matrix[8] * velocity[2])
    return f32(f32(y + x) + z)


class ContactRumbleProducer:
    """Transactional ten-ms projection; no inferred source fields or labels."""

    def __init__(self, initial: RumbleState) -> None:
        if type(initial) is not RumbleState:
            raise ValueError("actual bootstrap rumble state required")
        self.state = initial
        self.native_t = 2190
        self.contacts = 0

    def advance(
        self, boundary: DeliveredContactBoundary, source: PhysicalRumbleInput
    ) -> PhysicalRumbleStep:
        if (
            type(boundary) is not DeliveredContactBoundary
            or type(source) is not PhysicalRumbleInput
        ):
            raise ValueError("decoded delivery and independently bound surface source required")
        receipt = boundary.receipt
        clock = receipt.native_t
        if clock != self.native_t + 10 or receipt.sequence != (clock - 2190) // 10 + 1:
            raise ValueError("every consecutive actual physical receipt required")
        if receipt.contacts != self.contacts + len(boundary.contacts):
            raise ValueError("receipt must cover its exact new contact prefix")
        normalized = [0.0, 0.0, 0.0]
        for offset, contact in enumerate(boundary.contacts, start=1):
            if contact.sequence != self.contacts + offset or contact.native_time != clock - 10:
                raise ValueError("contact belongs to another sequence or delivery tick")
            # Windows848100..848413 updates the three scalar maxima during
            # delivery, without reading its saved timestamp argument. Actual
            # straight487 includes667 callbacks stamped callback+15ms. Their
            # timestamps remain in the raw source; they do not defer this
            # scalar accumulator. Goal timestamps use a separate contract.
            impact = project_prepared_impact(contact)
            if impact is not None:
                normalized[impact.channel] = max(
                    normalized[impact.channel], impact.normalized_strength
                )
        material = select_rumble_material(source.visual, source.wheel_models).material
        speed = longitudinal_speed(source.visual)
        if type(source.surface_enabled) is not bool:
            raise ValueError("explicit actual surface-enable flag required")
        channels = (normalized[0], normalized[1], normalized[2])
        if material == 80:
            # Literal source7cecdb skips the whole surface block for sentinel80.
            state = combine_contact_rumble(self.state, channels)
        else:
            parameters = source.palette.get(material)
            if type(parameters) is not bytes or len(parameters) != 0x90:
                raise ValueError("complete selected actual material parameters required")
            sensitivity, cap, divisor, gain = (
                unpack_from("<f", parameters, offset)[0] for offset in (0x54, 0x58, 0x7C, 0x80)
            )
            surface = SurfaceRumbleInput(
                speed, material, source.surface_enabled, sensitivity, cap, divisor, gain
            )
            state = produce_rumble(self.state, surface, channels)
        result = PhysicalRumbleStep(clock, state, channels, material, speed, len(boundary.contacts))
        self.state, self.native_t, self.contacts = state, clock, receipt.contacts
        return result
