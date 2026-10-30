"""Material-only projection of the pinned original732f10 producer helper.

Per-wheel script getters are different: this helper retains the raw fallback
material when a wheel lacks contact, and selects a front/back group by counts.
No amplitude, clock, motor callback or collision label is supplied here.
"""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from


@dataclass(frozen=True, slots=True)
class RumbleMaterial:
    material: int
    front_contacts: int
    back_contacts: int


def select_rumble_material(visual: bytes, wheel_models: tuple[bytes, ...]) -> RumbleMaterial:
    if type(visual) is not bytes or len(visual) != 864:
        raise ValueError("complete actual visual source required")
    if (
        type(wheel_models) is not tuple
        or len(wheel_models) != 4
        or any(type(model) is not bytes or len(model) != 32 for model in wheel_models)
    ):
        raise ValueError("four independently bound32-byte wheel model snapshots required")
    contacts = [0, 0]  # Original helper group A=back, B=front.
    material = [80, 80]
    fallback = [0, 0]  # Literal original helper initial register values.
    for wheel, model in enumerate(wheel_models):
        longitudinal = unpack_from("<f", model, 0x18)[0]
        if not isfinite(longitudinal):
            raise ValueError("finite actual wheel model longitudinal position required")
        group = int(longitudinal > 0)
        raw_material = visual[0xB8 + wheel * 0x2C]
        flags = unpack_from("<I", visual, 0xD0 + wheel * 0x2C)[0]
        if raw_material > 80:
            raise ValueError("raw wheel material outside the qualified domain")
        if flags & 4:
            contacts[group] += 1
            if material[group] != 13:
                material[group] = raw_material
        else:
            fallback[group] = raw_material
        if flags & 2:
            material[group] = 13
    for group in (0, 1):
        if material[group] >= 80:
            material[group] = fallback[group]
    selected = 0 if contacts[0] > contacts[1] else 1
    return RumbleMaterial(material[selected], contacts[1], contacts[0])
