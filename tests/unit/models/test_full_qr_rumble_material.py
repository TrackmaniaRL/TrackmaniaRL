"""Synthetic wheel and material inputs for the selection contract."""

from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_material import (
    RumbleMaterial,
    select_rumble_material,
)


def sources() -> tuple[bytes, tuple[bytes, ...]]:
    visual = bytearray(864)
    models = []
    for wheel, (position, material) in enumerate(((-1.0, 3), (-1.0, 4), (1.0, 5), (1.0, 6))):
        model = bytearray(32)
        pack_into("<f", model, 0x18, position)
        models.append(bytes(model))
        visual[0xB8 + wheel * 0x2C] = material
        pack_into("<I", visual, 0xD0 + wheel * 0x2C, 4)
    return bytes(visual), tuple(models)


def test_synthetic_tie_selects_front_material() -> None:
    assert select_rumble_material(*sources()) == RumbleMaterial(6, 2, 2)


def test_synthetic_special_material_has_priority() -> None:
    visual, models = sources()
    changed = bytearray(visual)
    pack_into("<I", changed, 0xD0 + 2 * 0x2C, 6)
    assert select_rumble_material(bytes(changed), models) == RumbleMaterial(13, 2, 2)


def test_synthetic_back_group_wins_when_it_has_more_contacts() -> None:
    visual, models = sources()
    changed = bytearray(visual)
    pack_into("<I", changed, 0xD0 + 3 * 0x2C, 0)
    assert select_rumble_material(bytes(changed), models) == RumbleMaterial(4, 1, 2)


def test_synthetic_out_of_domain_material_is_rejected() -> None:
    visual, models = sources()
    changed = bytearray(visual)
    changed[0xB8] = 81
    with pytest.raises(ValueError, match="outside the qualified domain"):
        select_rumble_material(bytes(changed), models)


def test_missing_wheel_binding_is_rejected() -> None:
    visual, _ = sources()
    with pytest.raises(ValueError, match="four independently bound"):
        select_rumble_material(visual, ())


def test_nonfinite_wheel_geometry_is_rejected() -> None:
    visual, bound = sources()
    models = list(bound)
    changed = bytearray(models[0])
    pack_into("<f", changed, 0x18, float("nan"))
    models[0] = bytes(changed)
    with pytest.raises(ValueError, match="finite actual wheel"):
        select_rumble_material(visual, tuple(models))
