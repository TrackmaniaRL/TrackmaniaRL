"""Source accumulator branches, unavailable provenance and literal real contact."""

import json
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_impacts import (
    ContactImpactInput,
    ImpactChannels,
    accumulate_contact_impact,
)
from trackmaniarl.trackmania.native_headless.full_qr.wheel_contact_capture import (
    WheelContactContext,
    decode_wheel_contact,
)


def state() -> ImpactChannels:
    return ImpactChannels(0.0, 0.0, 0.0, 80, 80, 80, 0)


def contact(**overrides: object) -> ContactImpactInput:
    return ContactImpactInput(
        **(
            {
                "normal": (0.0, 1.0, 0.0),
                "contact_velocity": (4.0, -8.0, 3.0),
                "material": 16,
                "wheel_index": 0,
                "wheel_category": 1,
                "contact_flags": 4,
            }
            | overrides
        )
    )


@pytest.mark.parametrize(("category", "expected"), [(0, (0.0, 0.0, 8.0)), (1, (0.0, 8.0, 0.0))])
def test_actual_wheel_word_selects_channel(category: int, expected: tuple) -> None:
    value = accumulate_contact_impact(state(), contact(wheel_category=category))
    assert (value.main, value.first, value.second) == expected
    assert value.flags == 4


def test_main_uses_contact_dot_not_vehicle_speed_or_vector_magnitude() -> None:
    value = accumulate_contact_impact(state(), contact(normal=(1.0, 0.0, 0.0), wheel_category=None))
    assert value.main == 4.0
    assert value.main_material == 16
    assert value.first == value.second == 0.0


def test_source_y_threshold_is_quarter_and_equality_reads_category() -> None:
    with pytest.raises(ValueError, match="wheel category source unavailable"):
        accumulate_contact_impact(state(), contact(normal=(1.0, 0.25, 0.0), wheel_category=None))
    value = accumulate_contact_impact(
        state(), contact(normal=(1.0, 0.249, 0.0), wheel_category=None)
    )
    assert value.main > 0
    assert value.first == value.second == 0


def test_non_wheel_sentinel_does_not_read_category() -> None:
    value = accumulate_contact_impact(state(), contact(wheel_index=-1, wheel_category=None))
    assert value.main == 8.0


def test_tiny_contact_speed_returns_before_flags_and_category() -> None:
    initial = state()
    assert (
        accumulate_contact_impact(
            initial, contact(contact_velocity=(0.0, 0.0, 0.0), wheel_category=None)
        )
        is initial
    )


def test_zero_normal_with_nonzero_velocity_still_accumulates_flags() -> None:
    result = accumulate_contact_impact(state(), contact(normal=(0.0, 0.0, 0.0)))
    assert result.main == result.first == result.second == 0.0
    assert result.flags == 4


def test_equal_or_smaller_impact_preserves_material_and_maximum() -> None:
    initial = ImpactChannels(0.0, 9.0, 0.0, 80, 9, 80, 2)
    actual = accumulate_contact_impact(initial, contact())
    assert actual.first == 9.0
    assert actual.first_material == 9
    assert actual.flags == 6
    equal = accumulate_contact_impact(actual, contact(contact_velocity=(0.0, -9.0, 0.0)))
    assert equal == actual


@pytest.mark.parametrize(
    "overrides",
    [
        {"normal": (float("nan"), 1.0, 0.0)},
        {"contact_velocity": (0.0, float("inf"), 0.0)},
        {"wheel_index": True},
        {"wheel_index": -2},
        {"wheel_category": True},
        {"contact_flags": -1},
        {"material": 81},
    ],
)
def test_unavailable_or_invalid_inputs_are_not_negative_collision(overrides: dict) -> None:
    with pytest.raises(ValueError, match="required"):
        accumulate_contact_impact(state(), contact(**overrides))


def test_saved_actual_contact_is_literal_input_not_callback_qualification() -> None:
    root = Path(__file__).parents[2] / "fixtures/full_qr/wheel_contacts_v1"
    context = json.loads((root / "context.json").read_text())
    source = context.get("context", context)
    bound = WheelContactContext(
        **{key: source[key] for key in WheelContactContext.__dataclass_fields__}
    )
    data = (root / "contacts.bin").read_bytes()
    contacts = [decode_wheel_contact(data[i : i + 2048], bound) for i in range(0, len(data), 2048)]
    side = next(value for value in contacts if abs(value.normal[1]) < 0.25)
    result = accumulate_contact_impact(
        state(),
        ContactImpactInput(
            normal=side.normal,
            contact_velocity=side.contact_velocity,
            material=side.material,
            wheel_index=side.wheel_index,
            wheel_category=None,
            contact_flags=0,
        ),
    )
    # Source execution110 produced literal float32 bits 6c8ee941 for record8.
    assert result.main == 29.194541931152344
    assert result.main_material == side.material == 9
    assert result.first == result.second == 0.0
