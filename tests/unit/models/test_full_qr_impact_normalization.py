"""Explicit source thresholds and branch boundaries, without a live source claim."""

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_impacts import (
    ImpactNormalizationParameters,
    normalize_contact_strength,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (2.0, 0.0),
        (6.0, 0.15625),
        (10.0, 0.5),
        (15.0, 0.84375),
        (20.0, 1.0),
        (30.0, 1.0),
    ],
)
def test_original_piecewise_knots_and_smoothstep(raw: float, expected: float) -> None:
    assert (
        normalize_contact_strength(raw, ImpactNormalizationParameters(1.0, 10.0, 20.0)) == expected
    )


def test_source_speed_scale_is_required_and_applied_before_helper() -> None:
    assert (
        normalize_contact_strength(3.0, ImpactNormalizationParameters(2.0, 10.0, 20.0)) == 0.15625
    )


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 10.0, 20.0),
        (1.0, 0.0, 20.0),
        (1.0, 10.0, 10.0),
        (1.0, 20.0, 10.0),
        (1.0, float("nan"), 20.0),
    ],
)
def test_unknown_or_invalid_parameters_are_not_defaulted(values: tuple) -> None:
    with pytest.raises(ValueError, match="required"):
        ImpactNormalizationParameters(*values)


@pytest.mark.parametrize("raw", [-1.0, float("nan"), float("inf")])
def test_invalid_impact_cannot_produce_a_false_negative(raw: float) -> None:
    with pytest.raises(ValueError, match="required"):
        normalize_contact_strength(raw, ImpactNormalizationParameters(1.0, 10.0, 20.0))
