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


def test_actual_rounded_contact_requires_exact_captured_projection() -> None:
    """Packet d9ee23b3 and original native helper both return float32 0x3f800002."""
    from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
        PreparedContactCapture,
    )
    from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact import (
        project_prepared_impact,
    )
    from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import (
        RumbleState,
        combine_contact_rumble,
    )

    contact = PreparedContactCapture(
        12338,
        20440,
        20455,
        0,
        1,
        9,
        (-0.5869570374488831, 0.008156714960932732, -0.8095770478248596),
        (0.9250243306159973, 0.3193965554237366, 2.1242616176605225),
        (-0.0036933422088623047, -0.38727688789367676, 49.39988708496094),
        (115.19999694824219, 21.599998474121094, 21.599998474121094, 144.0),
    )
    impact = project_prepared_impact(contact)
    assert impact is not None
    assert impact.normalized_strength == 1.000000238418579
    channels = (impact.normalized_strength, 0.0, 0.0)
    with pytest.raises(ValueError, match="bounded"):
        combine_contact_rumble(RumbleState(0.0, 0.0), channels)
    assert combine_contact_rumble(
        RumbleState(0.0, 0.0), channels, source_contacts=(contact,)
    ) == RumbleState(4.0, 1.0)
    with pytest.raises(ValueError, match="exact source projection"):
        combine_contact_rumble(
            RumbleState(0.0, 0.0), (1.0000001192092896, 0.0, 0.0), source_contacts=(contact,)
        )
