"""Scalar projection of witnessed prepared contacts, without a collision label."""

from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.contact_impacts import (
    ImpactNormalizationParameters,
    contact_magnitude,
    normalize_contact_strength,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
    PreparedContactCapture,
)


@dataclass(frozen=True, slots=True)
class PreparedImpact:
    native_time: int
    event_time: int
    channel: int
    raw_strength: float
    normalized_strength: float


def project_prepared_impact(contact: PreparedContactCapture) -> PreparedImpact | None:
    """Apply measured arguments and owner thresholds to the source's scalar path.

    This omits unconsumed flag/material writes rather than inventing their
    values. It neither aggregates a frame nor clears maxima, emits a motor
    callback, or establishes interval coverage. The provider must still
    qualify the complete producer, its cadence and other contact paths.
    """
    if type(contact) is not PreparedContactCapture:
        raise ValueError("typed prepared contact capture required")
    magnitude = contact_magnitude(
        contact.normal, contact.contact_velocity, contact.wheel_index, contact.wheel_category
    )
    if magnitude is None:
        return None
    slot, strength = magnitude
    wheel_high, wheel_low, main_low, main_high = contact.normalization_thresholds
    low, high = (main_low, main_high) if slot == 0 else (wheel_low, wheel_high)
    normalized = normalize_contact_strength(strength, ImpactNormalizationParameters(3.6, low, high))
    return PreparedImpact(contact.native_time, contact.event_time, slot, strength, normalized)
