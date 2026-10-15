"""Optional policy-side components independent of game and map integrations."""

from trackmaniarl.policies.reference_support import (
    ReferenceActionSupport,
    ReferenceSupportConfig,
    ReferenceSupportDecision,
    ReferenceSupportRequest,
)

__all__ = [
    "ReferenceActionSupport",
    "ReferenceSupportConfig",
    "ReferenceSupportDecision",
    "ReferenceSupportRequest",
]
