"""Offline diagnostics for saved captures from the pinned Linux vehicle.

Equations follow ELF 3b610782...779 functions 9d6ad0 and 1194748. This
module performs no process reads and does not authorize training observations.
Modeled values are separate from the participant's captured API caches.
"""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from


@dataclass(frozen=True, slots=True)
class VehicleDiagnostics:
    slip_native_order: tuple[float, ...]
    material_native_order: tuple[int, ...]
    modeled_skidding_count: int
    cached_skidding_count: int
    cached_flying_duration: int
    flying_accumulator_branch: bool
    modeled_adherence_default: float | None


def _exact(value: bytes, size: int) -> None:
    if type(value) is not bytes or len(value) != size:
        raise ValueError(f"exactly {size} captured bytes required")


def diagnose_vehicle(
    visual_state: bytes, player_cache: bytes, global_owner: bytes
) -> VehicleDiagnostics:
    """Compare literal caches with source equations, without assuming freshness.

    Native wheel order is FL/FR/RR/RL. A nonnull adherence owner requires
    the separately qualified resolver; its absence never selects a fallback.
    The flying predicate alone does not establish an accumulated duration.
    """
    _exact(visual_state, 864)
    _exact(player_cache, 0x8C)
    _exact(global_owner, 8)
    slips = tuple(unpack_from("<f", visual_state, 0xBC + i * 0x2C)[0] for i in range(4))
    if not all(isfinite(value) for value in slips):
        raise ValueError("nonfinite captured wheel coefficient")
    flags = tuple(unpack_from("<I", visual_state, 0xD0 + i * 0x2C)[0] for i in range(4))
    source_flags = unpack_from("<I", visual_state, 0x88)[0]
    materials = tuple(
        13 if flag & 2 else visual_state[0xB8 + i * 0x2C] if flag & 4 else 80
        for i, flag in enumerate(flags)
    )
    skidding = sum(
        bool(flag & 4) and (slip > 0.0 or bool(source_flags & 32))
        for flag, slip in zip(flags, slips, strict=True)
    )
    return VehicleDiagnostics(
        slip_native_order=slips,
        material_native_order=materials,
        modeled_skidding_count=skidding,
        cached_skidding_count=unpack_from("<I", player_cache, 0x4C)[0],
        cached_flying_duration=unpack_from("<I", player_cache, 0x5C)[0],
        flying_accumulator_branch=(
            (((source_flags & 0x10008) ^ 0x10000) | unpack_from("<I", visual_state, 0x1BC)[0]) == 0
        ),
        modeled_adherence_default=1.0 if unpack_from("<Q", global_owner)[0] == 0 else None,
    )
