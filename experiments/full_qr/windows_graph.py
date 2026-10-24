"""Full V5 inputs for Windows comparison only; never a training environment.

The caller owns the same mutex/process birth/module-hash/read-quota guards as
WindowsTerminalReader. Scene traversal follows the installed VehicleState
sources, with actual native field registration/getters pinned separately.
GameTime is the Connect fallback counter; rules.Now equivalence remains open.
"""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from struct import unpack, unpack_from
from time import monotonic, sleep
from typing import Any

import numpy as np
import torch

from experiments.full_qr.windows_terminal import (
    WindowsPhysicsSample,
    WindowsPhysicsSampleChangedError,
    WindowsPhysicsUnavailableError,
    WindowsTerminalReader,
    WindowsTerminalSnapshot,
)
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.trackmania.native_headless.full_qr.native_features import GRAPH_FIELDS, _speed


class WindowsObservationUnavailableError(ValueError):
    """No complete current source frame; never converted into an observation."""


@dataclass(frozen=True, slots=True)
class WindowsGraphObservation:
    terminal: WindowsTerminalSnapshot
    physics: WindowsPhysicsSample
    entity_id: int
    vis: int
    state: int
    game_time_ms: int
    start_time_ms: int
    channels: tuple[float, ...]
    scalar_cache_hex: str
    visual_state_hex: str
    adherence_owner: int

    def feature_carrier(self) -> np.ndarray:
        # Private indexing carrier, not invented telemetry for unused API fields.
        if len(self.channels) != len(GRAPH_FIELDS) or not all(
            isfinite(value) for value in self.channels
        ):
            raise ValueError("all 23 actual comparison channels must be finite")
        values = np.full(33, np.nan, dtype=np.float32)
        values[list(GRAPH_FIELDS)] = self.channels
        return values


class WindowsGraphReader:
    """Current player/entity/visual state joined to current physics and API getters."""

    _SOURCE_FUNCTIONS = (
        (0x726440, 4263, "b18cc971488a49b44282ee98422ce02a60ec1cc115255ba1e844ce2c100c34c8"),
        (0x7261A0, 283, "ebf2454e8cc91c845f5d86953878bcb922b1bdaa617ab4b629c43d49c5aaca6f"),
        (0x12DD7B0, 197, "275d27c1bde165dd398fdfe09296d50a2dccc8a113e6dd5e7cf589b78ca878e0"),
        (0x1353790, 105, "92685300899169ceb0b9452ffd3102c3d3ab223876f82525148af9d470257fba"),
        (0x1353C70, 5276, "4a5bdbc64c8eb7937bdd7ded9854da261739bda6b41bfe08b52ec054b9f0b39b"),
        (0x1353BE0, 131, "374b1c9f35ed76bab83e68d5a0ca240938557fb65b8f8f16d7b691cb7831058a"),
        (0x6C2BB0, 37, "36ba605baf0a6882bd7e9b0ab1ed2f8d9c235dfab24aa0b6d79daf764d6f395d"),
        (0x6C2BD5, 58, "33100697bf7bbe5ba2df126559fcc01ba59ba0b7cdf4107f1f246465387fb90f"),
        (0x6D2610, 197, "d14ec392ae48e99b443183e943456d9c7e35cf9d834ffc04213ab9531530591d"),
    )

    def __init__(self, terminal: WindowsTerminalReader) -> None:
        self.terminal = terminal
        self._r, self._q = terminal._read, terminal._q
        self.base = terminal.modules.game_base
        for rva, size, digest in self._SOURCE_FUNCTIONS:
            raw = b"".join(
                self._r(self.base + rva + offset, min(4096, size - offset))
                for offset in range(0, size, 4096)
            )
            if sha256(raw).hexdigest() != digest:
                raise ValueError("pinned Windows comparison source function changed")
        if self._r(self.base + 0x12DD880, 21) != bytes.fromhex(
            "488d050de1a000c7410804000000488901488bc1c3"
        ):
            raise ValueError("pinned current entity selector table route changed")
        if self._r(self.base + 0x7276D0, 26) != bytes.fromhex(
            "f30f10494cf30f1041408b4134f30f114204f30f114a088902c3"
        ):
            raise ValueError("pinned visual Dir getter changed")
        if self._r(self.base + 0x6D2500, 12) != bytes.fromhex("4863c248c1e00448034130c3"):
            raise ValueError("pinned physics reference array lookup changed")

    def _entity(self, snapshot: WindowsTerminalSnapshot) -> tuple[int, bytes, int, int | None]:
        player = snapshot.player
        if self._q(self._q(player) + 0x108) != self.base + 0x12DD7B0:
            raise ValueError("current player has a different entity selector")
        fields = self._r(player + 0x278, 0x34)
        entity = 0x0FF00000
        context = 0
        context_kind = None
        for offset in (0x28C, 0x280):
            candidate = unpack_from("<I", fields, offset - 0x278)[0]
            if candidate != 0x0FF00000:
                entity = candidate
                break
        else:
            context = self._q(player + 0xC60)
            selected = 6
            if context:
                context_kind = self._r(context + 12, 1)[0]
                if context_kind < 4:
                    selected = self._r(self.base + 0x1CEB994, 4)[context_kind]
                if selected > 6:
                    raise ValueError("unknown current entity selector slot")
            offset = 0x278 if selected == 6 else 0x294 + selected * 4
            entity = unpack_from("<I", fields, offset - 0x278)[0]
        if entity & 0xFF000000 != 0x02000000:
            raise ValueError("one actual driving entity is required; no replay/editor fallback")
        return entity, fields, context, context_kind

    def _block(self, address: int, size: int) -> bytes:
        return b"".join(
            self._r(address + offset, min(4096, size - offset)) for offset in range(0, size, 4096)
        )

    def _adherence(
        self, snapshot: WindowsTerminalSnapshot, entity: int, physical: WindowsPhysicsSample
    ) -> tuple[float, int, int, bytes, int, bytes]:
        # The global context1fbce20 is temporarily set to this actual CSmArena
        # during script calls, then cleared (actual313/314). Resolve the same
        # typed arena through CurrentPlayground.Arena, not through a null default.
        self.terminal._property(0x1F55170, "Arena", 0xD78)
        arena = self._q(snapshot.playground + 0xD78)
        self.terminal._class(arena, "CSmArena", 0x2D019000)
        pool = self._q(arena + 0x7C8)
        table = pool + 0x8E0
        header = self._r(table, 0x38)
        capacity, count = unpack_from("<II", header, 4)
        flags, keys, values = unpack_from("<QQQ", header, 0x20)
        if not 0 < count <= capacity <= 65536 or capacity & (capacity - 1):
            raise ValueError("bounded nonempty native entity reference table required")
        slot = entity & (capacity - 1)
        initial = slot
        for step in range(capacity):
            bits = unpack("<I", self._r(flags + (slot >> 4) * 4, 4))[0]
            flag = (bits >> ((slot & 15) * 2)) & 3
            if flag & 2:
                raise ValueError("actual current entity is absent from the arena physics pool")
            if flag == 0 and unpack("<I", self._r(keys + slot * 4, 4))[0] == entity:
                break
            slot = (slot + step + 1) & (capacity - 1)
            if slot == initial:
                raise ValueError("native entity reference lookup exhausted")
        else:
            raise ValueError("native entity reference lookup exceeded bounded capacity")
        reference_address = values + slot * 16
        reference = self._r(reference_address, 16)
        if reference.hex() != physical.reference_hex:
            raise ValueError(
                "arena entity resolver disagrees with the current player/body reference"
            )
        coefficient = unpack("<f", self._r(physical.vehicle + 0x1760, 4))[0]
        if not isfinite(coefficient) or coefficient < 0:
            raise ValueError("actual physical adherence coefficient is unavailable")
        return coefficient, arena, table, header, reference_address, reference

    def sample(self, *, tolerance: float = 0.2) -> WindowsGraphObservation:
        snapshot = self.terminal.sample()
        try:
            physical = self.terminal.physics(snapshot, tolerance=tolerance)
        except (WindowsPhysicsUnavailableError, WindowsPhysicsSampleChangedError) as error:
            raise WindowsObservationUnavailableError(str(error)) from error
        self.terminal._property(0x1ED0BC0, "GameScene", 0x2A0)
        entity, entity_fields, context, context_kind = self._entity(snapshot)
        scene = self._q(snapshot.app + 0x2A0)
        manager_count = unpack("<I", self._r(scene + 8, 4))[0]
        if not 13 < manager_count <= 128:
            raise ValueError("current vehicle scene manager is unavailable")
        manager = self._q(scene + 0x10 + 13 * 8)
        container = self._r(manager + 0x210, 16)
        pointer, count, capacity = unpack("<QII", container)
        if not pointer or pointer & 15 or not 0 < count <= capacity <= 1000:
            raise ValueError("bounded aligned current visual container required")
        objects = self._block(pointer, count * 8)
        matches = [
            address
            for address in unpack("<" + "Q" * count, objects)
            if unpack("<I", self._r(address, 4))[0] == entity
        ]
        if len(matches) != 1:
            raise ValueError("one unique visual for the current GUI entity is required")
        vis = matches[0]
        state = self._q(vis + 0x130)  # Native registration306, AsyncState reference.
        if state == 0:
            # VehicleState.ViewingPlayerState returns null here; Connect yields
            # without sending a packet. Actual321 observed this during reset.
            raise WindowsObservationUnavailableError("current visual AsyncState is unavailable")
        coefficient, owner, table, table_header, reference_address, reference = self._adherence(
            snapshot, entity, physical
        )
        if self._q(self.base + 0x1FBCE20) not in (0, owner):
            raise ValueError("active script adherence context belongs to a different arena")
        start = self._r(snapshot.player + 0x218, 4)
        game_time = self.terminal.game_time(snapshot)
        for _ in range(3):
            cache = self._r(snapshot.player + 0xE24, 0xFC)
            visual = self._r(state, 864)
            if self._r(snapshot.player + 0xE24, 0xFC) == cache and self._r(state, 864) == visual:
                break
        else:
            raise WindowsObservationUnavailableError(
                "comparison scalar/visual buffers did not produce a coherent sample"
            )
        position = unpack_from("<3f", cache)
        visual_position = unpack_from("<3f", visual, 0x50)
        if (
            not all(isfinite(value) for value in position + visual_position)
            or max(abs(a - b) for a, b in zip(position, visual_position, strict=True)) > tolerance
            or max(abs(a - b) for a, b in zip(position, physical.gui_position, strict=True))
            > tolerance
        ):
            raise ValueError("current visual/scalar/physics positions do not match")

        def f32(offset: int) -> float:
            return unpack_from("<f", cache, offset - 0xE24)[0]

        def u32(offset: int) -> int:
            return unpack_from("<I", cache, offset - 0xE24)[0]

        velocity = unpack_from("<3f", cache, 0xC)
        slips = tuple(unpack_from("<f", visual, offset)[0] for offset in (0xBC, 0xE8, 0x140, 0x114))
        skid, flying = u32(0xF0C), u32(0xF1C)
        if (
            skid > 4
            or flying > 0x7FFFFFFF
            or u32(0xEFC) > 16
            or not all(0 <= value <= 1 for value in slips)
            or not -1 <= f32(0xEEC) <= 1
            or not 0 <= f32(0xEF0) <= 1
        ):
            raise ValueError("actual native graph getter value outside its source domain")
        start_time = unpack("<i", start)[0]
        if start_time <= 0:
            raise ValueError("actual positive race StartTime is required")
        channels = (
            float(max(0, game_time - start_time)),
            *position,
            *velocity,
            *(unpack_from("<f", visual, offset)[0] for offset in (0x34, 0x40, 0x4C)),
            _speed(velocity),
            f32(0xEF8),
            float(u32(0xEFC)),
            *slips,
            float(skid),
            float(flying),
            coefficient,
            f32(0xEEC),
            f32(0xEF0),
            float(u32(0xEF4) != 0),
        )
        if len(channels) != 23 or not all(isfinite(value) for value in channels):
            raise ValueError("complete finite actual comparison graph inputs required")
        current = self.terminal.sample()
        unchanged = {
            "terminal": current == snapshot,
            "start": self._r(snapshot.player + 0x218, 4) == start,
            "entity_fields": self._r(snapshot.player + 0x278, 0x34) == entity_fields,
            "context": self._q(snapshot.player + 0xC60) == context,
            "context_kind": not context or self._r(context + 12, 1)[0] == context_kind,
            "scene": self._q(snapshot.app + 0x2A0) == scene,
            "manager": self._q(scene + 0x10 + 13 * 8) == manager,
            "container": self._r(manager + 0x210, 16) == container,
            "objects": self._block(pointer, count * 8) == objects,
            "vis_entity": unpack("<I", self._r(vis, 4))[0] == entity,
            "state": self._q(vis + 0x130) == state,
            "adherence_arena": self._q(snapshot.playground + 0xD78) == owner,
            "adherence_table": self._r(table, 0x38) == table_header,
            "adherence_reference": self._r(reference_address, 16) == reference,
            "adherence_value": unpack("<f", self._r(physical.vehicle + 0x1760, 4))[0]
            == coefficient,
            "adherence_context": self._q(self.base + 0x1FBCE20) in (0, owner),
        }
        if not all(unchanged.values()):
            changed_names = {name for name, value in unchanged.items() if not value}
            changed = ", ".join(sorted(changed_names))
            # VehicleState can publish a different object-list container while
            # this read spans two render updates. Discard the entire attempt;
            # wait_sample rebinds the full chain under its unchanged guard.
            if changed_names <= {"state", "context", "context_kind", "container"}:
                raise WindowsObservationUnavailableError(
                    f"current visual/entity context changed before publication: {changed}"
                )
            raise ValueError(f"current comparison chain changed during sample: {changed}")
        return WindowsGraphObservation(
            snapshot,
            physical,
            entity,
            vis,
            state,
            game_time,
            start_time,
            channels,
            cache.hex(),
            visual.hex(),
            owner,
        )

    def wait_sample(
        self, *, guard: Callable[[], object], tolerance: float = 0.2, timeout_s: float = 1.0
    ) -> WindowsGraphObservation:
        """Bounded equivalent of Connect yielding without a packet.

        Each attempt rebinds everything. Only explicitly unavailable/unstable
        source buffers wait; missing identity, position mismatches, bad getters,
        read failures and other source errors remain fatal. The caller's guard
        checks focus, process birth and the actual UI while no new input is sent.
        """
        if not isfinite(timeout_s) or not 0 < timeout_s <= 1.0:
            raise ValueError("comparison source wait must be bounded by one second")
        deadline = monotonic() + timeout_s
        while True:
            guard()
            try:
                return self.sample(tolerance=tolerance)
            except WindowsObservationUnavailableError:
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "complete current comparison source frame did not arrive"
                    ) from None
                sleep(0.001)


class WindowsComparisonGraphPipeline(BoundaryGraphFeaturePipelineV5):
    """The same full V5 calculations, with actual live comparison timestamps.

    This does not change NativeGraphFrame's exact50ms training contract.
    Unused API fields remain absent; no synthetic packet is sent to the SDK.
    """

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if not isinstance(observation, WindowsGraphObservation):
            raise TypeError("actual Windows comparison graph observation required")
        values = observation.feature_carrier()
        position = values[[4, 5, 6]]
        nearest = self._nearest(position)
        track, curvature = self._track_features(position, values[[10, 12]], nearest)
        result = {
            "physics": torch.from_numpy(self._physics_v2(values, nearest, curvature)),
            "track": torch.from_numpy(track),
            "context": torch.from_numpy(self._context_v3(values, nearest)),
            "recovery": torch.from_numpy(self._recovery_v5(values, nearest)),
        }
        self._validate_v5_tensors(result)
        return result
