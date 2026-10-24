"""Read-only terminal identity for the pinned Windows comparison client.

The caller must own the controller mutex, verify the current process birth and
both module hashes, and bound its ReadProcessMemory callback. This neither
authorizes inputs nor turns a client UI sample into a native terminal event.
Missing pointers/type mismatches remain errors, never Finish=False.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from struct import unpack, unpack_from
from time import monotonic, sleep


class WindowsPhysicsSampleChangedError(ValueError):
    """An initially valid physics binding changed before its snapshot was complete."""


class WindowsPhysicsUnavailableError(ValueError):
    """The current actor has no resolved physics reference; no sample exists."""


@dataclass(frozen=True, slots=True)
class WindowsModuleContext:
    game_base: int
    openplanet_base: int


@dataclass(frozen=True, slots=True)
class WindowsTerminalSnapshot:
    app: int
    playground: int
    terminal: int
    player: int
    script_api: int
    root_map: int
    map_info: int
    map_identifier: int
    resolved_map_identifier: str
    ui_sequence: int
    ui_change_human_ms: int
    ui_sequence_number: int

    @property
    def finish(self) -> bool:
        return self.ui_sequence == 11


@dataclass(frozen=True, slots=True)
class WindowsPhysicsSample:
    body: int
    vehicle: int
    model: int
    reference_hex: str
    physical40_hex: str
    gui_position: tuple[float, float, float]
    physical_position: tuple[float, float, float]
    position_error: float


class WindowsTerminalReader:
    """Rebind the actual client chain every sample; never reuse stale pointers."""

    def __init__(self, read: Callable[[int, int], bytes], modules: WindowsModuleContext) -> None:
        self.read, self.modules = read, modules

    def _read(self, address: int, size: int) -> bytes:
        if not 4096 <= address < address + size < 0x800000000000 or not 0 < size <= 4096:
            raise ValueError("bounded canonical user-space read required")
        raw = self.read(address, size)
        if type(raw) is not bytes or len(raw) != size:
            raise ValueError("complete literal Windows memory read required")
        return raw

    def _q(self, address: int) -> int:
        return unpack("<Q", self._read(address, 8))[0]

    def _class(self, address: int, name: str, identifier: int) -> None:
        vtable = self._q(address)
        getter = self._q(vtable + 16)  # Windows ABI: GetClassInfo is slot2.
        code = self._read(getter, 8)
        if code[:3] not in (b"\x48\x8d\x05", b"\x48\x8b\x05") or code[7] != 0xC3:
            raise ValueError("unsupported pinned Windows GetClassInfo getter")
        target = getter + 7 + unpack_from("<i", code, 3)[0]
        descriptor = self._q(target) if code[1] == 0x8B else target
        raw = self._read(descriptor, 32)
        if (
            unpack_from("<I", raw, 24)[0] != identifier
            or self._read(unpack_from("<Q", raw, 8)[0], len(name) + 1)
            != name.encode("ascii") + b"\0"
        ):
            raise ValueError(f"current Windows object is not {name}")

    def _property(self, rva: int, name: str, expected_offset: int) -> None:
        raw = self._read(self.modules.game_base + rva, 0x48)
        if (
            unpack_from("<I", raw, 0x18)[0] != expected_offset
            or self._read(unpack_from("<Q", raw, 0x20)[0], len(name) + 1)
            != name.encode("ascii") + b"\0"
        ):
            raise ValueError("pinned Windows property descriptor changed")

    def _identifier(self, value: int) -> str:
        # Literal read equivalent of pinned MwId lookup1402d47e0. Other flag
        # classes are unavailable, not rendered through a guessed fallback.
        if value & 0xC0000000 != 0x40000000:
            raise ValueError("a live interned map identifier is required")
        base = self.modules.game_base
        signature = bytes.fromhex("8b118bc225000000c03d00000040752881")
        if self._read(base + 0x2D47E0, len(signature)) != signature:
            raise ValueError("pinned MwId lookup changed")
        index = value & 0x3FFFFFFF
        table = self._q(base + 0x1FFB1B0 + (index >> 16) * 16)
        offset = unpack("<I", self._read(table + (index & 65535) * 4, 4))[0]
        pool = self._q(base + 0x1FFB1A0)
        raw = self._read(pool + offset, 64)
        end = raw.find(b"\0")
        if not 0 < end < 64:
            raise ValueError("bounded literal map identifier required")
        return raw[:end].decode("ascii")

    def _map_uid_source(self, map_info: int) -> None:
        # Native GetParam member03044000 selects this+54, then MwId.GetName.
        # Actual source258 and current read266 establish the semantic binding.
        base = self.modules.game_base
        if (
            self._q(self._q(map_info) + 9 * 8) != base + 0xC456C0
            or self._read(base + 0xC45714, 11) != bytes.fromhex("2d004004030f840b010000")
            or self._read(base + 0xC4584E, 12) != bytes.fromhex("488d4f544803f0e886ef68ff")
        ):
            raise ValueError("pinned native MapUid property route changed")

    def position(self, snapshot: WindowsTerminalSnapshot) -> tuple[float, float, float]:
        """Read native Position only while its actual availability predicate holds.

        The engine's unavailable branch returns a default vector. That branch is
        deliberately refused here; a default cannot bind the GUI to a physics body.
        """
        base = self.modules.game_base
        player = snapshot.player
        if (
            self._q(self._q(snapshot.script_api) + 9 * 8) != base + 0x1353C70
            or self._read(base + 0x1354F50 + 57 * 4, 4) != bytes.fromhex("cc3f3501")
            or self._read(base + 0x1354001, 13) != bytes.fromhex("498b5350488d4dd0e8e2f6ffff")
            or self._read(base + 0x1353700, 12) != bytes.fromhex("488bcae808a6f8ff85c07510")
            or self._read(base + 0x135371C, 14) != bytes.fromhex("f20f1087240e00008b872c0e0000")
            or self._read(base + 0x12DDD10, 13) != bytes.fromhex("33c083b91c020000ff0f95c0c3")
        ):
            raise ValueError("pinned native Position property route changed")
        identity = (
            "app",
            "playground",
            "terminal",
            "player",
            "script_api",
            "root_map",
            "map_info",
            "map_identifier",
        )
        # Live read292 returned zero xyz; its next read had actual coordinates.
        # The precise producer phase is unproven. A finite first read alone
        # is insufficient. Retry only a changed vector, before publishing any
        # sample/input; missing availability, identity and finite guards stay fatal.
        for _ in range(3):
            available = self._read(player + 0x21C, 4)
            if available == b"\xff" * 4:
                raise ValueError("native GUI Position is unavailable")
            raw = self._read(player + 0xE24, 12)
            values = unpack("<3f", raw)
            current = self.sample()
            if (
                any(getattr(current, key) != getattr(snapshot, key) for key in identity)
                or self._read(player + 0x21C, 4) != available
                or not all(isfinite(value) for value in values)
            ):
                raise ValueError("current GUI position binding changed or is nonfinite")
            if self._read(player + 0xE24, 12) == raw:
                return values
        raise ValueError("GUI Position did not produce a stable native sample")

    def _physics_once(
        self, snapshot: WindowsTerminalSnapshot, *, tolerance: float = 0.2
    ) -> WindowsPhysicsSample:
        """Bind the current player's physics reference, never a previous body's address.

        Pinned CSmPlayer constructor12dd220 installs vtable1ceb998; the arena's
        reference resolution12f2244 writes the whole sixteen-byte reference at
        player+1110. This is a physics reference, not a ScriptAPI vehicle wrapper.
        The body's literal header must match that reference including generation.
        """
        if not isfinite(tolerance) or not 0 < tolerance <= 2.0:
            raise ValueError("existing comparison tolerance must remain at most two units")
        base = self.modules.game_base
        if (
            self._q(snapshot.player) != base + 0x1CEB998
            or self._read(base + 0x12DD230, 10) != bytes.fromhex("488d0561e7a000488903")
            or self._read(base + 0x12F2244, 7) != bytes.fromhex("0f11b710110000")
        ):
            raise ValueError("pinned current CSmPlayer physics reference route changed")
        gui_position = self.position(snapshot)
        reference = self._read(snapshot.player + 0x1110, 16)
        kind, generation, vehicle = unpack("<IIQ", reference)
        if kind == 0 and generation in (0, 0xFFFFFFFF) and vehicle == 0:
            # Native constructors/reset12f2140 explicitly clear this reference
            # before its arena entity resolver writes the new complete value.
            raise WindowsPhysicsUnavailableError(
                f"current player physics reference is unavailable: {reference.hex()}"
            )
        if kind != 0x0A020000 or generation == 0xFFFFFFFF:
            raise ValueError("current player physics reference is unavailable")
        if unpack("<I", self._read(vehicle + 0x48, 4))[0] != 1:
            raise ValueError("one current reverse-bound physics body is required")
        body = self._q(vehicle + 0x50)
        if self._read(body, 16) != reference:
            raise ValueError("current player/body physics reference does not match")
        model = self._q(vehicle + 0x88)
        if unpack("<I", self._read(model + 0x1790, 4))[0] != 5:
            raise ValueError("current player physics model must be the qualified car")
        physical = self._read(vehicle + 0x12E0, 40)
        position = unpack_from("<3f", physical, 16)
        error = max(abs(a - b) for a, b in zip(position, gui_position, strict=True))
        if not all(isfinite(value) for value in position):
            raise ValueError("current physics position is nonfinite")
        if error > tolerance:
            raise ValueError(
                f"current GUI/body position mismatch: GUI={gui_position}, "
                f"physics={position}, error={error}, tolerance={tolerance}"
            )
        if self._read(snapshot.player + 0x1110, 16) != reference:
            raise WindowsPhysicsSampleChangedError(
                "current player physics reference changed during sample"
            )
        if self._read(body, 16) != reference:
            raise WindowsPhysicsSampleChangedError(
                "current body physics reference changed during sample"
            )
        if self._q(vehicle + 0x50) != body:
            raise WindowsPhysicsSampleChangedError(
                "current vehicle/body reverse binding changed during sample"
            )
        if self._q(vehicle + 0x88) != model:
            raise WindowsPhysicsSampleChangedError(
                "current vehicle/model binding changed during sample"
            )
        return WindowsPhysicsSample(
            body, vehicle, model, reference.hex(), physical.hex(), gui_position, position, error
        )

    def physics(
        self, snapshot: WindowsTerminalSnapshot, *, tolerance: float = 0.2
    ) -> WindowsPhysicsSample:
        """Publish only a coherent current binding, with at most three fresh reads.

        Actual324 caught a reference replacement after all initial type/body/
        position checks had passed. Only that class of changed read retries.
        Unavailable pointers, bad identities/models and position mismatch remain
        fatal. Each attempt starts again from the actual player and source roots.
        """
        for attempt in range(3):
            try:
                return self._physics_once(snapshot, tolerance=tolerance)
            except WindowsPhysicsSampleChangedError:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable bounded physics read")

    def wait_physics(  # noqa: PLR0913 - Explicit guard and unchanged position/deadline bounds.
        self,
        snapshot: WindowsTerminalSnapshot,
        *,
        guard: Callable[[], object],
        tolerance: float = 0.2,
        timeout_s: float = 1.0,
    ) -> WindowsPhysicsSample:
        """Wait for an actual complete reference, never reuse an old sample.

        Only the exact source-defined empty reference and a changed read wait.
        The full map/player/body/model/position checks run again before return.
        Other errors are fatal; the caller checks focus/birth/UI on every attempt.
        """
        if not isfinite(timeout_s) or not 0 < timeout_s <= 1.0:
            raise ValueError("comparison physics wait must be bounded by one second")
        deadline = monotonic() + timeout_s
        while True:
            guard()
            try:
                return self.physics(snapshot, tolerance=tolerance)
            except (WindowsPhysicsUnavailableError, WindowsPhysicsSampleChangedError):
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "complete current player physics reference did not arrive"
                    ) from None
                sleep(0.001)

    def game_time(self, snapshot: WindowsTerminalSnapshot) -> int:
        """Read the actual client API GameTime counter, not the temporary script context.

        This is the Connect plugin's fallback counter. It does not by itself
        qualify a rules.Now clock, a native episode clock or any terminal event.
        """
        base = self.modules.game_base
        self._property(0x1ED0A08, "Network", 0x220)
        self._property(0x1ED0818, "PlaygroundClientScriptAPI", 0x3308)
        self._property(0x1F22A10, "GameTime", 0xFFFFFFFF)
        network = self._q(snapshot.app + 0x220)
        self._class(network, "CTrackManiaNetwork", 0x2402F000)
        api = self._q(network + 0x3308)
        self._class(api, "CGamePlaygroundClientScriptAPI", 0x03187000)
        if (
            self._q(self._q(api) + 9 * 8) != base + 0xFCC9C0
            or self._read(base + 0xFCCD70 + 4, 4) != bytes.fromhex("a8cafc00")
            or self._read(base + 0xFCCAA8, 12) != bytes.fromhex("488d0dd1e20201e81c6530ff")
            or self._read(base + 0x2D2FD0, 4) != bytes.fromhex("8b4118c3")
        ):
            raise ValueError("pinned client GameTime source route changed")
        value = unpack("<i", self._read(base + 0x1FFAD98, 4))[0]
        current = self.sample()
        if (
            value < 0
            or current.app != snapshot.app
            or current.player != snapshot.player
            or current.map_identifier != snapshot.map_identifier
            or self._q(snapshot.app + 0x220) != network
            or self._q(network + 0x3308) != api
        ):
            raise ValueError("current client GameTime binding changed or is unavailable")
        return value

    def sample(self) -> WindowsTerminalSnapshot:
        dll = self.modules.openplanet_base
        if self._read(dll + 0xBEB20, 8) != bytes.fromhex("488b05b1da8f00c3"):
            raise ValueError("pinned Openplanet GetApp getter changed")
        self._property(0x1ED0C98, "CurrentPlayground", 0x3D8)
        self._property(0x1ED0AE0, "RootMap", 0x360)
        self._property(0x1ED1690, "MapInfo", 0x310)
        app = self._q(dll + 0x9BC5D8)
        self._class(app, "CTrackMania", 0x24001000)
        playground = self._q(app + 0x3D8)
        self._class(playground, "CSmArenaClient", 0x2D004000)
        root_map = self._q(app + 0x360)
        self._class(root_map, "CGameCtnChallenge", 0x03043000)
        map_info = self._q(root_map + 0x310)
        self._class(map_info, "CGameCtnChallengeInfo", 0x03044000)
        self._map_uid_source(map_info)
        map_identifier = unpack("<I", self._read(map_info + 0x54, 4))[0]
        name = self._identifier(map_identifier)
        buffer = self._read(playground + 0x9F8, 16)
        pointer, count, capacity = unpack("<QII", buffer)
        if not pointer or count != 1 or not count <= capacity <= 8:
            raise ValueError("one actual current client terminal is required")
        terminal = self._q(pointer)
        self._class(terminal, "CGameTerminal", 0x03003000)
        player_pair = self._read(terminal + 0x18, 16)
        controlled, player = unpack("<2Q", player_pair)
        if controlled != player:
            raise ValueError("GUI and controlled player must be the same actual player")
        self._class(player, "CSmPlayer", 0x2D008000)
        script = self._q(player + 0x180)
        self._class(script, "CSmScriptPlayer", 0x2D02E000)
        if self._q(script + 0x50) != player:
            raise ValueError("ScriptAPI must point back to the current GUI player")
        ui = self._read(terminal + 0x250, 9)
        sequence, changed, number = unpack("<IIB", ui)
        if sequence > 11:
            raise ValueError("unknown current UI sequence")
        if (
            self._q(dll + 0x9BC5D8) != app
            or self._q(app + 0x3D8) != playground
            or self._q(app + 0x360) != root_map
            or self._read(playground + 0x9F8, 16) != buffer
            or self._q(pointer) != terminal
            or self._read(terminal + 0x18, 16) != player_pair
            or self._q(player + 0x180) != script
            or self._q(script + 0x50) != player
            or self._q(root_map + 0x310) != map_info
            or unpack("<I", self._read(map_info + 0x54, 4))[0] != map_identifier
        ):
            raise ValueError("current client root/player chain changed during sample")
        return WindowsTerminalSnapshot(
            app,
            playground,
            terminal,
            player,
            script,
            root_map,
            map_info,
            map_identifier,
            name,
            sequence,
            changed,
            number,
        )
