"""Synthetic Windows memory graph; missing roots never become Finish=False."""

import json
from dataclasses import asdict
from hashlib import sha256
from struct import pack, pack_into, unpack
from typing import Any, ClassVar

import pytest

from experiments.full_qr.windows_terminal import WindowsModuleContext, WindowsTerminalReader


class SyntheticTerminalReader(WindowsTerminalReader):
    """Exercise real control flow against artificial signatures and addresses."""

    _CLASS_GETTERS: ClassVar[dict[str, bool]] = {
        sha256(b"\x11\x22\x33").hexdigest(): False,
        sha256(b"\x44\x55\x66").hexdigest(): True,
    }
    _CLASS_GETTER_RET = 0xFE

    def _signature(self, address: int, size: int, expected_sha256: str) -> bool:
        return super()._signature(address, size, sha256(b"\xa5" * size).hexdigest())


def fixture() -> tuple[dict[str, Any], dict[tuple[int, int], bytes]]:
    """Construct a fictitious graph; no process-memory capture is included.

    Signature checks use fictitious markers through a test-only reader.
    """
    base, dll = 0x10000000, 0x30000000
    app, playground, terminal, player, script, root_map, map_info = (
        0x50000000 + index * 0x10000 for index in range(1, 8)
    )
    network, api, vehicle, body, model, arena, pool = (
        0x50000000 + index * 0x10000 for index in range(8, 15)
    )
    scene, manager, objects, vis, state, flags, keys, values = (
        0x50000000 + index * 0x10000 for index in range(15, 23)
    )
    reads: dict[tuple[int, int], bytes] = {}

    def put(address: int, value: bytes) -> None:
        reads[address, len(value)] = value

    def q(address: int, value: int) -> None:
        put(address, pack("<Q", value))

    next_ptr = 0x60000000

    def alloc() -> int:
        nonlocal next_ptr
        result = next_ptr
        next_ptr += 0x1000
        return result

    def class_object(address: int, class_name: str, ident: int) -> int:
        vtable, getter, desc, text = alloc(), alloc(), alloc(), alloc()
        q(address, vtable)
        q(vtable + 16, getter)
        put(getter, b"\x11\x22\x33" + pack("<i", desc - getter - 7) + b"\xfe")
        descriptor = bytearray(32)
        pack_into("<Q", descriptor, 8, text)
        pack_into("<I", descriptor, 24, ident)
        put(desc, bytes(descriptor))
        put(text, class_name.encode("ascii") + b"\0")
        return vtable

    def prop(rva: int, prop_name: str, offset: int) -> None:
        text = alloc()
        record = bytearray(0x48)
        pack_into("<I", record, 0x18, offset)
        pack_into("<Q", record, 0x20, text)
        put(base + rva, bytes(record))
        put(text, prop_name.encode("ascii") + b"\0")

    for addr, class_name, ident in (
        (app, "CTrackMania", 0x24001000),
        (playground, "CSmArenaClient", 0x2D004000),
        (root_map, "CGameCtnChallenge", 0x03043000),
        (map_info, "CGameCtnChallengeInfo", 0x03044000),
        (terminal, "CGameTerminal", 0x03003000),
        (script, "CSmScriptPlayer", 0x2D02E000),
        (network, "CTrackManiaNetwork", 0x2402F000),
        (api, "CGamePlaygroundClientScriptAPI", 0x03187000),
        (arena, "CSmArena", 0x2D019000),
    ):
        vtable = class_object(addr, class_name, ident)
        if addr == map_info:
            q(vtable + 9 * 8, base + 0xC456C0)
        if addr == script:
            q(vtable + 9 * 8, base + 0x1353C70)
        if addr == api:
            q(vtable + 9 * 8, base + 0xFCC9C0)
    # This identity must also satisfy the pinned physics vtable route.
    player_vtable = base + 0x1CEB998
    q(player, player_vtable)
    getter, desc, text = alloc(), alloc(), alloc()
    q(player_vtable + 16, getter)
    put(getter, b"\x11\x22\x33" + pack("<i", desc - getter - 7) + b"\xfe")
    descriptor = bytearray(32)
    pack_into("<Q", descriptor, 8, text)
    pack_into("<I", descriptor, 24, 0x2D008000)
    put(desc, bytes(descriptor))
    put(text, b"CSmPlayer\0")
    q(player_vtable + 0x108, base + 0x12DD7B0)

    for rva, pname, offset in (
        (0x1ED0C98, "CurrentPlayground", 0x3D8),
        (0x1ED0AE0, "RootMap", 0x360),
        (0x1ED1690, "MapInfo", 0x310),
        (0x1ED0A08, "Network", 0x220),
        (0x1ED0818, "PlaygroundClientScriptAPI", 0x3308),
        (0x1F22A10, "GameTime", 0xFFFFFFFF),
        (0x1F55170, "Arena", 0xD78),
        (0x1ED0BC0, "GameScene", 0x2A0),
    ):
        prop(rva, pname, offset)

    q(dll + 0x9BC5D8, app)
    q(app + 0x3D8, playground)
    q(app + 0x360, root_map)
    q(root_map + 0x310, map_info)
    put(map_info + 0x54, pack("<I", 0x40000001))
    table, text_pool = alloc(), alloc()
    q(base + 0x1FFB1B0, table)
    put(table + 4, pack("<I", 0))
    q(base + 0x1FFB1A0, text_pool)
    put(text_pool, b"synthetic-map\0" + bytes(50))
    terminal_array = alloc()
    put(playground + 0x9F8, pack("<QII", terminal_array, 1, 1))
    q(terminal_array, terminal)
    put(terminal + 0x18, pack("<2Q", player, player))
    q(player + 0x180, script)
    q(script + 0x50, player)
    put(terminal + 0x250, pack("<IIB", 1, 123, 2))
    q(app + 0x220, network)
    q(network + 0x3308, api)
    put(base + 0x1FFAD98, pack("<i", 2000))
    put(player + 0x21C, pack("<I", 0))
    position = (1.0, 2.0, 3.0)
    cache = bytearray(0xFC)
    pack_into("<3f", cache, 0, *position)
    put(player + 0xE24, bytes(cache))
    put(player + 0xE24, pack("<3f", *position))
    ref = pack("<IIQ", 0x0A020000, 7, vehicle)
    put(player + 0x1110, ref)
    put(vehicle + 0x48, pack("<I", 1))
    q(vehicle + 0x50, body)
    put(body, ref)
    q(vehicle + 0x88, model)
    put(model + 0x1790, pack("<I", 5))
    physical = bytearray(40)
    pack_into("<3f", physical, 16, *position)
    put(vehicle + 0x12E0, bytes(physical))

    # Synthetic inputs for the comparison graph and adherence-table joins.
    put(player + 0x218, pack("<i", 1500))
    entity = 0x02000001
    entity_fields = bytearray(0x34)
    pack_into("<I", entity_fields, 0x28C - 0x278, entity)
    put(player + 0x278, bytes(entity_fields))
    q(player + 0xC60, 0)
    q(app + 0x2A0, scene)
    put(scene + 8, pack("<I", 14))
    q(scene + 0x10 + 13 * 8, manager)
    put(manager + 0x210, pack("<QII", objects, 1, 1))
    q(objects, vis)
    put(vis, pack("<I", entity))
    q(vis + 0x130, state)
    visual = bytearray(864)
    pack_into("<3f", visual, 0x50, *position)
    put(state, bytes(visual))
    q(playground + 0xD78, arena)
    q(arena + 0x7C8, pool)
    header = bytearray(0x38)
    pack_into("<II", header, 4, 4, 1)
    pack_into("<3Q", header, 0x20, flags, keys, values)
    put(pool + 0x8E0, bytes(header))
    put(flags, pack("<I", 0))
    put(keys + 4, pack("<I", entity))
    put(values + 16, ref)
    put(vehicle + 0x1760, pack("<f", 0.75))
    q(base + 0x1FBCE20, 0)
    _add_signature_sentinels(reads, base, dll)

    snapshot = {
        "app": app,
        "playground": playground,
        "terminal": terminal,
        "player": player,
        "script_api": script,
        "root_map": root_map,
        "map_info": map_info,
        "map_identifier": 0x40000001,
        "resolved_map_identifier": "synthetic-map",
        "ui_sequence": 1,
        "ui_change_human_ms": 123,
        "ui_sequence_number": 2,
    }
    physics = {
        "body": body,
        "vehicle": vehicle,
        "model": model,
        "reference_hex": ref.hex(),
        "physical40_hex": bytes(physical).hex(),
        "gui_position": list(position),
        "physical_position": list(position),
        "position_error": 0.0,
    }
    data = {
        "modules": {"trackmania.exe": {"base": base}, "openplanet.dll": {"base": dll}},
        "snapshot": snapshot,
        "position": position,
        "physics": physics,
        "game_time_ms": 2000,
        "observation": {
            "terminal": snapshot,
            "physics": physics,
            "entity_id": entity,
            "vis": vis,
            "state": state,
            "game_time_ms": 2000,
            "start_time_ms": 1500,
            "channels": [500.0, *position, *([0.0] * 15), 0.75, 0.0, 0.0, 0.0],
            "scalar_cache_hex": bytes(cache).hex(),
            "visual_state_hex": bytes(visual).hex(),
            "adherence_owner": arena,
        },
    }
    return data, reads


def _add_signature_sentinels(reads: dict[tuple[int, int], bytes], base: int, dll: int) -> None:
    specs = (
        (dll + 0xBEB20, 8),
        (base + 0x2D47E0, 17),
        (base + 0xC45714, 11),
        (base + 0xC4584E, 12),
        (base + 0x1354F50 + 57 * 4, 4),
        (base + 0x1354001, 13),
        (base + 0x1353700, 12),
        (base + 0x135371C, 14),
        (base + 0x12DDD10, 13),
        (base + 0x12DD230, 10),
        (base + 0x12F2244, 7),
        (base + 0xFCCD70 + 4, 4),
        (base + 0xFCCAA8, 12),
        (base + 0x2D2FD0, 4),
        (base + 0x12DD880, 21),
        (base + 0x7276D0, 26),
        (base + 0x6D2500, 12),
    )
    for address, size in specs:
        reads[address, size] = b"\xa5" * size


def reader(data: dict[str, Any], reads: dict[tuple[int, int], bytes]) -> WindowsTerminalReader:
    return SyntheticTerminalReader(
        lambda address, size: reads[address, size],
        WindowsModuleContext(
            data["modules"]["trackmania.exe"]["base"],
            data["modules"]["openplanet.dll"]["base"],
        ),
    )


def test_signature_digest_checks_exact_bytes_and_length() -> None:
    data, reads = fixture()
    instance = reader(data, reads)
    address = data["modules"]["openplanet.dll"]["base"] + 0xBEB20
    digest = sha256(b"\xa5" * 8).hexdigest()
    assert WindowsTerminalReader._signature(instance, address, 8, digest)
    assert not WindowsTerminalReader._signature(instance, address, 8, sha256(b"x" * 8).hexdigest())
    reads[address, 8] = b"\xa5" * 7
    with pytest.raises(ValueError, match="complete literal"):
        WindowsTerminalReader._signature(instance, address, 8, digest)


def test_synthetic_class_getter_supports_indirect_descriptor() -> None:
    data, reads = fixture()
    instance = reader(data, reads)
    terminal = data["snapshot"]["terminal"]
    vtable = unpack("<Q", reads[terminal, 8])[0]
    getter = unpack("<Q", reads[vtable + 16, 8])[0]
    original = reads[getter, 8]
    descriptor = getter + 7 + unpack("<i", original[3:7])[0]
    pointer = 0x6F000000
    reads[pointer, 8] = pack("<Q", descriptor)
    reads[getter, 8] = b"\x44\x55\x66" + pack("<i", pointer - getter - 7) + b"\xfe"
    instance.sample()


def test_synthetic_class_getter_rejects_unknown_prefix() -> None:
    data, reads = fixture()
    instance = reader(data, reads)
    terminal = data["snapshot"]["terminal"]
    vtable = unpack("<Q", reads[terminal, 8])[0]
    getter = unpack("<Q", reads[vtable + 16, 8])[0]
    original = reads[getter, 8]
    reads[getter, 8] = b"\x10" + original[1:]
    with pytest.raises(ValueError, match="GetClassInfo getter"):
        instance.sample()


def test_synthetic_client_playing_sample_is_decoded() -> None:
    data, reads = fixture()
    sample = reader(data, reads).sample()
    assert asdict(sample) == data["snapshot"]
    assert sample.finish is False  # The synthetic Playing state is not a missing-data fallback.


def test_missing_client_playground_is_unavailable_not_false_finish() -> None:
    data, reads = fixture()
    reads[data["snapshot"]["app"] + 0x3D8, 8] = b"\0" * 8
    with pytest.raises(ValueError, match="bounded canonical"):
        reader(data, reads).sample()


def test_spectator_player_cannot_be_mistaken_for_controlled_player() -> None:
    data, reads = fixture()
    player = data["snapshot"]["player"]
    reads[data["snapshot"]["terminal"] + 0x18, 16] = pack("<2Q", player + 8, player)
    with pytest.raises(ValueError, match="GUI and controlled"):
        reader(data, reads).sample()


def test_unknown_ui_value_remains_an_explicit_error() -> None:
    data, reads = fixture()
    address = data["snapshot"]["terminal"] + 0x250
    reads[address, 9] = pack("<I", 12) + reads[address, 9][4:]
    with pytest.raises(ValueError, match="unknown current UI"):
        reader(data, reads).sample()


def test_changed_getapp_code_is_refused_before_following_a_pointer() -> None:
    data, reads = fixture()
    reads[data["modules"]["openplanet.dll"]["base"] + 0xBEB20, 8] = b"\0" * 8
    with pytest.raises(ValueError, match="GetApp getter"):
        reader(data, reads).sample()


def test_foreign_class_descriptor_is_not_accepted_by_pointer_shape() -> None:
    data, reads = fixture()
    name = b"CGameTerminal\0"
    key = next(key for key, value in reads.items() if value == name)
    reads[key] = b"X" + name[1:]
    with pytest.raises(ValueError, match="not CGameTerminal"):
        reader(data, reads).sample()


def test_synthetic_position_matches_generated_fixture() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    assert native.position(native.sample()) == tuple(data["position"])


def test_map_uid_semantic_route_change_is_refused() -> None:
    data, reads = fixture()
    reads[data["modules"]["trackmania.exe"]["base"] + 0xC4584E, 12] = b"\0" * 12
    with pytest.raises(ValueError, match="MapUid property route"):
        reader(data, reads).sample()


def test_unavailable_position_never_returns_native_default_vector() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x21C, 4] = b"\xff" * 4
    with pytest.raises(ValueError, match="Position is unavailable"):
        native.position(snapshot)


def test_position_getter_change_is_refused() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x135371C, 14] = b"\0" * 14
    with pytest.raises(ValueError, match="Position property route"):
        native.position(snapshot)


def test_nonfinite_gui_position_cannot_bind_a_physics_body() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0xE24, 12] = pack("<3f", float("nan"), 0, 0)
    with pytest.raises(ValueError, match="nonfinite"):
        native.position(snapshot)


def test_synthetic_current_player_reference_selects_the_physics_body() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    actual = native.physics(native.sample())
    assert json.loads(json.dumps(asdict(actual))) == data["physics"]


def test_missing_current_physics_reference_never_reuses_an_old_body() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = b"\0" * 16
    with pytest.raises(ValueError, match="physics reference is unavailable"):
        native.physics(snapshot)


def test_recycled_body_generation_must_match_the_current_player_reference() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    body = data["physics"]["body"]
    previous = reads[body, 16]
    reads[body, 16] = previous[:4] + pack("<I", 99) + previous[8:]
    with pytest.raises(ValueError, match="reference does not match"):
        native.physics(snapshot)


def test_changed_physics_reference_route_is_refused() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x12F2244, 7] = b"\0" * 7
    with pytest.raises(ValueError, match="physics reference route changed"):
        native.physics(snapshot)


def test_current_car_model_kind_remains_required() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["physics"]["model"] + 0x1790, 4] = pack("<I", 4)
    with pytest.raises(ValueError, match="qualified car"):
        native.physics(snapshot)


def test_physics_reference_change_during_read_is_not_published() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    original = native.read
    visits = 0

    def changing_reference(address: int, size: int) -> bytes:
        nonlocal visits
        if address == snapshot.player + 0x1110 and size == 16:
            visits += 1
            if visits % 2 == 0:
                return b"\0" * 16
        return original(address, size)

    native.read = changing_reference
    with pytest.raises(ValueError, match="physics reference changed during sample"):
        native.physics(snapshot)
    assert visits == 6


def test_replaced_physics_read_is_discarded_before_a_fresh_complete_snapshot() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    original = native.read
    visits = 0

    def transient_reference(address: int, size: int) -> bytes:
        nonlocal visits
        if address == snapshot.player + 0x1110 and size == 16:
            visits += 1
            if visits == 2:
                return b"\0" * 16
        return original(address, size)

    native.read = transient_reference
    actual = native.physics(snapshot)
    assert actual.reference_hex == data["physics"]["reference_hex"]
    assert visits == 4


def test_comparison_position_tolerance_cannot_be_relaxed() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    with pytest.raises(ValueError, match="at most two units"):
        native.physics(native.sample(), tolerance=2.01)


def test_empty_current_reference_waits_for_a_fresh_complete_sample() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    original = native.read
    visits = 0
    guards = []

    def unavailable_once(address: int, size: int) -> bytes:
        nonlocal visits
        if address == snapshot.player + 0x1110 and size == 16:
            visits += 1
            if visits == 1:
                return b"\0" * 16
        return original(address, size)

    native.read = unavailable_once
    actual = native.wait_physics(snapshot, guard=lambda: guards.append(True))
    assert actual.reference_hex == data["physics"]["reference_hex"]
    assert visits == 3
    assert len(guards) == 2


def test_missing_reference_wait_has_a_deadline_and_publishes_nothing() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = b"\0" * 16
    with pytest.raises(TimeoutError, match="did not arrive"):
        native.wait_physics(snapshot, guard=lambda: None, timeout_s=0.001)


def test_wrong_reference_kind_is_fatal_without_waiting() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = pack("<IIQ", 0x032E2000, 1, 12345)
    guards = []
    with pytest.raises(ValueError, match="physics reference is unavailable"):
        native.wait_physics(snapshot, guard=lambda: guards.append(True))
    assert len(guards) == 1


def test_wait_guard_failure_stops_before_any_physics_read() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()

    def lost_focus() -> None:
        raise RuntimeError("focus lost")

    native.read = lambda address, size: pytest.fail("physics read after guard failure")
    with pytest.raises(RuntimeError, match="focus lost"):
        native.wait_physics(snapshot, guard=lost_focus)


def test_synthetic_client_game_time_uses_the_api_counter() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    assert native.game_time(native.sample()) == data["game_time_ms"]


def test_changed_client_game_time_getter_is_refused() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x2D2FD0, 4] = b"\0" * 4
    with pytest.raises(ValueError, match="GameTime source route changed"):
        native.game_time(snapshot)


def test_missing_network_api_never_uses_a_temporary_script_context_clock() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.app + 0x220, 8] = b"\0" * 8
    with pytest.raises(ValueError, match="bounded canonical"):
        native.game_time(snapshot)


def test_invalid_client_clock_sentinel_is_not_an_elapsed_race_time() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x1FFAD98, 4] = b"\xff" * 4
    with pytest.raises(ValueError, match="GameTime binding changed or is unavailable"):
        native.game_time(snapshot)


def test_transient_zero_position_is_not_published() -> None:
    # Controlled synthetic read race, not proof of production clock parity.
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    original = native.read
    visits = 0

    def transient_position(address: int, size: int) -> bytes:
        nonlocal visits
        if address == snapshot.player + 0xE24 and size == 12:
            visits += 1
            if visits == 1:
                return b"\0" * 12
        return original(address, size)

    native.read = transient_position
    assert native.position(snapshot) == tuple(data["position"])
    assert visits == 4


def test_continuously_changing_position_exhausts_bounded_read_attempts() -> None:
    data, reads = fixture()
    native = reader(data, reads)
    snapshot = native.sample()
    original = native.read
    visits = 0

    def changing_position(address: int, size: int) -> bytes:
        nonlocal visits
        if address == snapshot.player + 0xE24 and size == 12:
            visits += 1
            return pack("<3f", visits, 10, 624)
        return original(address, size)

    native.read = changing_position
    with pytest.raises(ValueError, match="stable native sample"):
        native.position(snapshot)
    assert visits == 6
