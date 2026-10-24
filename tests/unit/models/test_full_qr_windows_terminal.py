"""Literal Playing capture; malformed/missing roots never become Finish=False."""

import json
from dataclasses import asdict
from pathlib import Path
from struct import pack
from typing import Any

import pytest

from experiments.full_qr.windows_terminal import WindowsModuleContext, WindowsTerminalReader


def fixture(
    name: str = "windows_terminal_position267.json",
) -> tuple[dict[str, Any], dict[tuple[int, int], bytes]]:
    path = Path(__file__).parents[2] / "fixtures/full_qr" / name
    data = json.loads(path.read_text())
    reads = {(row["address"], row["bytes"]): bytes.fromhex(row["hex"]) for row in data["reads"]}
    return data, reads


def reader(data: dict[str, Any], reads: dict[tuple[int, int], bytes]) -> WindowsTerminalReader:
    return WindowsTerminalReader(
        lambda address, size: reads[address, size],
        WindowsModuleContext(
            data["modules"]["trackmania.exe"]["base"],
            data["modules"]["openplanet.dll"]["base"],
        ),
    )


def test_literal_actual_client_playing_sample_is_decoded() -> None:
    data, reads = fixture()
    sample = reader(data, reads).sample()
    assert asdict(sample) == data["snapshot"]
    assert sample.finish is False  # Actual captured Playing=1, never a missing-data fallback.


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


def test_actual_position_matches_literal_native_getter_capture() -> None:
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


def test_actual_current_player_reference_selects_the_literal_physics_body() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    actual = native.physics(native.sample())
    assert json.loads(json.dumps(asdict(actual))) == data["physics"]


def test_missing_current_physics_reference_never_reuses_an_old_body() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = b"\0" * 16
    with pytest.raises(ValueError, match="physics reference is unavailable"):
        native.physics(snapshot)


def test_recycled_body_generation_must_match_the_current_player_reference() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    body = data["physics"]["body"]
    previous = reads[body, 16]
    reads[body, 16] = previous[:4] + pack("<I", 99) + previous[8:]
    with pytest.raises(ValueError, match="reference does not match"):
        native.physics(snapshot)


def test_changed_physics_reference_route_is_refused() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x12F2244, 7] = b"\0" * 7
    with pytest.raises(ValueError, match="physics reference route changed"):
        native.physics(snapshot)


def test_current_car_model_kind_remains_required() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["physics"]["model"] + 0x1790, 4] = pack("<I", 4)
    with pytest.raises(ValueError, match="qualified car"):
        native.physics(snapshot)


def test_physics_reference_change_during_read_is_not_published() -> None:
    data, reads = fixture("windows_player_physics280.json")
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
    data, reads = fixture("windows_player_physics280.json")
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
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    with pytest.raises(ValueError, match="at most two units"):
        native.physics(native.sample(), tolerance=2.01)


def test_empty_current_reference_waits_for_a_fresh_complete_sample() -> None:
    data, reads = fixture("windows_player_physics280.json")
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
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = b"\0" * 16
    with pytest.raises(TimeoutError, match="did not arrive"):
        native.wait_physics(snapshot, guard=lambda: None, timeout_s=0.001)


def test_wrong_reference_kind_is_fatal_without_waiting() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.player + 0x1110, 16] = pack("<IIQ", 0x032E2000, 1, 12345)
    guards = []
    with pytest.raises(ValueError, match="physics reference is unavailable"):
        native.wait_physics(snapshot, guard=lambda: guards.append(True))
    assert len(guards) == 1


def test_wait_guard_failure_stops_before_any_physics_read() -> None:
    data, reads = fixture("windows_player_physics280.json")
    native = reader(data, reads)
    snapshot = native.sample()

    def lost_focus() -> None:
        raise RuntimeError("focus lost")

    native.read = lambda address, size: pytest.fail("physics read after guard failure")
    with pytest.raises(RuntimeError, match="focus lost"):
        native.wait_physics(snapshot, guard=lost_focus)


def test_literal_client_game_time_uses_the_actual_api_counter() -> None:
    data, reads = fixture("windows_client_clock290.json")
    native = reader(data, reads)
    assert native.game_time(native.sample()) == data["game_time_ms"]


def test_changed_client_game_time_getter_is_refused() -> None:
    data, reads = fixture("windows_client_clock290.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x2D2FD0, 4] = b"\0" * 4
    with pytest.raises(ValueError, match="GameTime source route changed"):
        native.game_time(snapshot)


def test_missing_network_api_never_uses_a_temporary_script_context_clock() -> None:
    data, reads = fixture("windows_client_clock290.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[snapshot.app + 0x220, 8] = b"\0" * 8
    with pytest.raises(ValueError, match="bounded canonical"):
        native.game_time(snapshot)


def test_invalid_client_clock_sentinel_is_not_an_elapsed_race_time() -> None:
    data, reads = fixture("windows_client_clock290.json")
    native = reader(data, reads)
    snapshot = native.sample()
    reads[data["modules"]["trackmania.exe"]["base"] + 0x1FFAD98, 4] = b"\xff" * 4
    with pytest.raises(ValueError, match="GameTime binding changed or is unavailable"):
        native.game_time(snapshot)


def test_transient_zero_position_observed_in_292_is_not_published() -> None:
    # Controlled read race using actual observed zero and actual capture267.
    # This is a reader regression, not proof of production event/clock parity.
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
