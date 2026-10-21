"""Synthetic contract fixtures only; none of these bytes is a physics experience."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime import create_backend, env, features, outcome
from trackmaniarl.trackmania.native_headless._runtime.artifacts import PINS
from trackmaniarl.trackmania.native_headless._runtime.shapes import StateSpec, VerifyCreditSpec
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


def _reject(call: Callable[[], object]) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected explicit rejection")


def _post(clock: int, channels: tuple[float, float, float]) -> bytes:
    """A protocol fixture, not a model of car dynamics."""
    raw = bytearray(256)
    raw[:8] = b"TMPOST03"
    for offset, value in {
        8: 256,
        12: 3,
        16: 42,
        20: 42,
        112: clock,
        116: clock,
        144: clock,
        136: 10,
        140: 10,
        148: 75,
        212: 10,
    }.items():
        struct.pack_into("<I", raw, offset, value)
    struct.pack_into("<Q", raw, 40, 100)
    struct.pack_into("<Q", raw, 48, 200)
    struct.pack_into("<4f", raw, 88, channels[1], channels[2], channels[0], 0.0)
    struct.pack_into(
        "<10ffIfII",
        raw,
        152,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1000.0,
        1,
        1000.0,
        1,
        0,
    )
    return bytes(raw)


def test_runtime_construct_and_close_are_inert() -> None:
    root = Path.cwd() / "nonexistent-runtime-fixture"
    config = NativeRuntimeConfig(
        runtime_root=root,
        trace_dir=root / "trace",
        work_dir=root / "work",
        asset_manifest=root / "manifest",
    )
    backend = create_backend(config)
    assert backend.capabilities["actionCount"] == 78
    assert backend.capabilities["clientParity"] is False
    _reject(backend.reset)
    _reject(backend.read_state)
    backend.close()
    backend.close()
    _reject(backend.reset)


def test_actions_and_delayed_pending_credit() -> None:
    for action in range(78):
        plan = features.action_sgb(action)
        assert len(plan) == 5
        assert all(-1 <= channels[0] <= 1 for channels in plan)
    _reject(lambda: features.action_sgb(True))
    _reject(lambda: features.action_sgb(78))
    plan = features.action_sgb(41)  # steer0, gas1, brake tap: first future only.
    assert plan[0] == (0.0, 1.0, 1.0)
    assert plan[1:] == [(0.0, 1.0, 0.0)] * 4
    pending = [0.0, 1.0, 0.0]
    applied = [tuple(pending), *plan[:4]]
    for count in range(1, 6):
        events = [_post(2230 + 10 * (index + 1), applied[index]) for index in range(count)]
        credit = features.verify_credit(VerifyCreditSpec(events, 2230, plan, pending))
        assert credit["intervals"] == count
        assert credit["nativeDelta"] == count * 10
        assert credit["pendingNotCredited"] is True
        corrupted = [*events]
        corrupted[0] = _post(2240, plan[0])
        _reject(
            lambda corrupted=corrupted: features.verify_credit(
                VerifyCreditSpec(corrupted, 2230, plan, pending)
            )
        )


def test_terminal_state_never_encodes_missing_pending() -> None:
    backend = env.Native16Env(lambda: None, root=Path("/unused-native-fixture"))
    backend.core = features
    backend.reference = features.Reference(
        [(float(i), 0.0, 0.0) for i in range(100)], {"sha256": "test-only"}
    )
    packet = {
        "profile": env.RUNTIME_PROFILE,
        "pid": 42,
        "meta": {"controller": 100, "sim": 200},
        "boundaryHex": _post(2230, (0.0, 1.0, 0.0)).hex(),
    }
    state = backend._state(StateSpec(packet, [41], 0, True))
    assert len(bytes.fromhex(state["native76Hex"])) == 76
    assert state["features"] is None
    assert state["pendingSGB"] is None
    assert state["readerHex"] is None
    assert state["pendingOwnP"] is None
    assert state["pendingProvenance"] is None
    assert state["historyActionIds"] == [41]


def test_full_result_qualifier_keeps_isvalid_separate_from_finish() -> None:
    value = {"Time": 17715, "Score": 0, "NbRespawns": 0, "NbCheckpoints": 1}
    validator = {
        "FileName": "g.Ghost.Gbx",
        "MapUid": outcome.MAP_UID,
        "IsValid": False,
        "ValidatedResult": value,
        "DeclaredResult": dict(value, Time=40000),
        "Desc": "synthetic fixture",
    }
    result = bytearray(768)
    result[:8] = b"TMRESULT"
    for offset, number in {
        8: 768,
        12: 1,
        16: 42,
        20: 42,
        148: 24575,
        240: 13,
        248: 1,
        128: 2230,
        152: 17715,
        156: 0,
        160: 0,
        184: 1,
        192: 8,
        256: 0,
        260: 17715,
    }.items():
        struct.pack_into("<I", result, offset, number)
    struct.pack_into("<Q", result, 24, 1)
    struct.pack_into("<Q", result, 40, 100)
    struct.pack_into("<Q", result, 48, 200)
    evidence = {
        "profile": env.RUNTIME_PROFILE,
        "pins": PINS,
        "reaped": {
            "pid": 42,
            "kind": "reaped",
            "process_gone": True,
            "parent_unchanged": True,
            "exit_status": None,
            "truncated_by_driver": False,
        },
        "validatorArrayHex": json.dumps([validator]).encode().hex(),
        "collectorHex": result.hex(),
    }
    packet = {
        "profile": env.RUNTIME_PROFILE,
        "resultEvidence": evidence,
        "pid": 42,
        "meta": {"controller": 100, "sim": 200},
        "boundaryHex": _post(2230, (0.0, 1.0, 0.0)).hex(),
    }
    qualified = outcome.classify(packet, env, Path("/unused"))
    assert qualified["terminated"] is True
    assert qualified["validationAcceptance"] is False
    assert qualified["physicalFinishNativeT"] is None
    evidence["collectorHex"] = result[:-1].hex()
    rejected = outcome.classify(packet, env, Path("/unused"))
    assert rejected["terminated"] is None
    assert rejected["status"] == "inconclusive"
