from __future__ import annotations

import io
import json
import time
from copy import deepcopy
from pathlib import Path
from struct import unpack_from
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.unit.trackmania._immediate_v7_synthetic import capture
from trackmaniarl.trackmania.native_headless import immediate_environment as immediate_module
from trackmaniarl.trackmania.native_headless._runtime import features
from trackmaniarl.trackmania.native_headless._runtime import immediate as immediate_runtime
from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession,
    physical_state,
    qualify_immediate,
)
from trackmaniarl.trackmania.native_headless.config import (
    NativeConfigurationError,
    NativeRuntimeConfig,
)
from trackmaniarl.trackmania.native_headless.environment import NativeHeadlessEnvironment
from trackmaniarl.trackmania.native_headless.immediate_environment import (
    NativeImmediateEnvironment,
    NativeImmediateEnvironmentFactory,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeExecutionDisabledError


def config(root: Path, **fields: Any) -> NativeRuntimeConfig:
    values: dict[str, Any] = {
        "runtime_root": root / "runtime",
        "trace_dir": root / "trace",
        "work_dir": root / "work",
        "asset_manifest": root / "assets.json",
        "protocol_profile": "immediate-current/v7",
        "reset_profile": "neutral-race-start/v7",
    }
    values.update(fields)
    return NativeRuntimeConfig(**values)


def test_profiles_require_explicit_compatible_selection(tmp_path: Path) -> None:
    immediate = config(tmp_path)
    with pytest.raises(ValueError, match="NativeImmediateEnvironment"):
        NativeHeadlessEnvironment(immediate)
    with pytest.raises(NativeConfigurationError, match="explicit neutral"):
        config(tmp_path, reset_profile="neutral-prelude/v1")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_trace_limits_rejected(tmp_path: Path, limit: Any) -> None:
    with pytest.raises(NativeConfigurationError, match="max_trace_bytes"):
        config(tmp_path, max_trace_bytes=limit)


def test_disabled_reset_does_not_read_assets(tmp_path: Path) -> None:
    immediate = NativeImmediateEnvironment(config(tmp_path))
    with pytest.raises(NativeExecutionDisabledError):
        immediate.reset()
    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize(
    "seed", [np.int32(-1), np.int64(12), np.random.default_rng(42).integers(100)]
)
def test_immediate_seed_accepts_numpy_integers(tmp_path: Path, seed: Any) -> None:
    settings = config(tmp_path)
    environment = NativeImmediateEnvironment(settings, seed=seed)
    assert type(environment.seed) is int
    assert environment.seed == int(seed)
    factory_environment = NativeImmediateEnvironmentFactory(settings).create(seed=seed)
    assert type(factory_environment.seed) is int
    assert factory_environment.seed == int(seed)


@pytest.mark.parametrize("seed", [True, False, np.bool_(True), 1.0, "1"])
def test_immediate_invalid_seed_is_rejected_before_resources(tmp_path: Path, seed: Any) -> None:
    settings = config(tmp_path, allow_native=True)
    with pytest.raises(ValueError, match="seed must be an integer"):
        NativeImmediateEnvironment(settings, seed=seed)
    with pytest.raises(ValueError, match="seed must be an integer"):
        NativeImmediateEnvironmentFactory(settings).create(seed=seed)
    environment = NativeImmediateEnvironment(settings, seed=2)
    with pytest.raises(ValueError, match="seed must be an integer"):
        environment.reset(seed=seed)
    assert environment.seed == 2
    assert environment._session is None
    assert not settings.trace_dir.exists()


def test_synthetic_n1_applies_current_action_without_pending_shift() -> None:
    before, _, packet = capture()
    posts = qualify_immediate(packet, before, 39, 1)
    assert len(posts) == 1
    state = physical_state(packet)
    assert state["appliedSGB"] == [0.0, 1.0, 0.0]
    assert state["pendingSGB"] is None
    assert state["features"] is None
    assert state["nativeT"] - before["nativeT"] == 10


def test_synthetic_n5_tap_has_exactly_one_applied_brake_interval() -> None:
    before, _, packet = capture(1)
    posts = qualify_immediate(packet, before, 2, 5)
    assert [
        physical_state({**packet, "boundaryHex": raw.hex()}, terminal=True)["appliedSGB"][2]
        for raw in posts
    ] == [1, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("index", "action", "expected"),
    [
        (0, 39, [(0.0, 1.0, 0.0)]),
        (1, 2, [(-1.0, 0.0, 1.0)] + [(-1.0, 0.0, 0.0)] * 4),
    ],
)
def test_synthetic_v7_actions_match_independent_wire_values(
    index: int, action: int, expected: list[tuple[float, float, float]]
) -> None:
    before, _, packet = capture(index)
    count = len(expected)
    # These expected controls are authored independently of the production
    # encoder and of the synthetic fixture generator.
    assert features.action_sgb(action)[:count] == expected
    posts = qualify_immediate(packet, before, action, count)
    receipt = bytes.fromhex(packet["authorization"]["receiptHex"])
    audits = packet["authorization"]["auditEventHex"]
    for number, (post, audit_hex, sgb) in enumerate(zip(posts, audits, expected, strict=True)):
        gas, brake, steer = unpack_from("<3f", post, 88)
        assert (steer, gas, brake) == sgb
        audit = bytes.fromhex(audit_hex)
        assert unpack_from("<3f", audit, 76) == sgb
        assert unpack_from("<3f", audit, 140) == expected[0]
        assert unpack_from("<3f", receipt, 28 + 24 * number) == sgb


def test_partial_eof_is_never_promoted_to_terminal() -> None:
    before, _, packet = capture()
    packet.update(
        eof=True,
        eofStage="boundary",
        receiveError={
            "reason": "EOFError: partial",
            "currentResponseBytes": 1,
            "currentResponseHex": "01",
        },
    )
    with pytest.raises(ValueError, match="zero-tail"):
        qualify_immediate(packet, before, 39, 1)


@pytest.mark.parametrize("corruption", ["action", "post", "audit", "receipt"])
def test_receipt_audit_post_mismatch_cannot_be_qualified(corruption: str) -> None:
    before, _, packet = capture()
    packet = deepcopy(packet)
    action = 36 if corruption == "action" else 39
    if corruption != "action":
        field, offset = {
            "post": ("nativeEventHex", 88),
            "audit": ("auditEventHex", 76),
            "receipt": ("receiptHex", 28),
        }[corruption]
        value = packet["authorization"][field]
        raw = bytearray.fromhex(value if isinstance(value, str) else value[0])
        raw[offset : offset + 4] = bytes.fromhex("0000003f")
        packet["authorization"][field] = raw.hex() if isinstance(value, str) else [raw.hex()]
    with pytest.raises(ValueError, match="V7"):
        qualify_immediate(packet, before, action, 1)


def test_trace_limit_counts_rx_work_and_output_before_append(tmp_path: Path) -> None:
    session = ImmediateSession.__new__(ImmediateSession)
    session.out, session.work, session.tmp = (tmp_path / name for name in ("output", "work", "rx"))
    session.max_trace_bytes = 10
    for directory in (session.out, session.work, session.tmp):
        directory.mkdir()
        (directory / "evidence.bin").write_bytes(b"123")
    session.check_budget(1)
    with pytest.raises(RxPersistenceError, match="trace budget"):
        session.check_budget(2)


def test_duration_cap_uses_one_remaining_interval_and_journals_it(tmp_path: Path) -> None:
    before, _, packet = capture()

    class Session:
        truncated = False

        def advance(self, action: int, n: int) -> dict[str, Any]:
            assert action == 39
            assert n == 1
            return packet

        def end(self, *, truncated: bool) -> None:
            self.truncated = truncated

        def check_budget(self, additional: int = 0) -> None:
            assert additional > 0

    immediate = NativeImmediateEnvironment(config(tmp_path, duration_limit=10))
    immediate._session = Session()
    immediate._current, immediate._ended = before, False
    journal = io.BytesIO()
    immediate._journal = journal
    result = immediate.step_native(39, native_iterations=5)
    assert result["actualIterations"] == result["effectiveRequestedIterations"] == 1
    assert result["requestedIterations"] == 5
    assert result["truncated"]
    assert not result["terminated"]
    assert result["reward"] is None
    assert result["trainingEligible"] is False
    assert json.loads(journal.getvalue())["result"]["observation"]["nativeT"] == 2210
    assert immediate._session.truncated


@pytest.mark.parametrize("action_type", [np.int32, np.int64, np.uint64])
def test_numpy_actions_and_iterations_are_normalized_before_native_call(
    tmp_path: Path, action_type: Any
) -> None:
    before, _, packet = capture()
    calls: list[tuple[int, int]] = []

    class Session:
        def advance(self, action: int, n: int) -> dict[str, Any]:
            assert type(action) is int
            assert type(n) is int
            calls.append((action, n))
            return packet

        def check_budget(self, additional: int = 0) -> None:
            assert additional > 0

    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = Session()
    immediate._current, immediate._ended = before, False
    immediate._journal = io.BytesIO()
    result = immediate.step_native(action_type(39), native_iterations=np.int64(1))
    assert calls == [(39, 1)]
    assert result["actualIterations"] == 1


def test_gym_discrete_sample_is_accepted_by_v7(tmp_path: Path) -> None:
    import gymnasium as gym

    before, _, packet = capture()
    mask = np.zeros(78, dtype=np.int8)
    mask[39] = 1
    sampled = gym.spaces.Discrete(78).sample(mask=mask)

    class Session:
        def advance(self, action: int, n: int) -> dict[str, Any]:
            assert type(action) is int
            assert (action, n) == (39, 1)
            return packet

        def check_budget(self, additional: int = 0) -> None:
            assert additional > 0

    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = Session()
    immediate._current, immediate._ended = before, False
    immediate._journal = io.BytesIO()
    assert immediate.step_native(sampled, native_iterations=1)["actualIterations"] == 1


@pytest.mark.parametrize("action", [-1, 78, True, False, np.bool_(True), 39.0, "39"])
def test_invalid_actions_fail_before_provider_access(tmp_path: Path, action: Any) -> None:
    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = SimpleNamespace(advance=lambda *_: pytest.fail("provider was accessed"))
    immediate._current, immediate._ended = {}, False
    with pytest.raises(ValueError, match="action"):
        immediate.step_native(action)


@pytest.mark.parametrize("iterations", [0, 6, True, False, np.bool_(True), 1.0, "1"])
def test_invalid_iterations_fail_before_provider_access(tmp_path: Path, iterations: Any) -> None:
    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = SimpleNamespace(advance=lambda *_: pytest.fail("provider was accessed"))
    immediate._current, immediate._ended = {}, False
    with pytest.raises(ValueError, match="native_iterations"):
        immediate.step_native(39, native_iterations=iterations)


@pytest.mark.parametrize("action", [np.int64(0), np.int64(77)])
def test_valid_action_boundaries_reach_provider_as_int(tmp_path: Path, action: Any) -> None:
    class ReachedProviderError(Exception):
        pass

    def advance(value: int, n: int) -> None:
        assert type(value) is int
        assert value == action
        assert type(n) is int
        assert n == 1
        raise ReachedProviderError

    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = SimpleNamespace(advance=advance, close=lambda **_: None)
    immediate._current, immediate._ended = {}, False
    with pytest.raises(ReachedProviderError):
        immediate.step_native(action, native_iterations=np.int64(1))


def test_repeated_reset_reuses_session_with_separate_duration_and_unique_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, _, packet = capture()
    original_physical_state = immediate_runtime.physical_state
    instances: list[Any] = []

    class Session:
        closed = False
        episode: int | None = None

        def __init__(self, settings: NativeRuntimeConfig, verified: Any) -> None:
            del verified
            settings.trace_dir.mkdir()
            instances.append(self)

        def reset(self) -> dict[str, Any]:
            self.episode = len(instances)
            return {"synthetic_reset": True, "pid": packet["pid"], "runtimeProvenance": {}}

        def advance(self, action: int, n: int) -> dict[str, Any]:
            assert (action, n) == (39, 1)
            return packet

        def end(self, *, truncated: bool) -> None:
            assert truncated
            self.episode = None

        def check_budget(self, additional: int = 0) -> None:
            assert additional > 0

        def close(self, *, error: bool) -> None:
            assert not error
            self.closed = True

    monkeypatch.setattr(immediate_module, "verify_assets", lambda _: object())
    monkeypatch.setattr(immediate_runtime, "ImmediateSession", Session)
    monkeypatch.setattr(
        immediate_runtime,
        "physical_state",
        lambda value, **kwargs: (
            deepcopy(before)
            if value.get("synthetic_reset")
            else original_physical_state(value, **kwargs)
        ),
    )
    settings = config(tmp_path, duration_limit=50, allow_native=True)
    immediate = NativeImmediateEnvironment(settings)
    try:
        _, first_info = immediate.reset(seed=11, duration_limit=10)
        first_path = immediate.journal_path
        result = immediate.step_native(39, native_iterations=5)
        assert result["actualIterations"] == 1
        assert result["truncated"]
        _, second_info = immediate.reset(seed=np.int64(12), duration_limit=20)
        assert len(instances) == 1
        assert immediate.config is settings
        assert settings.duration_limit == 50
        assert first_info["episodeDurationLimit"] == 10
        assert second_info["episodeDurationLimit"] == immediate._duration_limit == 20
        assert second_info["requestedSeed"] == 12
        assert type(second_info["requestedSeed"]) is int
        assert second_info["seedChangesEngineReset"] is False
        assert immediate.journal_path != first_path
        assert first_path is not None
        assert len(first_path.read_bytes().splitlines()) == 2
        assert immediate._elapsed == 0
        assert not immediate._ended
    finally:
        immediate.close()
    assert instances[0].closed


@pytest.mark.parametrize("duration", [True, 0, 11, 60])
def test_episode_duration_override_cannot_exceed_provisioned_limit(
    tmp_path: Path, duration: Any
) -> None:
    immediate = NativeImmediateEnvironment(config(tmp_path, duration_limit=50, allow_native=True))
    with pytest.raises(ValueError, match="episode duration"):
        immediate.reset(duration_limit=duration)
    assert immediate._session is None
    assert immediate.config.duration_limit == immediate._duration_limit == 50


@pytest.mark.parametrize(
    ("remaining", "used", "reason"),
    [
        (100.0, 20, None),
        (10.0, 20, "session deadline reserve"),
        (100.0, 61, "session evidence reserve"),
    ],
)
def test_session_reuse_guard_counts_future_episode_and_known_close_copies(  # noqa: PLR0913
    tmp_path: Path, remaining: float, used: int, reason: str | None
) -> None:
    calls: list[bool] = []

    def quota_status() -> dict[str, int]:
        calls.append(True)
        return {
            "used_bytes": used,
            "max_bytes": 100,
            "close_duplicate_reserve_bytes": 5,
            "close_metadata_reserve_bytes": 25,
            "close_reserve_bytes": 30,
        }

    immediate = NativeImmediateEnvironment(config(tmp_path))
    immediate._session = SimpleNamespace(
        closed=False,
        episode=None,
        deadline=time.monotonic() + remaining,
        quota_status=quota_status,
    )
    status = immediate.session_reuse_status(minimum_remaining_seconds=30, additional_trace_bytes=10)
    assert status["eligible"] is (reason is None)
    assert status["reason"] == reason
    assert status["close_reserve_bytes"] == 30
    assert calls == [True]
    immediate._session.episode = 0
    assert not immediate.session_reuse_status(
        minimum_remaining_seconds=30, additional_trace_bytes=10
    )["eligible"]
    assert calls == [True]  # An active branch cannot be reused even with quota left.
