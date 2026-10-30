"""Decision scheduling with synthetic transport, not native source qualification."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.unit.models.test_full_qr_environment import frame
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.full_qr import native_provider
from trackmaniarl.trackmania.native_headless.full_qr.environment import FullQrStep
from trackmaniarl.trackmania.native_headless.full_qr.profile import PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.protocols import NativeInconclusiveError


class RecordingSession:
    def __init__(self) -> None:
        self.native_t = 2200
        self.applied: list[tuple[float, float, float]] = []
        self.observed: list[int] = []
        self.qualified: list[int] = []
        self.eof = False

    def advance(self, action: int, stride: int) -> dict[str, Any]:
        self.applied.extend(action_sgb(action)[:stride])
        self.native_t += 10 * stride
        return {"eof": self.eof, "state": {"nativeT": self.native_t}, "action": action}


@pytest.fixture
def recording_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[native_provider.NativeFullQrProvider, RecordingSession]:
    session = RecordingSession()
    events = SimpleNamespace(
        physics_stride=1,
        observe_physics=lambda unused, clock: session.observed.append(clock),
        deliver=lambda *args: None,
    )
    # Explicit synthetic configuration: the constructor performs no native IO.
    settings = SimpleNamespace(
        runtime=SimpleNamespace(protocol_profile=PROTOCOL, allow_native=True),
        assets=SimpleNamespace(profile=PROFILE),
        actor="synthetic",
        retention=None,
        driver_episode_limit=1,
        next_episode_reserve_bytes=1,
    )
    provider = native_provider.NativeFullQrProvider(settings, events)
    provider._session = session
    provider._before = {"nativeT": 2200}
    provider._identity = object()
    provider._collector = SimpleNamespace(peek=lambda *args: None)
    monkeypatch.setattr(provider, "_frame", lambda: frame(session.native_t - 2200))
    monkeypatch.setattr(provider, "close", Mock())
    monkeypatch.setattr(native_provider, "physical_state", lambda packet: packet["state"])

    def qualify(  # noqa: PLR0913 -- existing native qualifier signature
        packet: dict[str, Any], before: Any, issued: int, ticks: int
    ) -> None:
        assert packet["action"] == issued
        assert packet["state"]["nativeT"] - before["nativeT"] == 10 * ticks
        session.qualified.append(issued)

    monkeypatch.setattr(native_provider, "qualify_immediate", qualify)
    return provider, session


# All six gas/brake combinations, including left/neutral/right steering. The
# runtime-profile test independently verifies the complete calibrated78 table.
@pytest.mark.parametrize(
    "action",
    [
        pytest.param(0, id="coast-left"),
        pytest.param(37, id="brake-neutral"),
        pytest.param(74, id="tap-right"),
        pytest.param(75, id="gas-right"),
        pytest.param(4, id="gas-brake-left"),
        pytest.param(41, id="gas-tap-neutral"),
    ],
)
@pytest.mark.parametrize("stride", [1, 5])
def test_decisions_preserve_action_plan_and_restart_only_at_decision_boundary(
    recording_provider: tuple[native_provider.NativeFullQrProvider, RecordingSession],
    action: int,
    stride: int,
) -> None:
    provider, session = recording_provider
    provider.events.physics_stride = stride
    expected = action_sgb(action)
    for decision in range(2):
        step = provider.advance(action)
        assert session.applied[5 * decision : 5 * (decision + 1)] == expected
        assert step.frame.channels[0] == 50 * (decision + 1)
        assert step.issued_steering == expected[0][0]
    assert len(session.qualified) == 10 // stride
    assert session.observed == (list(range(2210, 2301, 10)) if stride == 1 else [])


def test_eof_without_final_evidence_remains_inconclusive(
    recording_provider: tuple[native_provider.NativeFullQrProvider, RecordingSession],
) -> None:
    provider, session = recording_provider
    session.eof = True
    with pytest.raises(NativeInconclusiveError):
        provider.advance(41)
    assert session.qualified == session.observed == []
    provider.close.assert_called_once()


@pytest.mark.parametrize("valid_source", [False, True])
def test_final_collector_must_return_a_typed_step(
    recording_provider: tuple[native_provider.NativeFullQrProvider, RecordingSession],
    *,
    valid_source: bool,
) -> None:
    provider, session = recording_provider
    session.eof = True
    provider._adherence_reader = object()
    final = FullQrStep(frame(50), 0.0) if valid_source else object()
    provider.events.final_step = Mock(return_value=final)
    if valid_source:
        assert provider.advance(41) is final
    else:
        with pytest.raises(TypeError, match="sourced final step"):
            provider.advance(41)
        provider.close.assert_called_once()
    provider.events.final_step.assert_called_once_with(
        session, 2200, provider._adherence_reader, 0.0
    )
    assert len(session.applied) == 1
