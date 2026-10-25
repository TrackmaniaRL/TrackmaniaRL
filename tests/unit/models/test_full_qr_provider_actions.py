"""Decision scheduling with synthetic transport, not native source qualification."""

from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.models.test_full_qr_environment import frame
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.full_qr import native_provider


class Session:
    def __init__(self) -> None:
        self.native_t = 2200
        self.applied: list[tuple[float, float, float]] = []

    def advance(self, action: int, stride: int) -> dict[str, Any]:
        applied = action_sgb(action)[:stride]
        self.applied.extend(applied)
        self.native_t += 10 * stride
        return {"eof": False, "state": {"nativeT": self.native_t}, "action": action}


@pytest.mark.parametrize("stride", [1, 5])
@pytest.mark.parametrize("action", range(78))
def test_one_decision_preserves_the_full_action_plan(
    monkeypatch: pytest.MonkeyPatch, action: int, stride: int
) -> None:
    session = Session()
    observed: list[int] = []
    qualified: list[int] = []
    provider = object.__new__(native_provider.NativeFullQrProvider)
    provider._closed = False
    provider._session = session
    provider._before = {"nativeT": 2200}
    provider._identity = object()
    provider._collector = SimpleNamespace(peek=lambda *args: None)
    provider.events = SimpleNamespace(
        physics_stride=stride,
        observe_physics=lambda session, clock: observed.append(clock),
        deliver=lambda *args: None,
    )
    monkeypatch.setattr(provider, "_frame", lambda: frame(session.native_t - 2200))
    monkeypatch.setattr(native_provider, "physical_state", lambda packet: packet["state"])

    def qualify(  # noqa: PLR0913 -- match the existing native qualifier signature
        packet: dict[str, Any], before: Any, issued: int, ticks: int
    ) -> None:
        assert packet["action"] == issued
        assert packet["state"]["nativeT"] - before["nativeT"] == 10 * ticks
        qualified.append(issued)

    monkeypatch.setattr(native_provider, "qualify_immediate", qualify)
    expected = action_sgb(action)
    for decision in range(2):
        step = provider.advance(action)
        assert session.applied[5 * decision : 5 * (decision + 1)] == expected
        assert step.frame.channels[0] == 50 * (decision + 1)
        assert step.issued_steering == expected[0][0]
    assert len(qualified) == 10 // stride
    assert observed == (list(range(2210, 2301, 10)) if stride == 1 else [])
