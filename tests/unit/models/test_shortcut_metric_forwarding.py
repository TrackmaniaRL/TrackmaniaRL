"""Shortcut policies must not label their action indexes as native pedal values."""

from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.experiments.shortcut_macro import (
    AutonomousMacroEnvironment,
    AutonomousMacroProvider,
)
from trackmaniarl.experiments.shortcut_residual import (
    ReferenceResidualProvider,
    SourceGuidedShortcutEnvironment,
)
from trackmaniarl.trackmania.native_headless.full_qr.environment import FullQrEnvironment


@pytest.mark.parametrize("kind", ["macro", "residual"])
def test_shortcut_step_forwards_action_without_fabricating_pedals(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    forwarded: list[Any] = []

    def base_step_info(_self: Any, _step: Any, _result: Any, action: Any) -> dict[str, float]:
        forwarded.append(action)
        return {"control_gas": 1.0, "control_brake": 0.0, "control_brake_tap": 0.0}

    monkeypatch.setattr(FullQrEnvironment, "_record_step_info", base_step_info)
    if kind == "macro":
        environment = object.__new__(AutonomousMacroEnvironment)
        provider = object.__new__(AutonomousMacroProvider)
        provider.program = (0, 1, 2, 3, 4)
        provider.policy_action = 7
    else:
        environment = object.__new__(SourceGuidedShortcutEnvironment)
        provider = object.__new__(ReferenceResidualProvider)
        provider._program = (0, 1, 2, 3, 4)
        provider.reference_ticks_in_plan = 5
    environment.provider = provider
    info = environment._record_step_info(None, SimpleNamespace(reason="running"), 7)
    assert forwarded == [7]
    assert not {"control_gas", "control_brake", "control_brake_tap"} & info.keys()
    assert info["reference_guided"] is (kind == "residual")
