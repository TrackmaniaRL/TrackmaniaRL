"""The source-guided task has an explicit action meaning and strict data domain."""

import pytest

from trackmaniarl.experiments.shortcut_residual import (
    REFERENCE_ACTION,
    ReferenceControls,
    residual_tick_action,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError


def test_neutral_correction_preserves_all_native_commands_and_tick_edges() -> None:
    for reference in range(78):
        for offset in range(5):
            assert residual_tick_action(reference, REFERENCE_ACTION, offset) == reference


def test_corrections_stay_on_action_table_and_brake_taps_release() -> None:
    for reference in range(78):
        for correction in range(78):
            for offset in (0, 1, 4):
                assert 0 <= residual_tick_action(reference, correction, offset) < 78
    assert residual_tick_action(39, 38, 0) % 6 == 1
    assert residual_tick_action(39, 38, 1) % 6 == 0
    for reference, correction, offset in ((-1, 39, 0), (39, 78, 0), (True, 39, 0), (39, 39, 5)):
        with pytest.raises(ValueError, match=r"action78|offsets"):
            residual_tick_action(reference, correction, offset)


def test_end_of_reference_is_a_command_fallback_not_episode_finish_or_error() -> None:
    controls = object.__new__(ReferenceControls)
    controls.first_reader_index = 151
    controls.actions = (39, 75)
    assert controls.at(152) == 75
    assert controls.at(153) == 36
    assert controls.at(15000) == 36
    with pytest.raises(NativeLifecycleError, match="precedes"):
        controls.at(150)
