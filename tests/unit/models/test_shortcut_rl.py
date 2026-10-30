import pytest

from trackmaniarl.distributed.coordinator_validation import _validate_episode_outcome
from trackmaniarl.experiments.shortcut_rl import native_goal_summary


def test_goal_scoring_keeps_physical_clock_and_distinguishes_record_boundary() -> None:
    info = {"termination_reason": "finished", "race_time_ms": 17750.0, "policy_version": 0}
    tied = native_goal_summary(1, {**info, "actual_goal_race_time_ms": 17715}, 355)
    record = native_goal_summary(1, {**info, "actual_goal_race_time_ms": 17714}, 355)
    assert tied["physical_race_time_s"] == record["physical_race_time_s"] == 17.75
    assert tied["finish_time_s"] == 17.715
    assert record["finish_time_s"] == 17.714
    assert tied["race_time_s"] == 17.715
    assert record["race_time_s"] == 17.714
    _validate_episode_outcome(tied)
    _validate_episode_outcome(record)
    with pytest.raises(ValueError, match="actual native Goal"):
        native_goal_summary(1, info, 355)
    with pytest.raises(ValueError, match="interval"):
        native_goal_summary(1, {**info, "actual_goal_race_time_ms": 17000}, 355)
    failed = native_goal_summary(1, {"termination_reason": "off_track", "policy_version": 0}, 20)
    assert not failed["finished"]
    assert failed["finish_time_s"] == 0
