import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools.record_rl1168.archive import archive_closed_trace
from tools.record_rl1168.episodes import InvalidProgress, episode
from tools.record_rl1168.train import (
    commit_complete_episode,
    evaluation_complete,
    transition_recorder,
)


class Environment:
    def __init__(self, config, folder):
        self.settings = SimpleNamespace(trace_dir=Path(folder) / "trace")
        self.tick = 0

    def reset(self):
        return self.state()

    def state(self):
        result = np.zeros(22, dtype=np.float32)
        result[1:10] = np.eye(3).reshape(-1)
        result[0] = self.tick * 10 / 30000
        result[10] = self.tick * 0.1 / 1000
        return result

    def step(self, command):
        self.tick += 1
        goal = self.tick == 3
        return (
            self.state(),
            0,
            goal,
            False,
            {"actual_goal_ms": 21 if goal else None, "race_ms": self.tick * 10},
        )

    def close(self):
        self.settings.trace_dir.mkdir(parents=True)
        (self.settings.trace_dir / "native-supervisor.json").write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "remainingOwned": [],
                    "cleanupErrors": [],
                    "cleanupPidfdSignals": [],
                    "supervisorError": None,
                }
            )
        )


class EpisodeTests(unittest.TestCase):
    def test_forced_native_cleanup_rejects_and_records_episode(self):
        class ForcedCleanup(Environment):
            def close(self):
                super().close()
                path = self.settings.trace_dir / "native-supervisor.json"
                receipt = json.loads(path.read_text())
                receipt["cleanupPidfdSignals"] = [{"pid": 123}]
                path.write_text(json.dumps(receipt))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=ForcedCleanup)),
        ):
            with pytest.raises(RuntimeError, match="closure failed"):
                episode(
                    {},
                    directory,
                    np.array([[x, 0, 0] for x in range(150)]),
                    lambda state: 0,
                    deadline=time.monotonic() + 5,
                )
            result = json.loads((Path(directory) / "result.json").read_text())
            assert not result["clean_closure"]
            assert result["closure"]["cleanupPidfdSignals"] == [{"pid": 123}]

    def test_close_exception_is_recorded_and_rejects_episode(self):
        class BrokenClose(Environment):
            def close(self):
                super().close()
                raise OSError("close failed")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=BrokenClose)),
        ):
            with pytest.raises(RuntimeError, match="closure failed"):
                episode(
                    {},
                    directory,
                    np.array([[x, 0, 0] for x in range(150)]),
                    lambda state: 0,
                    deadline=time.monotonic() + 5,
                )
            result = json.loads((Path(directory) / "result.json").read_text())
            assert not result["clean_closure"]
            assert "close failed" in result["cleanupErrors"][0]["error"]

    def test_episode_recorder_collects_five_field_transitions(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
        ):
            rows = []
            result = episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
                training=True,
                on_transition=transition_recorder(rows),
            )
            assert result["status"] == "NATIVE_GOAL"
            assert len(rows) == 3
            assert all(len(row) == 5 for row in rows)
            assert rows[-1][-1] is True

    def test_incomplete_final_evaluation_is_not_completed(self):
        completed = [{"status": "DRIVING_FAILURE"} for _ in range(100)]
        assert not evaluation_complete([], 100)
        assert not evaluation_complete(completed[:99], 100)
        assert evaluation_complete(completed, 100)
        completed[-1] = {"status": "STOPPED_BUDGET_OR_OPERATOR"}
        assert not evaluation_complete(completed, 100)
        completed[-1] = {"status": "INVALID_PROGRESS"}
        assert evaluation_complete(completed, 100)

    def test_only_progress_guard_raises_retryable_training_exception(self):
        geometry = np.array([[x, 0, 0] for x in range(150)])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
        ):
            with (
                patch(
                    "tools.record_rl1168.episodes.shaped_reward",
                    side_effect=ValueError("unphysical progress jump; discard transition"),
                ),
                pytest.raises(InvalidProgress),
            ):
                episode(
                    {},
                    Path(directory) / "progress",
                    geometry,
                    lambda state: 0,
                    deadline=time.monotonic() + 5,
                    training=True,
                )
            with patch(
                "tools.record_rl1168.episodes.shaped_reward",
                side_effect=ValueError("other corruption"),
            ):
                with pytest.raises(ValueError, match="other corruption") as error:
                    episode(
                        {},
                        Path(directory) / "other",
                        geometry,
                        lambda state: 0,
                        deadline=time.monotonic() + 5,
                        training=True,
                    )
                assert not isinstance(error.value, InvalidProgress)

    def test_only_complete_terminal_episodes_reach_replay(self):
        class Learner:
            def __init__(self):
                self.rows = []

            def add(self, *row):
                self.rows.append(row)

        learner = Learner()
        incomplete = [("state", 0, 0.0, "following", False)]
        for status in (
            "STOPPED_BUDGET_OR_OPERATOR",
            "STEP_LIMIT_INCOMPLETE",
            "FAILED",
            "INVALID_PROGRESS",
        ):
            assert not commit_complete_episode({"status": status}, incomplete, learner)
        assert learner.rows == []
        with pytest.raises(ValueError, match="terminal transition"):
            commit_complete_episode({"status": "DRIVING_FAILURE"}, incomplete, learner)
        complete = [("state", 0, 0.0, "following", True)]
        assert commit_complete_episode({"status": "NATIVE_GOAL"}, complete, learner)
        assert learner.rows == complete

    def test_stop_is_explicit_and_does_not_return_a_finish(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
        ):
            stop = Path(directory) / "STOP"
            stop.touch()
            result = episode(
                {},
                Path(directory) / "episode",
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
                training=True,
                stop_file=stop,
            )
            assert result["status"] == "STOPPED_BUDGET_OR_OPERATOR"
            assert result["actual_goal_ms"] is None
            assert result["clean_closure"]

    def test_step_limit_is_explicitly_incomplete(self):
        class LongEpisode(Environment):
            def step(self, command):
                self.tick += 1
                return (
                    self.state(),
                    0,
                    False,
                    False,
                    {"actual_goal_ms": None, "race_ms": self.tick * 10},
                )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=LongEpisode)),
        ):
            result = episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(400)]),
                lambda state: 0,
                deadline=time.monotonic() + 30,
                training=True,
            )
            assert result["status"] == "STEP_LIMIT_INCOMPLETE"
            assert result["actual_goal_ms"] is None
            assert result["policy_steps"] == 3000
            assert result["clean_closure"]

    def test_invalid_progress_counts_as_failed_trial_and_next_trial_runs(self):
        geometry = np.array([[x, 0, 0] for x in range(150)])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
        ):
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            with patch(
                "tools.record_rl1168.episodes.shaped_reward",
                side_effect=ValueError("unphysical progress jump; discard transition"),
            ):
                failed = episode(
                    {}, first, geometry, lambda state: 0, deadline=time.monotonic() + 5
                )
            success = episode({}, second, geometry, lambda state: 0, deadline=time.monotonic() + 5)
            rows = [failed, success]
            assert len(rows) == 2
            assert sum(r["status"] == "NATIVE_GOAL" for r in rows) == 1
            assert failed["status"] == "INVALID_PROGRESS"
            assert failed["actual_goal_ms"] is None
            assert failed["clean_closure"]
            assert json.loads((first / "result.json").read_text())["status"] == "INVALID_PROGRESS"

    def test_invalid_progress_does_not_hide_bad_closure(self):
        class BadClosure(Environment):
            def close(self):
                super().close()
                p = self.settings.trace_dir / "native-supervisor.json"
                d = json.loads(p.read_text())
                d["cleanupErrors"] = ["test failure"]
                p.write_text(json.dumps(d))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=BadClosure)),
            patch(
                "tools.record_rl1168.episodes.shaped_reward",
                side_effect=ValueError("unphysical progress jump; discard transition"),
            ),
            pytest.raises(RuntimeError, match="closure failed"),
        ):
            episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
            )

    def test_training_progress_error_does_not_hide_bad_closure(self):
        class BadClosure(Environment):
            def close(self):
                super().close()
                path = self.settings.trace_dir / "native-supervisor.json"
                receipt = json.loads(path.read_text())
                receipt["cleanupErrors"] = ["test failure"]
                path.write_text(json.dumps(receipt))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=BadClosure)),
            patch(
                "tools.record_rl1168.episodes.shaped_reward",
                side_effect=ValueError("unphysical progress jump; discard transition"),
            ),
            pytest.raises(RuntimeError, match="closure failed"),
        ):
            episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
                training=True,
            )

    def test_unrelated_evaluation_error_still_raises(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
            patch(
                "tools.record_rl1168.episodes.shaped_reward",
                side_effect=ValueError("nonfinite reward input"),
            ),
            pytest.raises(ValueError, match="nonfinite"),
        ):
            episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
            )

    def test_goal_reward_uses_measured_partial_final_tick(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, environment=SimpleNamespace(FallEnvironment=Environment)),
        ):
            rows = []
            result = episode(
                {},
                directory,
                np.array([[x, 0, 0] for x in range(150)]),
                lambda state: 0,
                deadline=time.monotonic() + 5,
                training=True,
                on_transition=lambda *row: rows.append(row),
            )
            assert result["actual_goal_ms"] == 21
            self.assertAlmostEqual(sum(row[2] for row in rows), 100 - 0.021)
            assert rows[-1][-1]
            assert result["clean_closure"]

    def test_evaluation_rejects_teacher_prefix(self):
        with pytest.raises(ValueError):
            episode({}, "unused", [], None, deadline=0, prefix_controls=[(0, 1, 0)])

    def test_archive_roundtrip_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "traces" / "owned"
            trace.mkdir(parents=True)
            (trace / "native-supervisor.json").write_text(
                json.dumps(
                    {
                        "returncode": 0,
                        "remainingOwned": [],
                        "cleanupErrors": [],
                        "cleanupPidfdSignals": [],
                        "supervisorError": None,
                    }
                )
            )
            (trace / "actual.bin").write_bytes(bytes(range(256)) * 100)
            receipt = archive_closed_trace(root, trace, root / "archive.tar.gz")
            assert receipt["roundtrip_verified"]
            assert not trace.exists()
            with pytest.raises(ValueError):
                archive_closed_trace(root, root / "other", root / "wrong.tar.gz")


if __name__ == "__main__":
    unittest.main()
