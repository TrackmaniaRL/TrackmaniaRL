"""Missing or ambiguous measured ticks cannot become fabricated intervals."""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def replay_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("scripts.headless_replay_windows_tapes")


def frame(clock: int, *, gas: int = 0) -> list[float]:
    raw = [0.0] * 33
    raw[3], raw[4:10], raw[31] = clock, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], gas
    return raw


def test_ambiguous_controls_at_same_clock_are_rejected_with_literal_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = replay_module(monkeypatch)
    source = {
        "initial_raw33": frame(10),
        "rows": [{"before_raw33": frame(10), "packets": [], "after_raw33": frame(10, gas=1)}],
    }
    with pytest.raises(ValueError, match="observed race clock 10"):
        replay.observed_tape(source)


def test_initial_only_complete_prefix_is_reported_without_a_policy_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replay = replay_module(monkeypatch)
    steps = []

    class Session:
        def __init__(self, *args: Any) -> None:
            self.cleanup = {"remainingOwned": []}

        def reset(self) -> None:
            pass

        def step(self, action: int, intervals: int) -> None:
            steps.append((action, intervals))
            assert (action, intervals) == (36, 1)

        def step_observed(self, source: dict) -> None:
            pytest.fail("no interval is supported after the first missing tick")

        def close(self, *, error: bool) -> None:
            assert not error

    monkeypatch.setattr(replay, "ObservedSession", Session)
    monkeypatch.setattr(
        replay,
        "state",
        lambda _: {"physicalPosition": [1.0, 2.0, 3.0], "physicalVelocity": [4.0, 5.0, 6.0]},
    )
    source = {
        "initial_raw33": frame(10),
        "rows": [{"before_raw33": frame(10), "packets": [], "after_raw33": frame(30)}],
    }
    report: dict[str, Any] = {"cases": {}}
    replay.replay_observed(
        {"runtime_root": str(tmp_path)},
        {},
        {"cases": {"missing": source}},
        tmp_path,
        lambda race: 2190 + race,
        "test",
        report,
    )
    case = report["cases"]["missing"]
    assert steps == [(36, 1)]  # Administrative neutral reset only, no policy interval.
    assert case["compared_samples_including_initial"] == 1
    assert case["missing_race_ticks"] == [20]
    assert case["endpoint_position_distance_readout_units"] == 0
    assert report["exact_physics_parity"] is False


@pytest.mark.parametrize("cases", [{}, {"clock_only": {"rows": []}}])
def test_no_replayed_intervals_cannot_qualify_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cases: dict[str, Any]
) -> None:
    replay = replay_module(monkeypatch)
    report: dict[str, Any] = {"cases": {}}
    replay.replay_observed(
        {"runtime_root": str(tmp_path)},
        {},
        {"cases": cases},
        tmp_path,
        lambda race: 2190 + race,
        "test",
        report,
    )
    assert report["cases"] == {}
    assert report["passed_collection"] is False
    assert report["exact_physics_parity"] is False


def test_nominal_cli_with_no_issued_tapes_cannot_qualify_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replay = replay_module(monkeypatch)
    manifest = tmp_path / "deployment.json"
    manifest.write_text(json.dumps({"driverBuildApproval": {"driverSha256": "test"}}))
    inputs = {
        "config": {
            "deployment_manifest": str(manifest),
            "deployment_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "receipt": {},
        "clock": {
            "relative_clock_alignment_qualified": True,
            "slope": {"numerator": 1, "denominator": 1},
            "intercept": {"numerator": -2190, "denominator": 1},
        },
        "windows": {"cases": {"clock_only": {"rows": []}}},
    }
    paths = []
    for name, value in inputs.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        paths.append(str(path))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["replay", *paths, str(output)])
    replay.main()
    report = json.loads((output / "report.json").read_bytes())
    assert report["cases"] == {}
    assert report["passed_collection"] is False
    assert report["exact_physics_parity"] is False
