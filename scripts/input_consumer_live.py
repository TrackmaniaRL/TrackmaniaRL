"""Explicit 007 diagnostic using the existing global lease and reset guards."""

from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.headless_windows_physical_probe import (
    MAX_PACKETS,
    ProbeRunContract,
    _raw33,
    run_live,
    select_actions,
)
from scripts.input_consumer_probe import InputConsumerProbeClient, verify_capture


def drive_three_phases(  # noqa: PLR0913 - shared driver contract plus explicit optional 006 transport
    env: Any,
    result: dict[str, Any],
    guard: Callable[[], None],
    *,
    fresh_006: bool = False,
    phase_profile: str = "default",
) -> None:
    """Retain literal observations; phase goals do not claim applied durations."""
    raw = _raw33(result["initial_raw33"])
    result["phases"], result["packets"], result["fresh006"] = [], [], []
    result["issued_command_ack_claimed"] = False
    result["phase_duration_source"] = "observed original Connect race timestamp"
    goals = {"default": (500, 750, 750), "long-mild-left": (150, 150, 2000)}
    if phase_profile not in goals:
        raise ValueError("explicit bounded input-consumer phase profile required")
    result["phase_profile"] = phase_profile
    gas, left = select_actions(env._action_table)
    neutral = next(
        (index for index, control in enumerate(env._action_table) if list(control) == [0, 0, 0]),
        None,
    )
    if neutral is None:
        raise ValueError("actual action table lacks neutral without braking")
    fresh = None
    if fresh_006:
        from scripts.headless_telemetry006 import Telemetry006Client, Telemetry006Config

        fresh = Telemetry006Client(Telemetry006Config(timeout_s=0.4))

    def retain_fresh(boundary: str) -> None:
        if fresh is None:
            return
        guard()
        row: dict[str, Any] = {"boundary": boundary}
        result["fresh006"].append(row)
        try:
            frame = fresh.read_next()
            row["raw33"] = _raw33(frame.values)
            row["raw_payload_hex"] = fresh.last_packet.hex() if fresh.last_packet else None
        except BaseException as error:
            row["error"] = repr(error)
            raise
        finally:
            row["metadata"] = fresh.last_metadata
        guard()

    try:
        for (name, action), goal in zip(
            (("neutral", neutral), ("gas", gas), ("mild_left", left)),
            goals[phase_profile],
            strict=True,
        ):
            guard()
            retain_fresh("before_" + name)
            phase: dict[str, Any] = {
                "name": name,
                "action": action,
                "before_raw33": raw,
                "issued_gbs": [float(value) for value in env._action_table[action]],
                "goal_observed_race_delta_ms": goal,
                "apply_start_host_ns": time.perf_counter_ns(),
            }
            result["phases"].append(phase)
            env.controller.apply_discrete(env._action_table[action].copy())
            phase["apply_finished_host_ns"] = time.perf_counter_ns()
            target = raw[3] + goal
            while raw[3] < target:
                guard()
                if len(result["packets"]) >= MAX_PACKETS:
                    raise RuntimeError("original telemetry packet budget exhausted")
                previous = raw
                raw = _raw33(env.client.read_next().values)
                if raw[3] < previous[3]:
                    raise RuntimeError("original race clock regressed during diagnostic")
                result["packets"].append(
                    {"raw33": raw, "host_perf_counter_ns": time.perf_counter_ns(), "phase": name}
                )
            phase["after_raw33"] = raw
            phase["observed_race_delta_ms"] = raw[3] - phase["before_raw33"][3]
            phase["completed"] = True
            retain_fresh("after_" + name)
    finally:
        try:
            env.release_controls()
        finally:
            if fresh is not None:
                fresh.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--no-artificial-lag", action="store_true")
    parser.add_argument("--fresh-telemetry006", action="store_true")
    parser.add_argument("--phase-profile", choices=("default", "long-mild-left"), default="default")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; import/tests never operate the game")
    contract = ProbeRunContract(
        client_factory=InputConsumerProbeClient,
        verify_capture=verify_capture,
        drive=functools.partial(
            drive_three_phases,
            fresh_006=args.fresh_telemetry006,
            phase_profile=args.phase_profile,
        ),
        report_name="input-consumer007.json",
        diagnostic_schema="tmrl.windows-input-consumer-diagnostic/v1",
    )
    return run_live(
        args.output.absolute(),
        game_dir=args.game_dir,
        no_artificial_lag=args.no_artificial_lag,
        probe_contract=contract,
    )


if __name__ == "__main__":
    raise SystemExit(main())
