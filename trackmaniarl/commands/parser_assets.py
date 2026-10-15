from __future__ import annotations

import argparse
from pathlib import Path

from trackmaniarl.commands.assets import (
    _build_geometry,
    _check_steering,
    _check_track_connection,
    _record_boundary,
    _record_demo,
    _record_trajectory,
)
from trackmaniarl.commands.human_recovery import _record_human_recovery
from trackmaniarl.commands.parser_types import CommandParsers
from trackmaniarl.commands.reward_points import record_reward_points


def register_asset_commands(commands: CommandParsers) -> None:
    track = commands.add_parser("track", help="TrackMania asset tools")
    track_commands = track.add_subparsers(dest="track_command", required=True)
    _register_trajectory_recorder(track_commands)
    _register_demo_recorder(track_commands)
    _register_human_recovery_recorder(track_commands)
    _register_boundary_recorder(track_commands)
    _register_geometry_builder(track_commands)
    _register_track_check(track_commands)
    _register_reward_points(track_commands)
    _register_steering_check(track_commands)


def _register_reward_points(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-reward-points", help="record a complete reward path without lidar boundaries"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--map-path", type=Path, help="optional local map file; otherwise bind to live map UID"
    )
    parser.add_argument("--max-duration", type=float, default=600.0)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--session-port", type=int, default=9001)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=record_reward_points)


def _add_telemetry_connection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=10.0)


def _register_trajectory_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser("record-trajectory", help="record XYZ points from OpenPlanet")
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--interval", type=float, default=1 / 30)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=_record_trajectory)


def _register_demo_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-demo", help="record finished human laps and drop outliers for replay seeding"
    )
    parser.add_argument("output", type=Path, help="directory that receives the kept .npz laps")
    parser.add_argument("--config", type=Path, required=True)
    _add_demo_recording_options(parser)
    parser.set_defaults(handler=_record_demo)


def _add_demo_recording_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1, help="laps to record in one session")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="discard laps slower than the best finish by more than this many seconds",
    )
    parser.add_argument("--start-timeout", type=float, default=120.0)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument(
        "--sampling-interval-ms",
        type=float,
        default=0.0,
        help="physical sampling interval; 0 records every new telemetry frame",
    )


def _register_human_recovery_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-recovery",
        help="inject one incident, hand control to a human, and save completed recovery laps",
    )
    _add_human_recovery_source_options(parser)
    _add_human_recovery_window_options(parser)
    _add_human_recovery_session_options(parser)
    parser.set_defaults(handler=_record_human_recovery)


def _add_human_recovery_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path, help="directory that receives recovery .npz files")
    parser.add_argument("--config", type=Path, required=True, help="run YAML matching the policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="policy checkpoint to drive")


def _add_human_recovery_session_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--count", type=int, default=36, help="laps; 36 gives two per stratified incident cell"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="stop after this many started laps (default: max(3*count, count+2))",
    )
    parser.add_argument("--seed", type=int, help="perturbation seed (default: run seed)")
    parser.add_argument("--max-duration", type=float, default=180.0)
    _add_human_recovery_handover_options(parser)
    parser.add_argument(
        "--minimum-context",
        type=float,
        default=1.0,
        help="minimum seconds of causal context before perturbation (at least 0.95)",
    )


def _add_human_recovery_handover_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--takeover-timeout",
        type=float,
        default=10.0,
        help="seconds to observe controller release and the first clear human input",
    )
    parser.add_argument(
        "--human-input-deadzone",
        type=float,
        default=0.10,
        help="minimum gas, brake, or steering magnitude that starts expert labels",
    )


def _add_human_recovery_window_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-progress-min",
        type=float,
        default=0.55,
        help="minimum perturbation progress as a fraction (default: 0.55)",
    )
    parser.add_argument(
        "--target-progress-max",
        type=float,
        default=0.83,
        help="maximum perturbation progress as a fraction (default: 0.83)",
    )
    _add_human_recovery_duration_options(parser)


def _add_human_recovery_duration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--perturbation-duration-min-ms", type=float, default=50.0, help="shortest incident"
    )
    parser.add_argument(
        "--perturbation-duration-max-ms", type=float, default=200.0, help="longest incident"
    )


def _register_boundary_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-boundary", help="record a manually driven left or right boundary"
    )
    parser.add_argument("side", choices=("left", "right"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-duration", type=float, default=300.0)
    parser.add_argument("--minimum-spacing", type=float, default=0.25)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=_record_boundary)


def _register_geometry_builder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "build-geometry", help="build a versioned lidar geometry .npz from two boundaries"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--map-path", type=Path, required=True)
    _add_geometry_options(parser)
    parser.set_defaults(handler=_build_geometry)


def _add_geometry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="odd moving-average window over resampled points (1 disables)",
    )
    parser.add_argument(
        "--lookahead-points",
        type=int,
        default=60,
        help="virtual points past the finish on open tracks so lidar look-ahead stays fresh",
    )


def _register_track_check(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "check", help="verify Openplanet telemetry, protocol, active map, and readiness"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--session-port", type=int, default=9001)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--config",
        type=Path,
        help="also require the active map UID to match a first-party run.yaml",
    )
    parser.set_defaults(handler=_check_track_connection)


def _register_steering_check(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "check-steering",
        help="measure how the game's analog settings bend the virtual stick's steering",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the curve here; set it as steering_curve_path in run.yaml",
    )
    parser.add_argument("--points", type=int, default=41)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=_check_steering)
