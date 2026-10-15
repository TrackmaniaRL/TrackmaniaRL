"""Record one complete human run for rewards without recording road boundaries."""

from __future__ import annotations

import argparse
from time import monotonic

import numpy as np

from trackmaniarl.trackmania.reward_points import save_reward_points
from trackmaniarl.trackmania.session import OpenPlanetSessionClient
from trackmaniarl.trackmania.telemetry import OpenPlanetClient, OpenPlanetClientConfig


def record_reward_points(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"reward points already exist: {args.output}")
    if args.map_path is not None and not args.map_path.is_file():
        raise ValueError(f"map file does not exist: {args.map_path}")
    if args.max_duration <= 0 or args.spacing <= 0:
        raise ValueError("max-duration and spacing must be positive")
    session = OpenPlanetSessionClient(args.host, args.session_port, timeout_s=args.timeout)
    active = session.inspect_loaded_map()
    session.confirm_ready(active.map_uid)
    client = OpenPlanetClient(OpenPlanetClientConfig(args.host, args.port, args.timeout))
    print(f"Map UID: {active.map_uid}. Restart the map, then drive once from start to finish.")
    try:
        points = _record_complete_run(client, args.max_duration)
    finally:
        client.close()
    session.verify_loaded_map(active.map_uid)
    path = save_reward_points(
        args.output, points, map_uid=active.map_uid, map_path=args.map_path, spacing_m=args.spacing
    )
    print(f"Recorded reward points: {path} (map_uid={active.map_uid})")


def _record_complete_run(client: OpenPlanetClient, duration: float) -> np.ndarray:
    deadline = monotonic() + duration
    points: list[np.ndarray] = []
    previous_time: float | None = None
    started = False
    while monotonic() < deadline:
        frame = client.read_next().values
        race_time, finished = float(frame[3]), bool(frame[2])
        if not started:
            if finished or not 0 <= race_time <= 1000:
                continue
            started = True
            print("Recording reward path; avoid respawns and finish the run.")
        if previous_time is not None and race_time < previous_time:
            raise ValueError("race restarted while recording; run the command again")
        previous_time = race_time
        position = frame[4:7].copy()
        if not points or np.linalg.norm(position - points[-1]) >= 0.25:
            points.append(position)
        if finished:
            if points and np.linalg.norm(position - points[-1]) > 1e-4:
                points.append(position)
            if len(points) < 2:
                raise ValueError("finished run contains too few reward points")
            return np.asarray(points, dtype=np.float32)
    raise TimeoutError("no complete run recorded within --max-duration")
