"""Passive human RGB/control recording at the configured decision cadence."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, time_ns
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.reward_points import RewardPoints
from trackmaniarl.trackmania.session import OpenPlanetSessionClient
from trackmaniarl.trackmania.telemetry import OpenPlanetClient, OpenPlanetClientConfig
from trackmaniarl.trackmania.vision_environment import CaptureRegion, ScreenFrameSource


def recording_contract(config_path: Path) -> dict[str, Any]:
    spec = RunSpec.from_yaml(config_path)
    assert spec.components.environment is not None
    kwargs = spec.components.environment.kwargs
    environment = kwargs["config"]
    points = RewardPoints(
        config_path.parent / environment["reward_points_path"],
        expected_map_uid=environment["expected_map_uid"],
    )
    return {
        "format": "trackmaniarl-driving-demo-v1",
        "map_uid": points.map_uid,
        "reward_sha256": points.sha256,
        "interval_ms": environment["decision_interval_ms"],
        "capture": kwargs["capture"],
        "image_size": [160, 90],
        "alignment": "image_before_control_window",
    }


def record_driving_demos(config_path: Path, output: Path, count: int) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    config_path = config_path.resolve()
    contract = recording_contract(config_path)
    spec = RunSpec.from_yaml(config_path)
    assert spec.components.environment is not None
    environment = spec.components.environment.kwargs["config"]
    host = environment.get("host", "127.0.0.1")
    session = OpenPlanetSessionClient(host, environment.get("session_port", 9001), timeout_s=10)
    session.verify_loaded_map(contract["map_uid"])
    source = ScreenFrameSource(CaptureRegion.model_validate(contract["capture"]))
    client = OpenPlanetClient(OpenPlanetClientConfig(host, environment.get("port", 9000)))
    output.mkdir(parents=True, exist_ok=True)
    try:
        index = 0
        while index < count:
            print(
                f"Lap {index + 1}/{count}: restart, then drive to the finish without respawns.",
                flush=True,
            )
            try:
                episode = _record_episode(client, source, float(contract["interval_ms"]))
            except ValueError as error:
                print(f"Discarded attempt: {error}. Restart for another attempt.", flush=True)
                continue
            session.verify_loaded_map(contract["map_uid"])
            path = output / f"lap-{time_ns()}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(
                    stream, **episode, metadata=json.dumps(contract, sort_keys=True)
                )
            print(f"Saved {path}: {len(episode['actions'])} decisions", flush=True)
            index += 1
    finally:
        source.close()
        client.close()


def _record_episode(client: OpenPlanetClient, source: Any, interval_ms: float) -> dict[str, Any]:
    deadline = monotonic() + 600
    frame = client.read().values
    while bool(frame[2]) or not 0 < float(frame[3]) <= 500:
        if monotonic() > deadline:
            raise TimeoutError("restart the map within ten minutes")
        frame = client.read().values
    print("Recording active; drive to the finish.", flush=True)
    images, telemetry, actions, timestamps = [], [], [], []
    previous_controls = np.zeros(3, dtype=np.float32)
    _, action_table = select_brake_tap_actions(None)
    table = np.asarray(action_table)
    allowed = np.flatnonzero(table[:, 1] >= 0)
    while monotonic() < deadline:
        state = frame.copy()
        started = float(state[3])
        rgb = _resize_rgb(source.capture())
        # Read after capture: every target control sample follows the image.
        frame = client.read().values
        controls = []
        while float(frame[3]) < started + interval_ms and not bool(frame[2]):
            if float(frame[3]) < started:
                raise ValueError("lap restarted; no partial demo was saved")
            controls.append(frame[[31, 32, 30]].copy())
            frame = client.read_next().values
        if bool(frame[2]):
            break
        if not controls or float(frame[3]) - started > interval_ms * 2:
            raise ValueError("capture/telemetry missed the decision cadence; reduce capture size")
        label_controls = np.mean(controls, axis=0)
        distances = np.square(table[allowed] - label_controls).sum(axis=1)
        action = int(allowed[np.argmin(distances)])
        # Only the preceding control window is observable; current labels never enter state.
        state[30:33] = previous_controls[[2, 0, 1]]
        previous_controls = label_controls
        images.append(rgb)
        telemetry.append(state)
        actions.append(action)
        timestamps.append(started)
    if not bool(frame[2]) or len(actions) < 64:
        raise ValueError("a complete lap with at least 64 decisions is required")
    return {
        "frames": np.stack(images),
        "telemetry": np.stack(telemetry),
        "actions": np.asarray(actions, dtype=np.int64),
        "timestamps_ms": np.asarray(timestamps, dtype=np.float64),
        "finish_time_s": float(frame[3]) / 1000,
    }


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    resized = F.interpolate(
        tensor, size=(90, 160), mode="bilinear", align_corners=False, antialias=True
    )
    return resized[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
