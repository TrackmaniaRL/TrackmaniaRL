"""Export an honest telemetry replay, not a game-authored GBX or client recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.ghost_divergence1103.prepare_archived_inputs import audited_controls

Array = np.ndarray[Any, Any]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract(study: Path) -> dict[str, Any]:
    evaluation = study / "pretraining-evaluation1093"
    result = read_json(evaluation / "evaluation-result.json")
    assert result["status"] == "PASS_FIVE_AUTONOMOUS_GREEDY_TRIALS"
    assert result["expert_file_access_denied"]
    assert result["updates_during_evaluation"] == 0
    assert result["trials"][-1]["finished"]
    assert result["trials"][-1]["finish_time_s"] == 17.715
    archive = next((evaluation / "closed-traces").rglob("session-00000006.tar.gz"))
    receipt = read_json(archive.with_suffix("").with_suffix(".json"))
    assert sha(archive) == receipt["archive_sha256"]
    with tarfile.open(archive, "r:gz") as source:

        def member(name: str) -> bytes:
            stream = source.extractfile(name)
            assert stream is not None
            return stream.read()

        meta = json.loads(member("trace/episode-0000/metadata.json"))
        reset = json.loads(member("trace/episode-0000/uncredited-neutral-reset-v7.json"))
        pid = int(meta["server_pid"])
        origin = int(reset["resetNativeT"])
        assert origin == 2200
        posts = member(f"trace/posts/post-{pid}.bin")
        goals = member(f"trace/posts/goal-receipt-{pid}.bin")
    assert len(posts) % 256 == len(goals) % 96 == 0
    commits = audited_controls(posts, pid)
    frames: list[dict[str, Any]] = []
    for at in range(0, len(posts), 256):
        row = posts[at : at + 256]
        if row[:8] != b"TMPOST03" or struct.unpack_from("<I", row, 148)[0] != 75:
            continue
        assert struct.unpack_from("<4I", row, 8) == (256, 3, pid, pid)
        clock = struct.unpack_from("<I", row, 112)[0]
        if clock < origin:
            continue
        assert clock == struct.unpack_from("<I", row, 116)[0]
        physical = struct.unpack_from("<10f", row, 152)
        gas, brake, steer = struct.unpack_from("<3f", row, 88)
        if clock > origin:
            assert commits[clock] == struct.pack("<3f", steer, gas, brake)
        assert all(math.isfinite(x) for x in physical)
        frames.append(
            {
                "t": clock - origin,
                "q": physical[:4],
                "p": physical[4:7],
                "v": physical[7:10],
                "sgb": [steer, gas, brake],
            }
        )
    assert [f["t"] for f in frames] == list(range(0, 17751, 10))
    goal_times = set()
    for at in range(0, len(goals), 96):
        row = goals[at : at + 96]
        assert row[:8] == b"TMGR0001"
        assert struct.unpack_from("<4I", row, 8) == (96, 1, pid, pid)
        _, last_goal, count = struct.unpack_from("<QII", row, 40)
        if count:
            goal_times.add(last_goal - origin)
    assert goal_times == {17715}
    # Confirm that archived controls are the actual network decisions of trial5.
    actions_file = evaluation / "actions-and-outcomes.jsonl"
    actions = [json.loads(line)["action"] for line in actions_file.read_text().splitlines()]
    assert len(actions) == 5 * 355
    with np.load(study / "training-data1091/macro-catalog.npz", allow_pickle=False) as data:
        commands = data["programs"][actions[-355:]].reshape(-1)
    for frame, command in zip(frames[1:], commands, strict=True):
        steer, variant = divmod(int(command), 6)
        gas, brake = divmod(variant, 3)
        expected = np.asarray([(steer - 6) / 6, gas, min(brake, 1)], dtype=np.float32)
        assert np.array_equal(np.asarray(frame["sgb"], dtype=np.float32), expected)
    return {
        "frames": frames,
        "goal_ms": 17715,
        "physical_end_ms": 17750,
        "provenance": {
            "archive": str(archive),
            "archive_sha256": sha(archive),
            "policy_sha256": result["policy_only_sha256"],
            "evaluation_sha256": sha(evaluation / "evaluation-result.json"),
            "model_action_log_sha256": sha(actions_file),
            "native_pid": pid,
            "actual_model_commands_verified": len(commands),
            "teacher_files_used": False,
            "new_simulations": 0,
        },
    }


def project(points: Array, camera: Array) -> Array:
    points = points - camera
    yaw = 0.35
    x = points[:, 0] * math.cos(yaw) - points[:, 2] * math.sin(yaw)
    z = points[:, 0] * math.sin(yaw) + points[:, 2] * math.cos(yaw)
    return np.column_stack((640 + x * 7, 410 + z * 4.4 - points[:, 1] * 6, z))


def render(  # noqa: PLR0913 -- explicit telemetry, geometry and output encoder
    data: dict[str, Any], geometry: dict[str, Array], output: Path, ffmpeg: Path
) -> None:
    frames = data["frames"]
    left, right = geometry["left"], geometry["right"]
    path = np.asarray([f["p"] for f in frames])
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 21)
    title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 30)
    clock_font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 38)
    # 25 fps, exact50? video frames select recorded10ms samples without interpolation.
    count = math.ceil(17.75 * 25) + 75
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x720",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with output.with_suffix(".ffmpeg.log").open("wb") as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        assert proc.stdin is not None
        try:
            for index in range(count):
                tick = min(index * 40, 17750) // 10
                frame = frames[tick]
                pos = np.asarray(frame["p"])
                image = Image.new("RGB", (1280, 720), "#101d2d")
                draw = ImageDraw.Draw(image)
                projected_left, projected_right = project(left, pos), project(right, pos)
                visible = [
                    (float((projected_left[i, 2] + projected_left[i + 1, 2]) / 2), i)
                    for i in range(len(projected_left) - 1)
                    if -150 < projected_left[i, 0] < 1430 and -150 < projected_left[i, 1] < 870
                ]
                for _, i in sorted(visible):
                    poly = [
                        tuple(projected_left[i, :2]),
                        tuple(projected_left[i + 1, :2]),
                        tuple(projected_right[i + 1, :2]),
                        tuple(projected_right[i, :2]),
                    ]
                    draw.polygon(poly, fill="#34485b")
                    draw.line(poly[:2], fill="#93a7b8", width=2)
                    draw.line(poly[2:], fill="#93a7b8", width=2)
                trail = project(path[max(0, tick - 250) : tick + 1], pos)
                if len(trail) > 1:
                    draw.line([tuple(p[:2]) for p in trail], fill="#52dfda", width=4)
                w, x, y, z = frame["q"]
                rot = np.asarray(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                body = np.asarray([[-1, 0.6, -2], [1, 0.6, -2], [1, 0.6, 2], [-1, 0.6, 2]])
                body = project(body @ rot.T + pos, pos)
                draw.polygon([tuple(p[:2]) for p in body], fill="#ffcf58", outline="white", width=2)
                front = project(np.asarray([[0, 0.8, 2.5]]) @ rot.T + pos, pos)[0]
                draw.line([(640, 406), tuple(front[:2])], fill="white", width=3)
                draw.rectangle((0, 0, 1280, 115), fill="#0b1320")
                draw.text(
                    (30, 16), "MODEL  |  AUTONOMOUS HEADLESS REPLAY", font=title, fill="white"
                )
                draw.text(
                    (30, 63),
                    "Recorded telemetry visualization - not game footage or a GBX ghost",
                    font=font,
                    fill="#a9bdcd",
                )
                draw.rectangle((25, 552, 470, 697), fill="#0b1320")
                draw.text(
                    (45, 566),
                    f"Physical clock  {frame['t'] / 1000:06.3f} s",
                    font=font,
                    fill="#c4d7e5",
                )
                if frame["t"] >= data["goal_ms"]:
                    draw.text((45, 600), "GOAL  17.715 s", font=clock_font, fill="#5be0bb")
                else:
                    draw.text(
                        (45, 600), f"{frame['t'] / 1000:06.3f} s", font=clock_font, fill="white"
                    )
                speed = float(np.linalg.norm(frame["v"])) * 3.6
                steer, gas, brake = frame["sgb"]
                draw.text(
                    (45, 650),
                    f"{speed:5.1f} km/h   S {steer:+.2f}   G {gas:.0f}   B {brake:.0f}",
                    font=font,
                    fill="#e4edf3",
                )
                # Full-map inset is measured boundary geometry and actual model trajectory.
                center = np.asarray(geometry["center"])
                minimum, maximum = center[:, [0, 2]].min(0), center[:, [0, 2]].max(0)
                scale = min(260 / (maximum[0] - minimum[0]), 250 / (maximum[1] - minimum[1]))

                def mini(
                    p: Array, low: Array = minimum, zoom: float = scale
                ) -> tuple[float, float]:
                    return (
                        float(960 + (p[0] - low[0]) * zoom),
                        float(420 + (p[2] - low[1]) * zoom),
                    )

                draw.rectangle((940, 395, 1260, 699), fill="#0b1320")
                draw.line([mini(p) for p in center], fill="#62778b", width=3)
                draw.line([mini(p) for p in path[: tick + 1 : 3]], fill="#52dfda", width=3)
                px, py = mini(pos)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#ffcf58")
                proc.stdin.write(image.tobytes())
                if index == 150:
                    image.save(output.with_suffix(".png"))
            proc.stdin.close()
            assert proc.wait(timeout=60) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = Path("\\\\?\\" + str(args.study.resolve()))
    output = study / "model-replay1109"
    output.mkdir(exist_ok=False)
    data = extract(study)
    geometry_path = Path(
        "F:/AITrackmania/reports/headless-full-qr-20261022-r11/"
        "prepared-full-qr/assets/trackmaniarl-test.geometry.npz"
    )
    with np.load(geometry_path, allow_pickle=False) as source:
        geometry = {key: source[key].copy() for key in ["left", "right", "center"]}
        assert str(source["map_uid"].item()) == "oqIJ5rQDRrNwLPTh9H2p_W4tLof"
    data["geometry_sha256"] = sha(geometry_path)
    (output / "recorded-model.json").write_text(json.dumps(data), encoding="utf-8")
    video = output / "model-17.715.mp4"
    render(data, geometry, video, args.study / "ffmpeg1107.exe")
    data["provenance"]["video_sha256"] = sha(video)
    (output / "provenance.json").write_text(json.dumps(data["provenance"], indent=2))
    print(
        json.dumps(
            {
                "video": str(video),
                "actual_samples": len(data["frames"]),
                "native_goal_ms": data["goal_ms"],
                "physical_end_ms": data["physical_end_ms"],
                "verified_model_commands": data["provenance"]["actual_model_commands_verified"],
            }
        )
    )


if __name__ == "__main__":
    main()
