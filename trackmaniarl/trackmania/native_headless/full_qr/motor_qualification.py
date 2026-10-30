"""Pinned actual client comparisons and full-domain live-source qualification."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import PhysicalRumbleStep
from trackmaniarl.trackmania.native_headless.full_qr.profile import PROFILE
from trackmaniarl.trackmania.native_headless.full_qr.render_rumble import Virtual150RumbleClock
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState

SOURCE_FILES = (
    "contact_delivery.py",
    "prepared_impact_capture.py",
    "prepared_impact.py",
    "contact_impacts.py",
    "rumble_material.py",
    "rumble_producer.py",
    "gameplay_rumble.py",
    "rumble_queue.py",
    "motor_feedback.py",
    "physical_rumble.py",
    "render_rumble.py",
    "native_rumble_bindings.py",
    "vehicle_capture.py",
    "vehicle_reader.py",
    "final_capture.py",
    "final_rumble.py",
)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate motor qualification field")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class QualifiedMotorReference:
    asset_pins: Any
    material14_uses_default: bool
    actual_client_callbacks: int
    clock_contract: str = "virtual150fps/v1"


def load_motor_reference(path: Path, expected_sha256: str) -> QualifiedMotorReference:
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("canonical pinned motor qualification manifest required")
    raw = path.read_bytes()
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("motor qualification manifest changed")
    manifest = json.loads(raw, object_pairs_hook=_unique)
    if (
        manifest["schema"] != "tmrl.full-qr-motor-reference/v1"
        or manifest["profile"] != PROFILE
        or manifest["clock_contract"] != "virtual150fps/v1"
        or manifest["client_nominal_max_fps"] != 150
    ):
        raise ValueError("explicit full QR virtual150FPS qualification required")
    if set(manifest["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("complete physical/motor source pins required")
    for name, pin in manifest["source_sha256"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != pin:
            raise ValueError("qualified physical/motor implementation changed")
    evidence = {}
    expected_files = {"live150", "client_settings"} | {
        f"{case}_{kind}"
        for case in ("right", "left", "straight")
        for kind in ("client", "comparison")
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("all actual client comparisons and full150s live evidence required")
    for name, item in manifest["files"].items():
        file = path.parent / item["name"]
        if file.resolve(strict=True) != file or file.parent != path.parent:
            raise ValueError("private canonical motor primary artifact required")
        data = file.read_bytes()
        if len(data) > 64 * 1024**2 or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("actual motor qualification evidence changed")
        evidence[name] = json.loads(data, object_pairs_hook=_unique)
    live = evidence["live150"]
    pins = manifest["asset_pins"]
    if (
        set(pins) != {"server", "driver", "shim", "map", "tape"}
        or live["source_pins"] != pins
        or live["status"] != "PASS_LIVE_STAGED_SOURCE_150S_ONLY"
        or live["training_started"] is not False
        or live["active_simulation_seconds"] != 150.0
        or live["closure"]["remainingOwned"] != []
        or live["closure"]["cleanupErrors"] != []
        or len(live["frames"]) != 15001
        or len(live["finish_boundaries"]) != 3001
    ):
        raise ValueError("complete actual full-domain source run and closure required")
    clock = Virtual150RumbleClock(RumbleState(0.0, 0.0))
    changes = []
    contacts = 0
    for index, row in enumerate(live["frames"]):
        if row["native_t"] != 2200 + index * 10:
            raise ValueError("live source lost a physical tick")
        fields = dict(row)
        fields["state"] = RumbleState(**row["state"])
        fields["normalized_impacts"] = tuple(row["normalized_impacts"])
        contacts += row["new_contacts"]
        for change in clock.advance(PhysicalRumbleStep(**fields)):
            changes.append(
                {
                    "sequence": change.sequence,
                    "race_time_ms": change.race_time_ms,
                    "large_word": change.large_word,
                    "small_word": change.small_word,
                }
            )
    if (
        changes != live["changes"]
        or clock.rendered_frames != 22501
        or live["virtual_frames"] != 22501
        or contacts != live["contacts"]
        or contacts < 1000
    ):
        raise ValueError("full live source and complete virtual render output disagree")
    for index, row in enumerate(live["finish_boundaries"]):
        if row["native_t"] != 2200 + index * 50 or type(row["finish_active"]) is not bool:
            raise ValueError("every full-domain paired Finish boundary required")
    settings = evidence["client_settings"]
    globals_ = settings["actual_rumble_globals"]
    if (
        settings["actual_receiver_188_float"] != 1.0
        or globals_["contact_scalar_skip"]["value"] != 0
        or globals_["producer_visual_skip"]["value"] != 0
        or globals_["material14_redirect"]["value"] not in (0, 1)
    ):
        raise ValueError("actual supported paired client rumble settings required")
    total = 0
    for case, action in (("right", 75), ("left", 3), ("straight", 39)):
        client, comparison = evidence[f"{case}_client"], evidence[f"{case}_comparison"]
        if (
            client["owned_controller_closed_under_mutex"] is not True
            or client["training_started"] is not False
            or not client["decisions"]
            or {row["action"] for row in client["decisions"]} != {action}
            or client["typed_terminal_sample"]["resolved_map_identifier"] != manifest["map_uid"]
            or client["modules"] != settings["modules"]
        ):
            raise ValueError("actual same-client forced-control primary comparison required")
        start = client["actual_reset_start_time"]
        callbacks = [
            row
            for row in client["full_notifications"]
            if row.get("actual_start_time") == start and row["game_time_before"] >= start
        ]
        offsets = {
            row["input_clock_before"] - row["game_time_before"]
            for row in callbacks
            if row["input_clock_before"] == row["input_clock_after"]
            and row["game_time_before"] == row["game_time_after"]
        }
        if len(offsets) != 1 or len(callbacks) < 20:
            raise ValueError("actual client callback/input-clock binding required")
        origin = start + offsets.pop()
        if len(comparison["comparisons"]) != len(callbacks):
            raise ValueError("comparison omitted actual client callbacks")
        positive_windows = set()
        for observed, predicted in zip(callbacks, comparison["comparisons"], strict=True):
            if (
                predicted["race_ms"] != observed["input_clock_before"] - origin
                or tuple(predicted["observed"])
                != (observed["large_motor"], observed["small_motor"])
                or (predicted["predicted"][0] > 100) != (observed["large_motor"] > 100)
            ):
                raise ValueError("native waveform changed the historical collision decision")
            if observed["large_motor"] > 100:
                positive_windows.add((predicted["race_ms"] + 49) // 50 * 50)
        if len(positive_windows) < 3:
            raise ValueError("actual sustained collision and queue-expiry comparison required")
        total += len(callbacks)
    return QualifiedMotorReference(
        MappingProxyType(dict(pins)), bool(globals_["material14_redirect"]["value"]), total
    )
