"""Paired actual client UI and native Goal qualification on one pinned map.

This supplies a Finish component, never motor coverage or a reward transition.
The raw native decoder continues to distinguish its facts from literal UI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.shapes import ParseRecordsSpec
from trackmaniarl.trackmania.native_headless._runtime.validator import parse_records
from trackmaniarl.trackmania.native_headless.assets import _unique_object
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import (
    HeldNativeGoalReader,
    NativeGoalBoundary,
    NativeGoalContext,
    NativeWaypointFact,
    decode_goal_facts,
)
from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


@dataclass(frozen=True, slots=True)
class PairedFinishReference:
    map_uid: str
    race_origin: int
    race_time_ms: int
    ui_observed_race_ms: int
    goal_waypoint_id: int
    asset_pins: dict[str, str]


def qualify_reference(
    evidence: dict[str, bytes], asset_pins: dict[str, str]
) -> PairedFinishReference:
    """Recompute the paired proof; accept no precomputed 'qualified' flag."""
    if set(asset_pins) != {"server", "map", "shim"} or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in asset_pins.values()
    ):
        raise ValueError("explicit paired server/map/observer asset pins required")
    client = json.loads(evidence["client"], object_pairs_hook=_unique_object)
    owned = json.loads(evidence["observer"], object_pairs_hook=_unique_object)
    snapshot = client["actual_finish_snapshot"]
    uid = snapshot["resolved_map_identifier"]
    if (
        client.get("status") != "PASS_ACTUAL_FULL_V6_REFERENCE_UI_FINISH"
        or client.get("training_started") is not False
        or client.get("learner_updates") != 0
        or client.get("owned_controller_closed_under_mutex") is not True
        or snapshot["ui_sequence"] != 11
        or owned["cleanup"]["remainingOwned"] != []
        or owned["cleanup"]["cleanupErrors"] != []
        or owned["cleanup"]["forced"] != []
        or owned["exits"] != {"plain-validator": 0}
    ):
        raise ValueError("actual client Finish and clean original observer execution required")
    plain = parse_records(
        ParseRecordsSpec(
            text=evidence["plain"].decode(),
            expected_files={"actual-window-pb.Replay.Gbx", "actual-window-pb.Ghost.Gbx"},
            expected_map_uid=uid,
            mode="plain",
            allow_inherited_array_close=False,
        )
    )
    observed = parse_records(
        ParseRecordsSpec(
            text=evidence["observed"].decode(),
            expected_files={"actual-window-pb.Ghost.Gbx"},
            expected_map_uid=uid,
            mode="plain",
            allow_inherited_array_close=False,
        )
    )
    records = [*plain, *observed]
    result = records[0]["ValidatedResult"]
    if (
        any(row["IsValid"] is not True for row in records)
        or any(row["ValidatedResult"] != row["DeclaredResult"] for row in records)
        or any(row["ValidatedResult"] != result for row in records)
        or any(row.get("Inputs") != records[0].get("Inputs") for row in records)
        or not isinstance(records[0].get("Inputs"), str)
        or not records[0]["Inputs"]
        or result["NbCheckpoints"] != 1
        or result["NbRespawns"] != 0
        or not 0 < result["Time"] < 150000
    ):
        raise ValueError("identical original accepted results and input summaries required")
    raw = evidence["goal"]
    if len(raw) != 1536:
        raise ValueError("one actual Goal entry/store pair required")
    pid = owned["identity"]["worker"]["pid"]
    context = NativeGoalContext(
        pid,
        {
            "participant": unpack_from("<Q", raw, 88)[0],
            "sim": unpack_from("<Q", raw, 96)[0],
            "controller": unpack_from("<Q", raw, 104)[0],
            "module_slide": unpack_from("<Q", raw, 752)[0],
        },
    )
    facts = decode_goal_facts(raw, context)
    writes = decode_writer_pairs(raw[240:752] + raw[1008:1520], pid=pid, meta=context.meta)
    origin, duplicate_origin = writes[0].race_base_words
    if (
        len(facts) != 1
        or not facts[0].endpoint_requested
        or origin != duplicate_origin
        or origin != 2200
        or facts[0].event_native_t - origin != result["Time"]
    ):
        raise ValueError("actual Goal timestamp and measured race origin must equal accepted score")
    episode = [
        row
        for row in client["ui_timing_observations"]
        if row["start_time_ms"] == client["actual_reset_start_time"] and row["race_ms"] >= 0
    ]
    positives = [row for row in episode if row["ui_sequence"] == 11]
    if (
        not positives
        or positives[0]["race_ms"] != result["Time"]
        or not any(row["ui_sequence"] != 11 for row in episode)
        or any(row["race_ms"] > result["Time"] for row in episode if row["ui_sequence"] != 11)
        or any(row["player"] != snapshot["player"] for row in episode)
    ):
        raise ValueError("paired actual UI transition must occur at the accepted race timestamp")
    return PairedFinishReference(
        uid,
        origin,
        result["Time"],
        positives[0]["race_ms"],
        facts[0].waypoint_id,
        dict(asset_pins),
    )


def load_reference(path: Path, expected_sha256: str) -> PairedFinishReference:
    """Load only SHA-pinned primary artifacts from a pinned reference manifest."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("paired Finish reference manifest changed")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if manifest.get("schema") != "tmrl.paired-finish-reference/v1":
        raise ValueError("paired Finish reference schema required")
    evidence = {}
    if set(manifest["files"]) != {"client", "observer", "plain", "observed", "goal"}:
        raise ValueError("all primary paired Finish evidence required")
    for name, entry in manifest["files"].items():
        file = path.parent / entry["name"]
        if (
            file.resolve(strict=True) != file
            or file.parent != path.parent.resolve(strict=True)
            or file.stat().st_size > 32 * 1024**2
        ):
            raise ValueError("bounded canonical private reference artifact required")
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("paired Finish primary artifact changed")
        evidence[name] = data
    return qualify_reference(evidence, manifest["asset_pins"])


class QualifiedGoalFinishReader:
    """An actual held Goal stream using the paired pinned-map Finish semantics."""

    def __init__(self, session: Any, reference: PairedFinishReference) -> None:
        if any(session.pins.get(role) != pin for role, pin in reference.asset_pins.items()):
            raise ValueError("paired Finish reference does not cover these runtime assets")
        self.reference = reference
        self.context = NativeGoalContext(session.pid, session.meta)
        self.reader = HeldNativeGoalReader(
            session.local_posts / f"goal-receipt-{session.pid}.bin",
            session.local_posts / f"goal-fact-{session.pid}.bin",
            self.context,
        )

    def read(self, native_t: int) -> bool:
        boundary = self.reader.read(native_t)
        if native_t < self.reference.race_origin or (native_t - self.reference.race_origin) % 50:
            raise ValueError("paired Finish requires an actual 50ms race boundary")
        return self._qualified_goal(boundary)

    def read_terminal(self, native_t: int) -> NativeGoalBoundary:
        """Separate actual ten-ms tail; never used to relax regular boundaries."""
        race = native_t - self.reference.race_origin
        if not 0 < race <= 150000 or race % 10 or race % 50 == 0:
            raise ValueError("actual partial terminal boundary required")
        boundary = self.reader.read(native_t)
        if not self._qualified_goal(boundary):
            raise ValueError("actual Goal event has not occurred by the terminal boundary")
        return boundary

    def read_committed_terminal(
        self, native_t: int
    ) -> tuple[NativeGoalBoundary, NativeWaypointFact]:
        """Return committed writer evidence; do not call the sampled clock Finish.

        The separate CommittedGoalCapture still has to validate both clocks,
        dispatch and callback. The existing occurred-Goal reader stays strict.
        """
        boundary = self.reader.read(native_t)
        self._qualified_goal(boundary)  # Validate the actual pinned-map Goal.
        endpoints = tuple(fact for fact in self.reader.waypoints if fact.endpoint_requested)
        if boundary.goal_active or len(endpoints) != 1:
            raise ValueError("one separately timed committed endpoint required")
        return boundary, endpoints[0]

    def _qualified_goal(self, boundary: NativeGoalBoundary) -> bool:
        for fact in self.reader.waypoints:
            # The complete map file is pinned. Table entries also contain live
            # relocated data, so they are not immutable file-byte identities.
            # decode_goal_facts already verifies the actual Goal endpoint bit,
            # source caller, selected entry and both literal writer stores.
            if fact.endpoint_requested and fact.waypoint_id != self.reference.goal_waypoint_id:
                raise ValueError("actual endpoint differs from the qualified map Goal")
        return boundary.goal_active
