from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    FeaturesSpec,
    TransitionRewardSpec,
    VerifyCreditSpec,
)

PROFILE = "tmre.native16-pending-reference78/v1"
ACTION_COUNT = 78


def canonical(v: Any) -> Any:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(v: Any) -> Any:
    return hashlib.sha256(canonical(v)).hexdigest()


def native16(raw: Any) -> Any:
    if type(raw) is not bytes or len(raw) != 76:
        raise ValueError("exact qualified native76 required")
    physical = struct.unpack_from("<10f", raw)
    rpm = struct.unpack_from("<f", raw, 40)[0]
    gear = struct.unpack_from("<I", raw, 44)[0]
    gbsf = struct.unpack_from("<4f", raw, 60)
    if not all(math.isfinite(v) for v in (*physical, rpm, *gbsf)):
        raise ValueError("nonfinite real state")
    if abs(sum(v * v for v in physical[:4]) - 1) > 0.02:
        raise ValueError("quaternion domain")
    if not (
        0 <= gbsf[0] <= 1
        and 0 <= gbsf[1] <= 1
        and (-1 <= gbsf[2] <= 1)
        and (-1.001 <= gbsf[3] <= 1.001)
    ):
        raise ValueError("input domain")
    return {
        "quaternionWXYZ": list(physical[:4]),
        "physicalPosition": list(physical[4:7]),
        "physicalVelocity": list(physical[7:]),
        "rpm": rpm,
        "gearRaw": gear,
        "gas": gbsf[0],
        "brake": gbsf[1],
        "steer": gbsf[2],
        "filteredSteer": gbsf[3],
        "raw76Hex": raw.hex(),
    }


def action_sgb(action: Any) -> Any:
    if type(action) is not int or not 0 <= action < 78:
        raise ValueError("action78 required")
    steerbin, variant = divmod(action, 6)
    gas, mode = divmod(variant, 3)
    steer = struct.unpack("<f", struct.pack("<f", (steerbin - 6) / 6))[0]
    return [(steer, float(gas), float(j == 0) if mode == 2 else float(mode)) for j in range(5)]


def local_vector(q: Any, v: Any) -> Any:
    w, x, y, z = q
    x, y, z = (-x, -y, -z)
    tx = 2 * (y * v[2] - z * v[1])
    ty = 2 * (z * v[0] - x * v[2])
    tz = 2 * (x * v[1] - y * v[0])
    return [
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    ]


class Reference:
    arc: Any
    fingerprint: Any
    points: Any
    source: Any

    def __init__(self, points: Any, source: Any) -> None:
        if len(points) < 100 or not isinstance(source, dict) or (not source.get("sha256")):
            raise ValueError("full pinned actual reference required")
        self.points = [tuple(map(float, p)) for p in points]
        if not all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError("reference coordinate domain")
        self.source = source
        self.arc = [0.0]
        for a, b in zip(self.points, self.points[1:], strict=False):
            self.arc.append(self.arc[-1] + math.dist(a, b))
        if self.arc[-1] <= 0:
            raise ValueError("nonzero reference length required")
        self.fingerprint = digest(
            {
                "points": self.points,
                "source": source,
                "projection": "nearest local index [-20,+80], lookahead native distance20/v1",
            }
        )

    def locate(self, position: Any, previous: Any) -> Any:
        if type(previous) is not int or not 0 <= previous < len(self.points):
            raise ValueError("explicit prior reference index required")
        low = max(0, previous - 20)
        high = min(len(self.points), previous + 81)
        index = min(range(low, high), key=lambda i: (math.dist(position, self.points[i]), i))
        distance = math.dist(position, self.points[index])
        target = index
        while target + 1 < len(self.points) and self.arc[target] < self.arc[index] + 20:
            target += 1
        return {
            "index": index,
            "progress": self.arc[index],
            "deviation": distance,
            "target": self.points[target],
            "targetIndex": target,
            "source": "authored reference geometry, not collision/CP",
            "referenceFingerprint": self.fingerprint,
        }


def features(spec: FeaturesSpec) -> Any:
    state, pending, clock, reference, projection, history = (
        spec.state,
        spec.pending,
        spec.clock,
        spec.reference,
        spec.projection,
        spec.history,
    )
    if len(pending) != 3 or not all(math.isfinite(v) for v in pending):
        raise ValueError("actual pending SGB required")
    if type(clock) is not int or not 0 <= clock <= 50000 or len(history) > 2:
        raise ValueError("profile clock/history domain")
    if projection["referenceFingerprint"] != reference.fingerprint:
        raise ValueError("reference mismatch")
    velocity = local_vector(state["quaternionWXYZ"], state["physicalVelocity"])
    delta = local_vector(
        state["quaternionWXYZ"],
        [a - b for a, b in zip(projection["target"], state["physicalPosition"], strict=False)],
    )
    gear_features = [float(state["gearRaw"] == g) for g in range(9)] + [
        float(state["gearRaw"] >= 9)
    ]
    x = [
        1.0,
        *state["quaternionWXYZ"],
        *[v / 100 for v in velocity],
        state["rpm"] / 10000,
        *gear_features,
        state["gas"],
        state["brake"],
        state["steer"],
        state["filteredSteer"],
        *pending,
        *[max(-10.0, min(10.0, v / 20)) for v in delta],
        min(projection["deviation"], 200) / 20,
        projection["progress"] / reference.arc[-1],
        clock / 50000,
    ]
    for j in range(2):
        if j >= len(history):
            x.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            action = history[-j - 1]
            sb, variant = divmod(action, 6)
            gas, mode = divmod(variant, 3)
            if type(action) is not int or not 0 <= action < 78:
                raise ValueError("actual history action required")
            x.extend(
                [1.0, (sb - 6) / 6, float(gas), *map(float, (mode == 0, mode == 1, mode == 2))]
            )
    if not all(math.isfinite(v) for v in x):
        raise ValueError("finite actual features required")
    return x


def transition_reward(spec: TransitionRewardSpec) -> Any:
    before, after, delta_native, finish = (spec.before, spec.after, spec.delta_native, spec.finish)
    if (
        type(delta_native) is not int
        or delta_native <= 0
        or delta_native % 10
        or (type(finish) is not bool)
    ):
        raise ValueError("real native duration and bound finish required")
    if before["referenceFingerprint"] != after["referenceFingerprint"]:
        raise ValueError("reference continuity")
    return (
        (after["progress"] - before["progress"]) / 10
        - 0.0002 * delta_native
        - 0.001 * min(after["deviation"], 50)
        + (20 if finish else 0)
    )


def verify_credit(spec: VerifyCreditSpec) -> Any:
    events, anchor, planned, pending = (spec.events, spec.anchor, spec.planned, spec.pending)
    # Only applied old pending + first N-1 future inputs are credited.
    if not 1 <= len(events) <= 5 or len(planned) != 5 or len(pending) != 3:
        raise ValueError("bounded N1..5 credit domain")
    expected = [tuple(pending)] + [tuple(v) for v in planned[:4]]
    for j, e in enumerate(events):
        if len(e) != 256 or e[:8] != b"TMPOST03" or struct.unpack_from("<II", e, 8) != (256, 3):
            raise ValueError("actual native event header required")
        if struct.unpack_from("<I", e, 112)[0] != anchor + 10 * (j + 1):
            raise ValueError("native grid/contiguity")
        gas, brake, steer = struct.unpack_from("<3f", e, 88)
        if struct.pack("<3f", steer, gas, brake) != struct.pack("<3f", *expected[j]):
            raise ValueError("actual applied input differs from authorized pending/plan")
    return {"intervals": len(events), "nativeDelta": 10 * len(events), "pendingNotCredited": True}
