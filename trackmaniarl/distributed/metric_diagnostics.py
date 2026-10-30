"""Optional, engine-independent metric availability and source-labelled values."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from math import isfinite
from typing import Any

METRIC_DIAGNOSTICS_KEY = "_trackmaniarl_metric_diagnostics"
METRIC_DIAGNOSTICS_SCHEMA = "tmrl.metric-diagnostics/v1"


def _label(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./:@-]{1,256}", value):
        raise ValueError("bounded diagnostic source/unit/name required")
    return value


def _counts(value: Any, steps: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or len(value) > 64:
        raise ValueError("bounded diagnostic coverage mapping required")
    result = {}
    for key, count in value.items():
        _label(key)
        if type(count) is not int or not 0 <= count <= steps:
            raise ValueError("diagnostic coverage must be an integer within episode steps")
        result[key] = count
    return result


def validate_metric_diagnostics(value: Any, *, steps: int | None = None) -> dict[str, Any]:
    """Snapshot a strict optional record; compatibility numeric fields stay numeric."""
    if (
        not isinstance(value, Mapping)
        or set(value)
        not in (
            {"schema", "scope", "source", "availability", "measurements", "termination_category"},
            {
                "schema",
                "scope",
                "source",
                "availability",
                "measurements",
                "termination_category",
                "coverage",
            },
        )
        or value.get("schema") != METRIC_DIAGNOSTICS_SCHEMA
    ):
        raise ValueError("explicit metric diagnostics schema required")
    scope = "final-step" if "coverage" in value else "step"
    if value.get("scope") != scope:
        raise ValueError("availability scope must explicitly describe step or final-step")
    source = _label(value["source"])
    category = value["termination_category"]
    if category not in ("ongoing", "finished", "time_limit", "unknown"):
        raise ValueError("explicit generic termination category required")
    available, measurements = value["availability"], value["measurements"]
    if not isinstance(available, Mapping) or not 1 <= len(available) <= 64:
        raise ValueError("bounded explicit availability required")
    availability = {}
    for key, flag in available.items():
        _label(key)
        if type(flag) is not bool:
            raise ValueError("metric availability must be boolean")
        availability[key] = flag
    if not isinstance(measurements, Mapping) or len(measurements) > 16:
        raise ValueError("bounded source-labelled measurements required")
    observed = {}
    for key, measurement in measurements.items():
        _label(key)
        if not isinstance(measurement, Mapping) or set(measurement) != {"value", "unit", "source"}:
            raise ValueError("measurement value/unit/source required")
        number = measurement["value"]
        if type(number) not in (int, float) or not isfinite(number):
            raise ValueError("finite measured scalar required")
        observed[key] = {
            "value": number,
            "unit": _label(measurement["unit"]),
            "source": _label(measurement["source"]),
        }
    result: dict[str, Any] = {
        "schema": METRIC_DIAGNOSTICS_SCHEMA,
        "scope": scope,
        "source": source,
        "termination_category": category,
        "availability": availability,
        "measurements": observed,
    }
    if steps is not None or "coverage" in value:
        coverage = value.get("coverage")
        if not isinstance(coverage, Mapping) or set(coverage) != {
            "steps",
            "availability",
            "measurements",
        }:
            raise ValueError("episode diagnostic coverage required")
        count = coverage["steps"]
        if type(count) is not int or count < 1 or (steps is not None and count != steps):
            raise ValueError("diagnostic coverage must match episode steps")
        flags = _counts(coverage["availability"], count)
        values = _counts(coverage["measurements"], count)
        if set(flags) != set(availability) or any(
            flags[k] < 1 for k in availability if availability[k]
        ):
            raise ValueError("availability coverage contradicts final measurement availability")
        if not set(observed) <= set(values) or any(values[k] < 1 for k in observed):
            raise ValueError("measurement coverage contradicts observed values")
        result["coverage"] = {"steps": count, "availability": flags, "measurements": values}
    return result


def metric_is_available(info: Mapping[str, Any], name: str) -> bool:
    """Absent optional contract keeps legacy behavior; explicit unknown is false."""
    if METRIC_DIAGNOSTICS_KEY not in info:
        return True
    diagnostics = validate_metric_diagnostics(info[METRIC_DIAGNOSTICS_KEY])
    return bool(diagnostics["availability"].get(name, False))


class MetricDiagnosticsTracker:
    """Retain final actual values and episode coverage; never carry stale values."""

    def __init__(self) -> None:
        self.steps = 0
        self.source: str | None = None
        self.flags: Counter[str] = Counter()
        self.values: Counter[str] = Counter()
        self.labels: dict[str, tuple[str, str]] = {}
        self.last: dict[str, Any] | None = None

    def record(self, info: Mapping[str, Any]) -> None:
        self.steps += 1
        self.last = None
        if METRIC_DIAGNOSTICS_KEY not in info:
            return
        record = validate_metric_diagnostics(info[METRIC_DIAGNOSTICS_KEY])
        if "coverage" in record:
            raise ValueError("a step cannot supply episode coverage")
        if self.source is not None and self.source != record["source"]:
            raise ValueError("diagnostic source changed within an episode")
        self.source = record["source"]
        for key, flag in record["availability"].items():
            self.flags[key] += int(flag)
        for key, measurement in record["measurements"].items():
            label = (measurement["unit"], measurement["source"])
            if key in self.labels and self.labels[key] != label:
                raise ValueError("measurement source/unit changed within an episode")
            self.labels[key] = label
            self.values[key] += 1
        self.last = record

    def summary(self) -> dict[str, Any] | None:
        if self.source is None:
            return None
        final = self.last or {
            "availability": {},
            "measurements": {},
            "termination_category": "unknown",
        }
        result = {
            "schema": METRIC_DIAGNOSTICS_SCHEMA,
            "scope": "final-step",
            "source": self.source,
            "termination_category": final["termination_category"],
            "availability": {key: final["availability"].get(key, False) for key in self.flags},
            "measurements": final["measurements"],
            "coverage": {
                "steps": self.steps,
                "availability": dict(self.flags),
                "measurements": dict(self.values),
            },
        }
        return validate_metric_diagnostics(result, steps=self.steps)
