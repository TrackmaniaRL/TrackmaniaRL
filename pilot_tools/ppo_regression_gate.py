"""Separate quality, timing and regression review; never selects/promotes a checkpoint."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from trackmaniarl.commands.evaluation import _has_runtime_errors, _runtime_trial_statistics


def _suite(payload: dict[str, Any]) -> dict[str, Any]:
    trials = payload.get("trials", [])
    maximum, counts_valid = _runtime_trial_statistics(trials)
    finishes = [t for t in trials if t.get("finished") is True]
    times = [t.get("finish_time_s") for t in finishes]
    times_valid = all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times)
    runtime_fields_present = all("telemetry_error" in t and "controller_error" in t for t in trials)
    valid = (payload.get("status") == "complete" and payload.get("expected_trials") == 10
             and len(trials) == 10 and {t.get("trial_index") for t in trials} == set(range(10))
             and all(t.get("map_uid") == "oqIJ5rQDRrNwLPTh9H2p_W4tLof" for t in trials)
             and runtime_fields_present and not _has_runtime_errors(trials)
             and counts_valid and times_valid
             and not any(t.get("termination_reason") in {"operator_interruption",
                         "capture_interruption", "telemetry_error"} for t in trials))
    return {"measurement_valid": valid, "finishes": len(finishes),
            "median_finish_s": median(times) if times and times_valid else None,
            "maximum_step_ms": maximum,
            "timing_gate_passed": bool(counts_valid and maximum is not None and maximum <= 100),
            "quality_gate_passed": valid and len(finishes) >= 8}


def regression_review(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before, after = _suite(reference), _suite(candidate)
    ratio = None
    if before["median_finish_s"] and after["median_finish_s"]:
        ratio = after["median_finish_s"] / before["median_finish_s"]
    valid = before["measurement_valid"] and after["measurement_valid"]
    fewer = valid and after["finishes"] < before["finishes"]
    slower = valid and ratio is not None and ratio > 1.10
    return {"reference": before, "candidate": after,
            "median_time_ratio": ratio, "review_required": not valid or fewer or slower,
            "reasons": (["invalid measurement"] if not valid else [])
            + (["fewer finishes"] if fewer else []) + (["median slowdown >10%"] if slower else []),
            "strict_gate_passed": after["quality_gate_passed"] and after["timing_gate_passed"],
            "automatic_checkpoint_selection": False, "automatic_full_training": False,
            "note": "Review trigger only; no statistical proof or replacement of final evaluation"}
