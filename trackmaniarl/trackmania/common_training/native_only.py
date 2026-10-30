"""Source-bound native-only training; never grants Windows/paired qualification.

The clock artifact must independently establish native simulation-time units.
The observed Windows/native affine mapping is deliberately not accepted here.
Raw endpoint, applied-input, terminal and reward validation is shared with the
existing V7 adapter rather than reimplemented by a training entrypoint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.trackmania.common_training.adapters import (
    KNOWN_MAP_SHA256,
    KNOWN_REFERENCE_SHA256,
    SCHEDULE,
    NativeCommonAdapter,
    checked_artifact,
    checked_json,
    collect_contract,
)
from trackmaniarl.trackmania.common_training.contract import (
    NATIVE_ONLY_CLOCK_SEMANTICS,
    ClockSpec,
    CommonTrainingContract,
    finite_number,
    require_digest,
)
from trackmaniarl.trackmania.common_training.geometry import CommonPolylineReference
from trackmaniarl.trackmania.native_headless._runtime.artifacts import PINS, ROLE_PATHS
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROFILE,
    IMMEDIATE_RESET_PROFILE,
)

CLOCK_SCHEMA = "tmrl.native-simulation-clock/v1"
UNIT_PROOF_SCHEMA = "tmrl.native-simulation-unit-source/v1"
UNIT_SEMANTICS = "nominal-engine-simulation-seconds; no wall/race origin mapping"
COUNTER_SOURCE = "actual POST.nativeT"
KNOWN_SERVER_SHA256 = PINS[ROLE_PATHS["server"]]
KNOWN_TAPE_SHA256 = PINS[ROLE_PATHS["tape"]]
RESET_NATIVE = 2200
RAW_INTERVAL = 10
# The pinned input tape has 4152 records. V7 reset leaves reader152 and ARM
# requires a subsequent reader: p+n < 4152. This is not a Windows clock bound.
LAST_CONTROLLED_NATIVE = RESET_NATIVE + (4152 - 1 - 152) * RAW_INTERVAL


def _counter(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value % RAW_INTERVAL:
        raise ValueError(name + ": actual nonnegative native counter on the10-unit grid required")
    return value


@dataclass(frozen=True, kw_only=True)
class NativeOnlyClockProof:
    """Explicit native unit proof and a bounded pinned tape/engine scope.

    Construction validates the selected manifest's declarations. Loading via
    ``from_artifact`` also hashes every referenced source. No source is executed.
    """

    sha256: str
    minimum_native: int
    maximum_native: int
    seconds_per_tick: float
    scope: Mapping[str, Any]
    source_artifacts: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_digest(self.sha256, "native clock proof")
        _counter(self.minimum_native, "native clock lower bound")
        _counter(self.maximum_native, "native clock upper bound")
        if self.minimum_native != RESET_NATIVE or not (
            RESET_NATIVE < self.maximum_native <= LAST_CONTROLLED_NATIVE
        ):
            raise ValueError("native clock scope must stay within the pinned V7 input readers")
        if finite_number(self.seconds_per_tick, "native source seconds per tick") != 0.001:
            raise ValueError("the pinned native simulation unit requires its proven0.001 scale")
        expected = {
            "native_profile": IMMEDIATE_PROFILE,
            "reset_profile": IMMEDIATE_RESET_PROFILE,
            "input_schedule": SCHEDULE,
            "map_sha256": KNOWN_MAP_SHA256,
            "reference_sha256": KNOWN_REFERENCE_SHA256,
            "server_sha256": KNOWN_SERVER_SHA256,
            "tape_sha256": KNOWN_TAPE_SHA256,
        }
        if not isinstance(self.scope, Mapping) or any(
            self.scope.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("explicit native clock engine/map/tape/reset scope required")
        if (
            not isinstance(self.source_artifacts, Mapping)
            or not {
                "unit_proof",
                "server",
                "tape",
            }
            <= self.source_artifacts.keys()
        ):
            raise ValueError(
                "independent native unit proof and actual server/tape sources required"
            )

    @classmethod
    def from_artifact(cls, path: Path, *, expected_sha256: str) -> NativeOnlyClockProof:
        proof = checked_json(path, expected_sha256)
        if (
            proof.get("schema") != CLOCK_SCHEMA
            or proof.get("native_clock_qualified") is not True
            or proof.get("counter_source") != COUNTER_SOURCE
            or type(proof.get("raw_interval")) is not int
            or proof["raw_interval"] != RAW_INTERVAL
        ):
            raise ValueError("independent source-bound native simulation clock proof required")
        result = cls(
            sha256=expected_sha256,
            minimum_native=proof["minimum_native"],
            maximum_native=proof["maximum_native"],
            seconds_per_tick=proof["seconds_per_tick"],
            scope=proof["scope"],
            source_artifacts=proof["sources"],
        )
        for name, source in result.source_artifacts.items():
            if not isinstance(source, Mapping):
                raise ValueError("native clock source must be an explicit artifact record")
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                raise ValueError("absolute native clock source evidence paths required")
            raw = checked_artifact(source_path, source["sha256"])
            if name == "unit_proof":
                units = json.loads(raw)
                if (
                    not isinstance(units, Mapping)
                    or units.get("schema") != UNIT_PROOF_SCHEMA
                    or units.get("semantics") != UNIT_SEMANTICS
                ):
                    raise ValueError("independent native nominal simulation unit evidence required")
                result._require_unit_consistency(units)
            if name == "server" and source["sha256"] != KNOWN_SERVER_SHA256:
                raise ValueError("native clock server source does not match its pinned scope")
            if name == "tape" and source["sha256"] != KNOWN_TAPE_SHA256:
                raise ValueError("native clock input tape source does not match its pinned scope")
        return result

    def _require_unit_consistency(self, units: Mapping[str, Any]) -> None:
        expected_sources = {
            name: source for name, source in self.source_artifacts.items() if name != "unit_proof"
        }
        if units.get("scope") != self.scope or units.get("sources") != expected_sources:
            raise ValueError("native unit evidence scope/sources differ from its clock manifest")
        scale = units.get("scale_constant")
        if not isinstance(scale, Mapping):
            raise ValueError("native unit evidence nominal scale mapping required")
        nominal = scale.get("nominal_scale")
        if (
            not isinstance(nominal, Mapping)
            or type(nominal.get("numerator")) is not int
            or type(nominal.get("denominator")) is not int
            or nominal != {"numerator": 1, "denominator": 1000}
            or finite_number(scale.get("nominal_seconds_per_tick"), "unit nominal tick scale")
            != self.seconds_per_tick
            or finite_number(
                scale.get("nominal_seconds_per_interval"), "unit nominal interval scale"
            )
            != self.seconds_per_tick * RAW_INTERVAL
        ):
            raise ValueError("native unit evidence nominal scale differs from its clock manifest")
        observed = units.get("observed_native_reference")
        if (
            not isinstance(observed, Mapping)
            or type(observed.get("controlled_maximum_from_reader_guard")) is not int
            or observed["controlled_maximum_from_reader_guard"] != LAST_CONTROLLED_NATIVE
        ):
            raise ValueError(
                "native unit evidence reader bound differs from its pinned clock scope"
            )

    def validate_timestamp(self, native: int) -> None:
        native = _counter(native, "actual nativeT")
        if not self.minimum_native <= native <= self.maximum_native:
            raise ValueError("native timestamp exceeds its independent source clock scope")

    def require_interval(self, before: int, after: int) -> None:
        self.validate_timestamp(before)
        self.validate_timestamp(after)
        if after <= before:
            raise ValueError("actual native simulation interval must increase")

    def clock_spec(self) -> ClockSpec:
        return ClockSpec(
            clock_id="native-POST.nativeT/source-nominal-unit-proof/v1",
            unit="native-ticks",
            seconds_per_tick=self.seconds_per_tick,
            semantics=NATIVE_ONLY_CLOCK_SEMANTICS,
            evidence_sha256=self.sha256,
            synthetic=False,
        )


def native_only_contract(
    proof: NativeOnlyClockProof, reference: CommonPolylineReference
) -> CommonTrainingContract:
    if type(proof) is not NativeOnlyClockProof:
        raise ValueError("explicit independent native-only clock proof required")
    return replace(collect_contract(proof, reference), purpose="native-only-training")


class NativeOnlyAdapter(NativeCommonAdapter):
    """V7 applied POST/EOF adapter with independent native simulation time."""

    def __init__(
        self,
        contract: CommonTrainingContract,
        *,
        proof: NativeOnlyClockProof,
        reference: CommonPolylineReference,
    ) -> None:
        native_only_contract(proof, reference).require_checkpoint(contract.envelope())
        super().__init__(contract, proof=proof, reference=reference)

    def bind_reset(
        self,
        state: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        episode_id: str,
    ) -> torch.Tensor:
        pins = info["runtimeProvenance"]["pins"]
        if pins.get("server") != KNOWN_SERVER_SHA256 or pins.get("tape") != KNOWN_TAPE_SHA256:
            raise ValueError("actual reset engine/tape differ from the native clock proof")
        return super().bind_reset(state, info, episode_id=episode_id)
