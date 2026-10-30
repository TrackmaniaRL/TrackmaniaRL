"""Evidence-bound physical clock and immediate-input common6 adapters.

The alignment proof establishes only its supplied map/reset/trace and observed
counter range. A separate complete paired gate is required before optimization.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import torch

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.common_training.contract import (
    COMMON6_PROFILE,
    ClockSpec,
    CommonTrainingContract,
    canonical,
    finite_number,
    require_digest,
    require_paired_comparison,
)
from trackmaniarl.trackmania.common_training.features import (
    COMMON_INTERVAL_KEY,
    CommonFeaturePipeline,
    CommonTransitionInput,
    make_transition,
)
from trackmaniarl.trackmania.common_training.geometry import CommonPolylineReference
from trackmaniarl.trackmania.native_headless._runtime.artifacts import PINS, ROLE_PATHS
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.config import IMMEDIATE_RESET_PROFILE

SCHEDULE = "current-record-before-load/v1"
KNOWN_REFERENCE_SHA256 = PINS[ROLE_PATHS["reference"]]
KNOWN_MAP_SHA256 = PINS[ROLE_PATHS["map"]]
CLOCK_SEMANTICS = "tmrl.observed-physical-race-duration/v1"


class NativeIntervalProof(Protocol):
    """Timestamp operations needed by the shared native endpoint adapter."""

    def clock_spec(self) -> ClockSpec: ...

    def validate_timestamp(self, native: int) -> None: ...

    def require_interval(self, before: int, after: int) -> None: ...


def checked_artifact(path: Path, expected_sha256: str) -> bytes:
    require_digest(expected_sha256, "selected evidence artifact")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("evidence artifact SHA-256 mismatch: " + str(path))
    return raw


def checked_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    value = json.loads(checked_artifact(path, expected_sha256))
    if not isinstance(value, Mapping):
        raise ValueError("evidence artifact must contain a JSON object")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(name + ": actual integer required")
    return value


@dataclass(frozen=True)
class PhysicalClockProof:
    sha256: str
    minimum_native: int
    maximum_native: int
    slope: Fraction
    intercept: Fraction
    scope: str
    source_artifacts: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: Path, *, expected_sha256: str) -> PhysicalClockProof:
        proof = checked_json(path, expected_sha256)
        if (
            proof.get("schema") != "tmrl.physical-sample-clock-alignment/v1"
            or proof.get("relative_clock_alignment_qualified") is not True
            or proof.get("fields")
            != ["positionX", "positionY", "positionZ", "velocityX", "velocityY", "velocityZ"]
            or proof.get("comparison") != "exact float32 bytes; no tolerance or rounding"
            or proof.get("relationship") != ("windows_race_ms = slope * native_counter + intercept")
            or not isinstance(proof.get("scope"), str)
            or not proof["scope"]
        ):
            raise ValueError("explicit qualified physical sampling proof required")
        slope = Fraction(**proof["slope"])
        intercept = Fraction(**proof["intercept"])
        if slope != 1 or intercept != -2190:
            raise ValueError("this adapter requires the observed native-minus-2190 mapping")
        sources = proof["sources"]
        if set(sources) != {"native", "windows"}:
            raise ValueError("both actual source traces are required")
        for source in sources.values():
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                raise ValueError("absolute source evidence paths required")
            checked_json(source_path, source["sha256"])
        matches = proof["matches"]
        if not isinstance(matches, list) or len(matches) < 3:
            raise ValueError("fit and heldout physical samples required")
        counters, splits = [], []
        for item in matches:
            native = _integer(item["native_counter"], "proof native counter")
            race = _integer(item["windows_race_ms"], "proof race milliseconds")
            if native % 10 or race != slope * native + intercept:
                raise ValueError("physical proof contradicts its exact clock relationship")
            actual = struct.pack("<6f", *item["position"], *item["velocity"])
            if actual.hex() != item["physics6Hex"]:
                raise ValueError("physical signature is not its literal six float32 fields")
            counters.append(native)
            splits.append(item["split"])
        if len(set(counters)) != len(counters) or min(counters) != 2200:
            raise ValueError("unique observed counters including literal reset2200 required")
        if splits.count("fit") != proof["fit_matches"] or (
            splits.count("heldout") != proof["heldout_matches"]
            or not splits.count("fit")
            or not splits.count("heldout")
        ):
            raise ValueError("exact fit/heldout evidence counts required")
        if max(counters) - min(counters) != proof["sampled_race_span_ms"]:
            raise ValueError("proof span does not match its observed sample bounds")
        return cls(
            expected_sha256, min(counters), max(counters), slope, intercept, proof["scope"], sources
        )

    def require_interval(self, before: int, after: int) -> None:
        before, after = _integer(before, "before nativeT"), _integer(after, "after nativeT")
        if not self.minimum_native <= before < after <= self.maximum_native:
            raise ValueError("native interval exceeds observed physical clock proof scope")
        if before % 10 or after % 10:
            raise ValueError("actual physical 10-unit sample grid required")

    def race_milliseconds(self, native: int) -> int:
        native = _integer(native, "nativeT")
        if not self.minimum_native <= native <= self.maximum_native:
            raise ValueError("native timestamp exceeds observed physical clock proof scope")
        value = self.slope * native + self.intercept
        if value.denominator != 1:
            raise ValueError("actual integral race milliseconds required")
        return value.numerator

    def clock_spec(self) -> ClockSpec:
        return ClockSpec(
            clock_id="native-POST.nativeT/observed-minus-2190",
            unit="native-ticks",
            seconds_per_tick=0.001,
            semantics=CLOCK_SEMANTICS,
            evidence_sha256=self.sha256,
            synthetic=False,
        )

    def validate_timestamp(self, native: int) -> None:
        self.race_milliseconds(native)


def collect_contract(
    proof: NativeIntervalProof,
    reference: CommonPolylineReference,
) -> CommonTrainingContract:
    if reference.file_sha256 != KNOWN_REFERENCE_SHA256:
        raise ValueError("the physical proof is scoped to the pinned reference/map only")
    return CommonTrainingContract(
        clock=proof.clock_spec(),
        reward=reference.reward_spec(),
        reset_profile=IMMEDIATE_RESET_PROFILE,
        input_schedule=SCHEDULE,
        observation_semantics="tmrl.physical-common6/observed-map-reset/v1",
        observation_profile=COMMON6_PROFILE,
        purpose="offline-plumbing",
    )


def model_weights_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def qualified_native_contract(
    gate_path: Path,
    *,
    expected_sha256: str,
    bindings: Mapping[str, Any],
) -> CommonTrainingContract:
    """Verify selected gate bytes and every receipt artifact before model use."""
    gate = checked_json(gate_path, expected_sha256)
    if gate.get("schema") != "tmrl.common-paired-qualification/v1":
        raise ValueError("complete evidence-bound paired qualification gate required")
    native = CommonTrainingContract.from_envelope(gate["native_contract"])
    windows = CommonTrainingContract.from_envelope(gate["windows_contract"])
    hashes = gate["initial_model_sha256"]
    require_paired_comparison(
        (native, windows), initial_model_sha256=(hashes["native"], hashes["windows"])
    )
    if hashes["native"] != bindings["initial_model_sha256"]:
        raise ValueError("gate does not bind the actual initialized shared model weights")
    for contract in (native, windows):
        for field, digest in asdict(contract.evidence).items():
            artifact = gate["evidence"][field]
            if artifact["sha256"] != digest:
                raise ValueError("gate receipt does not match its paired evidence artifact")
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_absolute():
                raise ValueError("absolute paired qualification evidence paths required")
            checked_artifact(artifact_path, artifact["sha256"])
    scope = gate["scope"]
    required = {
        "map_sha256": KNOWN_MAP_SHA256,
        "reference_sha256": KNOWN_REFERENCE_SHA256,
        "clock_proof_sha256": bindings["clock_proof_sha256"],
        "native_iterations": bindings["native_iterations"],
        "seed": bindings["seed"],
    }
    if any(scope.get(key) != value for key, value in required.items()) or (
        type(scope.get("max_decisions")) is not int
        or scope["max_decisions"] < bindings["decisions"]
    ):
        raise ValueError("paired gate does not cover this map/time/action/seed/decision scope")
    return native


class NativeCommonAdapter:
    def __init__(
        self,
        contract: CommonTrainingContract,
        *,
        proof: NativeIntervalProof,
        reference: CommonPolylineReference,
    ) -> None:
        expected = collect_contract(proof, reference)
        for key in (
            "clock",
            "reward",
            "reset_profile",
            "input_schedule",
            "observation_semantics",
            "observation_profile",
        ):
            if canonical(asdict(contract)[key]) != canonical(asdict(expected)[key]):
                raise ValueError(
                    "native common adapter contract differs from physical evidence: " + key
                )
        self.contract, self.proof, self.reference = contract, proof, reference
        self.pipeline = CommonFeaturePipeline(contract)
        self._before: Mapping[str, Any] | None = None
        self._episode_id, self._step = "", 0

    def bind_reset(
        self,
        state: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        episode_id: str,
    ) -> torch.Tensor:
        if (
            state.get("schema") != "tmrl.native-immediate-state/v1"
            or state.get("resetProfile") != IMMEDIATE_RESET_PROFILE
            or state.get("inputSchedule") != SCHEDULE
            or state.get("nativeT") != 2200
            or info["runtimeProvenance"]["pins"]["map"] != KNOWN_MAP_SHA256
        ):
            raise ValueError("actual literal V7 reset and pinned map provenance required")
        self.proof.validate_timestamp(state["nativeT"])
        tensor = self.pipeline.transform_observation({"backend": "native-headless", "raw": state})
        self._before, self._episode_id, self._step = state, episode_id, 0
        return tensor

    def transition(self, action: int, result: Mapping[str, Any]) -> Transition:
        if self._before is None:
            raise ValueError("successful physical reset required before common transitions")
        before, after = self._before, result["observation"]
        if result.get("rawIntervalQualified") is not True or type(action) is not int:
            raise ValueError("actual qualified immediate input interval required")
        raw_result = dict(result)
        digest = raw_result.pop("rawTransitionSha256")
        raw_row = {
            "kind": "raw-transition",
            "before": before,
            "action": action,
            "result": raw_result,
        }
        if hashlib.sha256(canonical(raw_row).encode()).hexdigest() != digest:
            raise ValueError("raw transition differs from its journal SHA-256")
        self.proof.require_interval(before["nativeT"], after["nativeT"])
        events = result["authorization"]["nativeEventHex"]
        count = _integer(result["actualIterations"], "actualIterations")
        if (
            not 1 <= count <= 5
            or len(events) != count
            or (
                result["deltaNative"] != count * 10
                or after["nativeT"] - before["nativeT"] != count * 10
            )
        ):
            raise ValueError("actual POST count and elapsed physical interval disagree")
        for index, (event, wanted) in enumerate(zip(events, action_sgb(action), strict=False)):
            raw = bytes.fromhex(event)
            if (
                len(raw) != 256
                or raw[:8] != b"TMPOST03"
                or (struct.unpack_from("<I", raw, 112)[0] != before["nativeT"] + (index + 1) * 10)
            ):
                raise ValueError("literal contiguous applied POST events required")
            gas, brake, steer = struct.unpack_from("<3f", raw, 88)
            if struct.pack("<3f", steer, gas, brake) != struct.pack("<3f", *wanted):
                raise ValueError("issued action differs from literally applied controls")
        last = bytes.fromhex(events[-1])
        if (last[152:212] + last[88:104]).hex() != after["native76Hex"]:
            raise ValueError("actual final physical endpoint differs from the last POST")
        finished = result["terminated"]
        if finished and (
            result["outcome"].get("qualifiedBy") != "env002.full-original-validator+TMRESULT"
            or result["outcome"].get("terminated") is not True
            or result["outcome"]["computedResult"]["NbCheckpoints"] != 1
        ):
            raise ValueError("actual EOF finish requires the original validator and TMRESULT")
        current = {"backend": "native-headless", "raw": before}
        endpoint = {"backend": "native-headless", "raw": after}
        measured = self.reference.measure_features(
            (
                self.pipeline.transform_observation(current),
                self.pipeline.transform_observation(endpoint),
            ),
            finished=finished,
        )
        row = make_transition(
            CommonTransitionInput(
                observation=current,
                next_observation=None if finished else endpoint,
                action=action,
                before_timestamp=before["nativeT"],
                after_timestamp=after["nativeT"],
                reward_measurement=measured,
                terminated=finished,
                truncated=result["truncated"],
                raw_transition_sha256=digest,
                episode_id=self._episode_id,
                step=self._step,
            ),
            self.contract,
        )
        self._before = None if row.terminated or row.truncated else after
        self._step += 1
        return row


@dataclass(frozen=True, kw_only=True)
class ObservedControlSample:
    """Actual telemetry channels at an actual race sample, never issued values."""

    race_ms: int
    steer: float
    gas: float
    brake: float

    def __post_init__(self) -> None:
        _integer(self.race_ms, "observed input race sample")
        if not -1 <= finite_number(self.steer, "observed steer") <= 1 or any(
            not 0 <= finite_number(value, "observed pedal") <= 1 for value in (self.gas, self.brake)
        ):
            raise ValueError("actual observed control channel domains required")


@dataclass(frozen=True, kw_only=True)
class WindowsStepRecord:
    before_connect33: Sequence[float]
    after_connect33: Sequence[float]
    before_race_ms: int
    after_race_ms: int
    issued_action: int
    observed_controls: tuple[ObservedControlSample, ...]
    terminated: bool
    truncated: bool
    finished: bool
    raw_transition_sha256: str
    episode_id: str | None = None
    step: int | None = None


class WindowsCommonAdapter:
    """Convert measured Windows rows without inferring when commands took effect.

    The producer owns telemetry acquisition, actual episode endpoints and its raw
    journal. This class controls no game/UI and never turns command receipts into
    observed input. Paired input equivalence remains a separate evidence gate.
    """

    def __init__(
        self,
        contract: CommonTrainingContract,
        *,
        proof: PhysicalClockProof,
        reference: CommonPolylineReference,
    ) -> None:
        if (
            contract.observation_profile != COMMON6_PROFILE
            or contract.clock.unit != "milliseconds"
            or contract.clock.seconds_per_tick != 0.001
            or contract.clock.evidence_sha256 != proof.sha256
            or contract.clock.semantics != CLOCK_SEMANTICS
            or canonical(asdict(contract.reward)) != canonical(asdict(reference.reward_spec()))
        ):
            raise ValueError("explicit common6 Windows race-sample/projection contract required")
        self.contract, self.proof, self.reference = contract, proof, reference
        self.pipeline = CommonFeaturePipeline(contract)

    def transition(self, source: WindowsStepRecord) -> Transition:
        before, after = source.before_race_ms, source.after_race_ms
        self.proof.require_interval(
            _integer(before, "actual Windows before race_ms") + 2190,
            _integer(after, "actual Windows after race_ms") + 2190,
        )
        if not source.observed_controls:
            raise ValueError("actual observed controls required separately from issued actions")
        samples = [sample.race_ms for sample in source.observed_controls]
        if samples != sorted(set(samples)) or any(
            not before <= stamp <= after for stamp in samples
        ):
            raise ValueError("actual ordered observed controls must lie within measured endpoints")
        current = {"backend": "windows-connect33", "raw": source.before_connect33}
        endpoint = {"backend": "windows-connect33", "raw": source.after_connect33}
        measured = self.reference.measure_features(
            (
                self.pipeline.transform_observation(current),
                self.pipeline.transform_observation(endpoint),
            ),
            finished=source.finished,
        )
        row = make_transition(
            CommonTransitionInput(
                observation=current,
                next_observation=None if source.terminated else endpoint,
                action=source.issued_action,
                before_timestamp=before,
                after_timestamp=after,
                reward_measurement=measured,
                terminated=source.terminated,
                truncated=source.truncated,
                raw_transition_sha256=source.raw_transition_sha256,
                episode_id=source.episode_id,
                step=source.step,
            ),
            self.contract,
        )
        interval = {
            **row.info[COMMON_INTERVAL_KEY],
            "input_observation": {
                "issued_action": source.issued_action,
                "issued_native_interval_sgb": action_sgb(source.issued_action),
                "observed_controls": [asdict(sample) for sample in source.observed_controls],
                "actual_elapsed_race_ms": after - before,
                "applied_equals_issued_qualified": False,
                "note": "actual sample timing retained; no command-to-actuation time inferred",
            },
        }
        return replace(row, info={COMMON_INTERVAL_KEY: interval})
