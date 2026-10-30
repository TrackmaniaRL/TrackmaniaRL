"""Persisted semantics and evidence for the common8 training experiment.

Evidence hashes refer to externally audited raw artifacts. A hash's presence is
not independent verification of its contents. Synthetic data explicitly remains
offline plumbing and can never qualify a live learning comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from trackmaniarl.trackmania.parity import COMMON_FIELDS, COMMON_PROFILE, COMMON_SCALES

SCHEMA = "tmrl.common-kinematics.training/v1"
COMMON6_PROFILE = "tmrl.common-kinematics6/v1"
NATIVE_ONLY_CLOCK_SEMANTICS = "tmrl.native-source-nominal-simulation-duration/v1"
MODEL_SPEC = {
    "sdk_version": "1.2.14",
    "algorithm": "existing-sdk-fixed-qr-dqn",
    "encoder": "MlpSensorEncoder",
    "input_dim": 8,
    "feature_dim": 256,
    "temporal": "IdentityTemporalCore",
    "action_count": 78,
    "quantile_count": 32,
    "learning_rate": 0.0001,
    "target_update_interval": 1000,
    "target_tau": 0.0,
    "gradient_clip_norm": 10.0,
    "exploration_epsilon": 0.1,
    "online_risk": "neutral",
    "evaluation_risk": "neutral",
    "optimizer": "torch.optim.Adam/defaults",
    "value_rescaling": False,
    "n_step": 1,
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_digest(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name}: lowercase SHA-256 of actual evidence required")


def finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: explicit finite number required")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class ClockSpec:
    """An actual timestamp source, not a step-count or wall-clock substitute."""

    clock_id: str
    unit: str
    seconds_per_tick: float
    semantics: str
    evidence_sha256: str
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.clock_id or not self.semantics or type(self.synthetic) is not bool:
            raise ValueError("explicit clock identity, semantics and synthetic flag required")
        scale = finite_number(self.seconds_per_tick, "seconds_per_tick")
        if scale <= 0:
            raise ValueError("positive qualified clock conversion required")
        fixed = {"seconds": 1.0, "milliseconds": 0.001, "microseconds": 1e-6}
        if self.unit not in (*fixed, "native-ticks"):
            raise ValueError("explicit known timestamp unit required")
        if self.unit in fixed and scale != fixed[self.unit]:
            raise ValueError("timestamp unit and conversion disagree")
        require_digest(self.evidence_sha256, "clock mapping")

    def elapsed(self, before: Any, after: Any) -> float:
        # Subtract integers before conversion to avoid losing precision at large stamps.
        finite_number(before, "before timestamp")
        finite_number(after, "after timestamp")
        seconds = (after - before) * self.seconds_per_tick
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("strictly increasing actual timestamps required")
        return float(seconds)


@dataclass(frozen=True, kw_only=True)
class QualificationEvidence:
    reset_sha256: str | None = None
    applied_inputs_sha256: str | None = None
    observation_semantics_sha256: str | None = None
    clock_mapping_sha256: str | None = None
    reward_semantics_sha256: str | None = None

    def require_complete(self) -> None:
        for name, value in asdict(self).items():
            require_digest(value, name)


@dataclass(frozen=True, kw_only=True)
class RewardMeasurement:
    """Actual projection results from the same pinned reference geometry."""

    progress_before_units: float
    progress_after_units: float
    deviation_after_units: float
    finished: bool

    def __post_init__(self) -> None:
        for name in ("progress_before_units", "progress_after_units", "deviation_after_units"):
            finite_number(getattr(self, name), name)
        if self.deviation_after_units < 0 or type(self.finished) is not bool:
            raise ValueError("nonnegative measured deviation and actual finish flag required")


@dataclass(frozen=True, kw_only=True)
class ProjectionLimits:
    """Optional rejection limits in authored coordinates; no implicit physics."""

    max_deviation_units: float | None = None
    max_progress_delta_units: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and finite_number(value, name) <= 0:
                raise ValueError("positive explicit projection rejection limits required")


@dataclass(frozen=True, kw_only=True)
class RewardSpec:
    """One explicitly shared reward; neither backend's opaque reward is used."""

    reference_sha256: str
    projection_profile: str
    coordinate_unit: str = "engine-world-unit"
    projection_limits: ProjectionLimits = field(default_factory=ProjectionLimits)
    profile: str = "tmrl.common-progress-deviation-time-finish/v1"
    progress_per_unit: float = 0.1
    time_per_second: float = 0.2
    deviation_per_unit: float = 0.001
    deviation_cap_units: float = 50.0
    finish_bonus: float = 20.0

    def __post_init__(self) -> None:
        require_digest(self.reference_sha256, "reward reference")
        if (
            not self.projection_profile
            or not self.coordinate_unit
            or self.profile != ("tmrl.common-progress-deviation-time-finish/v1")
        ):
            raise ValueError("explicit supported projection/reward profile required")
        for name in (
            "progress_per_unit",
            "time_per_second",
            "deviation_per_unit",
            "deviation_cap_units",
            "finish_bonus",
        ):
            if finite_number(getattr(self, name), name) < 0:
                raise ValueError("nonnegative reward coefficients required")

    def score(self, measurement: RewardMeasurement, elapsed_s: float) -> float:
        if finite_number(elapsed_s, "reward duration") <= 0:
            raise ValueError("positive actual reward duration required")
        limits = self.projection_limits
        if limits.max_deviation_units is not None and (
            measurement.deviation_after_units > limits.max_deviation_units
        ):
            raise ValueError("actual projection exceeds declared deviation rejection limit")
        if (
            limits.max_progress_delta_units is not None
            and abs(measurement.progress_after_units - measurement.progress_before_units)
            > limits.max_progress_delta_units
        ):
            raise ValueError("actual projection exceeds declared progress rejection limit")
        return (
            self.progress_per_unit
            * (measurement.progress_after_units - measurement.progress_before_units)
            - self.time_per_second * elapsed_s
            - self.deviation_per_unit
            * min(measurement.deviation_after_units, self.deviation_cap_units)
            + self.finish_bonus * measurement.finished
        )


@dataclass(frozen=True, kw_only=True)
class CommonTrainingContract:
    clock: ClockSpec
    reward: RewardSpec
    reset_profile: str
    input_schedule: str
    observation_semantics: str
    observation_profile: str = COMMON_PROFILE
    evidence: QualificationEvidence = field(default_factory=QualificationEvidence)
    purpose: str = "offline-plumbing"
    gamma: float = 0.995
    discount_reference_seconds: float = 0.05
    updates_per_reference: float = 0.25

    def __post_init__(self) -> None:
        if not all((self.reset_profile, self.input_schedule, self.observation_semantics)):
            raise ValueError("explicit reset/input/observation semantic profiles required")
        if self.purpose not in ("offline-plumbing", "native-only-training", "qualified-training"):
            raise ValueError("explicit common training purpose required")
        if self.observation_profile not in (COMMON_PROFILE, COMMON6_PROFILE):
            raise ValueError("explicit common6 or common8 observation profile required")
        if not 0 < finite_number(self.gamma, "gamma") < 1:
            raise ValueError("gamma must be strictly between zero and one")
        if finite_number(self.discount_reference_seconds, "discount reference") <= 0:
            raise ValueError("positive discount reference required")
        if finite_number(self.updates_per_reference, "update rate") <= 0:
            raise ValueError("positive update rate required")
        if self.purpose == "native-only-training" and (
            self.clock.synthetic
            or self.clock.unit != "native-ticks"
            or self.clock.semantics != NATIVE_ONLY_CLOCK_SEMANTICS
        ):
            raise ValueError("native-only training requires its independent native source clock")
        if self.purpose == "qualified-training":
            self.require_qualified()

    def require_qualified(self) -> None:
        if self.clock.synthetic:
            raise ValueError("synthetic clocks cannot qualify physical training/comparison")
        self.evidence.require_complete()
        if self.evidence.clock_mapping_sha256 != self.clock.evidence_sha256:
            raise ValueError("clock evidence must bind the declared timestamp conversion")

    def envelope(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "observation": {
                "profile": self.observation_profile,
                "fields": COMMON_FIELDS[: self.feature_dim],
                "scales": COMMON_SCALES[: self.feature_dim],
            },
            "action_profile": "discrete78/v1",
            "model": {**MODEL_SPEC, "input_dim": self.feature_dim},
            **asdict(self),
        }
        return {**body, "sha256": sha256(body)}

    @property
    def feature_dim(self) -> int:
        return 6 if self.observation_profile == COMMON6_PROFILE else 8

    @property
    def digest(self) -> str:
        return str(self.envelope()["sha256"])

    def require_checkpoint(self, saved: Mapping[str, Any]) -> None:
        if canonical(saved) != canonical(self.envelope()):
            raise ValueError(
                "incompatible common observation/reset/input/reward/time/model checkpoint"
            )

    @classmethod
    def from_envelope(cls, saved: Mapping[str, Any]) -> CommonTrainingContract:
        """Parse a complete, self-consistent manifest; no clocks receive defaults."""
        values = dict(saved)
        for key in ("schema", "observation", "action_profile", "model", "sha256"):
            values.pop(key, None)
        values["clock"] = ClockSpec(**values["clock"])
        reward = dict(values["reward"])
        reward["projection_limits"] = ProjectionLimits(**reward["projection_limits"])
        values["reward"] = RewardSpec(**reward)
        values["evidence"] = QualificationEvidence(**values["evidence"])
        result = cls(**values)
        result.require_checkpoint(saved)
        return result

    def semantic_body(self) -> dict[str, Any]:
        body = self.envelope()
        for key in ("sha256", "clock", "evidence"):
            body.pop(key)
        body["clock_semantics"] = self.clock.semantics
        return body


def require_paired_comparison(
    contracts: tuple[CommonTrainingContract, CommonTrainingContract],
    *,
    initial_model_sha256: tuple[str, str],
) -> None:
    """Check semantic eligibility; raw evidence still needs an independent audit."""
    first, second = contracts
    for contract in contracts:
        if contract.purpose != "qualified-training":
            raise ValueError(
                "offline plumbing/native-only training cannot qualify a learning comparison"
            )
        contract.require_qualified()
    if canonical(first.semantic_body()) != canonical(second.semantic_body()):
        raise ValueError("paired model/reward/reset/input/observation/time/budget contracts differ")
    for digest in initial_model_sha256:
        require_digest(digest, "initial model")
    if initial_model_sha256[0] != initial_model_sha256[1]:
        raise ValueError("paired runs must start from identical model weights")
