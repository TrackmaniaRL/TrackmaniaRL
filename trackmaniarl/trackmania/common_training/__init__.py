"""Explicit shared common8 training, independent of either engine's lifecycle.

Importing this package neither starts an engine nor qualifies physical equivalence.
Use :func:`require_paired_comparison` before making cross-engine learning claims.
"""

from trackmaniarl.trackmania.common_training.contract import (
    COMMON6_PROFILE,
    ClockSpec,
    CommonTrainingContract,
    ProjectionLimits,
    QualificationEvidence,
    RewardMeasurement,
    RewardSpec,
    require_paired_comparison,
)
from trackmaniarl.trackmania.common_training.features import (
    COMMON_INTERVAL_KEY,
    CommonFeaturePipeline,
    CommonTransitionInput,
    make_transition,
    validate_transition,
)
from trackmaniarl.trackmania.common_training.geometry import (
    PROJECTION_PROFILE,
    CommonPolylineReference,
    Projection,
)
from trackmaniarl.trackmania.common_training.learning import CommonLearner, make_learner
from trackmaniarl.trackmania.common_training.replay import (
    CommonReplayStore,
    CommonSampler,
    DurationUpdateBudget,
)

__all__ = [
    "COMMON6_PROFILE",
    "COMMON_INTERVAL_KEY",
    "PROJECTION_PROFILE",
    "ClockSpec",
    "CommonFeaturePipeline",
    "CommonLearner",
    "CommonPolylineReference",
    "CommonReplayStore",
    "CommonSampler",
    "CommonTrainingContract",
    "CommonTransitionInput",
    "DurationUpdateBudget",
    "Projection",
    "ProjectionLimits",
    "QualificationEvidence",
    "RewardMeasurement",
    "RewardSpec",
    "make_learner",
    "make_transition",
    "require_paired_comparison",
    "validate_transition",
]
