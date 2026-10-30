"""A search-only environment with ranking independent of the reward clock.

Keep the original native provider, event admission, feature pipeline and historical
reward/termination rules. Do not apply the experimental autonomous learner's
terminal time correction: its final-50ms assumption failed in run1096. Search
selects exclusively on independently sourced actual Goal, never on this reward.
This factory must not be used to collect training demonstrations or train a model.
"""

from typing import Any

from trackmaniarl.experiments.shortcut_macro import (
    AutonomousMacroEnvironment,
    AutonomousMacroShortcutFactory,
)
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment,
    FullQrEnvironmentSettings,
    FullQrStep,
    FullQrTerminalStep,
    NativeObservationPipeline,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import NativeFullQrProvider
from trackmaniarl.trackmania.reward_types import RewardResult


class SequenceSearchEnvironment(AutonomousMacroEnvironment):
    def _observe_and_reward(
        self, step: FullQrStep | FullQrTerminalStep
    ) -> tuple[dict[str, Any], RewardResult]:
        return FullQrEnvironment._observe_and_reward(self, step)


class SequenceSearchFactory(AutonomousMacroShortcutFactory):
    def _create_environment(
        self,
        provider: NativeFullQrProvider,
        pipeline: NativeObservationPipeline,
        settings: FullQrEnvironmentSettings,
    ) -> FullQrEnvironment:
        return SequenceSearchEnvironment(provider, pipeline, settings)

    def load_demonstration(self, path: Any, pipeline: Any) -> list[Any]:
        raise RuntimeError("search-only factory; training data import forbidden")
