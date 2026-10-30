"""Synthetic SDK integration, never native source qualification."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from tests.unit.experiments.test_graph_iqn_v5 import _geometry
from trackmaniarl.algorithms import DiscreteValueLearner, StableDiscreteSoftActorCritic
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import tree_collate
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics, summarize_episode
from trackmaniarl.models.encoders import StructuredMlpSensorEncoder
from trackmaniarl.models.factory import CompositeValueModelFactory
from trackmaniarl.models.sensor_actor_critic import SensorActorCriticModelFactory
from trackmaniarl.trackmania.diagnostics import ProgressBinDiagnostics
from trackmaniarl.trackmania.native_headless import NativeTelemetryEnvironment
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment,
    FullQrEnvironmentSettings,
    FullQrReset,
    FullQrStep,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGraphFeaturePipeline,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
from trackmaniarl.trackmania.reward_config import RewardConfig


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_structured_mlp_preserves_axes_and_uses_canonical_sensor_order(
    leading: tuple[int, ...],
) -> None:
    encoder = StructuredMlpSensorEncoder({"b": [3], "a": [2, 2]}, 8, hidden_dim=16)
    observation = {"a": torch.randn(*leading, 2, 2), "b": torch.randn(*leading, 3)}
    actual = encoder(observation)
    assert actual.shape == (*leading, 8)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, encoder(dict(reversed(list(observation.items())))), rtol=0, atol=0
    )
    actual.sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    with pytest.raises(ValueError, match="exactly"):
        encoder({"a": observation["a"]})
    with pytest.raises(ValueError, match="shape differs"):
        encoder({**observation, "b": torch.zeros(*leading, 4)})


@pytest.mark.parametrize(
    "invalid",
    [
        {"bad": np.array([float("nan")])},
        {"bad": np.float32(float("inf"))},
        {"bad": {}},
        {"bad": []},
        {"good": 1.0, "bad": ()},
        {"bad": np.empty(0)},
        {"bad": torch.empty(0)},
        {"bad": "not numeric"},
        {"bad": np.array([np.datetime64("2026-01-01")])},
        {"bad": np.array([np.timedelta64(5, "ms")])},
        {"bad": np.timedelta64(5, "ms")},
    ],
)
def test_custom_features_fail_before_reward_consumption(
    monkeypatch: pytest.MonkeyPatch,
    invalid: dict[str, Any],
) -> None:
    environment, provider, pipeline = setup()
    environment.reset()
    monkeypatch.setattr(pipeline, "transform_observation", lambda _: invalid)
    with pytest.raises(NativeLifecycleError, match="no transition"):
        environment.step(39)
    assert provider.closed == 1
    assert provider.events.consumed_through_ms == 0


def test_numpy_scalar_observations_are_accepted_on_reset_and_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, provider, pipeline = setup()
    monkeypatch.setattr(
        pipeline,
        "transform_observation",
        lambda _: {"speed": np.float32(1.25), "gear": np.int64(2)},
    )
    initial, _ = environment.reset()
    observed, _, _, _, _ = environment.step(39)
    assert initial["speed"] == observed["speed"] == np.float32(1.25)
    assert initial["gear"] == observed["gear"] == np.int64(2)
    assert provider.events.consumed_through_ms == 50
    environment.close()


def test_boolean_arrays_and_complex_values_remain_collatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, provider, pipeline = setup()
    monkeypatch.setattr(
        pipeline,
        "transform_observation",
        lambda frame: {
            "boolean": np.array([True, False], dtype=np.bool_),
            "boolean_scalar": np.bool_(frame.channels[0] == 0),
            "complex_array": np.array([1 + 2j], dtype=np.complex64),
            "complex_scalar": np.complex64(3 + 4j),
        },
    )
    initial, _ = environment.reset()
    observed, _, _, _, _ = environment.step(0)
    batch = tree_collate([initial, observed])
    assert batch["boolean"].dtype == torch.bool
    assert batch["boolean_scalar"].dtype == torch.bool
    torch.testing.assert_close(batch["boolean_scalar"], torch.tensor([True, False]))
    assert batch["complex_array"].dtype == torch.complex64
    assert batch["complex_scalar"].dtype == torch.complex64
    assert provider.events.consumed_through_ms == 50
    environment.close()


def test_observations_own_reused_numpy_and_torch_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, provider, pipeline = setup()
    numpy_buffer = np.zeros(2, dtype=np.float32)
    tensor_buffer = torch.zeros(2, dtype=torch.float32)

    def reused_buffers(frame: NativeGraphFrame) -> dict[str, Any]:
        numpy_buffer.fill(frame.channels[0])
        tensor_buffer.fill_(frame.channels[0])
        return {"nested": ({"numpy": numpy_buffer}, [tensor_buffer])}

    monkeypatch.setattr(pipeline, "transform_observation", reused_buffers)
    initial, _ = environment.reset()
    first, _, _, _, _ = environment.step(0)
    second, _, _, _, _ = environment.step(0)
    for observation, expected in ((initial, 0), (first, 50), (second, 100)):
        nested = observation["nested"]
        assert isinstance(nested, tuple)
        assert isinstance(nested[0], dict)
        assert isinstance(nested[1], list)
        assert nested[0]["numpy"].dtype == np.float32
        assert nested[1][0].dtype == torch.float32
        np.testing.assert_array_equal(nested[0]["numpy"], [expected, expected])
        torch.testing.assert_close(nested[1][0], torch.full((2,), expected, dtype=torch.float32))
    first["nested"][0]["numpy"][0] = -1
    first["nested"][1][0][0] = -1
    assert numpy_buffer[0] == 100
    assert tensor_buffer[0].item() == 100
    assert initial["nested"][0]["numpy"][0] == 0
    assert second["nested"][1][0][0].item() == 100
    assert provider.events.consumed_through_ms == 100
    environment.close()


def _native_learner(algorithm: str, encoder: str) -> tuple[Any, torch.nn.Module]:
    shapes = {"physics": [60], "track": [3, 88], "context": [29], "recovery": [8]}
    dimension = 192 if encoder == "graph" else 32
    component = (
        {"class_path": "trackmaniarl.experiments.full_qr.components:BatchedIncidentEncoder"}
        if encoder == "graph"
        else {
            "class_path": "trackmaniarl.models.encoders:StructuredMlpSensorEncoder",
            "kwargs": {"input_shapes": shapes, "output_dim": dimension, "hidden_dim": 32},
        }
    )
    if algorithm == "sd-sac":
        model = SensorActorCriticModelFactory(
            algorithm, component, {"feature_dim": dimension, "action_count": 78}
        ).build()
        learner = StableDiscreteSoftActorCritic(model, execution={"device": "cpu"})
    else:
        heads = {"dqn": "ScalarQHead", "qr": "FixedQuantileHead", "iqn": "ImplicitQuantileHead"}
        strategies = {
            "dqn": "ScalarValueStrategy",
            "qr": "FixedQuantileStrategy",
            "iqn": "RandomQuantileStrategy",
        }
        head_config = {"feature_dim": dimension, "action_count": 78}
        strategy_config = {}
        if algorithm == "qr":
            head_config["quantile_count"] = 8
            strategy_config = {"quantile_count": 8}
        elif algorithm == "iqn":
            strategy_config = {"train_quantile_count": 8, "target_quantile_count": 8}
        model = CompositeValueModelFactory(
            encoder=component,
            temporal={
                "class_path": "trackmaniarl.models.temporal:IdentityTemporalCore",
                "kwargs": {"input_dim": dimension},
            },
            head={
                "class_path": f"trackmaniarl.models.heads:{heads[algorithm]}",
                "kwargs": head_config if algorithm == "dqn" else {"config": head_config},
            },
            strategy={
                "class_path": f"trackmaniarl.models.strategies:{strategies[algorithm]}",
                "kwargs": strategy_config,
            },
        ).build()
        learner = DiscreteValueLearner(model, execution={"device": "cpu"})
    return learner, model


@pytest.mark.parametrize("algorithm", ["dqn", "qr", "iqn", "sd-sac"])
@pytest.mark.parametrize("encoder", ["graph", "mlp"])
def test_native_observations_support_discrete_learners_and_encoders(
    algorithm: str, encoder: str, tmp_path: Path
) -> None:
    """Synthetic SDK transitions test model interchange, not native qualification."""
    environment, _, _ = setup()
    assert isinstance(environment, NativeTelemetryEnvironment)
    environment.pipeline = NativeGraphFeaturePipeline(_geometry(tmp_path / "g.npz"), "test-map")
    learner, model = _native_learner(algorithm, encoder)
    learner.setup({"seed": 17})
    try:
        observation, _ = environment.reset(seed=17)
        next_observation, reward, terminated, truncated, _ = environment.step(39)

        def stack(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {name: torch.stack([value] * 4) for name, value in values.items()}

        batch = TrainingBatch(
            data={},
            observations=stack(observation),
            next_observations=stack(next_observation),
            actions=torch.full((4,), 39),
            rewards=torch.full((4,), reward),
            terminated=torch.full((4,), terminated),
            truncated=torch.full((4,), truncated),
            bootstrap_discounts=torch.full((4,), 0.995),
            transition_ids=list(range(4)),
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics, _ = learner.update(batch)
        assert metrics
        assert all(np.isfinite(value) for value in metrics.values())
        assert any(
            not torch.equal(old, current)
            for old, current in zip(before, model.parameters(), strict=True)
        )
        action = int(learner.policy().act(next_observation, PolicyMode.EVALUATION))
        assert 0 <= action < 78
        environment.step(action)
    finally:
        environment.close()


def frame(clock: int) -> NativeGraphFrame:
    return NativeGraphFrame(
        (clock, clock / 20, 0, 0, 50, 0, 0, 1, 0, 0, 50, 5000, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0)
    )


class Pipeline:
    def reset_episode(self) -> None:
        self.times = []

    def transform_observation(self, value: NativeGraphFrame) -> dict[str, np.ndarray]:
        self.times.append(value.channels[0])
        return {"synthetic": np.asarray(value.channels, dtype=np.float32)}


class Provider:
    def __init__(self) -> None:
        self.closed = 0
        self.generation = 0
        self.clock = 0
        self.coverage = True
        self.foreign = False
        self.eof = False
        self.reuse = False
        self.finish = False

    def reset(self, *, seed: int | None = None) -> FullQrReset:
        if not self.reuse:
            self.generation += 1
        self.clock = 0
        self.identity = EventStreamIdentity(
            "foreign" if self.foreign else "actor0", str(self.generation)
        )
        self.events = RewardEventCollector(self.identity)
        return FullQrReset(frame(0), self.events)

    def advance(self, action: int) -> FullQrStep | None:
        self.clock += 50
        if self.eof:
            return None
        self.events.record_motor(
            self.identity, sequence=self.clock // 50, race_time_ms=self.clock, large_motor=255
        )
        self.events.record_ui_boundary(
            self.identity, race_time_ms=self.clock, finish_active=self.finish
        )
        if self.coverage:
            self.events.seal_motor_through(self.identity, self.clock)
        return FullQrStep(frame(self.clock), 0.0)

    def close(self) -> None:
        self.closed += 1


def setup() -> tuple[FullQrEnvironment, Provider, Pipeline]:
    provider, pipeline = Provider(), Pipeline()
    trajectory = np.zeros((10_001, 3), dtype=np.float32)
    trajectory[:, 0] = np.arange(len(trajectory))
    environment = FullQrEnvironment(
        provider,
        pipeline,
        FullQrEnvironmentSettings(
            trajectory,
            RewardConfig(maximum_race_time_s=150, collision_penalty=0.15, collision_cooldown_s=0),
            actor="actor0",
        ),
    )
    return environment, provider, pipeline


def test_sdk_mapping_collision_delivery_and_fresh_reset_history() -> None:
    environment, provider, pipeline = setup()
    initial, info = environment.reset(seed=17)
    assert initial["synthetic"].shape == (23,)
    assert info["episode"] == "1"
    observation, reward, terminated, truncated, info = environment.step(77)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["collision_detected"]
    assert info["reward_collision"] == -0.15
    assert observation["synthetic"][0] == 50
    assert pipeline.times == [0, 50]
    assert provider.events.consumed_through_ms == 50
    environment.reset()
    assert pipeline.times == [0]
    assert provider.events.consumed_through_ms == 0
    assert environment.step(0)[4]["episode"] == "2"
    environment.close()
    environment.close()
    assert provider.closed == 1


@pytest.mark.parametrize("finish", [False, True])
def test_actor_metrics_retain_actual_reward_components_and_collisions(*, finish: bool) -> None:
    environment, provider, _ = setup()
    # Let this short synthetic trace exercise the actual Finish reward path.
    environment.reward_config = replace(
        environment.reward_config, minimum_finish_steps=1, finish_progress=0.0001
    )
    metrics = EpisodeMetrics(ProgressBinDiagnostics(78, bin_count=20))
    try:
        environment.reset()
        provider.finish = finish
        _, reward, terminated, _, info = environment.step(41)
        metrics.record_reward(reward, info)
        summary = summarize_episode(
            reward, metrics.summary_info(epsilon=0.0, version=0, transitions=1), 1
        )
        components = {key: value for key, value in summary.items() if key.startswith("reward/")}
        assert sum(components.values()) == pytest.approx(reward)
        assert components["reward/time"] < 0
        assert bool(components["reward/pbrs"]) is (not finish)
        assert components["reward/progress"] > 0
        assert components["reward/terminal"] == (30 if finish else 0)
        assert terminated is finish
        assert info["collision"] is True
        assert info["collision_detected"] is True
        assert summary["collision/count"] == summary["collision/detected_count"] == 1
        assert info["accepted_progress_delta_m"] > 0
        assert info["projected_velocity_mps"] > 0
    finally:
        environment.close()


@pytest.mark.parametrize("fault", ["coverage", "eof"])
def test_missing_source_or_eof_is_fatal_without_transition(fault: str) -> None:
    environment, provider, _ = setup()
    environment.reset()
    if fault == "coverage":
        provider.coverage = False
    else:
        provider.eof = True
    with pytest.raises(NativeLifecycleError, match="no transition returned"):
        environment.step(0)
    assert provider.closed == 1
    assert provider.events.consumed_through_ms == 0
    with pytest.raises(NativeLifecycleError):
        environment.reset()


@pytest.mark.parametrize("fault", ["foreign", "reuse"])
def test_reset_rejects_foreign_or_reused_episode(fault: str) -> None:
    environment, provider, _ = setup()
    environment.reset()
    setattr(provider, fault, True)
    with pytest.raises(NativeLifecycleError, match="reset failed"):
        environment.reset()
    assert provider.closed == 1


def test_bad_action_does_not_advance_engine_or_close_valid_episode() -> None:
    environment, provider, _ = setup()
    environment.reset()
    for action in (True, np.bool_(True), -1, 78, 1.0, np.int64(-1), np.int32(78)):
        with pytest.raises(ValueError, match="78 actions"):
            environment.step(action)
    assert provider.clock == 0
    assert provider.closed == 0
    environment.step(0)
    assert provider.clock == 50
    environment.close()


@pytest.mark.parametrize(
    "action", [np.int32(0), np.int64(77), gym.spaces.Discrete(78, seed=17).sample()]
)
def test_numpy_integer_actions_reach_provider_as_python_int(
    action: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment, provider, _ = setup()
    environment.reset()
    advance = provider.advance
    calls = []

    def record_advance(received: int) -> FullQrStep | None:
        calls.append(received)
        return advance(received)

    monkeypatch.setattr(provider, "advance", record_advance)
    environment.step(action)
    assert calls == [int(action)]
    assert type(calls[0]) is int
    assert provider.events.consumed_through_ms == 50
    environment.close()


@pytest.mark.parametrize("invalid_seed", [True, False, np.bool_(True), 1.0, "not an integer"])
def test_invalid_seed_does_not_close_or_reset_a_valid_episode(invalid_seed: Any) -> None:
    environment, provider, _ = setup()
    with pytest.raises(ValueError, match="seed must be an integer"):
        environment.reset(seed=invalid_seed)
    assert provider.generation == 0
    assert provider.closed == 0
    environment.reset(seed=17)
    with pytest.raises(ValueError, match="seed must be an integer"):
        environment.reset(seed=invalid_seed)
    assert provider.generation == 1
    assert provider.closed == 0
    environment.step(0)
    environment.close()


@pytest.mark.parametrize(
    "seed", [np.int32(-1), np.int64(12), np.random.default_rng(42).integers(100)]
)
def test_numpy_seed_reaches_provider_as_python_int(
    monkeypatch: pytest.MonkeyPatch, seed: Any
) -> None:
    environment, provider, _ = setup()
    original_reset = provider.reset
    received = []

    def record_seed(*, seed: int | None = None) -> FullQrReset:
        received.append(seed)
        return original_reset(seed=seed)

    monkeypatch.setattr(provider, "reset", record_seed)
    environment.reset(seed=seed)
    assert received == [int(seed)]
    assert type(received[0]) is int
    environment.close()


def test_interrupt_preserves_graceful_sdk_shutdown_and_closes_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, provider, _ = setup()
    environment.reset()

    def interrupt(action: int) -> FullQrStep:
        raise KeyboardInterrupt

    monkeypatch.setattr(provider, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        environment.step(0)
    assert provider.closed == 1
    assert provider.events.consumed_through_ms == 0
