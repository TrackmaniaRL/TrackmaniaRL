"""Stdlib lifecycle tokens only: no process, socket, Torch, physics or replay."""

import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trackmaniarl.trackmania.native_headless import environment as facade
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.config import NATIVE_PROFILE, NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.protocols import (
    NativeExecutionDisabledError,
    NativeInconclusiveError,
    NativeLifecycleError,
    NativeObservationUnavailableError,
)


class BackendFixture:
    def __init__(self) -> None:
        self.capabilities = {"fixtureOnly": True}
        self.state = {
            "features": list(range(44)),
            "featureProfile": "tmre." + NATIVE_PROFILE,
            "nativeT": 2230,
            "fixtureOnly": True,
        }
        self.result: dict[str, Any] = {
            "status": "continue",
            "trainingEligible": True,
            "terminated": False,
            "truncated": False,
            "reward": 0.25,
            "observation": deepcopy(self.state),
        }
        self.reset_error: Exception | None = None
        self.step_error: Exception | None = None
        self.steps = 0
        self.close_calls = 0

    def reset(self) -> dict[str, Any]:
        if self.reset_error:
            raise self.reset_error
        return deepcopy(self.state)

    def step(self, action: int, native_iterations: int = 5) -> dict[str, Any]:
        del action, native_iterations
        self.steps += 1
        if self.step_error:
            raise self.step_error
        self.state = deepcopy(self.result["observation"])
        return deepcopy(self.result)

    def read_state(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def close(self) -> None:
        self.close_calls += 1


class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).absolute().parent
        self.config = NativeRuntimeConfig(
            runtime_root=root,
            trace_dir=root / "unused-traces",
            work_dir=root / "unused-work",
            asset_manifest=root / "not-read.json",
            allow_native=True,
        )
        self.backend = BackendFixture()
        self.verified = VerifiedAssets(NATIVE_PROFILE, root, {}, "fixture-sha")
        self.loader = patch.object(facade, "_create_backend", return_value=self.backend)
        self.verifier = patch.object(facade, "verify_assets", return_value=self.verified)
        self.create = self.loader.start()
        self.verify = self.verifier.start()
        self.addCleanup(self.loader.stop)
        self.addCleanup(self.verifier.stop)

    def test_inert_factory_and_explicit_native_opt_in(self) -> None:
        from dataclasses import replace

        factory = facade.NativeHeadlessEnvironmentFactory(replace(self.config, allow_native=False))
        env = factory.create(seed=7)
        assert not self.create.called
        assert not self.verify.called
        with pytest.raises(NativeExecutionDisabledError):
            env.reset()
        assert not self.create.called
        assert not self.verify.called
        env.close()
        assert self.backend.close_calls == 0

    def test_factory_leases_distinct_absolute_children_without_io(self) -> None:
        factory = facade.NativeHeadlessEnvironmentFactory(self.config)
        with patch("pathlib.Path.open", side_effect=AssertionError("construction must be inert")):
            instances = [
                factory.create(seed=7),
                factory.create(seed=7),
                factory.create(seed=8),
                facade.NativeHeadlessEnvironmentFactory(self.config).create(seed=7),
            ]
        traces = [env.config.trace_dir for env in instances]
        work = [env.config.work_dir for env in instances]
        assert len(set(traces)) == 4
        assert len(set(work)) == 4
        assert not self.create.called
        assert not self.verify.called
        for env in instances:
            assert env.config.trace_dir.is_absolute()
            assert env.config.work_dir.is_absolute()
            assert env.config.trace_dir.parent == self.config.trace_dir
            assert env.config.work_dir.parent == self.config.work_dir
            assert env.config.trace_dir.name == env.config.work_dir.name
            lease = env.config.trace_dir.name
            assert lease.startswith("env-")
            assert len(lease) == 36
            assert all(char in "0123456789abcdef" for char in lease[4:])
            assert env.config.runtime_root == self.config.runtime_root
            assert env.config.asset_manifest == self.config.asset_manifest
            env.close()
        assert self.backend.close_calls == 0

    def test_normal_raw_and_sdk_step_preserve_actual_fixture_fields(self) -> None:
        env = facade.NativeHeadlessEnvironmentFactory(self.config).create(seed=7)
        state, info = env.reset()
        assert info["requestedSeed"] == 7
        assert not info["seedChangesEngineReset"]
        assert state == self.backend.state
        assert info["assetManifestSha256"] == "fixture-sha"
        raw = env.step_native(36, native_iterations=3)
        assert raw == self.backend.result
        observed, reward, terminated, truncated, details = env.step(36)
        assert observed == state
        assert reward == 0.25
        assert not terminated
        assert not truncated
        assert details["nativeResult"] == raw
        raw["observation"]["features"][0] = 999
        assert env.last_result["observation"]["features"][0] == 0
        env.close()
        env.close()
        assert self.backend.close_calls == 1

    def test_real_terminal_absence_is_preserved_not_transformed_to_zeros(self) -> None:
        env = facade.NativeHeadlessEnvironment(self.config)
        env.reset()
        terminal = {
            "features": None,
            "featureProfile": "tmre." + NATIVE_PROFILE,
            "pendingSGB": None,
            "readerHex": None,
            "fixtureOnly": True,
        }
        self.backend.result.update(status="terminated", terminated=True, observation=terminal)
        state, _, terminated, truncated, _ = env.step(36)
        assert terminated
        assert not truncated
        assert state == terminal
        assert env.read_state()["features"] is None
        with pytest.raises(NativeObservationUnavailableError):
            facade.NativeFeaturePipeline().transform_observation(state)
        with pytest.raises(NativeLifecycleError):
            env.step(36)
        assert self.backend.steps == 1
        env.close()

    def test_inconclusive_flags_raise_with_raw_evidence(self) -> None:
        env = facade.NativeHeadlessEnvironment(self.config)
        env.reset()
        self.backend.result.update(status="inconclusive", terminated=None, truncated=None)
        with pytest.raises(NativeInconclusiveError) as caught:
            env.step(36)
        assert caught.value.raw_result == self.backend.result
        assert caught.value.raw_result == env.last_result
        with pytest.raises(NativeLifecycleError):
            env.step(36)
        env.close()

    def test_failed_reset_and_transport_do_not_silently_retry(self) -> None:
        env = facade.NativeHeadlessEnvironment(self.config)
        self.backend.reset_error = OSError("fixture reset failure")
        with pytest.raises(OSError, match=r"fixture.*failure"):
            env.reset()
        assert self.backend.close_calls == 1
        with pytest.raises(NativeLifecycleError):
            env.reset()
        self.backend.reset_error = None
        other = facade.NativeHeadlessEnvironment(self.config)
        other.reset()
        self.backend.step_error = OSError("fixture transport failure")
        with pytest.raises(OSError, match=r"fixture.*failure"):
            other.step(0)
        with pytest.raises(NativeLifecycleError):
            other.step(0)
        assert self.backend.steps == 1
        other.close()

    def test_invalid_actions_and_missing_current_features_are_rejected(self) -> None:
        env = facade.NativeHeadlessEnvironment(self.config)
        with pytest.raises(NativeLifecycleError):
            env.read_state()
        env.reset()
        for invalid in (-1, 78, True):
            with pytest.raises(ValueError, match="action must"):
                env.step(invalid)
        assert self.backend.steps == 0
        self.backend.result["observation"]["features"] = None
        with pytest.raises(NativeInconclusiveError):
            env.step(0)
        env.close()

    def test_feature_pipeline_accepts_only_finite_actual44(self) -> None:
        pipeline = facade.NativeFeaturePipeline()
        assert pipeline.transform_observation(self.backend.state) == tuple(
            float(x) for x in range(44)
        )
        for state in (
            {},
            {"features": None},
            {"features": [0] * 43},
            {"features": [float("nan")] * 44},
        ):
            with pytest.raises(NativeObservationUnavailableError):
                pipeline.transform_observation(state)
