"""Experimental, explicitly configured OpenPlanet map-switching environment.

This backend requires a separately installed protocol-3 plugin and exclusive
control of the game instance. It has not yet been validated against a real game.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from trackmaniarl.core.runtime import _resolve_evaluation_map
from trackmaniarl.distributed.evaluation_contract import validate_pipeline_binding
from trackmaniarl.trackmania.bound_telemetry import BoundOpenPlanetClient
from trackmaniarl.trackmania.environment import OpenPlanetEnvironment, OpenPlanetEnvironmentFactory
from trackmaniarl.trackmania.map_staging import StagedAssets
from trackmaniarl.trackmania.map_switch_controller import (
    MapSwitchController,
    OpenPlanetSwitchTransport,
)
from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal
from trackmaniarl.trackmania.reward_points import load_reward_asset
from trackmaniarl.trackmania.telemetry import OpenPlanetClientConfig

_BOUND_TELEMETRY_PORT = 9002


class _TrainingSession:
    """Own the game controller and the instance lock throughout collection."""

    def __init__(  # noqa: PLR0913 -- explicit session-owned resources
        self,
        environment: OpenPlanetEnvironment,
        switch: MapSwitchController,
        uid: str,
        staged: StagedAssets,
        training_path: Path,
        training_asset: Any,
        release_suite_cache: Callable[[], None],
    ) -> None:
        self.environment, self.switch, self.uid = environment, switch, uid
        self.staged, self.training_path, self.training_asset = staged, training_path, training_asset
        self.release_suite_cache = release_suite_cache
        self._failed = False
        self._closed = False
        self._close_complete = False
        self._evaluation_active = False
        self._evaluation_open = False

    def reset(self, *, seed: int | None = None) -> Any:
        self._require_training_ready()
        try:
            self.training_asset.validate_map(self.training_path)
            # A plugin restart invalidates the switch lease even if protocol 2
            # reports the same UID. Do not issue reset controls in that state.
            # At a finish screen the player need not be ready until after reset;
            # OpenPlanetEnvironment confirms readiness once reset has completed.
            self.switch.verify_active_map(self.uid, require_player=False)
            return self.environment.reset(seed=seed)
        except BaseException as error:
            self.fail_closed(error)
            raise

    def step(self, action: Any) -> Any:
        self._require_training_ready()
        try:
            return self.environment.step(action)
        except BaseException as error:
            self.fail_closed(error)
            raise

    def _require_active(self) -> None:
        if self._failed or self._closed:
            raise RuntimeError("multi-map session is quarantined or closed")

    def _require_training_ready(self) -> None:
        self._require_active()
        if self._evaluation_active:
            raise RuntimeError("training is suspended until the training map is restored")

    def fail_closed(self, error: BaseException) -> None:
        """Neutralize control and keep the already armed session guard."""
        self._failed = True
        try:
            self.release_controls()
        except Exception as release_error:
            error.add_note(f"Failed to release controls: {release_error!r}")
        try:
            self.switch.quarantine_uncertain_telemetry()
        except Exception as quarantine_error:
            error.add_note(f"Failed to quarantine telemetry: {quarantine_error!r}")

    def release_controls(self) -> None:
        self.environment.release_controls()

    def close(self) -> None:
        if self._close_complete:
            return
        self._closed = True
        incomplete_evaluation = self._evaluation_active and not self._failed
        if incomplete_evaluation:
            self._failed = True
        errors: list[BaseException] = []

        def cleanup(action: Callable[[], Any]) -> None:
            try:
                action()
            except BaseException as error:
                errors.append(error)

        cleanup(self.release_controls)
        cleanup(self.environment.close)
        cleanup(self.release_suite_cache)
        cleanup(self.staged.close)
        if incomplete_evaluation:
            errors.append(RuntimeError("evaluation closed before training map restoration"))
        if not self._failed and not errors:
            cleanup(lambda: self.training_asset.validate_map(self.training_path))
            if not errors:
                cleanup(lambda: self.switch.verify_active_map(self.uid, require_player=False))
            if not errors:
                cleanup(self.switch.finish_evaluation_guard)
        cleanup(self.switch.close)
        if errors:
            self._failed = True
            first, *remaining = errors
            for additional in remaining:
                first.add_note(f"Additional session cleanup failure: {additional!r}")
            raise first
        self._close_complete = True


class _EvaluationSession:
    """Guard evaluation calls even when a caller catches their errors."""

    def __init__(self, environment: OpenPlanetEnvironment, training: _TrainingSession) -> None:
        self._environment, self._training = environment, training
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def reset(self, *, seed: int | None = None) -> Any:
        self._require_active()
        try:
            return self._environment.reset(seed=seed)
        except BaseException as error:
            self._training.fail_closed(error)
            raise

    def step(self, action: Any) -> Any:
        self._require_active()
        try:
            return self._environment.step(action)
        except BaseException as error:
            self._training.fail_closed(error)
            raise

    def _require_active(self) -> None:
        self._training._require_active()
        if self._closed:
            raise RuntimeError("evaluation session is closed")

    def close(self) -> None:
        """Close evaluation telemetry without closing the shared controller."""
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            self._training.release_controls()
        except BaseException as release_error:
            error = release_error
        try:
            self._environment.client.close()
        except BaseException as close_error:
            if error is None:
                error = close_error
            else:
                error.add_note(f"Failed to close evaluation telemetry: {close_error!r}")
        if error is not None:
            self._training.fail_closed(error)
            raise error


class OpenPlanetMultiMapEnvironmentFactory:
    """Opt-in map switching, requiring a pre-provisioned durable journal.

    `instance_id` and `journal_path` must uniquely and persistently identify the
    exclusively owned game. A missing journal is never silently recreated.
    """

    def __init__(  # noqa: PLR0913 -- each identity and transport parameter is explicit
        self,
        config: Any,
        *,
        training_map_path: str | Path,
        journal_path: str | Path,
        instance_id: str,
        staging_root: str | Path,
        enable_experimental_multimap: bool = False,
        assume_exclusive_no_unobserved_map_changes: bool = False,
        base_dir: str | Path = ".",
    ) -> None:
        self.config = OpenPlanetEnvironmentFactory(config, base_dir=base_dir).config
        base = Path(base_dir).resolve()
        self.training_map_path = (base / training_map_path).resolve()
        self.journal_path = (base / journal_path).resolve()
        self.staging_root = (base / staging_root).resolve()
        if self.config.host != "127.0.0.1" or self.config.session_port != 9001:
            raise ValueError(
                "experimental switching requires the fixed Connect 127.0.0.1:9001 endpoint"
            )
        if not instance_id.strip() or not self.config.expected_map_uid:
            raise ValueError("explicit game instance and training map UID are required")
        self.instance_id = instance_id
        if type(enable_experimental_multimap) is not bool:
            raise ValueError("enable_experimental_multimap must be a boolean")
        if type(assume_exclusive_no_unobserved_map_changes) is not bool:
            raise ValueError("assume_exclusive_no_unobserved_map_changes must be a boolean")
        self.enable_experimental_multimap = enable_experimental_multimap
        self.assume_exclusive_no_unobserved_map_changes = assume_exclusive_no_unobserved_map_changes
        self._training_asset_sha256: str | None = None
        self._training_map_sha256: str | None = None
        self._suite_cache: dict[str, tuple[StagedAssets, Any]] = {}

    def _asset(self, item: Any) -> Any:
        if item.map_path is None:
            raise ValueError("multi-map evaluation requires an explicit map file")
        asset = load_reward_asset(
            item.geometry_path, item.reward_points_path, item.expected_map_uid
        )
        if not asset.map_sha256:
            raise ValueError("multi-map reward assets require a map checksum")
        asset.validate_map(item.map_path)
        return asset

    def evaluation_pipeline(self, pipeline: Any, item: Any) -> Any:
        """Copy and attest map-dependent pipeline state for one evaluation map."""
        from trackmaniarl.core.builtins import IdentityFeaturePipeline
        from trackmaniarl.trackmania.features import LidarFeaturePipeline, TelemetryFeaturePipeline

        if type(pipeline) not in (
            IdentityFeaturePipeline,
            TelemetryFeaturePipeline,
            LidarFeaturePipeline,
        ):
            raise ValueError("multi-map backend requires a reviewed telemetry feature pipeline")
        if type(pipeline) is LidarFeaturePipeline and pipeline._expert_guidance_line is not None:
            raise ValueError("multi-map lidar does not support a training-map expert guidance line")
        candidate = deepcopy(pipeline)
        set_map = getattr(candidate, "set_evaluation_map", None)
        if callable(set_map):
            set_map(item)
        validate_pipeline_binding(candidate, self._asset(item), item)
        return candidate

    def validate_scheduled_evaluation_suite(
        self, suite: Any, pipeline: Any, base_dir: Path
    ) -> None:
        from trackmaniarl.trackmania.features import LidarFeaturePipeline

        self._training_asset_sha256 = None
        self._training_map_sha256 = None
        if type(self) is not OpenPlanetMultiMapEnvironmentFactory:
            raise ValueError("multi-map factory subclasses cannot inherit attestation")
        self._require_experimental_consent()
        if self.config.pace_reference_path is not None:
            raise ValueError("multi-map evaluation requires map-specific pace references")
        training = load_reward_asset(
            self.config.geometry_path, self.config.reward_points_path, self.config.expected_map_uid
        )
        if not training.map_sha256:
            raise ValueError("training reward asset requires a map checksum")
        training.validate_map(self.training_map_path)
        if type(pipeline) is LidarFeaturePipeline:
            geometry = pipeline.geometry
            with geometry.path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if (
                digest != geometry.sha256
                or geometry.map_uid != self.config.expected_map_uid
                or geometry.map_sha256 != training.map_sha256
            ):
                raise ValueError("training feature geometry differs from the attested map")
        for item in suite.maps:
            self.evaluation_pipeline(pipeline, _resolve_evaluation_map(item, base_dir))
        self._training_asset_sha256 = training.sha256
        self._training_map_sha256 = training.map_sha256

    def create(self, *, seed: int, evaluation_map: Any | None = None) -> _TrainingSession:
        self._require_experimental_consent()
        if evaluation_map is not None:
            raise ValueError(
                "multi-map factory creates its evaluation environments through its session"
            )
        if not self._training_asset_sha256 or not self._training_map_sha256:
            raise RuntimeError("training assets must be attested before creating the game session")
        staged = StagedAssets(self.staging_root)
        try:
            return self._create_staged(seed, staged)
        except BaseException:
            staged.close()
            raise

    def _require_experimental_consent(self) -> None:
        if (
            self.enable_experimental_multimap is not True
            or self.assume_exclusive_no_unobserved_map_changes is not True
        ):
            raise ValueError(
                "multi-map evaluation requires explicit experimental opt-in and acceptance "
                "of exclusive control with no unobserved map changes"
            )

    def _create_staged(self, seed: int, staged: StagedAssets) -> _TrainingSession:
        training_path = staged.stage(self.training_map_path, "training.Map.Gbx")
        field = "geometry_path" if self.config.geometry_path is not None else "reward_points_path"
        original_asset = getattr(self.config, field)
        asset_path = staged.stage(original_asset, "training-asset.npz")
        config = self.config.model_copy(update={field: asset_path})
        asset = load_reward_asset(
            config.geometry_path, config.reward_points_path, config.expected_map_uid
        )
        if (
            asset.sha256 != self._training_asset_sha256
            or asset.map_sha256 != self._training_map_sha256
        ):
            raise RuntimeError("staged training asset differs from the attested asset")
        asset.validate_map(training_path)
        journal = MapSwitchJournal(self.journal_path, self.instance_id)
        transport = OpenPlanetSwitchTransport(self.config.host, self.config.session_port)
        switch = MapSwitchController(journal, transport)
        environment = None
        holder: list[OpenPlanetEnvironment] = []
        try:
            if journal.pending() is not None:
                raise RuntimeError("unresolved map switch prevents starting collection")

            def prepare() -> None:
                environment = OpenPlanetEnvironmentFactory(config).create(seed=seed)
                holder.append(environment)
                environment.release_controls()
                environment.client.close()

            switch.switch(
                training_path, self.config.expected_map_uid or "", before_dispatch=prepare
            )
            environment = holder[0]
            switch.verify_active_map(self.config.expected_map_uid or "")
            # A process crash or failed telemetry read must leave a durable
            # record before the first bound frame can be consumed.
            switch.begin_evaluation_guard()
            self._bind_telemetry(environment, switch, self.config.expected_map_uid or "")
            session = _TrainingSession(
                environment,
                switch,
                self.config.expected_map_uid or "",
                staged,
                training_path,
                asset,
                self._release_suite_cache,
            )
            return session
        except BaseException:
            try:
                created = (
                    environment if environment is not None else (holder[0] if holder else None)
                )
                if created is not None:
                    created.close()
            finally:
                switch.close()
            raise

    @contextmanager
    def evaluation_map(self, training: _TrainingSession, item: Any) -> Iterator[_EvaluationSession]:
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        training._require_active()
        if training._evaluation_open:
            raise RuntimeError("an evaluation map context is already open")
        evaluation = None
        training._evaluation_open = True
        training._evaluation_active = True
        try:
            self._asset(item)
            training.switch.switch(
                item.map_path,
                item.expected_map_uid,
                before_dispatch=lambda: self._quiesce(training),
            )
            evaluation = OpenPlanetEnvironment(
                self.config, training.environment.controller, evaluation_map=item
            )
            self._bind_telemetry(evaluation, training.switch, item.expected_map_uid)
        except BaseException as error:
            training.fail_closed(error)
            training._evaluation_open = False
            if evaluation is not None:
                try:
                    evaluation.client.close()
                except Exception as close_error:
                    error.add_note(f"Failed to close evaluation telemetry: {close_error!r}")
            raise
        session = _EvaluationSession(evaluation, training)
        try:
            yield session
        except BaseException as error:
            from trackmaniarl.distributed.actor_evaluation import _EvaluationStoppedError

            safe_stop = isinstance(error, _EvaluationStoppedError) and error.safe_boundary
            if not safe_stop:
                training.fail_closed(error)
            try:
                session.close()
            except Exception as close_error:
                error.add_note(f"Failed to close evaluation telemetry: {close_error!r}")
                if isinstance(error, _EvaluationStoppedError):
                    error.safe_boundary = False
            raise
        else:
            session.close()
        finally:
            training._evaluation_open = False

    def restore_training(self, training: _TrainingSession) -> None:
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        training._require_active()
        if training._evaluation_open:
            raise RuntimeError("close the evaluation map context before restoring training")
        try:
            training.training_asset.validate_map(training.training_path)
            training.switch.switch(
                training.training_path,
                training.uid,
                before_dispatch=lambda: self._quiesce(training),
            )
            training.switch.verify_active_map(training.uid)
            self._bind_telemetry(training.environment, training.switch, training.uid)
        except BaseException as error:
            training.fail_closed(error)
            raise
        training._evaluation_active = False

    def verify_evaluation_map(self, training: _TrainingSession, item: Any) -> None:
        training._require_active()
        try:
            self._asset(item)
            training.switch.verify_active_map(item.expected_map_uid, require_player=False)
        except BaseException as error:
            training.fail_closed(error)
            raise

    @staticmethod
    def abort_evaluation(training: _TrainingSession, error: BaseException) -> None:
        if not isinstance(training, _TrainingSession):
            error.add_note("Cannot neutralize an unknown multi-map training session")
            return
        training.fail_closed(error)

    def require_switch_capacity(self, training: _TrainingSession, required: int) -> None:
        """Check available plugin operation IDs before leaving the training map."""
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        training._require_training_ready()
        try:
            training.switch.require_capacity(training.uid, required)
        except BaseException as error:
            training.fail_closed(error)
            raise

    def _bind_telemetry(
        self, environment: OpenPlanetEnvironment, switch: MapSwitchController, uid: str
    ) -> None:
        """Consume only frames tagged with the verified map operation."""
        identity = switch.telemetry_identity(uid)
        environment.client.close()
        environment.client = BoundOpenPlanetClient(
            OpenPlanetClientConfig(self.config.host, _BOUND_TELEMETRY_PORT, self.config.timeout_s),
            identity,
        )

    @staticmethod
    def _quiesce(training: _TrainingSession) -> None:
        training.release_controls()
        training.environment.client.close()

    @contextmanager
    def staged_suite(self, suite: Any, base_dir: Path, expected: str) -> Iterator[Any]:
        from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest

        cached = self._suite_cache.get(expected)
        if cached is None:
            if self._suite_cache:
                raise RuntimeError("suite identity changed; restart with a fresh session")
            staged = StagedAssets(self.staging_root)
            try:
                copied = staged.stage_suite(suite, base_dir, expected)
            except BaseException as error:
                try:
                    staged.close()
                except BaseException as cleanup:
                    error.add_note(f"Suite staging cleanup failed: {cleanup!r}")
                raise
            cached = (staged, copied)
            self._suite_cache[expected] = cached
        if evaluation_assets_digest(cached[1], Path(".")) != expected:
            raise RuntimeError("cached suite assets differ from the assigned suite")
        yield cached[1]

    def _release_suite_cache(self) -> None:
        errors: list[BaseException] = []
        for key, (stage, _) in list(self._suite_cache.items()):
            try:
                stage.close()
            except BaseException as error:
                errors.append(error)
            else:
                del self._suite_cache[key]
        if errors:
            first, *remaining = errors
            for additional in remaining:
                first.add_note(f"Additional suite cleanup failure: {additional!r}")
            raise first
