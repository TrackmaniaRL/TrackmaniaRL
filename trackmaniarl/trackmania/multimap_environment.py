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

    def reset(self, *, seed: int | None = None) -> Any:
        try:
            self.training_asset.validate_map(self.training_path)
            # A plugin restart invalidates the switch lease even if protocol 2
            # reports the same UID. Do not issue reset controls in that state.
            # At a finish screen the player need not be ready until after reset;
            # OpenPlanetEnvironment confirms readiness once reset has completed.
            self.switch.verify_active_map(self.uid, require_player=False)
            return self.environment.reset(seed=seed)
        except BaseException as error:
            try:
                self.release_controls()
            except Exception as release_error:
                error.add_note(f"Failed to release reset controls: {release_error!r}")
            raise

    def step(self, action: Any) -> Any:
        return self.environment.step(action)

    def release_controls(self) -> None:
        self.environment.release_controls()

    def close(self) -> None:
        try:
            self.environment.close()
        finally:
            try:
                self.release_suite_cache()
            finally:
                try:
                    self.switch.close()
                finally:
                    self.staged.close()


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
        self.enable_experimental_multimap = enable_experimental_multimap
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

        if type(self) is not OpenPlanetMultiMapEnvironmentFactory:
            raise ValueError("multi-map factory subclasses cannot inherit attestation")
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
        self._training_asset_sha256 = training.sha256
        self._training_map_sha256 = training.map_sha256
        for item in suite.maps:
            self.evaluation_pipeline(pipeline, _resolve_evaluation_map(item, base_dir))

    def create(self, *, seed: int, evaluation_map: Any | None = None) -> _TrainingSession:
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
    def evaluation_map(
        self, training: _TrainingSession, item: Any
    ) -> Iterator[OpenPlanetEnvironment]:
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        self._asset(item)
        training.switch.switch(
            item.map_path, item.expected_map_uid, before_dispatch=lambda: self._quiesce(training)
        )
        evaluation = OpenPlanetEnvironment(
            self.config, training.environment.controller, evaluation_map=item
        )
        try:
            self._bind_telemetry(evaluation, training.switch, item.expected_map_uid)
            yield evaluation
        finally:
            # The training session owns the shared input device; only close the
            # evaluation telemetry connection here.
            evaluation.client.close()

    def restore_training(self, training: _TrainingSession) -> None:
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        training.training_asset.validate_map(training.training_path)
        training.switch.switch(
            training.training_path, training.uid, before_dispatch=lambda: self._quiesce(training)
        )
        training.switch.verify_active_map(training.uid)
        self._bind_telemetry(training.environment, training.switch, training.uid)

    def verify_evaluation_map(self, training: _TrainingSession, item: Any) -> None:
        self._asset(item)
        training.switch.verify_active_map(item.expected_map_uid, require_player=False)

    def require_switch_capacity(self, training: _TrainingSession, required: int) -> None:
        """Check available plugin operation IDs before leaving the training map."""
        if not isinstance(training, _TrainingSession):
            raise TypeError("multi-map evaluation requires its owning training session")
        training.switch.require_capacity(training.uid, required)

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
