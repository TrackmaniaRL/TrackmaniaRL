"""Actual V8 frame provider; a complete motor/UI source remains mandatory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from struct import pack, unpack
from time import monotonic
from typing import Any, Protocol, cast

from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession,
    physical_state,
    qualify_immediate,
)
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import HeldAdherenceReader
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrReset,
    FullQrStep,
    FullQrTerminalStep,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame
from trackmaniarl.trackmania.native_headless.full_qr.profile import PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.full_qr.trace_retention import ClosedTraceRetention
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCaptureContext
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_reader import HeldVehicleReader
from trackmaniarl.trackmania.native_headless.protocols import (
    NativeInconclusiveError,
    NativeLifecycleError,
)


def _action_for_tick(action: int, offset: int) -> int:
    """Release a decision's brake tap after its first native tick."""
    brake_mode_count, tap_mode = 3, 2
    is_tap = action % brake_mode_count == tap_mode
    return action - tap_mode if is_tap and offset > 0 else action


def _issued_steering(action: int) -> float:
    """Keep the historical float32 command distinct from measured calibration."""
    variants_per_steering, center_bin = 6, 6
    nominal = (action // variants_per_steering - center_bin) / center_bin
    return float(unpack("<f", pack("<f", nominal))[0])


class FullQrEventSource(Protocol):
    """One provider's sourced delivery, with no default negative event values.

    reset binds a fresh stream to the current owned native child. deliver must
    observe complete motor delivery and UI at this boundary, using their actual
    qualified clocks. A contact tape or EOF cannot satisfy this interface by
    itself. Each factory instance creates a separate source per environment.
    """

    def reset(
        self,
        session: ImmediateSession,
        identity: EventStreamIdentity,
    ) -> RewardEventCollector: ...

    def deliver(self, session: ImmediateSession, frame: NativeGraphFrame) -> None: ...

    def release_episode(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeFullQrProviderSettings:
    runtime: NativeRuntimeConfig
    assets: VerifiedAssets
    actor: str
    retention: ClosedTraceRetention
    driver_episode_limit: int = 8
    next_episode_reserve_bytes: int = 192 * 1024**2

    def __post_init__(self) -> None:
        if self.driver_episode_limit < 1 or self.next_episode_reserve_bytes < 1:
            raise ValueError("positive driver cohort and next-episode reserve required")


class NativeFullQrProvider:
    """Bounded warm-reset cohorts, with verified retention after driver rollover."""

    def __init__(
        self,
        config: NativeFullQrProviderSettings,
        events: FullQrEventSource,
    ) -> None:
        settings, assets, actor = config.runtime, config.assets, config.actor
        if settings.protocol_profile != PROTOCOL or assets.profile != PROFILE:
            raise ValueError("the separately deployed full QR V8 profile is required")
        if not settings.allow_native or not actor:
            raise ValueError("explicit native execution and actor identity required")
        if events is None:
            raise ValueError("a real motor/UI event source is required")
        self.settings, self.assets, self.events, self.actor = settings, assets, events, actor
        self.retention = config.retention
        self.driver_episode_limit = config.driver_episode_limit
        self.next_episode_reserve_bytes = config.next_episode_reserve_bytes
        self._driver_generation, self._driver_episodes = 0, 0
        self.archive_receipts: list[dict[str, Any]] = []
        self._children: list[dict[str, Any]] = []
        self._session: ImmediateSession | None = None
        self._reader: HeldVehicleReader | None = None
        self._adherence_reader: HeldAdherenceReader | None = None
        self._before: dict[str, Any] | None = None
        self._collector: RewardEventCollector | None = None
        self._identity: EventStreamIdentity | None = None
        self._generation = 0
        self._closed = False

    def _frame(self) -> NativeGraphFrame:
        if self._reader is None or self._adherence_reader is None or self._before is None:
            raise NativeLifecycleError("native full QR reset is required")
        native_t = self._before["nativeT"]
        return NativeGraphFrame.from_measured_capture(
            self._reader.read(native_t), self._adherence_reader.read(native_t)
        )

    def reset(self, *, seed: int | None = None) -> FullQrReset:
        if self._closed:
            raise NativeLifecycleError("create a fresh provider after failure or close")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer; it does not alter native map reset")
        try:
            if self._session is not None:
                self.events.release_episode()
                if not self._can_reuse_driver():
                    self._finish_session("driver_rollover")
            if self._session is None:
                self._driver_generation += 1
                self._driver_episodes = 0
                name = f"driver-{self._driver_generation:08d}"
                settings = replace(
                    self.settings,
                    trace_dir=self.settings.trace_dir / name,
                    work_dir=self.settings.work_dir / name,
                )
                self._session = ImmediateSession(settings, self.assets)
            session = self._session
            packet = session.reset()
            identity = process_identity(session.pid)
            if identity is None:
                raise NativeLifecycleError("native child identity disappeared at reset")
            self._children.append(identity)
            self._driver_episodes += 1
            self._before = physical_state(packet)
            meta = session.meta
            self._reader = HeldVehicleReader(
                session.local_posts / f"vehicle-{session.pid}.bin",
                VehicleCaptureContext(
                    session.pid,
                    meta["participant"],
                    meta["vehicle"],
                    meta["module_slide"],
                    self._before["nativeT"],
                ),
            )
            self._adherence_reader = HeldAdherenceReader(
                session.local_posts / f"adherence-{session.pid}.bin", self._reader.context
            )
            initial = self._frame()
            if initial.channels[0] != 0:
                raise ValueError("native full QR reset must expose race time0")
            self._generation += 1
            identity = EventStreamIdentity(self.actor, f"{self._generation}:{session.pid}")
            collector = self.events.reset(session, identity)
            if (
                type(collector) is not RewardEventCollector
                or collector.identity != identity
                or collector.consumed_through_ms != 0
            ):
                raise ValueError("event source did not bind the fresh native episode")
            self._identity, self._collector = identity, collector
            return FullQrReset(initial, collector)
        except BaseException as error:
            self._fail(error)
            raise

    def advance(self, action: int) -> FullQrStep | FullQrTerminalStep:
        if type(action) is not int or not 0 <= action < 78:
            raise ValueError("one of the historical 78 actions is required")
        if (
            self._closed
            or self._session is None
            or self._before is None
            or self._collector is None
            or self._identity is None
        ):
            raise NativeLifecycleError("an active native full QR reset is required")
        try:
            return self._advance_decision(action)
        except BaseException as error:
            self._fail(error)
            raise

    def _physics_delivery(self) -> tuple[int, Callable[[ImmediateSession, int], None] | None]:
        stride = getattr(self.events, "physics_stride", 5)
        if type(stride) is not int or stride not in (1, 5):
            raise ValueError("event source must require one or five native ticks")
        if stride == 5:
            return stride, None
        observe = getattr(self.events, "observe_physics", None)
        if not callable(observe):
            raise ValueError("per-tick event source must expose its held observation hook")
        return stride, observe

    def _advance_decision(self, action: int) -> FullQrStep | FullQrTerminalStep:
        assert self._session is not None
        assert self._before is not None
        previous_native_t = self._before["nativeT"]
        if previous_native_t >= 152200:
            raise NativeLifecycleError("full150s domain ended; reset is required")
        stride, observe = self._physics_delivery()
        issued = self._decision_steering(action)
        for offset in range(0, 5, stride):
            tick_action = self._control_for_tick(action, offset)
            packet = self._session.advance(tick_action, stride)
            if packet["eof"]:
                final = self._final_step(previous_native_t, issued) if stride == 1 else None
                if final is not None:
                    return final
                raise NativeInconclusiveError(packet)
            qualify_immediate(packet, self._before, tick_action, stride)
            self._before = physical_state(packet)
            if observe is not None:
                observe(self._session, self._before["nativeT"])
        return self._complete_step(issued)

    def _decision_steering(self, action: int) -> float:
        return _issued_steering(action)

    def _control_for_tick(self, action: int, offset: int) -> int:
        return _action_for_tick(action, offset)

    def _final_step(
        self, previous_native_t: int, issued: float
    ) -> FullQrStep | FullQrTerminalStep | None:
        """EOF permits a source read; only actual collector/Goal data can end a step."""
        assert self._session is not None
        assert self._before is not None
        if self._adherence_reader is None:
            return None
        collect_final = getattr(self.events, "final_step", None)
        if callable(collect_final):
            final = collect_final(self._session, previous_native_t, self._adherence_reader, issued)
            if final is not None:
                if type(final) not in (FullQrStep, FullQrTerminalStep):
                    raise TypeError("an actual sourced final step is required")
                return cast(FullQrStep | FullQrTerminalStep, final)
        terminal = getattr(self.events, "terminal_frame", None)
        native_t = self._before["nativeT"]
        if callable(terminal) and 0 <= native_t - previous_native_t < 50:
            frame = terminal(
                self._session, previous_native_t, self._adherence_reader.read(native_t)
            )
            return FullQrTerminalStep(frame, issued)
        return None

    def _complete_step(self, issued: float) -> FullQrStep:
        assert self._session is not None
        assert self._collector is not None
        assert self._identity is not None
        frame = self._frame()
        self.events.deliver(self._session, frame)
        # Coverage is checked here; reward success owns subsequent consumption.
        self._collector.peek(self._identity, int(frame.channels[0]))
        return FullQrStep(frame, issued)

    def _fail(self, error: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"native provider cleanup failed: {cleanup!r}")

    def _can_reuse_driver(self) -> bool:
        session = self._session
        if session is None or self._driver_episodes >= self.driver_episode_limit:
            return False
        # A warm reset inherits this driver's original deadline. Reserve a
        # conservative three minutes for a new full150s episode; never extend
        # the original <=240s native watchdog to make reuse succeed.
        remaining = session.deadline - monotonic()
        if not isfinite(remaining):
            raise NativeLifecycleError("finite native driver deadline required")
        if remaining < 180.0:
            return False
        monitor = session._quota_monitor
        if monitor is None:
            raise NativeLifecycleError("a live native trace quota monitor is required")
        usage = monitor.usage(force=True)
        persisted: int = session.capture.persisted
        # Account for closing/publishing the current capture AND all eventual
        # post copies, before the configured next-episode reserve. The original
        # session monitor remains the authority throughout the next episode.
        required = (
            usage.used_bytes
            + usage.post_copy_bytes
            + 2 * persisted
            + self.next_episode_reserve_bytes
        )
        return required <= session.max_trace_bytes

    def finish_episode(self) -> None:
        """Close a consumed driver before any between-episode learner wait."""
        if self._session is not None and not self._can_reuse_driver():
            self.events.release_episode()
            self._finish_session("episode_complete")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.events.close()
        finally:
            if self._session is not None:
                self._finish_session("provider_close")

    def _finish_session(self, reason: str) -> None:
        session = self._session
        if session is None:
            return
        session.close()
        self._session = None
        self._reader, self._before = None, None
        self._adherence_reader = None
        receipt = self.retention.archive(session, self._children, reason=reason)
        self.archive_receipts.append(
            {
                key: receipt[key]
                for key in (
                    "archive",
                    "archive_sha256",
                    "archive_bytes",
                    "sources_pruned",
                    "archive_seconds",
                )
            }
        )
        self._children = []
