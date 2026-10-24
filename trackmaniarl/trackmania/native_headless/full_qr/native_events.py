"""Full native delivery using qualified physics/virtual-input/paired-Finish sources."""

from pathlib import Path
from struct import unpack_from
from typing import Any

from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import MeasuredAdherence
from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.contact_delivery import (
    HeldContactDeliveryReader,
)
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrStep,
    FullQrTerminalStep,
)
from trackmaniarl.trackmania.native_headless.full_qr.final_capture import (
    FinalCollectorBoundary,
    decode_final_collector_boundary,
    read_final_stream,
)
from trackmaniarl.trackmania.native_headless.full_qr.finish_reference import (
    QualifiedGoalFinishReader,
    load_reference,
)
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext
from trackmaniarl.trackmania.native_headless.full_qr.motor_qualification import load_motor_reference
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_rumble_bindings import (
    HeldRumbleBindings,
)
from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import (
    ContactRumbleProducer,
    PhysicalRumbleInput,
)
from trackmaniarl.trackmania.native_headless.full_qr.render_rumble import Virtual150RumbleClock
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import (
    CommittedGoalCapture,
    PartialGoalCapture,
)
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCaptureContext
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_reader import HeldVehicleStream


class QualifiedNativeEvents:
    """One actor's complete declared virtual150FPS source; no default negatives.

    The provider must hold and expose each ten-ms physical boundary before
    asking for a fifty-ms reward boundary. Client wall-clock jitter is not
    replayed: the pinned qualification explicitly names virtual150fps/v1.
    """

    physics_stride = 1

    def __init__(  # noqa: PLR0913 - Independent primary reference paths and their pins.
        self,
        *,
        finish_reference_path: str,
        finish_reference_sha256: str,
        motor_reference_path: str,
        motor_reference_sha256: str,
    ) -> None:
        self.finish_reference = load_reference(Path(finish_reference_path), finish_reference_sha256)
        self.motor_reference = load_motor_reference(
            Path(motor_reference_path), motor_reference_sha256
        )
        self._bindings: HeldRumbleBindings | None = None
        self._session: Any = None
        self._collector: RewardEventCollector | None = None
        self._closed = False

    def reset(self, session: Any, identity: EventStreamIdentity) -> RewardEventCollector:
        if self._closed:
            raise ValueError("create a fresh event source after close")
        self.release_episode()
        if any(
            session.pins.get(role) != pin for role, pin in self.motor_reference.asset_pins.items()
        ):
            raise ValueError("motor qualification does not cover the current native assets")
        try:
            self._session = session
            self._bindings = HeldRumbleBindings(
                session, material14_uses_default=self.motor_reference.material14_uses_default
            )
            meta = session.meta
            context = VehicleCaptureContext(
                session.pid, meta["participant"], meta["vehicle"], meta["module_slide"], 2200
            )
            self._vehicle = HeldVehicleStream(
                session.local_posts / f"vehicle-{session.pid}.bin", context
            )
            self._contacts = HeldContactDeliveryReader(
                session.local_posts / f"contact-receipt-{session.pid}.bin",
                session.local_posts / f"prepared-impact-{session.pid}.bin",
                self._bindings.contact_context,
            )
            self._finish = QualifiedGoalFinishReader(session, self.finish_reference)
            captures, boundaries = self._vehicle.take(2200), self._contacts.take(2200)
            if (
                len(captures) != 2
                or len(boundaries) != 2
                or captures[0].native_t != 2190
                or boundaries[0].receipt.native_t != 2190
                or boundaries[0].contacts
            ):
                raise ValueError("complete neutral bootstrap physical/delivery prefix required")
            initial = RumbleState(*unpack_from("<ff", captures[0].visual_state, 0x32C))
            if initial != RumbleState(0.0, 0.0):
                raise ValueError("unexpected bootstrap waveform")
            self._producer = ContactRumbleProducer(initial)
            self._render = Virtual150RumbleClock(initial)
            self._collector = RewardEventCollector(identity)
            self._observe(captures[1], boundaries[1])
            self._finish.read(2200)
            return self._collector
        except BaseException:
            self.release_episode()
            raise

    def _observe(
        self, capture: Any, boundary: Any, *, final_source: PhysicalRumbleInput | None = None
    ) -> None:
        if self._bindings is None or self._collector is None:
            raise ValueError("an active independently bound source is required")
        if capture.native_t != boundary.receipt.native_t:
            raise ValueError("physical visual and contact receipts do not share a held boundary")
        source = self._bindings.source(capture) if final_source is None else final_source
        if type(source) is not PhysicalRumbleInput or source.visual != capture.visual_state:
            raise ValueError("actual complete physical rumble source required")
        step = self._producer.advance(boundary, source)
        for change in self._render.advance(step):
            self._collector.record_motor(
                self._collector.identity,
                sequence=change.sequence,
                race_time_ms=change.race_time_ms,
                large_motor=change.large_motor,
            )
        self._last_capture = capture

    def _check_session(self, session: Any) -> None:
        if (
            self._closed
            or session is not self._session
            or self._bindings is None
            or session.pid != self._bindings.pid
            or self._collector is None
        ):
            raise ValueError("current owned episode source is required")

    def observe_physics(self, session: Any, native_t: int) -> None:
        self._check_session(session)
        assert self._collector is not None
        assert self._bindings is not None
        captures, boundaries = self._vehicle.take(native_t), self._contacts.take(native_t)
        if len(captures) != 1 or len(boundaries) != 1:
            raise ValueError("each actual physical tick must be observed while held")
        self._observe(captures[0], boundaries[0])

    def deliver(self, session: Any, frame: NativeGraphFrame) -> None:
        self._check_session(session)
        assert self._collector is not None
        assert self._bindings is not None
        end = int(frame.channels[0])
        native = 2200 + end
        if self._producer.native_t != native or self._render.native_t != native:
            raise ValueError("complete physical and virtual render delivery required")
        finish = self._finish.read(native)
        self._collector.record_ui_boundary(
            self._collector.identity, race_time_ms=end, finish_active=finish
        )
        self._collector.seal_motor_through(self._collector.identity, end)

    def terminal_frame(  # noqa: PLR0913 - session, causal clock, measured coefficient and final proof.
        self,
        session: Any,
        previous_native_t: int,
        adherence: MeasuredAdherence,
        *,
        final_boundary: FinalCollectorBoundary | None = None,
    ) -> NativeGoalTerminalFrame | NativeCommittedGoalFrame:
        self._check_session(session)
        assert self._collector is not None
        assert self._bindings is not None
        native = self._last_capture.native_t
        if self._producer.native_t != native or self._render.native_t != native:
            raise ValueError("complete actual terminal physical/render coverage required")
        boundary = self._finish.reader.read(native)
        if not boundary.goal_active:
            return self._committed_terminal_frame(previous_native_t, adherence, final_boundary)
        goal = self._finish.read_terminal(native)
        frame = NativeGoalTerminalFrame(
            PartialGoalCapture(previous_native_t, self._last_capture, goal), adherence
        )
        end = native - 2200
        self._collector.record_terminal_boundary(
            self._collector.identity, race_time_ms=end, finish_active=True
        )
        self._collector.seal_motor_through(self._collector.identity, end)
        return frame

    def _committed_terminal_frame(
        self,
        previous_native_t: int,
        adherence: MeasuredAdherence,
        final_boundary: FinalCollectorBoundary | None = None,
    ) -> NativeCommittedGoalFrame:
        if self._collector is None:
            raise ValueError("committed terminal requires an active source collector")
        native = self._last_capture.native_t
        boundary, fact = self._finish.read_committed_terminal(native)
        capture = CommittedGoalCapture(
            previous_native_t, self._last_capture, boundary, fact, final_boundary
        )
        frame = NativeCommittedGoalFrame(capture, adherence)
        for change in self._render.finish_committed(fact.event_native_t):
            self._collector.record_motor(
                self._collector.identity,
                sequence=change.sequence,
                race_time_ms=change.race_time_ms,
                large_motor=change.large_motor,
            )
        if (
            self._producer.native_t != native
            or self._render.rendered_through_native_t != fact.event_native_t
        ):
            raise ValueError(
                "separate actual physical and complete terminal render clocks required"
            )
        self._collector.record_committed_boundary(
            self._collector.identity,
            sampled_ms=native - 2200,
            score_ms=frame.score_race_time_ms,
            final_sample_completed=(
                final_boundary is not None and native - previous_native_t == 50
            ),
        )
        self._collector.seal_motor_through(self._collector.identity, frame.score_race_time_ms)
        return frame

    def final_step(  # noqa: PLR0913 - independently bound session, clock, reader and issued command.
        self,
        session: Any,
        previous_native_t: int,
        adherence_reader: Any,
        issued_steering: float,
    ) -> FullQrStep | FullQrTerminalStep | None:
        """Consume only an actually completed extra native collector sample.

        EOF is the occasion for this read, never its physical/Finish authority.
        The normal live binding and held-reader guards remain unchanged.
        """
        self._check_session(session)
        assert self._collector is not None
        assert self._bindings is not None
        try:
            records = read_final_stream(
                session.local_posts / f"final-boundary-{session.pid}.bin", 896, 5000
            )
        except FileNotFoundError:
            return None
        facts = read_final_stream(session.local_posts / f"goal-fact-{session.pid}.bin", 768, 10000)
        proof = decode_final_collector_boundary(
            records[-896:], facts, context=NativeGoalContext(session.pid, session.meta)
        )
        if proof.native_t <= self._last_capture.native_t:
            return None
        if proof.previous_native_t != self._last_capture.native_t:
            raise ValueError("final collector source has an unobserved physical gap")
        captures = self._vehicle.take(proof.native_t)
        boundaries = self._contacts.take(proof.native_t)
        if len(captures) != 1 or len(boundaries) != 1:
            raise ValueError("one actual extra collector physical/contact sample required")
        material = read_final_stream(
            session.local_posts / f"final-material-{session.pid}.bin", 1536, 5000
        )
        source = self._bindings.final_source(captures[0], proof, material[-1536:])
        self._observe(captures[0], boundaries[0], final_source=source)
        adherence = adherence_reader.read(proof.native_t)
        goal = self._finish.reader.read(proof.native_t)
        if goal.goal_active and (proof.native_t - 2200) % 50 == 0:
            frame = NativeGraphFrame.from_measured_capture(captures[0], adherence)
            if proof.native_t != previous_native_t + 50:
                raise ValueError("actual final regular physical step must still be50ms")
            self.deliver(session, frame)
            self._collector.peek(self._collector.identity, proof.native_t - 2200)
            return FullQrStep(frame, issued_steering)
        return FullQrTerminalStep(
            self.terminal_frame(session, previous_native_t, adherence, final_boundary=proof),
            issued_steering,
        )

    def release_episode(self) -> None:
        if self._bindings is not None:
            self._bindings.close()
            self._bindings = None
        self._session, self._collector = None, None

    def close(self) -> None:
        self._closed = True
        self.release_episode()
