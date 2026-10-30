"""Paused TICK physics with real game-camera observations and explicit clocks."""
import time
import numpy as np
from PIL import Image
from tools.vision_qr1167.tick_client import TickClient
from tools.vision_qr1167.hud_clock import HudClock


class FramePairingError(RuntimeError):
    """No admissible rendered observation; discard the unobserved transition."""


class NativeStepError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class TickBridge:
    def __init__(self, installation, calibration, *, fast_steps=False):
        import vgamepad as vg
        from trackmaniarl.trackmania.vision_environment import WindowFrameSource
        from trackmaniarl.trackmania.window_capture import WindowCaptureConfig
        self.fast_steps = fast_steps
        self.step_timings = []
        self.vg = vg
        self.client = TickClient(installation)
        self.original = self.client.request('settings')
        self.old_speed = self.client.require_map()['effectiveGameSpeed']
        self.revision = None
        self.pad = None
        self.camera = None
        self.last = None
        self.clock = HudClock(calibration)
        try:
            if self.client.require_map()['inputExecutionEffective']:
                raise RuntimeError('policy requires disabled demonstration playback')
            bindings = self.original['inputBindings']
            if any(b.get('pad') == 'right' for b in bindings):
                raise RuntimeError('step button already assigned')
            bindings = [b for b in bindings if b['action'] != 'stepForward'] + [
                dict(action='stepForward', keyboard=None, pad='right', speed=None)]
            self.revision = self.client.request('settings', dict(
                expectedRevision=self.original['revision'], inputBindings=bindings,
                drawPredictedTrajectoryEnabled=False, drawOptimizationTrajectoryEnabled=False), 'PATCH')['revision']
            self.client.speed(0)
            self.pad = vg.VX360Gamepad()
            self.pad.reset(); self.pad.update()
            self.camera = WindowFrameSource(WindowCaptureConfig(client_width=1536, client_height=864))
            self.camera.prepare_episode()
            time.sleep(.5)
        except BaseException:
            self.close()
            raise

    def newest(self):
        event = self.newest_event()
        return event['payload'] if event else None

    def newest_event(self):
        return next((e for e in reversed(self.client.events)
                     if e['eventType'] == 'vehicle-state'), None)

    def tick(self, action):
        status = self.client.require_map()
        if status['effectiveGameSpeed'] != 0 or status['inputExecutionEffective']:
            raise RuntimeError(f'lost exclusive paused physics control: speed={status["effectiveGameSpeed"]}, playback={status["inputExecutionEffective"]}')
        before_event = self.newest_event()
        before = before_event['payload'] if before_event else None
        receive_index = before_event['_receive_index'] if before_event else -1
        steer, accel, brake = action
        self.pad.left_joystick_float(x_value_float=float(steer), y_value_float=0)
        self.pad.right_trigger_float(float(accel))
        self.pad.left_trigger_float(float(brake))
        self.pad.update()
        started = time.monotonic()
        if self.last is not None and before and before['raceTick'] != self.last['raceTick']:
            raise RuntimeError('unsolicited physics step before command')
        time.sleep(.01 if self.fast_steps else .04)
        self.pad.press_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.pad.update()
        pressed_at = time.monotonic()
        try:
            if not self.fast_steps:
                time.sleep(.2)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                current = self.newest_event()
                if current and current['_receive_index'] > receive_index:
                    break
                time.sleep(.001 if self.fast_steps else .01)
            else:
                raise TimeoutError('no fresh native telemetry after step')
        finally:
            self.pad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
            self.pad.update()
            released_at = time.monotonic()
        # A release edge must be polled by the game before another press.
        # Its conservative delay is admitted by the separate timing probe.
        time.sleep(.04 if self.fast_steps else .2)
        if self.fast_steps:
            deadline = time.monotonic()+2
            while True:
                settled = self.client.require_map()
                if settled['inputExecutionEffective']:
                    raise RuntimeError('teacher playback during step')
                if settled['effectiveGameSpeed'] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('step did not return to pause')
                time.sleep(.005)
        events = [e['payload'] for e in list(self.client.events)
                  if e['eventType']=='vehicle-state' and e['_receive_index'] > receive_index]
        if len(events) != 1:
            raise NativeStepError(f'expected exactly one native step, received {len(events)}', dict(
                before=before, received=events, requested_action=list(action),
                button_hold_seconds=released_at-pressed_at,
                observed_seconds=time.monotonic()-started, admitted_to_replay=False))
        current = events[0]
        self.last = current
        self.step_timings.append(time.monotonic()-started)
        if len(self.step_timings)>1000:
            del self.step_timings[:500]
        return current

    def capture_paused_frame(self, timeout=15):
        """Retry visibility loss only while the exact native state stays paused."""
        from trackmaniarl.core.environment_errors import EnvironmentPausedError
        started = time.monotonic()
        interrupted = False
        expected = {k: self.last[k] for k in ('raceTick', 'simulationTick')}

        def unchanged():
            status = self.client.require_map()
            latest = self.newest()
            if (status['effectiveGameSpeed'] != 0 or status['inputExecutionEffective'] or
                    latest is None or any(latest[k] != value for k, value in expected.items())):
                raise FramePairingError('native state changed during visibility interruption')

        while True:
            try:
                frame = self.camera.capture()
            except EnvironmentPausedError:
                interrupted = True
                unchanged()
                if time.monotonic() - started >= timeout:
                    raise
                time.sleep(.1)
                continue
            if interrupted:
                unchanged()
            return frame, time.monotonic() - started if interrupted else 0.

    def observe(self):
        if self.last is None:
            raise RuntimeError('fresh native step required before observation')
        deadline = time.monotonic() + 2
        last_error = None
        visibility_wait_seconds = 0.
        while time.monotonic() < deadline:
            frame, waited = self.capture_paused_frame(timeout=max(0., 15.-visibility_wait_seconds))
            deadline += waited
            visibility_wait_seconds += waited
            self.last_frame = frame
            self.last_frame_native_state = dict(self.last)
            try:
                clock, errors = self.clock.read(frame)
                if clock == self.last['raceTick'] + 10:
                    # Exclude HUD clock, scoreboard and control hints from the policy.
                    image = Image.fromarray(frame[200:790, 180:1170]).convert('L').resize((84, 84))
                    return np.asarray(image), dict(rendered_ms=clock,
                        telemetry_ms=self.last['raceTick'], clock_label_offset_ms=10,
                        glyph_errors=errors, telemetry=self.last,
                        visibility_wait_seconds=visibility_wait_seconds)
                last_error = f'rendered={clock}, native={self.last["raceTick"]}'
            except ValueError as error:
                last_error = str(error)
            time.sleep(.02)
        raise FramePairingError('rendered/native pairing failed: ' + str(last_error))

    def teleport(self, revision, milliseconds):
        self.pad.reset(); self.pad.update()
        self.revision = self.client.request('settings', dict(expectedRevision=self.revision,
            activeInputCollectionId=self.original['activeInputCollectionId']), 'PATCH')['revision']
        deadline = time.monotonic() + 3
        while self.client.require_map()['loadedInputRevisionId'] != revision:
            if time.monotonic() > deadline:
                raise TimeoutError('reference revision not loaded')
            time.sleep(.05)
        self.client.command('teleport-to-revision-time', dict(
            revisionId=revision, raceTimeMilliseconds=milliseconds), timeout=15)
        time.sleep(.3)
        # Isolate subsequent policy steps from the revision rewind cache.
        self.revision = self.client.request('settings', dict(expectedRevision=self.revision,
            activeInputCollectionId=None), 'PATCH')['revision']
        deadline = time.monotonic() + 3
        while self.client.require_map()['loadedInputRevisionId'] is not None:
            if time.monotonic() > deadline:
                raise TimeoutError('reference revision not unloaded')
            time.sleep(.05)
        # A neutral preparation tick is explicit training-only initialization.
        # It must never be counted as a full-start evaluation.
        self.last = None
        self.tick((0, 0, 0))
        return self.observe()

    def advance(self, action, ticks=5):
        if not 1 <= ticks <= 5 or self.last is None:
            raise ValueError('invalid transition request')
        records = []
        for _ in range(ticks):
            prior = self.last['raceTick']
            sample = self.tick(action)
            if sample['currentCheckpointNumber'] >= sample['totalCheckpointCount'] > 0:
                # Absorbing terminal observation; no claim of a real camera frame.
                return np.zeros((84,84), dtype=np.uint8), dict(telemetry=sample,
                    native_steps=records+[sample], requested_action=list(action),
                    absorbing_terminal=True, goal=True)
            if sample['raceTick'] - prior != 10:
                raise RuntimeError('physics clock discontinuity')
            records.append(sample)
        image, info = self.observe()
        info['requested_action'] = list(action)
        info['native_steps'] = records
        return image, info

    def reset_prefix(self, milliseconds):
        """Training-only real reset and teacher prefix; no cached teleport.

        The actual handoff time is measured, not claimed equal to requested time.
        Prefix wall time is irrelevant to policy stepping and is excluded from RL.
        """
        if not 0 <= milliseconds <= 16000:
            raise ValueError('training prefix out of range')
        self.recover_finish()
        self.pad.reset(); self.pad.update()
        self.revision = self.client.request('settings', dict(expectedRevision=self.revision,
            activeInputCollectionId=self.original['activeInputCollectionId'],
            inputExecutionEnabled=milliseconds > 0), 'PATCH')['revision']
        self.client.speed(1)
        reset_cursor = self.client.received_events
        self.pad.press_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.pad.update()
        try:
            time.sleep(1.2)
        finally:
            self.pad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.pad.update()
        deadline = time.monotonic() + 25
        observed_reset = False
        while time.monotonic() < deadline:
            current = self.newest()
            if current:
                observed_reset |= any(e['eventType']=='vehicle-state' and
                    e['_receive_index']>reset_cursor and e['payload']['raceTick']==0
                    for e in list(self.client.events))
                if observed_reset and current['raceTick'] >= max(10, milliseconds):
                    break
            time.sleep(.01)
        else:
            raise TimeoutError('actual reset/prefix not observed')
        self.client.speed(0)
        self.revision = self.client.request('settings', dict(expectedRevision=self.revision,
            inputExecutionEnabled=False), 'PATCH')['revision']
        deadline = time.monotonic() + 3
        while self.client.require_map()['inputExecutionEffective']:
            if time.monotonic() > deadline:
                raise TimeoutError('teacher input did not release')
            time.sleep(.02)
        self.last = None
        self.tick((0, 0, 0))
        frame, info = self.observe()
        info['training_prefix_requested_ms'] = milliseconds
        info['full_start_evaluation'] = False
        return frame, info

    def recover_finish(self):
        from tools.vision_qr1167.tick_client import UID
        status = self.client.request('runtime/status')
        last = self.last or self.newest()
        finished = last is not None and last['currentCheckpointNumber'] >= last['totalCheckpointCount'] > 0
        self.finish_recovery_trace = []
        if status['currentMapUid'] == UID and not finished:
            return
        if status['currentMapUid'] not in (None,UID) or not finished:
            raise RuntimeError('map unavailable without an observed finish')
        self.camera.validate()
        # A paused terminal physics step may leave a pending zero-speed request
        # even while the finish replay reports an effective speed of one.
        self.client.speed(1)
        before = self.newest_event()
        marker = before['_receive_index'] if before else -1
        # Controller reset protocol: acknowledge finish replay, then Improve.
        # Map UID alone can also describe the terminal/replay transition. Require
        # a fresh early vehicle state AND the actual racing HUD before returning.
        for attempt in range(9):
            status = self.client.request('runtime/status')
            if status['currentMapUid'] not in (None,UID):
                raise RuntimeError('unexpected map after finish')
            event = self.newest_event()
            current = event['payload'] if event else None
            evidence = dict(attempt=attempt,map_uid=status['currentMapUid'],
                native_ms=current['raceTick'] if current else None)
            if (status['currentMapUid']==UID and event and event['_receive_index']>marker and
                    0<=current['raceTick']<1000 and current['currentCheckpointNumber']==0):
                try:
                    clock,_=self.clock.read(self.camera.capture())
                    evidence['hud_ms']=clock
                    if 0<=clock<1000:
                        self.finish_recovery_trace.append(evidence)
                        self.last=None
                        return
                except ValueError:
                    evidence['hud_unreadable']=True
            self.finish_recovery_trace.append(evidence)
            if attempt==8:
                break
            self.pad.press_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self.pad.update(); time.sleep(.2)
            self.pad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self.pad.update(); time.sleep(.5)
        raise TimeoutError('finish reset did not return a fresh racing HUD: '+str(self.finish_recovery_trace[-3:]))

    def close(self):
        errors = []
        if self.pad:
            self.pad.reset(); self.pad.update()
        try:
            if self.revision is not None:
                current = self.client.request('settings')
                if current['revision'] != self.revision:
                    raise RuntimeError('concurrent settings change; refusing overwrite')
                self.client.request('settings', dict(expectedRevision=self.revision,
                    inputBindings=self.original['inputBindings'],
                    inputExecutionEnabled=self.original['inputExecutionEnabled'],
                    drawPredictedTrajectoryEnabled=self.original.get('drawPredictedTrajectoryEnabled', False),
                    drawOptimizationTrajectoryEnabled=self.original.get('drawOptimizationTrajectoryEnabled', False),
                    activeInputCollectionId=self.original['activeInputCollectionId']), 'PATCH')
                self.revision = None
            self.client.speed(self.old_speed)
        except Exception as e:
            errors.append(str(e))
        if self.camera:
            self.camera.close()
            self.camera = None
        self.client.close()
        if errors:
            raise RuntimeError('cleanup: ' + '; '.join(errors))
