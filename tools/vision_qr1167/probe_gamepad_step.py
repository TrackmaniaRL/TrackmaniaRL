"""Bounded environment-controller test, with original settings restored.

Uses the same virtual gamepad interface as the RL runtime, not host-time
driving. A held D-pad button tests whether TICK detects an actual step edge.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('installation')
    parser.add_argument('dependencies')
    parser.add_argument('output', type=Path)
    parser.add_argument('--camera-calibration', type=Path)
    parser.add_argument('--pulse-seconds', type=float, default=.2)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path.insert(0, args.dependencies)
    import vgamepad as vg
    from tools.vision_qr1167.tick_client import TickClient
    client = TickClient(args.installation)
    original = client.request('settings')
    old_speed = client.require_map()['effectiveGameSpeed']
    pad = None
    report = {'status': 'FAIL', 'snapshots': [], 'training_started': False}
    installed_revision = None
    camera = clock = None
    try:
        if not .03 <= args.pulse_seconds <= .2:
            raise ValueError('pulse duration out of bounded probe range')
        if args.camera_calibration:
            from tools.vision_qr1167.hud_clock import HudClock
            from trackmaniarl.trackmania.vision_environment import WindowFrameSource
            from trackmaniarl.trackmania.window_capture import WindowCaptureConfig
            clock = HudClock(args.camera_calibration)
            camera = WindowFrameSource(WindowCaptureConfig(client_width=1536, client_height=864))
            camera.prepare_episode()
            report['camera_samples'] = []
        bindings = original['inputBindings']
        if any(b.get('pad') == 'right' for b in bindings):
            raise RuntimeError('D-pad right already assigned')
        patched = [b for b in bindings if b['action'] != 'stepForward']
        patched.append(dict(action='stepForward', keyboard=None, pad='right', speed=None))
        changed = client.request('settings', dict(expectedRevision=original['revision'],
                                                  inputBindings=patched), 'PATCH')
        installed_revision = changed['revision']
        client.speed(0)
        pad = vg.VX360Gamepad()
        pad.reset(); pad.update()
        time.sleep(.5)
        # TICK reports pre-tick telemetry but the capture catalogue labels the
        # post-tick clock. Prime that boundary explicitly; never admit it to RL.
        pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        pad.update(); time.sleep(.2)
        pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        pad.update(); time.sleep(.2)
        report['administrative_neutral_prime_steps'] = 1
        for sample in range(7):
            client.require_map()
            if sample:
                pad.right_trigger_float(1.0)
                pad.update(); time.sleep(.1)
                pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
                pad.update()
                time.sleep(args.pulse_seconds)
                pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
                pad.update()
                time.sleep(args.pulse_seconds)
            result = client.command('capture-state-clip', {})
            clip_id = result['result']['runtimeClipId']
            deadline = time.monotonic() + 3
            clip = None
            while time.monotonic() < deadline:
                for event in reversed(client.events):
                    if event['eventType'] == 'state-clip-catalog':
                        clip = next((c for c in event['payload'].get('clips', [])
                                     if c['runtimeClipId'] == clip_id), None)
                        if clip:
                            break
                if clip:
                    break
                time.sleep(.05)
            if not clip:
                raise RuntimeError('capture not observed in native catalogue')
            report['snapshots'].append(clip)
            if camera:
                from PIL import Image
                frame = camera.capture()
                Image.fromarray(frame).save(args.output / f'frame-{sample}.png')
                rendered_ms, errors = clock.read(frame)
                vehicles = [e for e in client.events if e['eventType'] == 'vehicle-state']
                report['camera_samples'].append(dict(rendered_ms=rendered_ms,
                    glyph_errors=errors, vehicle=vehicles[-1] if vehicles else None))
        times = [x['raceTime'] for x in report['snapshots']]
        report['race_times_ms'] = times
        report['single_tick_verified'] = all(b - a == 10 for a, b in zip(times, times[1:]))
        report['status'] = 'PASS_CLOCK_ONLY' if report['single_tick_verified'] else 'FAIL_NO_EXACT_STEP'
        if camera:
            report['rendered_clock_matches_capture'] = all(
                s['rendered_ms'] == t for s, t in zip(report['camera_samples'], times))
            report['status'] = 'PASS_CLOCK_AND_RENDER' if report['single_tick_verified'] and report['rendered_clock_matches_capture'] else 'FAIL_FRAME_PAIRING'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if pad:
            pad.reset(); pad.update()
        try:
            if installed_revision is not None:
                current = client.request('settings')
                if current['revision'] != installed_revision:
                    raise RuntimeError('settings changed concurrently; refusing overwrite')
                client.request('settings', dict(expectedRevision=installed_revision,
                                                inputBindings=original['inputBindings']), 'PATCH')
            client.speed(old_speed)
            report['settings_restored'] = True
        except Exception as error:
            report['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        report['events'] = [e for e in client.events if e['eventType'] in
                             ('vehicle-state', 'state-replay', 'revision-rewind')]
        report['reader_errors'] = client.errors
        client.close()
        if camera:
            camera.close()
        (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('events', 'snapshots')}))


if __name__ == '__main__':
    main()
