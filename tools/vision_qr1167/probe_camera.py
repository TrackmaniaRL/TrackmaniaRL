"""Bounded real-camera/teleport calibration; never admits samples to replay."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('installation')
    p.add_argument('dependencies')
    p.add_argument('revision')
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, a.dependencies)
    from PIL import Image
    from tools.vision_qr1167.tick_client import TickClient
    from trackmaniarl.trackmania.vision_environment import WindowFrameSource
    from trackmaniarl.trackmania.window_capture import WindowCaptureConfig
    client = TickClient(a.installation)
    camera = WindowFrameSource(WindowCaptureConfig(client_width=1536, client_height=864))
    old_speed = client.require_map()['effectiveGameSpeed']
    report = dict(status='FAIL', training_started=False, samples=[])
    try:
        camera.prepare_episode()
        client.speed(0)
        for digit in range(10):
            requested = 10000 + digit * 110
            command = client.command('teleport-to-revision-time',
                dict(revisionId=a.revision, raceTimeMilliseconds=requested), timeout=15)
            time.sleep(.3)
            frame = camera.capture()
            Image.fromarray(frame).save(a.output / f'camera-{requested}.png')
            vehicles = [e for e in client.events if e['eventType'] == 'vehicle-state']
            report['samples'].append(dict(requested_ms=requested, command=command,
                vehicle=vehicles[-1] if vehicles else None,
                paused=client.require_map()['effectiveGameSpeed'] == 0))
        report['status'] = 'CAPTURED_REQUIRES_CLOCK_READBACK'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        try:
            client.speed(old_speed)
            report['speed_restored'] = True
        except Exception as error:
            report['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        camera.close()
        client.close()
        (a.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                         samples=len(report['samples']))))


if __name__ == '__main__':
    main()
