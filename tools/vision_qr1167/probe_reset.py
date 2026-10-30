"""Bounded actual GiveUp reset using the training controller, not a teleport."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('installation')
    p.add_argument('dependencies')
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, a.dependencies)
    import vgamepad as vg
    from tools.vision_qr1167.tick_client import TickClient
    c = TickClient(a.installation)
    original = c.require_map()
    pad = vg.VX360Gamepad()
    report = dict(status='FAIL', training_started=False)
    try:
        if original['inputExecutionEffective']:
            raise RuntimeError('playback must be disabled')
        pad.reset(); pad.update()
        c.speed(1)
        time.sleep(.1)
        pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        pad.update(); time.sleep(.2)
        pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        pad.update()
        c.speed(0)
        time.sleep(.2)
        report['capture'] = c.command('capture-state-clip', {})
        report['status'] = 'RESET_OBSERVED_REQUIRES_BOUNDARY_CHECK'
    except Exception as e:
        report['error'] = type(e).__name__ + ': ' + str(e)
    finally:
        pad.reset(); pad.update()
        report['events'] = list(c.events)
        c.speed(original['effectiveGameSpeed'])
        c.close()
        (a.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    vehicles = [e['payload']['raceTick'] for e in report['events'] if e['eventType']=='vehicle-state']
    print(json.dumps(dict(status=report['status'], error=report.get('error'), clocks=vehicles,
                         event_types=sorted({e['eventType'] for e in report['events']}))))


if __name__ == '__main__':
    main()
