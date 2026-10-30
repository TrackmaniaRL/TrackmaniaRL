"""Collect an ordered reward path from real execution of the frozen teacher."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('installation')
    p.add_argument('dependencies')
    p.add_argument('calibration')
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, a.dependencies)
    from tools.vision_qr1167.bridge import TickBridge
    bridge = None
    report = dict(status='FAIL', training_started=False)
    try:
        bridge = TickBridge(a.installation, a.calibration)
        bridge.revision = bridge.client.request('settings', dict(
            expectedRevision=bridge.revision, inputExecutionEnabled=True), 'PATCH')['revision']
        bridge.client.speed(1)
        bridge.pad.press_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update(); time.sleep(.2)
        bridge.pad.release_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update()
        deadline = time.monotonic()+30
        saw_reset = False
        while time.monotonic() < deadline:
            last = bridge.newest()
            if last and last['raceTick'] == 0 and last['currentCheckpointNumber'] == 0:
                saw_reset = True
            if saw_reset and last and last['currentCheckpointNumber'] >= last['totalCheckpointCount'] > 0:
                report['finish_telemetry'] = last
                bridge.last = last
                break
            time.sleep(.01)
        bridge.client.speed(0)
        # Find the final actual reset; exclude old retained socket samples.
        vehicles = [e['payload'] for e in bridge.client.events if e['eventType']=='vehicle-state']
        start = max(i for i,r in enumerate(vehicles) if r['raceTick']==0)
        records = []
        for r in vehicles[start:]:
            if not records or r['raceTick'] > records[-1]['raceTick']:
                records.append(r)
        report['records'] = records
        report['status'] = 'PASS_REWARD_REFERENCE' if 'finish_telemetry' in report else 'FAIL_FINISH_NOT_OBSERVED'
        report['record_claim'] = False
        report['teacher_handoff_neutral_tick_ms'] = 0
        if 'finish_telemetry' in report:
            bridge.reset_prefix(0)
    except Exception as error:
        report['path_captured'] = report['status'] == 'PASS_REWARD_REFERENCE'
        report['status'] = 'FAIL'
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if bridge:
            report['events'] = list(bridge.client.events)
            try:
                bridge.close()
                report['cleanup_complete'] = True
            except Exception as error:
                report['cleanup_error'] = str(error)
        (a.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','records')}))


if __name__=='__main__':
    main()
