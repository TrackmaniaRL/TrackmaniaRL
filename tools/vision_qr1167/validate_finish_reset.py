"""Bounded teacher-only lifecycle check, with no policy or learner updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('preparation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan = json.loads((args.preparation/'campaign-08/plan.json').read_text())
    sys.path.insert(0, plan['dependencies'])
    from tools.vision_qr1167.bridge import TickBridge
    from tools.vision_qr1167.tick_client import UID
    report = dict(status='FAIL', teacher_only=True, learner_updates=0,
                  autonomous_record=False, cleanupErrors=[])
    for name in ('bridge.py', 'tick_client.py'):
        report[name+'_sha256'] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    bridge = None
    start = time.monotonic()
    try:
        bridge = TickBridge(plan['installation'], plan['calibration'], fast_steps=True)
        status = bridge.client.require_map()
        report['initial'] = status
        if status['loadedInputRevisionId'] != '1ccf57c0-fe6f-4f44-8ef5-dcc602bd05d8':
            raise RuntimeError('unexpected teacher revision')
        latest = bridge.newest_event()
        marker = latest['_receive_index'] if latest else -1
        bridge.revision = bridge.client.request('settings', dict(
            expectedRevision=bridge.revision, inputExecutionEnabled=True), 'PATCH')['revision']
        bridge.client.speed(1)
        bridge.pad.press_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update()
        time.sleep(.2)
        bridge.pad.release_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update()
        deadline = time.monotonic()+25
        while time.monotonic() < deadline:
            fresh = [e for e in list(bridge.client.events)
                     if e['eventType']=='vehicle-state' and e['_receive_index']>marker]
            resets = [e for e in fresh if e['payload']['raceTick']==0]
            reset_marker = resets[0]['_receive_index'] if resets else float('inf')
            goals = [e for e in fresh if e['_receive_index']>reset_marker
                     and e['payload']['currentCheckpointNumber'] >= e['payload']['totalCheckpointCount']>0]
            if goals:
                goal_event = goals[0]
                break
            time.sleep(.01)
        else:
            raise TimeoutError('teacher did not reach real finish')
        bridge.last = goal_event['payload']
        report['teacher_goal'] = bridge.last
        if bridge.last['raceTick'] != 17601:
            raise RuntimeError('teacher finish changed')
        if time.monotonic()-start > 60:
            raise TimeoutError('lifecycle wall budget')
        # Keep the same event stream, including all high-sequence old-race events.
        _, reset = bridge.reset_prefix(0)
        report['reset'] = reset
        report['retained_old_goal_event'] = goal_event in bridge.client.events
        report['native_sequence_restarted'] = reset['telemetry']['sequence'] < goal_event['payload']['sequence']
        if reset['telemetry']['raceTick'] >= 1000:
            raise RuntimeError('reset did not return to start')
        report['steps'] = []
        for _ in range(3):
            before = bridge.last['raceTick']
            _, info = bridge.advance((0,1,0))
            clocks = [before]+[s['raceTick'] for s in info['native_steps']]
            if len(clocks)!=6 or any(b-a!=10 for a,b in zip(clocks,clocks[1:])):
                raise RuntimeError('post-finish physics step mismatch')
            if info['rendered_ms'] != clocks[-1]+10:
                raise RuntimeError('post-finish camera mismatch')
            report['steps'].append(info)
        if bridge.client.require_map()['currentMapUid'] != UID:
            raise RuntimeError('unexpected map')
        report['status'] = 'PASS_FINISH_RESET_EXACT_STEPS_CAMERA'
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if bridge:
            try:
                bridge.close()
            except Exception as error:
                report['cleanupErrors'].append(str(error))
        report['wall_seconds'] = time.monotonic()-start
        (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('initial','reset','teacher_goal','steps')}))
    if not report['status'].startswith('PASS') or report['cleanupErrors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
