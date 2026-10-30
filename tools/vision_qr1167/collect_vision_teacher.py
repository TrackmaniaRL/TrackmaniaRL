"""Collect measured camera/teacher pairs after the campaign closes.

Labels are native applied controls, not inferred from the image or goal time.
This creates demonstration data only; it never claims an autonomous result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--game-speed', type=float, default=.25)
    p.add_argument('--max-seconds', type=int, default=120)
    a = p.parse_args()
    if not .05 <= a.game_speed <= 1 or not 30 <= a.max_seconds <= 240:
        raise ValueError('teacher collection limits')
    closed = json.loads((a.campaign/'completion.json').read_text())
    if closed['cleanupErrors']:
        raise RuntimeError('previous cleanup incomplete')
    owner = json.loads((a.campaign/'windows-launch.json').read_text())
    subprocess.run(['powershell','-NoProfile','-Command',
        f"$p=Get-Process -Id {int(owner['pid'])} -ErrorAction SilentlyContinue;"
        f"if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq '{int(owner['birthUtcTicks'])}'){{exit 9}};exit 0"], check=True)
    plan = json.loads((a.campaign/'plan.json').read_text())
    sys.path.insert(0, plan['dependencies'])
    import numpy as np
    from tools.vision_qr1167.bridge import TickBridge, FramePairingError
    a.output.mkdir(exist_ok=False)
    report = dict(status='FAIL', autonomous=False, teacher_playback=True,
                  samples=[], cleanupErrors=[], training_updates=0,
                  game_speed=a.game_speed,max_seconds=a.max_seconds)
    bridge = None
    started = time.monotonic()
    try:
        bridge = TickBridge(plan['installation'],plan['calibration'])
        status = bridge.client.require_map()
        report['map_uid'] = status['currentMapUid']
        report['revision_id'] = status['loadedInputRevisionId']
        if report['revision_id'] != '1ccf57c0-fe6f-4f44-8ef5-dcc602bd05d8':
            raise RuntimeError('expected frozen 17.601 teacher revision')
        bridge.revision = bridge.client.request('settings',dict(
            expectedRevision=bridge.revision,inputExecutionEnabled=True),'PATCH')['revision']
        bridge.client.speed(a.game_speed)
        bridge.pad.press_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update();time.sleep(.2)
        bridge.pad.release_button(button=bridge.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        bridge.pad.update()
        reset_seen = False
        last_clock = -1
        while time.monotonic()-started < a.max_seconds and len(report['samples']) < 500:
            sample = bridge.newest()
            if sample:
                reset_seen |= sample['raceTick']==0
            if not reset_seen or sample is None:
                time.sleep(.001)
                continue
            if sample['currentCheckpointNumber'] >= sample['totalCheckpointCount'] > 0:
                bridge.last = sample
                report['teacher_goal_ms'] = sample['raceTick']
                report['status'] = 'COLLECTED_TEACHER_CAMERA'
                break
            if sample['raceTick'] < last_clock+50:
                time.sleep(.001)
                continue
            # Capture the running game once, decode its own clock, and pair with
            # an exact native clock in the same reset's buffered stream.
            frame = bridge.camera.capture()
            try:
                rendered, errors = bridge.clock.read(frame)
            except ValueError as error:
                report['unreadable_frames'] = report.get('unreadable_frames',0)+1
                continue
            if rendered-10 <= last_clock:
                continue
            matching = None
            deadline=time.monotonic()+.25
            while time.monotonic()<deadline:
                matching=next((e['payload'] for e in reversed(list(bridge.client.events))
                    if e['eventType']=='vehicle-state' and e['payload']['raceTick']==rendered-10),None)
                if matching is not None:break
                time.sleep(.001)
            if matching is None:
                report['unpaired_frames'] = report.get('unpaired_frames',0)+1
                continue
            from PIL import Image
            image=np.asarray(Image.fromarray(frame[200:790,180:1170]).convert('L').resize((84,84)))
            info=dict(rendered_ms=rendered,telemetry_ms=matching['raceTick'],
                telemetry=matching,clock_label_offset_ms=10,glyph_errors=errors)
            filename=f'frame-{len(report["samples"]):04d}.npy'
            np.save(a.output/filename,image,allow_pickle=False)
            report['samples'].append(dict(path=filename,
                sha256=hashlib.sha256((a.output/filename).read_bytes()).hexdigest(),
                evidence=info,label_semantics='native applied controls at observed clock; future interval alignment required'))
            last_clock=matching['raceTick']
        if report['status']=='FAIL':
            raise TimeoutError('bounded collection ended without a measured teacher goal')
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        if bridge:
            report['native_events']=list(bridge.client.events)
            try:bridge.close()
            except Exception as error:report['cleanupErrors'].append(str(error))
        report['wall_seconds']=time.monotonic()-started
        (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','native_events')}))


if __name__=='__main__':main()
