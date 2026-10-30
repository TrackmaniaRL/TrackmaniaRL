"""Bounded raw HUD evidence collection; rejected images never enter training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        raise ValueError('clock probe requires 1..200 steps')
    plan = json.loads((args.campaign/'plan.json').read_text())
    sys.path.insert(0, plan['dependencies'])
    from PIL import Image
    from tools.vision_qr1167.bridge import TickBridge
    args.output.mkdir(exist_ok=False)
    report = dict(status='FAIL', training_updates=0, diagnostic_only=True,
                  samples=[], cleanupErrors=[])
    bridge = None
    started = time.monotonic()
    try:
        bridge = TickBridge(plan['installation'], plan['calibration'], fast_steps=True)
        for i in range(args.steps):
            if time.monotonic()-started > 90:
                raise TimeoutError('clock probe budget')
            prior = bridge.last
            native = bridge.tick((0, 0, 0))
            if prior and native['raceTick']-prior['raceTick'] != 10:
                raise RuntimeError('native clock discontinuity')
            frame = bridge.camera.capture()
            filename = f'frame-{i:04d}.png'
            Image.fromarray(frame).save(args.output/filename)
            row = dict(path=filename, native=native,
                       sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest())
            try:
                value, scores = bridge.clock.read(frame)
                row.update(rendered_ms=value, matches_native=value==native['raceTick']+10,
                           scores=scores)
            except ValueError as error:
                row['read_error'] = str(error)
            report['samples'].append(row)
        report['status'] = 'COLLECTED_RAW_CLOCK_DIAGNOSTIC'
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        if bridge:
            try:
                bridge.close()
            except Exception as error:
                report['cleanupErrors'].append(str(error))
        report['elapsed_seconds'] = time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report['status'], samples=len(report['samples']),
        rejected=sum(bool(r.get('read_error')) or not r.get('matches_native',False)
                     for r in report['samples']), error=report.get('error'),
        cleanupErrors=report['cleanupErrors'])))


if __name__ == '__main__':
    main()
