"""Compare two short rollouts from the same TICK reference state."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('installation')
    p.add_argument('dependencies')
    p.add_argument('calibration')
    p.add_argument('revision')
    p.add_argument('output', type=Path)
    p.add_argument('--real-prefix', action='store_true')
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, a.dependencies)
    from tools.vision_qr1167.bridge import TickBridge
    report = dict(status='FAIL', training_started=False, runs=[])
    bridge = None
    try:
        bridge = TickBridge(a.installation, a.calibration)
        for repeat in range(2):
            frame, info = bridge.reset_prefix(10000) if a.real_prefix else bridge.teleport(a.revision, 10000)
            run = [info]
            report['runs'].append(run)
            for _ in range(4):
                frame, info = bridge.advance((0, 1, 0))
                run.append(info)
            np.save(a.output/f'policy-frame-{repeat}.npy', frame, allow_pickle=False)
        differences = []
        for left, right in zip(*report['runs']):
            lp, rp = left['telemetry'], right['telemetry']
            differences.append(max(abs(lp['position'][k]-rp['position'][k]) for k in ('x','y','z')))
        report['max_position_difference_m'] = max(differences)
        report['initial_speeds'] = [r[0]['telemetry']['speedMetersPerSecond'] for r in report['runs']]
        report['status'] = 'PASS_SHORT_RESTORE_PARITY' if max(differences) < 1e-4 else 'FAIL_RESTORE_PARITY'
        if a.real_prefix:
            report['status'] = 'PASS_MEASURED_PREFIX_HANDOFF'
            report['same_state_parity_claimed'] = False
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if bridge:
            report['events'] = list(bridge.client.events)
            report['last_native'] = bridge.last
            if hasattr(bridge, 'last_frame'):
                from PIL import Image
                Image.fromarray(bridge.last_frame).save(a.output/'last-camera.png')
            try:
                bridge.close()
                report['controls_released_settings_restored'] = True
            except Exception as error:
                report['cleanup_error'] = str(error)
        (a.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs', 'events')}))


if __name__ == '__main__':
    main()
