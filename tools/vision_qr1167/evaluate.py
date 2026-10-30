"""Bounded frozen-policy evaluation; requires the training owner to be gone."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def checkpoint_source(campaign, completion, initialization=None):
    if initialization is None:
        if completion['checkpoint'] != 'latest.pt':
            raise ValueError('closed checkpoint name mismatch')
        path = campaign/'latest.pt'
        expected = completion['checkpoint_sha256']
        provenance = dict(policy_kind='RL', training_updates=completion['learner_updates'])
    else:
        receipt = json.loads((initialization/'result.json').read_text())
        if receipt['status'] != 'BC_INITIALIZATION_ONLY' or receipt['rl_updates'] != 0:
            raise ValueError('expected a supervised-only initialization receipt')
        path = initialization/'initialization.pt'
        expected = receipt['policy_sha256']
        provenance = dict(policy_kind='BC_ONLY', training_updates=0,
                          supervised_updates=receipt['bc_updates'],
                          initialization_receipt=str(initialization/'result.json'))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != expected:
        raise ValueError('checkpoint SHA mismatch')
    return path, sha, provenance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--race-ms', type=int, default=30000)
    parser.add_argument('--initialization', type=Path,
                        help='Evaluate a separately admitted BC initialization; never report it as RL.')
    args = parser.parse_args()
    if not 1 <= args.trials <= 100 or not 30 <= args.seconds <= 1800:
        raise ValueError('evaluation budget out of bounds')
    if not 1000 <= args.race_ms <= 30000 or (args.race_ms != 30000 and not args.diagnostic):
        raise ValueError('short driving horizon is diagnostic only')
    completion = json.loads((args.campaign/'completion.json').read_text())
    if completion['cleanupErrors']:
        raise RuntimeError('training cleanup incomplete')
    owner = json.loads((args.campaign/'windows-launch.json').read_text())
    # PID reuse is harmless; refuse only the exact previous owner.
    script = (f"$p=Get-Process -Id {int(owner['pid'])} -ErrorAction SilentlyContinue;"
              f"if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq '{int(owner['birthUtcTicks'])}'){{exit 9}};exit 0")
    subprocess.run(['powershell','-NoProfile','-Command',script], check=True)
    plan = json.loads((args.campaign/'plan.json').read_text())
    sys.path.insert(0, plan['dependencies'])
    import numpy as np
    import torch
    from tools.vision_qr1167.bridge import TickBridge
    from tools.vision_qr1167.learning import Learning
    from tools.vision_qr1167.model import ACTIONS
    checkpoint, sha, provenance = checkpoint_source(args.campaign, completion, args.initialization)
    args.output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', policy_checkpoint_sha256=sha,
                  trials=[],
                  evaluation_updates=0, epsilon=0, teacher_playback=False,
                  public_record_claim=False, cleanupErrors=[])
    report.update(provenance)
    report.update(diagnostic=args.diagnostic, visual_ghost_absence_verified=False,
                  independent_qualification=False, race_horizon_ms=args.race_ms)
    learner = Learning(device='cpu')
    learner.restore(torch.load(checkpoint, map_location='cpu', weights_only=False))
    initial_updates = learner.learner.update_count
    if initial_updates != report['training_updates']:
        raise ValueError('checkpoint update count disagrees with provenance')
    bridge = None
    started = time.monotonic()
    try:
        bridge = TickBridge(plan['installation'], plan['calibration'],
                            fast_steps=plan.get('fast_steps', False))
        with (args.output/'native-traces.jsonl').open('x', buffering=1) as trace:
            for trial in range(args.trials):
                if time.monotonic()-started >= args.seconds:
                    break
                result = dict(trial=trial, goal_ms=None, status='INCOMPLETE')
                report['trials'].append(result)
                try:
                    frame, info = bridge.reset_prefix(0)
                    # This fixed neutral startup is part of the controller protocol,
                    # included in the native race clock. No teacher or perturbation.
                    result['neutral_startup_race_ms'] = info['telemetry']['raceTick']
                    history = deque([frame.copy() for _ in range(4)], maxlen=4)
                    for step in range(600):
                        if time.monotonic()-started >= args.seconds:
                            break
                        action = learner.action(np.stack(history), 0)
                        frame, info = bridge.advance(ACTIONS[action])
                        if args.diagnostic and step % 20 == 0 and not info.get('goal'):
                            from PIL import Image
                            Image.fromarray(bridge.last_frame).save(args.output/f'trial-{trial}-step-{step}.png')
                        trace.write(json.dumps(dict(trial=trial, step=step,
                                                    action=action, evidence=info))+'\n')
                        history.append(frame)
                        if info.get('goal'):
                            result.update(status='NATIVE_GOAL', goal_ms=info['telemetry']['raceTick'])
                            break
                        if info['telemetry']['raceTick'] >= args.race_ms:
                            result['status'] = 'DIAGNOSTIC_HORIZON' if args.diagnostic else 'NO_GOAL_30S'
                            break
                except Exception as error:
                    result.update(status='FAIL_TRIAL', error=type(error).__name__+': '+str(error))
                    if hasattr(error, 'evidence'):
                        result['failure_evidence'] = error.evidence
                    if args.diagnostic and getattr(bridge, 'last_frame', None) is not None:
                        from PIL import Image
                        filename = f'trial-{trial}-last-captured-frame.png'
                        Image.fromarray(bridge.last_frame).save(args.output/filename)
                        result['last_captured_frame'] = filename
                        result['captured_frame_native_state'] = getattr(bridge, 'last_frame_native_state', None)
                        result['last_native_state'] = bridge.last
                    # Preserve the failure and stop; do not make additional state
                    # transitions after a failed physics or camera contract.
                    break
                finally:
                    (args.output/'result.json').write_text(json.dumps(report, indent=2))
        assert learner.learner.update_count == initial_updates
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
        report['status'] = 'EVALUATED' if len(report['trials']) == args.trials and all(
            t['status'] in ('NATIVE_GOAL','NO_GOAL_30S','DIAGNOSTIC_HORIZON') for t in report['trials']) else 'INCOMPLETE'
    except Exception as error:
        report.update(status='FAIL', error=type(error).__name__+': '+str(error))
    finally:
        if bridge:
            try:
                bridge.close()
            except Exception as error:
                report['cleanupErrors'].append(str(error))
        report['elapsed_seconds'] = time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
