"""Bounded real-camera QR-DQN campaign. Records never certify a public WR."""
import argparse
from collections import deque
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix('.stage')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    # Windows readers/antivirus may briefly hold the destination without delete
    # sharing. Preserve atomic replacement, but tolerate a bounded sharing clash.
    deadline = time.monotonic() + 2
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.02)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    root = a.root
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    # Exclusive launch receipt, separate from mutable monitoring state.
    with (root/'runner-started.json').open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), f)
    sys.path.insert(0, plan['dependencies'])
    import numpy as np
    import torch
    from trackmaniarl.core.environment_errors import EnvironmentPausedError
    from tools.vision_qr1167.bridge import TickBridge, FramePairingError
    from tools.vision_qr1167.curriculum import ProgressReward, ProgressDiscontinuity
    from tools.vision_qr1167.learning import Learning
    from tools.vision_qr1167.model import ACTIONS
    from tools.vision_qr1167.reference import Reference
    reference_bytes = Path(plan['reference']).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != plan['reference_sha256']:
        raise ValueError('reference changed since launch preparation')
    reference_data = json.loads(reference_bytes)
    if reference_data['status'] != 'PASS_REWARD_REFERENCE':
        raise ValueError('reward reference not admitted')
    learning = None
    bridge = None
    rng = random.Random(1167)
    start = time.monotonic()
    state = dict(phase='starting', training_started=False, learner_updates=0,
        transitions=0, episodes=0, goals=0, full_start_goals=0,
        best_goal_ms=None, public_record_claim=False, remainingOwned=[], cleanupErrors=[])

    def publish():
        state['heartbeat_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state['elapsed_seconds'] = time.monotonic()-start
        if learning:
            state['learner_updates'] = learning.learner.update_count
            state['transitions'] = learning.transitions
            state['auxiliary_imitation_updates'] = learning.retention.steps if learning.retention else 0
        atomic_json(root/'state.json', state)

    def checkpoint(name='latest.pt'):
        if learning is None:
            return
        temporary = root/(name+'.stage')
        torch.save(learning.checkpoint(), temporary)
        os.replace(temporary, root/name)
        state['checkpoint'] = name
        state['checkpoint_sha256'] = hashlib.sha256((root/name).read_bytes()).hexdigest()

    try:
        publish()
        learning = Learning(device=plan['device'])
        if plan.get('resume_checkpoint'):
            source = Path(plan['resume_checkpoint'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != plan['resume_sha256']:
                raise ValueError('resume checkpoint SHA changed')
            learning.restore(torch.load(source, map_location='cpu', weights_only=False))
            state['resumed_transitions'] = learning.transitions
        if plan.get('retention_dataset'):
            learning.configure_retention(plan['retention_dataset'])
            if plan.get('admit_extended_teacher_actions'):
                state['admitted_action_extensions']=learning.extend_actions_from_retention()
        bridge = TickBridge(plan['installation'], plan['calibration'], fast_steps=plan.get('fast_steps', False))
        state['phase'] = 'collecting'
        with (root/'transitions.jsonl').open('x', encoding='utf-8', buffering=1) as trace:
            while time.monotonic()-start < plan['max_seconds'] and learning.transitions < plan['max_transitions']:
                if (root/'STOP').exists():
                    state['stop_reason'] = 'operator_stop'
                    break
                prefix = 0 if rng.random() < .5 else rng.choice(list(range(2000,16001,1000)))
                state['prefix_requested_ms'] = prefix
                state['phase'] = 'resetting'
                publish()
                try:
                    frame, info = bridge.reset_prefix(prefix)
                    # State perturbations are real steering actions, not edited physics.
                    frame, info = bridge.advance((rng.uniform(-.12,.12), 1, 0), rng.randint(1,5))
                except FramePairingError as error:
                    number = state.get('rejected_resets', 0) + 1
                    state['rejected_resets'] = number
                    atomic_json(root/f'rejected-reset-{number}.json', dict(
                        status='FAIL_RESET_OBSERVATION', error=str(error), admitted_to_replay=False))
                    state['consecutive_bad_resets'] = state.get('consecutive_bad_resets', 0) + 1
                    if state['consecutive_bad_resets'] >= 5:
                        raise RuntimeError('five resets without an admissible camera observation') from error
                    checkpoint()
                    publish()
                    continue
                state['consecutive_bad_resets'] = 0
                reference = Reference(reference_data['records'])
                distance, lateral = reference.reset(info['telemetry']['raceTick'], info['telemetry']['position'])
                rewarder = ProgressReward(distance)
                history = deque([frame.copy() for _ in range(4)], maxlen=4)
                observation = np.stack(history)
                episode_start = info['telemetry']['raceTick']
                learning.begin_episode()
                last_progress = distance
                stalled = 0.
                state['phase'] = 'training'
                for step in range(600):
                    if (root/'STOP').exists() or time.monotonic()-start >= plan['max_seconds']:
                        break
                    if step % 32 == 0:
                        if shutil.disk_usage('C:/').free < 10*2**30 or shutil.disk_usage(root).free < 4*2**30:
                            raise RuntimeError('disk reserve guard')
                    epsilon = max(plan.get('epsilon_minimum', .08), plan.get('epsilon_initial', .5) - learning.transitions/plan.get('epsilon_decay_transitions', 50000))
                    action = learning.action(observation, epsilon)
                    try:
                        frame, info = bridge.advance(ACTIONS[action])
                    except FramePairingError as error:
                        from PIL import Image
                        state['rejected_observations'] = state.get('rejected_observations', 0)+1
                        number = state['rejected_observations']
                        Image.fromarray(bridge.last_frame).save(root/f'rejected-frame-{number}.png')
                        atomic_json(root/f'rejected-frame-{number}.json', dict(status='FAIL_FRAME_PAIRING',
                            error=str(error), native=bridge.last, admitted_to_replay=False))
                        break
                    dt = len(info['native_steps'])*.01
                    telemetry = info['telemetry']
                    distance, lateral = reference.project(telemetry['position'], dt)
                    goal = bool(info.get('goal', False))
                    try:
                        reward = rewarder.step(distance, lateral, dt, goal=goal)
                    except ProgressDiscontinuity as error:
                        number = state.get('rejected_progress', 0) + 1
                        state['rejected_progress'] = number
                        atomic_json(root/f'rejected-progress-{number}.json', dict(
                            status='FAIL_PROGRESS_TRANSITION', error=str(error),
                            episode=state['episodes'], step=step, evidence=info,
                            projected_distance=distance, previous_maximum=rewarder.maximum,
                            lateral_m=lateral, dt=dt, admitted_to_replay=False))
                        state['last_episode_end'] = 'rejected_progress_reset'
                        break
                    stalled = stalled+dt if distance-last_progress < .01 else 0.
                    last_progress = max(last_progress, distance)
                    truncated = stalled >= 3 or lateral > 25 or telemetry['raceTick']-episode_start >= 30000 or step==599
                    history.append(frame)
                    next_observation = np.stack(history)
                    learning.add(observation, action, reward, next_observation, goal, truncated)
                    metrics = None
                    # Collection is ~2.2s per transition; updates take ~20ms.
                    # Only increase replay reuse once a diverse buffer exists.
                    updates = plan.get('updates_per_transition', 1) if len(learning.replay) >= 512 else 1
                    for _ in range(updates):
                        metrics = learning.update()
                    observation = next_observation
                    state.update(training_started=True, epsilon=epsilon, latest_reward=reward,
                        latest_race_ms=telemetry['raceTick'], progress_m=distance, lateral_m=lateral)
                    if metrics:
                        state['metrics'] = metrics
                    trace.write(json.dumps(dict(episode=state['episodes'], step=step,
                        action=action, reward=reward, terminated=goal, truncated=truncated,
                        prefix_ms=prefix, evidence=info))+'\n')
                    if learning.transitions % 100 == 0:
                        checkpoint()
                    publish()
                    if goal:
                        state['goals'] += 1
                        if prefix == 0:
                            state['full_start_goals'] += 1
                            measured = telemetry['raceTick']
                            if state['best_goal_ms'] is None or measured < state['best_goal_ms']:
                                state['best_goal_ms'] = measured
                                checkpoint('best-candidate.pt')
                                # Training includes perturbations; record requires a
                                # separate frozen-policy actual-start evaluation.
                                state['independent_evaluation_pending'] = True
                        break
                    if truncated:
                        break
                state['episodes'] += 1
                checkpoint()
                publish()
            state['stop_reason'] = state.get('stop_reason', 'bounded_budget')
            state['phase'] = 'completed_training_budget'
    except EnvironmentPausedError as error:
        # No transition is admitted after a missing image. Preserve the learner
        # and close this run; resuming requires a fresh environment/reset.
        state['phase'] = 'stopped_environment_unavailable'
        state['stop_reason'] = 'environment_unavailable'
        state['error'] = type(error).__name__ + ': ' + str(error)
        atomic_json(root/'environment-unavailable.json', dict(
            status='STOPPED_ENVIRONMENT_UNAVAILABLE', error=state['error'],
            admitted_to_replay=False, fresh_reset_required=True))
    except BaseException as error:
        state['phase'] = 'failed'
        state['error'] = type(error).__name__ + ': ' + str(error)
        if hasattr(error, 'evidence'):
            atomic_json(root/'failure-evidence.json', error.evidence)
        (root/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        try:
            checkpoint()
        except Exception as error:
            state['cleanupErrors'].append('checkpoint: '+str(error))
        if bridge:
            try:
                bridge.close()
            except Exception as error:
                state['cleanupErrors'].append('bridge: '+str(error))
        publish()
        atomic_json(root/'completion.json', state)
    if state['phase']=='failed' or state['cleanupErrors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
