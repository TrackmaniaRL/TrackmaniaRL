"""Bounded recorded Windows inference with a protected policy-only checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .adapter import ExactDigitalPad, control, execute

UID = 'oqIJ5rQDRrNwLPTh9H2p_W4tLof'
POLICY_SHA = 'eaedead68139309da2e89e2732a31caeba917666095af7ee08152b83186c1762'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_path(value: str) -> Path:
    if value.startswith('/mnt/'):
        return Path(value[5].upper() + ':' + value[6:])
    return Path(value)


def load_policy(study: Path, output: Path) -> tuple[Any, Any, tuple[tuple[int, ...], ...]]:
    from trackmaniarl.algorithms.value_based.policy import (
        DiscreteValuePolicy,
        DiscreteValuePolicyConfig,
    )
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
    from trackmaniarl.models.contracts import RiskSpec
    from trackmaniarl.models.factory import CompositeValueModelFactory

    source = study / 'pretraining1092/policy-only.pt'
    if sha(source) != POLICY_SHA:
        raise ValueError('protected evaluated BC policy SHA mismatch')
    copy = output / 'policy.pt'
    shutil.copyfile(source, copy)
    # Codec uses a temporary next to its input: only load our private copy.
    checkpoint = str(copy.resolve())
    if not checkpoint.startswith('\\\\?\\'):
        checkpoint = '\\\\?\\' + checkpoint
    weights = TorchCheckpointCodec().load(Path(checkpoint))
    spec = yaml.safe_load((study / 'inference1093.yaml').read_text())
    components = spec['components']
    model = CompositeValueModelFactory(**components['model_factory']['kwargs']).build().eval()
    if model.architecture_fingerprint() != weights['architecture_fingerprint']:
        raise ValueError('architecture mismatch')
    for name, state in weights['online'].items():
        getattr(model, name).load_state_dict(state, strict=True)
    policy = DiscreteValuePolicy(model, torch.device('cpu'),
                                DiscreteValuePolicyConfig(0, None, RiskSpec(), RiskSpec()))
    kwargs = components['feature_pipeline']['kwargs']
    pipeline = BoundaryGraphFeaturePipelineV5(windows_path(kwargs['geometry_path']), UID)
    catalog_path = windows_path(components['environment']['kwargs']['config']
                                ['macro_catalog']['path'])
    if sha(catalog_path) != weights['macro_catalog_sha256']:
        raise ValueError('catalog/model mismatch')
    with np.load(catalog_path, allow_pickle=False) as data:
        programs = tuple(tuple(int(a) for a in row) for row in data['programs'])
    if len(programs) != 132:
        raise ValueError('132-program protected model required')
    return policy, pipeline, programs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--controller', choices=('gamepad', 'keyboard'), default='gamepad')
    parser.add_argument('--warmup-seconds', type=int, default=12)
    args = parser.parse_args()
    if not args.live or sys.platform != 'win32' or not 0 <= args.warmup_seconds <= 30:
        raise RuntimeError('explicit bounded Windows live invocation required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    output = Path('\\\\?\\' + str(args.output.resolve()))
    output.mkdir(exist_ok=False)
    policy, pipeline, programs = load_policy(args.study, output)
    # Once weights and unordered vocabulary are loaded, forbid any expert files.
    def deny_teacher(event: str, values: tuple[Any, ...]) -> None:
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            path = str(values[0]).lower()
            if any(word in path for word in ('teacher', 'controls.npz', 'controls.csv',
                                             'demonstration', 'labels-', '.ghost.gbx')):
                # The study directory contains "multi-demonstration": match filenames only.
                name = path.replace('\\', '/').split('/')[-1]
                if any(word in name for word in ('teacher', 'controls', 'demonstration',
                                                 'labels-', '.ghost.gbx')):
                    raise RuntimeError('teacher file access denied during autonomous trial')
    sys.addaudithook(deny_teacher)
    from trackmaniarl.core.contracts import PolicyMode
    from trackmaniarl.trackmania.control import GamepadController, KeyboardController
    from trackmaniarl.trackmania.recording import RecordingOptions, record_window
    from trackmaniarl.trackmania.session import OpenPlanetSessionClient
    from trackmaniarl.trackmania.telemetry import OpenPlanetClient, OpenPlanetClientConfig

    report: dict[str, Any] = {'status': 'STARTING', 'policy_sha256': POLICY_SHA,
                              'learner_updates': 0, 'teacher_sequences_accessible': False,
                              'control_timing': 'host deadlines, not physics ACK',
                              'finish_clock': 'Windows telemetry, not headless native Goal',
                              'decisions': 0, 'finished': False}
    report['controller'] = args.controller
    pad: Any = None
    client = OpenPlanetClient(OpenPlanetClientConfig(timeout_s=1))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    mutex = kernel.CreateMutexW(None, False, 'Global\\TrackmaniaRL.ComparisonController')
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    if not mutex or kernel.WaitForSingleObject(mutex, 0) not in (0, 0x80):
        raise RuntimeError('another Windows controller owns the game')
    timer = ctypes.WinDLL('winmm')
    timer.timeBeginPeriod(1)
    deadline = time.monotonic() + 90
    def guard() -> None:
        if time.monotonic() > deadline or (output / 'STOP').exists():
            raise TimeoutError('own bounded test deadline/STOP')
    try:
        session = OpenPlanetSessionClient('127.0.0.1', 9001, timeout_s=2)
        session.verify_loaded_map(UID)
        session.confirm_ready(UID)
        ffmpeg = args.study / 'ffmpeg1107.exe'
        recording = RecordingOptions(output / 'attempt.mkv', 'Trackmania',
                                     str(ffmpeg.resolve()))
        print('READY_SWITCH_TO_TRACKMANIA', flush=True)
        time.sleep(args.warmup_seconds)
        guard()
        pad = (GamepadController(restart_input='gamepad') if args.controller == 'gamepad'
               else KeyboardController())
        motor = ExactDigitalPad(pad) if args.controller == 'keyboard' else pad
        with record_window(recording):
            pad.reset()
            reset_deadline = time.monotonic() + 12
            while True:
                guard()
                raw = client.read().values
                if raw[3] == 0 and raw[2] == 0:
                    break
                if time.monotonic() > reset_deadline:
                    raise RuntimeError('gamepad reset did not produce a real zero race clock')
            pipeline.reset_episode()
            policy.reset_episode()
            report['initial_raw33'] = raw.tolist()
            observation = pipeline.transform_observation(raw)
            action = policy.act(observation, PolicyMode.EVALUATION)
            # First action is chosen from the real reset observation, before green.
            motor.apply(control(programs[action][0]))
            start_deadline = time.monotonic() + 10
            while raw[3] == 0:
                guard()
                raw = client.read().values
                if time.monotonic() > start_deadline:
                    raise RuntimeError('race did not start after countdown')
            report['first_positive_race_ms'] = float(raw[3])
            epoch = time.perf_counter() - float(raw[3]) / 1000
            report['schedule'] = 'fixed host epoch estimated from first positive actual race clock'
            last_clock = 0.0
            race_deadline = time.monotonic() + 30
            with (output / 'decisions.jsonl').open('x', encoding='utf-8') as log:
                while True:
                    guard()
                    if time.monotonic() > race_deadline:
                        report['status'] = 'DNF_30S_BUDGET'
                        break
                    # Validate the whole selected macro before issuing its first tick.
                    if args.controller == 'keyboard':
                        for command in programs[action]:
                            if int(command) // 6 not in (0, 6, 12):
                                raise ValueError('non-digital model steering; no quantization')
                    ticks = execute(programs[action], motor, guard,
                                    anchor=epoch + int(report['decisions']) * 0.05)
                    raw = client.read().values
                    clock = float(raw[3])
                    if raw[2] == 1:
                        report.update(status='FINISHED_WINDOWS_TELEMETRY', finished=True,
                                      finish_telemetry_ms=clock)
                    elif clock <= last_clock or clock - last_clock > 200:
                        raise RuntimeError('stale/regressed or >200ms telemetry gap')
                    log.write(json.dumps({'decision': report['decisions'], 'action': action,
                              'raw33': raw.tolist(), 'commands': ticks,
                              'q_margin': policy.last_q_margin}, allow_nan=False) + '\n')
                    log.flush()
                    report['decisions'] += 1
                    if report['finished']:
                        break
                    last_clock = clock
                    observation = pipeline.transform_observation(raw)
                    action = policy.act(observation, PolicyMode.EVALUATION)
            pad.apply(np.zeros(3, dtype=np.float32))
            time.sleep(2)
    except BaseException as error:
        report.update(status='FAILED_OR_BLOCKED', error=repr(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        if pad is not None:
            pad.close()
        client.close()
        timer.timeEndPeriod(1)
        kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)
        report['controls_released'] = True
        report['policy_unchanged'] = sha(output / 'policy.pt') == POLICY_SHA
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
