"""Owned, bounded unmodified native physics batches; compressed reproducible evidence."""
from __future__ import annotations
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from core import native_goal

BASE = Path('/home/kuba/tmrl-full-qr-20261022-011')
SERVER = BASE / 'waypoint-runtime002/build/server-probe/linux-validate-map2-001'
MAP = BASE / 'waypoint-runtime002/analysis/tmrl-shortcut/assets-001/tmrl-test.Map.Gbx'
SERVER_SHA = '3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779'
MAP_SHA = 'ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc'
GHOST_SHA = '221d9960f6dda43bd71b1c5205ee4e13c87466080e1224a11c63999fe36d78a9'


def clock() -> float:
    value = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    if not math.isfinite(value) or value < 0:
        raise RuntimeError('Invalid monotonic clock')
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    stage = path.with_suffix('.stage')
    stage.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    stage.replace(path)


def identity(pid: int) -> dict[str, Any] | None:
    p = Path(f'/proc/{pid}')
    try:
        s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(s[1]), pgid=int(s[2]), sid=int(s[3]),
                    birth=int(s[19]), uid=p.stat().st_uid,
                    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def descendants(parent: int) -> list[dict[str, Any]]:
    rows = [r for p in Path('/proc').iterdir() if p.name.isdigit()
            if (r := identity(int(p.name))) is not None]
    owned = {parent}
    while True:
        more = {r['pid'] for r in rows if r['ppid'] in owned}
        if more <= owned:
            break
        owned |= more
    return [r for r in rows if r['pid'] in owned and r['pid'] != parent]


def reserves(root: Path, private: Path) -> None:
    for path, gib in ((Path('/mnt/c'), 10), (root, 4), (BASE, 4)):
        if shutil.disk_usage(path).free < gib * 2**30:
            raise RuntimeError('Storage reserve')
    size = 0
    for directory in (root, private):
        for p in directory.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():
                    size += p.stat().st_size
            except FileNotFoundError:
                continue
    if size >= 8 * 2**30:
        raise RuntimeError('Own8GiB quota')


def run_batch(root: Path, private: Path, number: int, baseline: tuple[int, ...],
              jobs: list[tuple[str, tuple[int, ...]]], stop: Callable[[], bool],
              heartbeat: Callable[[dict[str, Any]], None], timeout: float = 240
              ) -> tuple[list[tuple[str, int | None]], dict[str, Any]]:
    reserves(root, private)
    if sha(SERVER / 'TrackmaniaServer') != SERVER_SHA or sha(MAP) != MAP_SHA:
        raise RuntimeError('Frozen native server/map fingerprint changed')
    source = root / 'protected/native17.712.Ghost.Gbx'
    binary = private / 'native-line1134'
    manifest = json.loads((root / 'plan.json').read_text())
    if sha(source) != GHOST_SHA or sha(binary) != manifest['writer_sha256']:
        raise RuntimeError('Frozen writer/source fingerprint changed')
    batch = root / 'batches' / f'batch-{number:06d}'
    work = private / f'batch-{number:06d}'
    batch.mkdir(parents=True, exist_ok=False)
    work.mkdir(exist_ok=False)
    spec = dict(baseline=baseline, active_ticks=1771,
                candidates=[dict(name=n, ticks=t) for n, t in jobs])
    data = json.dumps(spec, separators=(',', ':')).encode()
    (batch / 'spec.json.gz').write_bytes(gzip.compress(data))
    (work / 'spec.json').write_bytes(data)
    subprocess.run([str(binary), str(source), str(work / 'spec.json'), str(work / 'tapes')],
                   check=True, timeout=30, stdout=subprocess.DEVNULL)
    proof = (work / 'tapes/encoding-proof.json').read_bytes()
    (batch / 'encoding-proof.json.gz').write_bytes(gzip.compress(proof))
    server_work = work / 'server'
    (server_work / 'UserData/Replays').mkdir(parents=True)
    (server_work / 'UserData/Maps').mkdir()
    (server_work / 'TrackmaniaServer').symlink_to(SERVER / 'TrackmaniaServer')
    (server_work / 'Packs').symlink_to(SERVER / 'Packs', target_is_directory=True)
    (server_work / 'UserData/Maps/tmrl-test.Map.Gbx').symlink_to(MAP)
    files = {}
    for name, _ in jobs:
        p = work / 'tapes' / f'INPUTONLY-{name}.Ghost.Gbx'
        files[p.name] = sha(p)
        shutil.copyfile(p, server_work / 'UserData/Replays' / p.name)
    env = os.environ.copy()
    for key in ('LD_PRELOAD', 'TMRE_SHIM', 'TMRE_TRACE_DIR'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    started = clock()
    report: dict[str, Any] = dict(status='RUNNING', inputs=files, native_count=len(jobs),
                                  server_sha256=SERVER_SHA, map_sha256=MAP_SHA,
                                  source_ghost_sha256=GHOST_SHA, timeout_seconds=timeout,
                                  source_shim_loaded=False, forced=[], cleanupErrors=[])
    output = work / 'stdout.json'
    with output.open('xb') as stdout, (batch / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(['./TrackmaniaServer', '/nodaemon', '/validatepath=.'],
                                 cwd=server_work, env=env, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        saved = identity(child.pid)
        if saved is None or saved['sid'] != child.pid or saved['pgid'] != child.pid:
            raise RuntimeError('Native session identity not isolated')
        report['native_identity'] = saved
        save(batch / 'launch.json', report)
        pidfd = os.pidfd_open(child.pid)
        last_size = 0
        last_growth = clock()
        failure = None
        try:
            while child.poll() is None:
                now = clock()
                raw = output.read_bytes()
                size = len(raw)
                if size != last_size:
                    last_growth = now
                    last_size = size
                heartbeat(dict(batch=number, native_identity=saved,
                               completed_native_records=len(re.findall(rb'"FileName"\s*:', raw)),
                               queued_native_records=len(jobs), native_stdout_bytes=size,
                               native_elapsed_seconds=now-started))
                if stop():
                    raise RuntimeError('Own search STOP/deadline')
                if now-started > timeout or now-last_growth > 120:
                    raise RuntimeError('Native batch timeout/output stagnation')
                if descendants(child.pid):
                    raise RuntimeError('Unexpected native descendant; preserve evidence')
                time.sleep(.25)
        except BaseException as error:
            failure = error
        finally:
            if child.poll() is None:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    report['forced'].append(saved)
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    child.wait(timeout=5)
            os.close(pidfd)
        report.update(exit_code=child.returncode, elapsed_seconds=clock()-started,
                      exact_native_identity_gone=identity(child.pid) != saved,
                      remaining_descendants=descendants(child.pid))
        if failure is not None:
            report.update(status='FAILED_PRESERVED', error=repr(failure))
            save(batch / 'result.json', report)
            raise failure
    if child.returncode != 0 or not report['exact_native_identity_gone'] or report['remaining_descendants']:
        save(batch / 'result.json', report)
        raise RuntimeError('Native exit/closure failure')
    raw = output.read_bytes()
    (batch / 'native-output.json.gz').write_bytes(gzip.compress(raw))
    records = json.loads(raw)
    wanted = {f'INPUTONLY-{n}.Ghost.Gbx': n for n, _ in jobs}
    if len(records) != len(wanted) or {r['FileName'] for r in records} != set(wanted):
        raise RuntimeError('Native records missing, duplicated or unbound')
    results = [(wanted[r['FileName']], native_goal(r)) for r in records]
    report.update(status='PASS_NATIVE_PHYSICS_BATCH_NOT_AUTONOMOUS_EVALUATION',
                  results=results, full_output_sha256=hashlib.sha256(raw).hexdigest(),
                  writer_sha256=sha(binary), input_files_unchanged=all(
                      sha(work / 'tapes' / n) == h for n, h in files.items()))
    if not report['input_files_unchanged'] or sha(SERVER / 'TrackmaniaServer') != SERVER_SHA or sha(MAP) != MAP_SHA:
        raise RuntimeError('Backend/input mutated during validation')
    # Keep the best input-only ghost from every batch. This is not a visual replay.
    finishes = [(goal, name) for name, goal in results if goal is not None]
    if finishes:
        goal, name = min(finishes)
        keep = batch / 'best-inputs-not-visual.Ghost.Gbx'
        shutil.copyfile(work / 'tapes' / f'INPUTONLY-{name}.Ghost.Gbx', keep)
        report['batch_best'] = dict(name=name, actual_native_goal_ms=goal,
                                   input_only_ghost=str(keep), sha256=sha(keep))
    save(batch / 'result.json', report)
    # Only this closed, generated batch directory. Symlinked packs/maps are not followed.
    resolved = work.resolve()
    if work.is_symlink() or resolved.parent != private.resolve() or not re.fullmatch(r'batch-[0-9]{6}', resolved.name):
        raise RuntimeError('Unsafe own scratch cleanup path')
    shutil.rmtree(resolved)
    return results, report
