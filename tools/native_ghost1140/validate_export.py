"""One bounded CPU functional validation, with original and same-time controls.

No training, benchmark, source shim, game window or upload.
"""
from __future__ import annotations
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

root=Path(sys.argv[1]).resolve();study=root.parent
sys.path.insert(0,str(study/'line-search1136/source'))
from backend import BASE,MAP,MAP_SHA,SERVER,SERVER_SHA,clock,descendants,identity,save,sha
from core import native_goal

def main() -> None:
    private=BASE/root.name
    if root.exists() or private.exists():raise RuntimeError('Fresh functional validation only')
    for path,need in [(Path('/mnt/c'),10),(study,4),(BASE,4)]:
        if shutil.disk_usage(path).free<need*2**30:raise RuntimeError('Storage reserve')
    if sha(SERVER/'TrackmaniaServer')!=SERVER_SHA or sha(MAP)!=MAP_SHA:raise RuntimeError('Frozen native identity')
    jobs=[('native17.712.Ghost.Gbx',study/'native-capture1126/TrackmaniaRL-AI-17.712-NATIVE.Ghost.Gbx',17712),
          ('measured17.712.Ghost.Gbx',study/'vis-export1142/MEASURED-SERVER-17.712-CONTROL.Ghost.Gbx',17712),
          ('autonomous17.705.Ghost.Gbx',study/'vis-export1144/GENERATED-AI-17.705-CHECK-PENDING.Ghost.Gbx',17705)]
    if len(sys.argv)>2:
        manifest=json.loads(Path(sys.argv[2]).read_text())
        jobs=[(entry['name'],Path(entry['path']).resolve(),entry['actual_goal_ms']) for entry in manifest]
        if not 1<=len(jobs)<=3 or len({name for name,_,_ in jobs})!=len(jobs):raise ValueError('One to three unique functional controls')
        for name,path,goal in jobs:
            if Path(name).name!=name or not path.is_relative_to(study) or type(goal) is not int or not 1<=goal<=17712:
                raise ValueError('Bound report inputs and integer actual Goals required')
    root.mkdir();private.mkdir(mode=0o700)
    (private/'UserData/Replays').mkdir(parents=True);(private/'UserData/Maps').mkdir()
    (private/'TrackmaniaServer').symlink_to(SERVER/'TrackmaniaServer')
    (private/'Packs').symlink_to(SERVER/'Packs',target_is_directory=True)
    (private/'UserData/Maps/tmrl-test.Map.Gbx').symlink_to(MAP)
    inputs={}
    for name,path,goal in jobs:
        inputs[name]=dict(source=str(path),sha256=sha(path),expected_actual_goal_ms=goal)
        shutil.copyfile(path,private/'UserData/Replays'/name)
    env=os.environ.copy()
    for key in ('LD_PRELOAD','TMRE_SHIM','TMRE_TRACE_DIR'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1')
    report=dict(status='RUNNING',inputs=inputs,remainingOwned=[],cleanupErrors=[],forced=[],
        source_shim_loaded=False,game_client_opened=False,training_updates=0,public_submission=False)
    started=clock()
    with (root/'stdout.json').open('xb') as out,(root/'stderr.log').open('xb') as err:
        child=subprocess.Popen(['./TrackmaniaServer','/nodaemon','/validatepath=.'],cwd=private,env=env,
            stdout=out,stderr=err,start_new_session=True)
        saved=identity(child.pid)
        if saved is None or saved['pgid']!=saved['sid'] or saved['sid']!=child.pid:raise RuntimeError('Own native session')
        report['native_identity']=saved;save(root/'launch.json',report)
        pidfd=os.pidfd_open(child.pid)
        try:child.wait(timeout=30)
        finally:
            if child.poll() is None:
                signal.pidfd_send_signal(pidfd,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    report['forced'].append(saved);signal.pidfd_send_signal(pidfd,signal.SIGKILL);child.wait(timeout=5)
            os.close(pidfd)
            report.update(exit_code=child.returncode,elapsed_seconds=clock()-started,
                exact_native_identity_gone=identity(child.pid)!=saved,remainingOwned=descendants(child.pid))
            save(root/'result.json',report)
    records=json.loads((root/'stdout.json').read_text())
    if len(records)!=len(jobs) or {r['FileName'] for r in records}!=set(inputs):raise RuntimeError('Actual native record binding')
    for row in records:
        if native_goal(row)!=inputs[row['FileName']]['expected_actual_goal_ms'] or row['IsValid'] is not True:
            report.update(status='FAIL_NATIVE_GENERATED_GHOST_PRESERVED',actual_records=records)
            save(root/'result.json',report);raise RuntimeError('Generated Ghost native functional validation failed')
    if child.returncode!=0 or not report['exact_native_identity_gone'] or report['remainingOwned'] or report['forced']:
        raise RuntimeError('Native exact closure')
    if sha(SERVER/'TrackmaniaServer')!=SERVER_SHA or sha(MAP)!=MAP_SHA or any(sha(Path(v['source']))!=v['sha256'] for v in inputs.values()):
        raise RuntimeError('Source mutation')
    report.update(status='PASS_UNMODIFIED_NATIVE_FUNCTIONAL_CONTROLS_NOT_CLIENT_VISUAL_CHECK',actual_records=records,
        server_sha256=SERVER_SHA,map_sha256=MAP_SHA,client_only_cosmetics_unverified=True)
    save(root/'result.json',report)
    print(json.dumps(dict(status=report['status'],actual_goals_ms=[native_goal(r) for r in records],exact_native_identity_gone=True)))

if __name__=='__main__':main()
