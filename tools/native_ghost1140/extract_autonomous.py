"""Bind a closed autonomous evaluation to actual native poses, visuals and inputs.

No learner, simulator, renderer, input playback or teacher files are opened.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import struct
import sys
import tarfile
from pathlib import Path
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,'/home/kuba/tmrl-full-qr-20261022-011/shortcut-multi-source1094')
sys.path.insert(0,'/mnt/c/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania')
from tools.ghost_divergence1103.prepare_archived_inputs import audited_controls
from tools.model_ghost1117.visual_state import visual_records,quantized_pose_agrees
from trackmaniarl.core.builtins import TorchCheckpointCodec

def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def physical_posts(raw: bytes,pid: int):
    layouts={b'TMPOST03':3,b'TMPOST02':2,b'TMREAD01':1,b'TMAUD003':2}
    if not raw or len(raw)%256:raise ValueError('Incomplete multiplexed native trace')
    for at in range(0,len(raw),256):
        row=raw[at:at+256];version=layouts.get(row[:8])
        if version is None or struct.unpack_from('<4I',row,8)!=(256,version,pid,pid):
            raise ValueError('Native trace owner/layout')
        if row[:8]==b'TMPOST03' and struct.unpack_from('<I',row,148)[0]==75:yield row

def extract(campaign: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Never overwrite a source snapshot')
    fitted=read(campaign/'finalization.json')
    policy=Path(fitted['policy']);stage=policy.parent.parent
    result=read(stage/'evaluation/result.json')
    goal=fitted['best_autonomous_goal_ms']
    if type(goal) is not int or result['actual_goals_ms']!=[goal]*5 or not result['weights_unchanged']:
        raise ValueError('Five autonomous actual Goals required')
    if result['teacher_files_opened']!=0 or result['learner_updates']!=0 or result['expert_access_attempts_blocked']:
        raise ValueError('Autonomous evaluation isolation required')
    if sha(policy)!=fitted['policy_sha256'] or result['policy_sha256']!=sha(policy):raise ValueError('Frozen policy hash')
    for name,role in [('collection','collect'),('training','train'),('evaluation','evaluate')]:
        report=read(stage/name/'supervision.json')
        if report['status']!='PASS_ROLE' or report['exits']!={role:0}:raise ValueError('Closed qualification roles required')
        if any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise ValueError('Exact role closure')
    checkpoint=TorchCheckpointCodec().load(policy)
    catalog=stage/'macro-catalog.npz'
    if checkpoint['macro_catalog_sha256']!=sha(catalog):raise ValueError('Exact unordered action mapping')
    with np.load(catalog,allow_pickle=False) as archive:programs=archive['programs'].copy()
    actions=[read_line for line in (stage/'evaluation/actions.jsonl').read_text().splitlines() if (read_line:=json.loads(line))]
    episodes=[];current=[]
    for row in actions:
        current.append(row['action'])
        if row['terminated'] or row['truncated']:
            if row['info']['termination_reason']!='finished':raise ValueError('DNF cannot supply a successful ghost')
            episodes.append(current);current=[]
    if len(episodes)!=5 or current or any(e!=episodes[0] for e in episodes):raise ValueError('Repeated autonomous action sequence binding')
    commands=programs[episodes[-1]].reshape(-1)
    archives=sorted((stage/'evaluation/closed-traces').rglob('*.tar.gz'),key=lambda p:p.stat().st_mtime_ns)
    if not archives:raise ValueError('Real closed trace required')
    source=archives[-1]
    receipt=read(source.with_suffix('').with_suffix('.json'))
    if sha(source)!=receipt['archive_sha256']:raise ValueError('Archive hash')
    with tarfile.open(source,'r:gz') as archive:
        def member(name: str) -> bytes:
            stream=archive.extractfile(name)
            if stream is None:raise ValueError(f'Actual archived member missing: {name}')
            return stream.read()
        metadata=json.loads(member('trace/episode-0000/metadata.json'))
        reset=json.loads(member('trace/episode-0000/uncredited-neutral-reset-v7.json'))
        pid=int(metadata['server_pid']);origin=int(reset['resetNativeT'])
        posts=member(f'trace/posts/post-{pid}.bin')
        goals=member(f'trace/posts/goal-receipt-{pid}.bin')
        raw_visual=member(f'trace/posts/vehicle-{pid}.bin')
    if len(posts)%256 or len(goals)%96:raise ValueError('Incomplete actual records')
    committed=audited_controls(posts,pid);visuals=visual_records(raw_visual,pid)
    frames=[]
    for row in physical_posts(posts,pid):
        clock=struct.unpack_from('<I',row,112)[0]
        if clock<origin:continue
        if clock!=struct.unpack_from('<I',row,116)[0]:raise ValueError('POST phase clock')
        physical=struct.unpack_from('<10f',row,152);gas,brake,steer=struct.unpack_from('<3f',row,88)
        if not all(math.isfinite(x) for x in physical+(gas,brake,steer)):raise ValueError('Nonfinite actual state')
        if clock>origin and committed[clock]!=struct.pack('<3f',steer,gas,brake):raise ValueError('Actual acknowledged input differs')
        state=visuals[clock]
        for offset,values in [(80,physical[4:7]),(92,physical[7:10])]:
            if not all(quantized_pose_agrees(p,v) for p,v in zip(values,struct.unpack_from('<3f',state,offset),strict=True)):
                raise ValueError('Independent visual/physical state disagreement')
        frames.append(dict(t=clock-origin,q=physical[:4],p=physical[4:7],v=physical[7:10],sgb=[steer,gas,brake]))
    if [f['t'] for f in frames]!=list(range(0,len(frames)*10,10)) or len(frames)-1!=len(commands):
        raise ValueError('Actual decision/tick clock binding')
    for frame,command in zip(frames[1:],commands,strict=True):
        steering,variant=divmod(int(command),6);gas,brake=divmod(variant,3)
        expected=np.asarray([(steering-6)/6,gas,min(brake,1)],dtype=np.float32)
        if not np.array_equal(expected,np.asarray(frame['sgb'],dtype=np.float32)):
            raise ValueError('Actual controls do not match the autonomous model action')
    observed_goals=set()
    for at in range(0,len(goals),96):
        row=goals[at:at+96]
        if row[:8]!=b'TMGR0001' or struct.unpack_from('<4I',row,8)!=(96,1,pid,pid):raise ValueError('Goal provenance')
        _,last,count=struct.unpack_from('<QII',row,40)
        if count:observed_goals.add(last-origin)
    if observed_goals!={goal}:raise ValueError('Independent actual Goal differs')
    output.mkdir()
    (output/'recorded-visual.bin').write_bytes(raw_visual)
    data=dict(status='VERIFIED_AUTONOMOUS_MODEL_TRACE_FOR_GHOST',goal_ms=goal,physical_end_ms=frames[-1]['t'],frames=frames,
        native_origin_ms=origin,provenance=dict(native_pid=pid,archive=str(source),archive_sha256=sha(source),
        policy=str(policy),policy_sha256=sha(policy),catalog_sha256=sha(catalog),
        evaluation_sha256=sha(stage/'evaluation/result.json'),action_log_sha256=sha(stage/'evaluation/actions.jsonl'),
        model_commands_verified=len(commands),actual_goals_ms=result['actual_goals_ms'],new_simulations=0,
        teacher_files_used=False,visual_capture_sha256=sha(output/'recorded-visual.bin')))
    (output/'physics.json').write_text(json.dumps(data,allow_nan=False))
    (output/'source-receipt.json').write_text(json.dumps({k:v for k,v in data.items() if k!='frames'},indent=2))
    print(json.dumps(dict(status=data['status'],goal_ms=goal,frames=len(frames),commands_verified=len(commands),output=str(output))))

if __name__=='__main__':extract(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
