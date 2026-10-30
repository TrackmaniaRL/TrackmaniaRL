"""Admit only clock-paired, temporally contiguous teacher windows for BC."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def history_window(frames,index):
    past=frames[max(0,index-3):index+1]
    return np.stack([past[0]]*(4-len(past))+past)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('collection',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--pad-initial-history',action='store_true',help='Repeat first measured image in missing history slots, as the runtime reset does')
    a=p.parse_args()
    raw=(a.collection/'result.json').read_bytes()
    report=json.loads(raw)
    if (report['status']!='COLLECTED_TEACHER_CAMERA' or report['cleanupErrors']
            or report['teacher_goal_ms']!=17601):
        raise ValueError('requires clean, measured 17.601 teacher completion')
    from tools.vision_qr1167.model import ACTIONS
    frames=[]; clocks=[]
    for row in report['samples']:
        payload=(a.collection/row['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('teacher frame changed')
        frame=np.load(a.collection/row['path'],allow_pickle=False)
        if frame.dtype!=np.uint8 or frame.shape!=(84,84):raise ValueError('frame contract')
        evidence=row['evidence']
        if evidence['rendered_ms']!=evidence['telemetry_ms']+10:raise ValueError('clock pairing')
        frames.append(frame);clocks.append(evidence['telemetry_ms'])
    if any(b<=a for a,b in zip(clocks,clocks[1:])):raise ValueError('unordered images')
    vehicles=[e['payload'] for e in report['native_events'] if e['eventType']=='vehicle-state']
    # Restrict to the last actual reset, excluding retained events from older laps.
    reset=max(i for i,v in enumerate(vehicles) if v['raceTick']==0)
    vehicles=vehicles[reset:]
    inputs=lambda v:tuple(float(v['inputs'][k]) for k in ('steer','accel','brake'))
    source_sha=None
    commands=None
    if a.source:
        payload=a.source.read_bytes();source_sha=hashlib.sha256(payload).hexdigest()
        if source_sha!='a005150fc6b77163ade531407f8a0014a038206af6890edbaeb43e7c54e0e2eb' or report['revision_id']!='1ccf57c0-fe6f-4f44-8ef5-dcc602bd05d8':
            raise ValueError('unqualified teacher source/revision')
        commands=[line.split() for line in payload.decode().splitlines()]
        def scheduled(clock):
            state=dict(steer=0.,accel=0.,brake=0.)
            for t,k,v in commands:
                if int(t)<=clock and k in state:state[k]=int(v)/(127 if k=='steer' else 1)
            return tuple(state[k] for k in ('steer','accel','brake'))
        if any(inputs(v)!=scheduled(v['raceTick']) for v in vehicles if v['raceTick']<17600):
            raise ValueError('actual teacher controls disagree with source schedule')
    x=[];y=[];times=[];padded=0;rejected=dict(gaps=0,unstable_controls=0,unsupported_controls=0,after_goal=0)
    for i in range(0 if a.pad_initial_history else 3,len(frames)):
        available=clocks[max(0,i-3):i+1]
        if any(not 40<=b-a<=60 for a,b in zip(available,available[1:])):
            rejected['gaps']+=1;continue
        if clocks[i]+60>report['teacher_goal_ms']:
            rejected['after_goal']+=1;continue
        future=[v for v in vehicles if clocks[i]+10<=v['raceTick']<=clocks[i]+60]
        # Require all six measured ticks; no interpolation across missing commands.
        if commands is not None:
            controls=[scheduled(t) for t in range(clocks[i]+10,clocks[i]+61,10)]
            if len(set(controls))!=1:
                rejected['unstable_controls']+=1;continue
            actual=controls[0]
        else:
            if sorted(set(v['raceTick'] for v in future))!=list(range(clocks[i]+10,clocks[i]+61,10)) or len(set(map(inputs,future)))!=1:
                rejected['unstable_controls']+=1;continue
            actual=inputs(future[0])
        matches=[j for j,action in enumerate(ACTIONS) if max(abs(a-b) for a,b in zip(actual,action))<1e-5]
        if not matches:rejected['unsupported_controls']+=1;continue
        x.append(history_window(frames,i));y.append(matches[0]);times.append(clocks[i])
        padded+=int(i<3)
    if not x:
        a.output.mkdir(exist_ok=False)
        (a.output/'receipt.json').write_text(json.dumps(dict(status='FAIL_NO_ADMISSIBLE_WINDOWS',rejected=rejected,source_sha256=hashlib.sha256(raw).hexdigest()),indent=2))
        raise ValueError('no admissible demonstration windows: '+str(rejected))
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'demonstrations.npz',observations=np.stack(x),actions=np.array(y),race_ms=np.array(times))
    receipt=dict(status='ADMITTED_TEACHER_WINDOWS',teacher_goal_ms=17601,
        source_sha256=hashlib.sha256(raw).hexdigest(),samples=len(x),rejected=rejected,
        dataset_sha256=hashlib.sha256((a.output/'demonstrations.npz').read_bytes()).hexdigest(),
        frame_spacing_ms=[40,60],future_control_ticks=6,autonomous=False,training_updates=0,
        padded_initial_windows=padded,initial_padding='repeat first measured image; runtime initialization convention' if a.pad_initial_history else None,
        teacher_source_sha256=source_sha,labels='complete source schedule verified against all measured native controls' if commands else 'six measured native ticks')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
