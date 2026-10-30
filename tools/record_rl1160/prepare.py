"""Read-only preflight of historical inputs, then prepare one fresh queued run."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import sys
import time

BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
OLD=BASE/'shortcut-multi-source1094'
REPO=Path('/mnt/c/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania')
STUDY=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/pilot-optimized-qr-7h690/pr-review778/joint-optimization846/target18-setup1019/autonomous1032/retained-rl1068/multi-demonstration1088')
ROOT=STUDY/'line-search1136'
QUAL=STUDY/'line-qualification1135/model-finalization1138'
FALL=Path('/mnt/f/AITrackmania/experiments/fall2026-05-ghosts-0749e38ff0f6')
DEP=BASE/'qualification-4h-0acad837f044470d9e8801c19fbec488'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import yaml
from contracts import remap_head
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.models.factory import CompositeValueModelFactory
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity


def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(1)
    name='record-rl1160-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    out=STUDY/name
    private=BASE/name
    out.mkdir(exist_ok=False)
    private.mkdir(mode=0o700,exist_ok=False)
    for folder in ('collection','bc','baseline-evaluation','bc-evaluation','final-100-evaluation','rl'):
        (out/folder).mkdir()
    # Freeze the known compatible SDK; no active deployment is changed.
    shutil.copytree(OLD/'trackmaniarl',private/'source/trackmaniarl',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for file in ('pipeline.py','contracts.py'):
        shutil.copyfile(Path(__file__).parent/file,private/file)
    manifest=[dict(path=str(p.relative_to(private)),sha256=sha(p)) for p in sorted(private.rglob('*')) if p.is_file()]
    save(out/'source-manifest.json',manifest)
    raw=yaml.safe_load((QUAL/'model-spec.yaml').read_text())
    cfg=read(QUAL/'environment.json')
    cfg.pop('training_demonstration',None)
    old_policy=ROOT/'protected/qualified-policy.pt'
    if sha(old_policy)!='2ba1621efadb03a693e08d3e262eeb6b8f3cfd57d5788c3bee0f8a5dbdf7781e':raise RuntimeError('Qualified policy hash changed')
    with np.load(ROOT/'protected/qualified-macro-catalog.npz',allow_pickle=False) as z:old=z['programs'].copy()
    tapes=[]
    for parent in (STUDY/'line-qualification1135',ROOT):
        best=read(parent/'best.json')
        ticks=read(Path(best['motors']))
        if len(ticks)%5 or hashlib.sha256(bytes(ticks)).hexdigest()!=best['motor_sha256']:raise RuntimeError('Tape shape/hash')
        tapes.append(dict(path=best['motors'],motor_sha256=best['motor_sha256'],goal_ms=best['actual_native_goal_ms']))
    programs=sorted(set(tuple(map(int,p)) for p in old)|{tuple(read(Path(t['path']))[i:i+5]) for t in tapes for i in range(0,len(read(Path(t['path']))),5)})
    np.savez_compressed(out/'macro-catalog.npz',schema=np.asarray('tmrl.autonomous-macro-action50ms-native10ms/v1'),programs=np.asarray(programs,dtype=np.int16))
    cfg['macro_catalog']=dict(path=str(out/'macro-catalog.npz'),sha256=sha(out/'macro-catalog.npz'))
    raw['components']['environment']['kwargs']['config']=cfg
    raw['components']['model_factory']['kwargs']['head']['kwargs']['config']['action_count']=len(programs)
    codec=TorchCheckpointCodec()
    policy=codec.load(old_policy)
    online=copy.deepcopy(policy['online'])
    online['head']=remap_head(online['head'],old,programs,64)
    model=CompositeValueModelFactory(**raw['components']['model_factory']['kwargs']).build()
    for key,value in online.items():getattr(model,key).load_state_dict(value,strict=True)
    tensors=[v for state in online.values() for v in state.values()]
    if len(tensors)!=90 or not all(torch.isfinite(t).all() for t in tensors):raise RuntimeError('Expected 90 finite module tensors')
    policy.update(online=online,architecture_fingerprint=model.architecture_fingerprint(),macro_catalog_sha256=sha(out/'macro-catalog.npz'),source_checkpoint_sha256=sha(old_policy),online_RL_updates=0)
    codec.save(policy,out/'baseline-policy.pt')
    (out/'environment.json').write_text(json.dumps(cfg,indent=2))
    (out/'model-spec.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
    RunSpec.from_yaml(out/'model-spec.yaml')
    ready=read(DEP/'parent-ready.json')
    identities=[process_identity(385)]
    if identities[0] is None or identities[0]['birth']!=9731:raise RuntimeError('Existing qualification parent identity changed')
    for child in ready['processes']:
        identities.extend(child[k] for k in ('worker','supervisor'))
    map_path=Path(raw['evaluation']['maps'][0]['map_path'])
    inputs=[old_policy,ROOT/'protected/qualified-macro-catalog.npz',map_path,Path(cfg['geometry_path']),Path(cfg['runtime']['asset_manifest']),
        Path(cfg['reward_reference']['path']),Path(cfg['event_source']['kwargs']['motor_reference_path']),Path(cfg['event_source']['kwargs']['finish_reference_path'])]
    inputs.extend(Path(t['path']) for t in tapes)
    save(out/'input-manifest.json',[dict(path=str(p),sha256=sha(p)) for p in inputs])
    save(out/'plan.json',dict(schema='tmrl.record-RL-pipeline/v1',report=str(out),private=str(private),source=str(private/'source'),
        map_uid=cfg['expected_map_uid'],map_sha256=sha(map_path),training_tapes=tapes,
        targets=[dict(map='tmrl-test',map_uid=cfg['expected_map_uid'],maximum_goal_ms=17600),dict(map='Fall 2026 #05',map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',maximum_goal_ms=21757)],
        acceptance='at least one actual native Goal below target in a frozen-model suite of 100 trials; no teacher or learner in evaluation',
        rl_seconds=14400,bc_seconds=1200,evaluation_seconds=3600,own_quota_gib=12,
        dependency=dict(directory=str(DEP),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),identities=identities),
        fresh_optimizer=True,resume=False,public_submission=False))
    # This map has a replay validator, not an admitted native observation/control
    # factory. Do not bind it to TMX99162's map-specific native event references.
    validation=read(FALL/'headless-fall05-validation-v26-candidate-rays.json')
    save(out/'fall05-admission.json',dict(status='BLOCKED_UNQUALIFIED_STEP_ENVIRONMENT',map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',
        map_sha256=sha(FALL/'map.Map.Gbx'),map_path=str(FALL/'map.Map.Gbx'),maximum_goal_ms=21757,
        existing_evidence=str(FALL/'headless-fall05-validation-v26-candidate-rays.json'),
        replay_validation_only=validation['scope'],
        required=['qualified reset/step action and causal-observation stream for this exact map and engine',
            'real native Goal and collision/material/contact events',
            'demonstration observations collected from that environment and matching model inputs',
            'independent model inference through TICK with matching observation contract'],
        reason='Existing full-QR runtime is pinned to TMX99162; Fall validator only evaluates complete replay files. No qualified Fall RL environment binding was found.',
        online_RL_started=False))
    save(out/'preflight.json',dict(status='PASS_OFFLINE_PREPARATION_NATIVE_PREFLIGHT_QUEUED',old_actions=len(old),new_actions=len(programs),
        finite_online_tensors=90,baseline_policy_sha256=sha(out/'baseline-policy.pt'),new_catalog_sha256=sha(out/'macro-catalog.npz'),
        qualification_not_modified=True,RL_started=False,fall05_admitted=False))
    print(json.dumps(dict(report=str(out),private=str(private),source=str(private/'source'),status='PREPARED_NOT_LAUNCHED')))


if __name__=='__main__':main()
