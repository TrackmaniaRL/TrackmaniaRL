"""New immutable recovery namespaces; failed campaigns remain untouched."""
from pathlib import Path
import ast,copy,hashlib,json,os,secrets,shutil,sys,time
BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
STUDY=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/pilot-optimized-qr-7h690/pr-review778/joint-optimization846/target18-setup1019/autonomous1032/retained-rl1068/multi-demonstration1088')
REPO=Path('/mnt/c/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania')
OLD=BASE/'qualification-4h-0acad837f044470d9e8801c19fbec488'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD/'deployment'))
import torch,yaml
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.checkpoints import validate_checkpoint_v2
from trackmaniarl.models.factory import CompositeValueModelFactory

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def copy_source(old,new):shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def lifecycle(source):
 for name in ('environment.py','native_provider.py'):
  rel=Path('trackmaniarl/trackmania/native_headless/full_qr')/name
  shutil.copy2(REPO/rel,source/rel)

def main():
 torch.set_num_threads(1)
 name='qualification-recovery1162-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
 out=STUDY/name;private=BASE/name;out.mkdir();private.mkdir(mode=0o700);source=private/'source'
 copy_source(OLD/'deployment',source);lifecycle(source)
 codec=TorchCheckpointCodec();origin=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/qualification-4h-0acad837f044470d9e8801c19fbec488/independent-readback-20261029/checkpoint.pt')
 expected='3027f2e2eb0a3a8833301bfe8b8a134a584a42a90afb5c6f03dc32486045704a'
 if sha(origin)!=expected:raise RuntimeError('Preserved checkpoint SHA changed')
 state=codec.load(origin);learner=state['learner'];validate_checkpoint_v2(learner)
 tensors=[]
 def visit(v):
  if isinstance(v,dict):
   for x in v.values():visit(x)
  elif isinstance(v,(tuple,list)):
   for x in v:visit(x)
  elif isinstance(v,torch.Tensor):tensors.append(v)
 visit(learner)
 if not tensors or not all(torch.isfinite(v).all() for v in tensors):raise RuntimeError('Nonfinite checkpoint tensor')
 raw=yaml.safe_load((OLD/'config.yaml').read_text());model=CompositeValueModelFactory(**raw['components']['model_factory']['kwargs']).build()
 if model.architecture_fingerprint()!=learner['architecture_fingerprint']:raise RuntimeError('Saved checkpoint architecture mismatch')
 for k,v in learner['online'].items():getattr(model,k).load_state_dict(v,strict=True)
 codec.save(dict(schema_version='2.0',learner=learner),private/'warm-policy.pt')
 raw['run_id']=name;raw['artifacts_dir']=str(out/'artifacts');(out/'artifacts').mkdir()
 env=raw['components']['environment']['kwargs']['config'];env['runtime']['trace_dir']=str(private/'traces');env['runtime']['work_dir']=str(BASE/'waypoint-runtime002/build/server-probe'/name);env['driver_episode_limit']=1;env['trace_archive_root']=str(out/'archives')
 raw['components']['learner']['kwargs'].update(model_initialization_checkpoint=str(private/'warm-policy.pt'),warm_start_submodules=['encoder','temporal','head','strategy'],warm_start_required_tensors=list(model.state_dict()))
 from trackmaniarl.models.loading import warm_start_composite_model,WarmStartOptions
 warm=warm_start_composite_model(model,private/'warm-policy.pt',WarmStartOptions(submodules=('encoder','temporal','head','strategy'),required_tensors=tuple(model.state_dict())))
 if warm.missing or warm.unexpected or warm.shape_mismatch:raise RuntimeError('Incomplete strict whole-policy warm start')
 raw['metadata'].update(initialization='User-authorized weights from independently read-back checkpoint; fresh optimizer and replay',recovery_initialization_sha256=sha(private/'warm-policy.pt'),recovery_original_checkpoint_sha256=expected,requested_wall_time_seconds=14400,resume_checkpoint=None)
 (private/'config.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
 worker=source/'trackmaniarl/experiments/full_qr/campaign_worker.py';text=worker.read_text();old='or spec.components.learner.kwargs.get("model_initialization_checkpoint") is not None'
 if text.count(old)!=1:raise RuntimeError('Warm-start guard preimage differs')
 check=f'''or spec.components.learner.kwargs.get("model_initialization_checkpoint") != {str(private/'warm-policy.pt')!r}
            or spec.metadata.get("recovery_initialization_sha256") != {sha(private/'warm-policy.pt')!r}
            or hashlib.sha256(Path({str(private/'warm-policy.pt')!r}).read_bytes()).hexdigest() != {sha(private/'warm-policy.pt')!r}'''
 text=text.replace(old,check);ast.parse(text);worker.write_text(text)
 (private/'token').write_text(secrets.token_hex(32));(private/'token').chmod(0o600);(private/'stop').write_bytes(b'\0')
 for f in ('run.py','smoke.py'):shutil.copy2(HERE/f,private/f)
 tmrl_old=STUDY/'record-rl1160-20261029-103800';tmrl=STUDY/name.replace('qualification-recovery','tmrl-recovery');tp=BASE/tmrl.name;tmrl.mkdir();tp.mkdir()
 for folder in ('collection','bc','baseline-evaluation','bc-evaluation','final-100-evaluation','rl'):(tmrl/folder).mkdir()
 copy_source(BASE/tmrl_old.name/'source',tp/'source');lifecycle(tp/'source')
 for f in ('baseline-policy.pt','macro-catalog.npz','model-spec.yaml','environment.json','offline-audit.json','fall05-admission.json'):
  shutil.copy2(tmrl_old/f,tmrl/f)
 shutil.copy2(REPO/'tools/record_rl1160/contracts.py',tp/'contracts.py')
 oldout=str(tmrl_old);newout=str(tmrl)
 for f in ('environment.json','model-spec.yaml'):(tmrl/f).write_text((tmrl/f).read_text().replace(oldout,newout))
 plan=json.loads((tmrl_old/'plan.json').read_text());plan.update(report=str(tmrl),private=str(tp),source=str(tp/'source'),dependency=dict(recovery_report=str(out),directory=str(private),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
 save(tmrl/'plan.json',plan);shutil.copy2(tmrl_old/'input-manifest.json',tmrl/'input-manifest.json')
 text=(BASE/tmrl_old.name/'pipeline.py').read_text();a=text.index('def wait_for_dependency():');b=text.index('\ndef collect():',a)
 text=text[:a]+'''def wait_for_dependency():
    dependency=read(OUT/'plan.json')['dependency']
    result=read(Path(dependency['directory'])/'parent-result.json')
    if not clean_dependency(result):raise RuntimeError('Recovered qualification did not complete cleanly')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=dependency['boot']:raise RuntimeError('Recovery boot changed')
    save(OUT/'dependency-closure.json',result)

'''+text[b:];ast.parse(text);(tp/'pipeline.py').write_text(text)
 manifest=[dict(path=str(p.relative_to(tp)),sha256=sha(p)) for p in tp.rglob('*.py')];save(tmrl/'source-manifest.json',manifest)
 oldfall=Path('/mnt/f/AITrackmania/experiments/fall2026-05-ghosts-0749e38ff0f6/fall-rl1161-20261029-110735');fall=oldfall.parent/name.replace('qualification-recovery','fall-recovery');fp=BASE/fall.name;fall.mkdir();fp.mkdir()
 copy_source(BASE/oldfall.name/'source',fp/'source');copy_source(BASE/oldfall.name/'runtime',fp/'runtime')
 # Runtime contains only this experiment's copied binaries/assets and old bounded
 # qualification work directories; the new settings use fresh per-episode roots.
 cfg=json.loads((oldfall/'environment-independent.json').read_text());assets=json.loads(Path(cfg['manifest']).read_text());assets['runtime_root']=str(fp/'runtime');save(fp/'assets.json',assets)
 cfg.update(private=str(fp),source=str(fp/'source'),runtime_root=str(fp/'runtime'),report=str(fall),manifest=str(fp/'assets.json'),manifest_sha256=sha(fp/'assets.json'))
 save(fall/'environment-independent.json',cfg);shutil.copy2(oldfall/'environment-admission.json',fall/'environment-admission.json')
 (fall/'qualification-v6').mkdir();shutil.copy2(oldfall/'qualification-v6/actual-rollout.npz',fall/'qualification-v6/actual-rollout.npz')
 save(out/'plan.json',dict(private=str(private),source=str(source),qualification_seconds=14400,tmrl=str(tmrl),tmrl_private=str(tp),fall=str(fall),fall_private=str(fp),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),old_qualification=str(OLD),original_checkpoint=str(origin),original_checkpoint_sha256=expected,fresh_optimizer=True,fresh_replay=True,original_counters=state['distributed']))
 save(out/'checkpoint-audit.json',dict(status='PASS_ARCHITECTURE_STRICT_TENSORS_AND_SHA',sha256=expected,tensor_count=len(tensors),nonfinite_tensors=0,original_update_count=learner['training']['update_count'],required_model_tensors=len(model.state_dict()),new_optimizer=True,new_replay=True))
 proof=STUDY/'qualification-recovery1162-20261029-133658'
 if (proof/'smoke-result.json').exists():
  prior=json.loads((proof/'smoke-result.json').read_text());supervision=json.loads((proof/'smoke-supervision.json').read_text());prior_source=BASE/proof.name/'source'
  same_code=all(sha(source/'trackmaniarl/trackmania/native_headless/full_qr'/f)==sha(prior_source/'trackmaniarl/trackmania/native_headless/full_qr'/f) for f in ('environment.py','native_provider.py'))
  if not same_code or prior['weights_source_sha256']!=expected or supervision['exits']!={'smoke':0} or any(supervision['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Prior native validation is not reusable')
  for row in prior['trials']:
   if sha(row['archive']['archive'])!=row['archive']['archive_sha256']:raise RuntimeError('Prior actual trace changed')
  shutil.copy2(proof/'smoke-result.json',out/'smoke-result.json')
  save(out/'smoke-proof-provenance.json',dict(source_report=str(proof),identical_native_lifecycle_source=True,identical_checkpoint_sha256=expected,new_native_benchmark_repeated=False,supervision=supervision))
 pins=[]
 for root in (private,tp,fp):
  pins.extend(dict(path=str(p),sha256=sha(p)) for p in root.rglob('*.py'))
 for p in (private/'config.yaml',private/'warm-policy.pt',out/'plan.json',tmrl/'plan.json',fall/'environment-independent.json',fp/'assets.json',fall/'qualification-v6/actual-rollout.npz'):pins.append(dict(path=str(p),sha256=sha(p)))
 save(out/'source-manifest.json',pins)
 print(json.dumps(dict(report=str(out),private=str(private),tmrl=str(tmrl),fall=str(fall))))

if __name__=='__main__':main()
