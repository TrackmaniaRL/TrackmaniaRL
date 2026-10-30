"""Owned sequential recovery: qualification -> tmrl RL -> Fall RL."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,traceback
out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text());private=Path(plan['private']);source=Path(plan['source']);sys.path.insert(0,str(source));sys.path.insert(0,str(private))
os.environ.update(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',WANDB_MODE='offline')
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity

def save(p,v):
 p=Path(p);stage=p.with_suffix('.stage');stage.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');stage.replace(p)
def reserves():
 for p,gib in (('/mnt/c',10),('/mnt/f',6),('/mnt/j',10),(str(private),6)):
  if shutil.disk_usage(p).free<gib*2**30:raise RuntimeError('Storage reserve '+p)
def pins():
 for row in json.loads((out/'source-manifest.json').read_text()):
  with Path(row['path']).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  if actual!=row['sha256']:raise RuntimeError('Frozen recovery input/source changed: '+row['path'])
def qualification():
 from trackmaniarl.experiments.full_qr.campaign_launcher import LaunchSettings,launch
 from trackmaniarl.experiments.full_qr.campaign_worker import WorkerSettings
 config=private/'config.yaml';settings=WorkerSettings(directory=private,config_path=config,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),token_path=private/'token',stop_path=private/'stop',endpoint='127.0.0.1:0',worker_name='learner',artifact_root=out/'artifacts')
 result=launch(LaunchSettings(settings,Path(sys.executable),Path(sys.executable),1,startup_seconds=240),source_root=source)
 save(out/'qualification-result.json',result)
def stage(name,args,limit,cwd):
 def stop():
  # Cohort.close invokes the callback even when its worker already exited.
  # A successful stage must not STOP the next, independently owned stage.
  if cohort.exited():return
  (out/'STOP').touch(exist_ok=True)
  if name=='qualification':(private/'stop').write_bytes(b'\1')
  elif name=='tmrl':(Path(plan['tmrl'])/'STOP').touch(exist_ok=True)
  elif name=='fall':(Path(plan['fall'])/'STOP').touch(exist_ok=True)
 cohort=CampaignProcessCohort(private/('cohort-'+name),stop);report=dict(status='FAILED',stage=name);started=time.monotonic()
 try:
  report['process']=cohort.start(name,args,cwd=cwd)
  while not cohort.exited():
   if (out/'STOP').exists() or time.monotonic()-started>limit:raise RuntimeError('Bounded stage deadline or STOP: '+name)
   reserves();save(out/'state.json',dict(phase=name,parent=process_identity(os.getpid()),boot=plan['boot'],stage_process=report['process'],elapsed_seconds=time.monotonic()-started,original_checkpoint_sha256=plan['original_checkpoint_sha256']))
   time.sleep(5)
  report['exits']=cohort.exited()
  if report['exits']!={name:0}:raise RuntimeError('Recovery stage failed; preserve diagnostics: '+name)
  report['status']='PASS_STAGE_EXIT0'
 finally:
  report['closure']=cohort.close(grace_seconds=120);report['wall_seconds']=time.monotonic()-started;save(out/(name+'-supervision.json'),report)
 if any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Stage closure not clean: '+name)
 return report
def main():
 # WSL may shut down while no experiment is running. Bind the fresh processes
 # to the actual launch boot, after verifying the prepared snapshot first.
 if (out/'launch-marker').exists():raise RuntimeError('No duplicate recovery launch')
 pins()
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if boot!=plan['boot']:
  save(out/'prelaunch-boot-binding.json',dict(preparation_boot=plan['boot'],launch_boot=boot,training_started=False))
  plan['boot']=boot;save(out/'plan.json',plan)
  tmrl_path=Path(plan['tmrl'])/'plan.json';tmrl_plan=json.loads(tmrl_path.read_text());tmrl_plan['dependency']['boot']=boot;save(tmrl_path,tmrl_plan)
  rows=json.loads((out/'source-manifest.json').read_text())
  for row in rows:
   if row['path'] in (str(out/'plan.json'),str(tmrl_path)):row['sha256']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
  save(out/'source-manifest.json',rows)
 with (out/'launch-marker').open('x') as f:f.write(str(os.getpid()))
 result=dict(status='FAILED',parent=process_identity(os.getpid()),boot=plan['boot'],stages=[])
 try:
  if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=plan['boot']:raise RuntimeError('Boot changed before recovery launch')
  pins();reserves()
  if not (out/'smoke-result.json').exists():result['stages'].append(stage('smoke',[sys.executable,'-B',str(private/'smoke.py'),str(out)],600,source))
  if json.loads((out/'smoke-result.json').read_text())['status']!='PASS_THREE_REAL_EPISODES_AND_EAGER_CLOSURE':raise RuntimeError('Actual native reset/closure validation required')
  result['stages'].append(stage('qualification',[sys.executable,'-B',str(private/'run.py'),str(out),'qualification'],15000,source))
  q=json.loads((private/'parent-result.json').read_text())
  if q.get('phase')!='completed':raise RuntimeError('Qualification failed its original budget/checkpoint/exit checks')
  pins()
  result['stages'].append(stage('tmrl',[sys.executable,'-B',str(Path(plan['tmrl_private'])/'pipeline.py'),plan['tmrl']],22500,Path(plan['tmrl_private'])))
  tmrl=json.loads((Path(plan['tmrl'])/'completion.json').read_text())
  if tmrl.get('status') not in ('RL_TARGET_REACHED_EXPORT_PENDING','RL_COMPLETED_TARGET_NOT_REACHED'):raise RuntimeError('tmrl did not complete; do not start Fall')
  pins()
  result['stages'].append(stage('fall',[sys.executable,'-B',str(Path(plan['fall_private'])/'source/train.py'),plan['fall']],21600,Path(plan['fall_private'])/'source'))
  result.update(status='COMPLETED_QUALIFICATION_AND_BOTH_RL_EXPERIMENTS',fall=json.loads((Path(plan['fall'])/'training-result.json').read_text()),tmrl=tmrl)
 except BaseException as error:
  result.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:save(out/'completion.json',result);save(out/'state.json',result)

if __name__=='__main__':
 if len(sys.argv)>2 and sys.argv[2]=='qualification':qualification()
 else:main()
