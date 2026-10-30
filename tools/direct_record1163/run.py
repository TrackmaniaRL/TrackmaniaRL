from pathlib import Path
import hashlib,json,os,shutil,sys,time,traceback
out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text());private=Path(plan['private']);sys.path.insert(0,plan['source'])
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',WANDB_MODE='offline')
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.stage');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def stage(name,args,limit,cwd):
 def stop():
  if cohort.exited():return
  (Path(plan[name])/'STOP').touch(exist_ok=True)
 cohort=CampaignProcessCohort(private/('cohort-'+name),stop);r=dict(status='FAILED',stage=name);started=time.monotonic()
 try:
  r['process']=cohort.start(name,args,cwd=cwd)
  while not cohort.exited():
   if (out/'STOP').exists() or time.monotonic()-started>limit:raise RuntimeError('Bounded stage deadline or STOP')
   for p,gib in (('/mnt/c',10),('/mnt/f',6),('/mnt/j',10),(str(private),6)):
    if shutil.disk_usage(p).free<gib*2**30:raise RuntimeError('Storage reserve '+p)
   save(out/'state.json',dict(phase=name,parent=process_identity(os.getpid()),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage_process=r['process'],elapsed_seconds=time.monotonic()-started))
   time.sleep(5)
  r['exits']=cohort.exited()
  if r['exits']!={name:0}:raise RuntimeError(name+' failed; preserve its evidence')
  r['status']='PASS_STAGE_EXIT0'
 finally:
  r['closure']=cohort.close(grace_seconds=120);save(out/(name+'-supervision.json'),r)
 if any(r['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Stage closure unclean')
 return r
def main():
 with (out/'launch-marker').open('x') as f:f.write(str(os.getpid()))
 r=dict(status='FAILED',stages=[],parent=process_identity(os.getpid()))
 try:
  for row in json.loads((out/'source-manifest.json').read_text()):
   if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Direct pipeline source/input changed')
  r['stages'].append(stage('tmrl',[sys.executable,'-B',plan['tmrl_entry'],plan['tmrl']],21600,Path(plan['tmrl_private'])))
  result=json.loads((Path(plan['tmrl'])/'completion.json').read_text())
  if result['status'] not in ('RL_TARGET_REACHED_EXPORT_PENDING','RL_COMPLETED_TARGET_NOT_REACHED'):raise RuntimeError('tmrl failed; keep Fall queued')
  r['stages'].append(stage('fall',[sys.executable,'-B',str(Path(plan['fall_private'])/'source/train.py'),plan['fall']],21600,Path(plan['fall_private'])/'source'))
  r.update(status='COMPLETED_BOTH_RECORD_RL_EXPERIMENTS',tmrl=result,fall=json.loads((Path(plan['fall'])/'training-result.json').read_text()))
 except BaseException as error:
  r.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:save(out/'completion.json',r);save(out/'state.json',r)
if __name__=='__main__':main()
