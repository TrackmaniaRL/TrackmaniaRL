from pathlib import Path
import hashlib,json,os,shutil,sys,time,traceback
out=Path(sys.argv[1]);cfg=json.loads((out/'environment.json').read_text());private=Path(cfg['private']);sys.path.insert(0,cfg['source']);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',WANDB_MODE='offline')
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
from atomic_json import save
def stage(name,args,limit):
 def stop():
  if not cohort.exited():(out/'STOP').touch(exist_ok=True)
 cohort=CampaignProcessCohort(private/('cohort-'+name),stop);report=dict(status='RUNNING');started=time.monotonic()
 try:
  report['process']=cohort.start(name,args,cwd=Path(cfg['source']))
  while not cohort.exited():
   if (out/'STOP').exists() or time.monotonic()-started>limit:raise RuntimeError('Bounded stage deadline or STOP')
   for path,gib in (('/mnt/c',10),('/mnt/f',6),(str(private),6)):
    if shutil.disk_usage(path).free<gib*2**30:raise RuntimeError('Storage reserve '+path)
   size=0
   for root in (private,out):
    for p in root.rglob('*'):
     try:
      if p.is_file():size+=p.stat().st_size
     except FileNotFoundError:pass
   if size>8*2**30:raise RuntimeError('Own8GiB quota')
   save(out/'state.json',dict(status='RUNNING',phase=name,parent=process_identity(os.getpid()),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage_process=report['process'],elapsed_seconds=time.monotonic()-started));time.sleep(5)
  report['exits']=cohort.exited()
  if report['exits']!={name:0}:raise RuntimeError('Stage failed; preserve stderr '+name)
  report['status']='PASS_STAGE_EXIT0'
 except BaseException as error:
  report.update(status='FAILED',error=repr(error));raise
 finally:
  report['closure']=cohort.close(grace_seconds=120);save(out/(name+'-supervision.json'),report)
 if any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Stage closure not clean')
 return report
def main():
 with (out/'launch-marker').open('x') as f:f.write(str(os.getpid()))
 result=dict(status='FAILED',stages=[])
 try:
  for row in json.loads((out/'source-manifest.json').read_text()):
   if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Frozen source/input changed')
  if not (out/'qualification-source/result.json').exists():result['stages'].append(stage('qualify',[sys.executable,'-B',str(Path(cfg['source'])/'run.py'),str(out),'qualify'],180))
  proof=json.loads((out/'qualification-source/result.json').read_text())
  if proof.get('info',{}).get('actual_goal_ms')!=17601:raise RuntimeError('10ms policy environment did not reproduce actual17601')
  result['stages'].append(stage('train',[sys.executable,'-B',str(Path(cfg['source'])/'train.py'),str(out)],21600))
  result.update(json.loads((out/'training-result.json').read_text()))
 except BaseException as error:
  result.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:save(out/'completion.json',result);save(out/'state.json',result)
if __name__=='__main__':
 if len(sys.argv)>2:
  from environment import qualify
  qualify(out/'environment.json',out/'qualification-source',out/'source-controls-TRAINING-ONLY.json')
 else:main()
