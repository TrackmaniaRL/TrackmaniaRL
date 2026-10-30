"""One new queued Fall05 experiment, bounded and supervised through pidfds."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,traceback
out=Path(sys.argv[1]).resolve();cfg=json.loads((out/'environment-independent.json').read_text());private=Path(cfg['private']);sys.path.insert(0,cfg['source'])
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity

def save(path,value):
 path=Path(path);stage=path.with_suffix('.stage');stage.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');stage.replace(path)

def stopped():return (out/'STOP').exists()

def reserves():
 for p,n in [('/mnt/c',10),('/mnt/f',6),('/mnt/j',10),(str(private),6)]:
  if shutil.disk_usage(p).free<n*2**30:raise RuntimeError(f'Storage reserve {p}')
 size=0
 for root in (out,private):
  for p in root.rglob('*'):
   try:
    if p.is_file():size+=p.stat().st_size
   except FileNotFoundError:pass
 if size>8*2**30:raise RuntimeError('Own Fall experiment quota8GiB')

def check_sources():
 for item in json.loads((out/'launch-source-manifest.json').read_text()):
  p=Path(item['path'])
  with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
  if actual!=item['sha256']:raise RuntimeError(f'Frozen Fall code or input changed: {p}')

def wait_dependencies():
 plan=json.loads((out/'pipeline-plan.json').read_text());started=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if boot!=plan['boot']:raise RuntimeError('Boot changed since launch admission')
 while time.monotonic()-started<43200:
  if stopped():raise RuntimeError('Own queued pipeline STOP')
  reserves();alive=[p for p in plan['dependency_identities'] if same_identity(p,process_identity(p['pid']))]
  save(out/'pipeline-state.json',dict(phase='WAITING_FOR_EXISTING_CAMPAIGNS',parent=process_identity(os.getpid()),boot=boot,dependency_alive=alive,training_started=False,monotonic=time.monotonic()))
  if not alive:
   if not all(Path(p).is_file() for p in plan['dependency_completion_files']):raise RuntimeError('Dependency disappeared without completion receipt')
   dependencies=[json.loads(Path(p).read_text()) for p in plan['dependency_completion_files']]
   for data in dependencies:
    closure=data.get('closure',{})
    if any(closure.get(k)!=[] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Dependency did not close cleanly')
    if not data.get('exits') or any(code!=0 for code in data['exits'].values()):raise RuntimeError('Dependency worker did not exit0; inspect before automatic launch')
   for root in plan['dependency_report_roots']:
    files=list(Path(root).glob('*-supervision.json'))+list(Path(root).glob('rl/completion.json'))
    for path in files:
     data=json.loads(path.read_text());closure=data.get('closure',{})
     if closure.get('remainingOwned')!=[] or closure.get('cleanupErrors')!=[] or closure.get('forced')!=[]:raise RuntimeError(f'Dependency closure not clean: {path}')
   for identity in plan['windows_dependencies']:
    pid,ticks=int(identity['pid']),int(identity['birthUtcTicks'])
    code=f'$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; if($p -and $p.StartTime.ToUniversalTime().Ticks -eq {ticks}){{exit 9}};exit 0'
    checked=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',code],capture_output=True,timeout=20)
    if checked.returncode!=0:raise RuntimeError('Exact Windows dependency owner still active or inspection failed')
   for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if 'campaign_entry' in cmd and '--role learner' in cmd:raise RuntimeError('Another learner appeared while queued')
   save(out/'dependency-closure.json',dict(status='PASS_EXACT_DEPENDENCIES_GONE',receipts=dependencies));return
  time.sleep(20)
 raise RuntimeError('Bounded12h dependency wait expired')

def main():
 with (out/'pipeline-launch-marker').open('x') as stream:stream.write(str(os.getpid()))
 result=dict(status='FAILED',parent=process_identity(os.getpid()),started_unix=time.time());cohort=None
 try:
  check_sources()
  if json.loads((out/'environment-admission.json').read_text())['status']!='PASS_FALL05_CAUSAL_PHYSICS_STEP_ENVIRONMENT':raise RuntimeError('Actual Fall environment admission required')
  wait_dependencies();check_sources();reserves()
  cohort=CampaignProcessCohort(private/'training-cohort',lambda:(out/'STOP').touch(exist_ok=True));started=time.monotonic()
  result['training_process']=cohort.start('fall-training',[sys.executable,'-B',str(Path(cfg['source'])/'train.py'),str(out)],cwd=Path(cfg['source']))
  save(out/'pipeline-state.json',dict(phase='BC_THEN_ACTUAL_PPO',**result,monotonic=time.monotonic()))
  while not cohort.exited():
   if stopped() or time.monotonic()-started>21600:raise RuntimeError('Own six-hour BC/RL/evaluation envelope or STOP')
   reserves();time.sleep(10)
  result['exits']=cohort.exited()
  if result['exits']!={'fall-training':0}:raise RuntimeError('Fall training worker failed; retain stderr and checkpoints')
  training=json.loads((out/'training-result.json').read_text());result.update(status=training['status'],training=training)
 except BaseException as error:
  result.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:
  if cohort is not None:
   result['closure']=cohort.close(grace_seconds=120)
   if any(result['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):result['status']='FAILED_CLOSURE'
  result['ended_unix']=time.time();save(out/'pipeline-completion.json',result);save(out/'pipeline-state.json',result)

if __name__=='__main__':main()
