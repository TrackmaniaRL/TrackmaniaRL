"""Repair only the closed, never-training Fall queue in a fresh namespace."""
from pathlib import Path
import ast,hashlib,json,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'environment-independent.json').read_text());q=out/'queue-v2'
completion=json.loads((out/'pipeline-completion.json').read_text())
if completion.get('status')!='FAILED' or completion.get('error')!="RuntimeError('Own queued pipeline STOP')" or 'training_process' in completion:raise RuntimeError('Only the explicitly stopped waiting queue may migrate')
sys.path.insert(0,cfg['source'])
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
if same_identity(completion['parent'],process_identity(completion['parent']['pid'])):raise RuntimeError('Original exact queue parent remains alive')
for file in ('windows-launch.json','power-guard-launch.json'):
 record=json.loads((out/file).read_text())
 import subprocess
 code=f"$p=Get-Process -Id {record['pid']} -ErrorAction SilentlyContinue; if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq '{record['birthUtcTicks']}'){{exit 9}};exit 0"
 checked=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',code],cwd='/mnt/c/Windows/System32',capture_output=True,timeout=20)
 if checked.returncode:raise RuntimeError(f'Original Windows queue owner/guard still active or inspection failed: {checked.returncode}, {checked.stderr.decode(errors="replace")}')
guard=json.loads((out/'power-guard-state.json').read_text())
if guard['phase']!='released':raise RuntimeError('Original queue power request not released')
q.mkdir(exist_ok=False)
source=(Path(cfg['source'])/'pipeline.py').read_text()
old="""   for data in dependencies:
    closure=data.get('closure',{})
    if any(closure.get(k)!=[] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Dependency did not close cleanly')
    if not data.get('exits') or any(code!=0 for code in data['exits'].values()):raise RuntimeError('Dependency worker did not exit0; inspect before automatic launch')"""
new="""   # The existing qualification has an aggregate closure. The queued tmrl
   # orchestrator writes individual phase receipts instead, not aggregate exits.
   first,tmrl=dependencies
   if first.get('phase')!='completed' or any(first.get('closure',{}).get(k)!=[] for k in ('remainingOwned','cleanupErrors','forced')) or not first.get('exits') or any(c!=0 for c in first['exits'].values()):raise RuntimeError('Qualification did not complete with clean exit0 closure')
   if tmrl.get('status') not in ('RL_COMPLETED_TARGET_NOT_REACHED','RL_TARGET_REACHED_EXPORT_PENDING'):raise RuntimeError('tmrl experiment failed; inspect its original failure before training Fall')
   tmrl_root=Path(plan['dependency_report_roots'][0])
   required=('collect-supervision.json','baseline-evaluate-supervision.json','bc-supervision.json','final-evaluate-supervision.json','rl/completion.json')
   if not all((tmrl_root/p).is_file() for p in required):raise RuntimeError('Missing actual tmrl phase closure receipt')"""
if source.count(old)!=1:raise RuntimeError('Gate patch preimage differs')
source=source.replace(old,new)
source=source.replace('capture_output=True,timeout=20','cwd="/mnt/c/Windows/System32",capture_output=True,timeout=20')
source=source.replace("out/'pipeline-launch-marker'","queue/'pipeline-launch-marker'").replace("out/'pipeline-state.json'","queue/'pipeline-state.json'").replace("out/'pipeline-completion.json'","queue/'pipeline-completion.json'")
source=source.replace("def stopped():return (out/'STOP').exists()","queue=out/'queue-v2'\ndef stopped():return (out/'STOP').exists() or (queue/'STOP').exists()")
source=source.replace("def check_sources():","def check_sources():\n for item in json.loads((queue/'source-manifest.json').read_text()):\n  if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Frozen queue-v2 code changed')")
ast.parse(source)
private=Path(cfg['private'])/'queue-v2';private.mkdir(exist_ok=False)
target=private/'pipeline.py';target.write_text(source)
manifest=[dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())]
(q/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'STOP').rename(out/'queue-v1-stop-request.txt')
(q/'repair-receipt.json').write_text(json.dumps(dict(status='REPAIRED_WAITING_QUEUE_ONLY',original_completion=str(out/'pipeline-completion.json'),original_power_guard_released=guard,training_started=False,reason='tmrl orchestrator emits phase closures, not top-level exits/closure',source=manifest),indent=2))
print(json.dumps(dict(queue=str(q),script=str(target))))
