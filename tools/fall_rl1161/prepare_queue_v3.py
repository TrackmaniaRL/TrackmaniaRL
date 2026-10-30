from pathlib import Path
import ast,hashlib,json,runpy,subprocess,sys
out=Path(sys.argv[1]);old_name=sys.argv[2] if len(sys.argv)>2 else 'queue-v2';new_name=sys.argv[3] if len(sys.argv)>3 else 'queue-v3';cfg=json.loads((out/'environment-independent.json').read_text());sys.path.insert(0,cfg['source'])
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
prior=out/old_name;completion=json.loads((prior/'pipeline-completion.json').read_text())
if 'training_process' in completion or same_identity(completion['parent'],process_identity(completion['parent']['pid'])):raise RuntimeError('Only closed never-training queue can migrate')
if json.loads((prior/'power-guard-state.json').read_text())['phase']!='released':raise RuntimeError('Old queue guard not released')
for file in ('windows-launch.json','power-guard-launch.json'):
 r=json.loads((prior/file).read_text());code=f"$p=Get-Process -Id {r['pid']} -ErrorAction SilentlyContinue;if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq '{r['birthUtcTicks']}'){{exit 9}};exit 0"
 if subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',code],cwd='/mnt/c/Windows/System32',capture_output=True,timeout=20).returncode:raise RuntimeError('Old exact Windows owner or guard still active')
plan=json.loads((out/'pipeline-plan.json').read_text());tmrl=Path(plan['dependency_report_roots'][0]);s=json.loads((tmrl/'queue-v3/state.json').read_text())
if s['phase']!='WAITING_FOR_EXISTING_QUALIFICATION':raise RuntimeError('tmrl replacement not waiting')
identity=s['parent'];live=process_identity(identity['pid'])
if live is None or any(live[k]!=identity[k] for k in ('pid','birth','ppid','uid','pgid','sid')):raise RuntimeError('tmrl full identity mismatch')
plan['dependency_identities']=[i for i in plan['dependency_identities'] if i['pid']!=2606]+[identity]
plan['dependency_completion_files'][1]=str(tmrl/'queue-v3/completion.json')
plan['windows_dependencies']=[i for i in plan['windows_dependencies'] if i['pid']!=47412]+[json.loads((tmrl/'queue-v3/windows-launch.json').read_text())]
q=out/new_name;q.mkdir(exist_ok=False);private=Path(cfg['private'])/new_name;private.mkdir(exist_ok=False)
(q/'pipeline-plan.json').write_text(json.dumps(plan,indent=2))
source=(Path(cfg['private'])/old_name/'pipeline.py').read_text().replace(repr(old_name),repr(new_name)).replace("(out/'pipeline-plan.json').read_text()","(queue/'pipeline-plan.json').read_text()")
bad="\n     if not data.get('exits') or any(c!=0 for c in data['exits'].values()):raise RuntimeError(f'Dependency phase exit not0: {path}')"
source=source.replace(bad,'')
source=source.replace("data=json.loads(path.read_text());closure=data.get('closure',{})","data=json.loads(path.read_text());closure=data.get('closure',{})\n     codes=list(data.get('exits',{}).values()) or [c.get('returncode') for c in closure.get('children',{}).values()]\n     if not codes or any(c!=0 for c in codes):raise RuntimeError(f'Dependency phase exit not0: {path}')")
ast.parse(source);target=private/'pipeline.py';target.write_text(source)
sys.argv=[str(target),str(out)];runpy.run_path(str(target),run_name='queue_import_preflight')
(q/'source-manifest.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (target,q/'pipeline-plan.json')],indent=2))
(q/'repair-receipt.json').write_text(json.dumps(dict(status='PASS_IMPORT_PREFLIGHT_NEW_DEPENDENCY_IDENTITIES',training_started=False,old_completion=completion,tmrl_parent=live),indent=2))
print('PASS_FALL_QUEUE_V3_IMPORT_AND_DEPENDENCIES')
