from pathlib import Path
import ast,hashlib,json,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'environment-independent.json').read_text());q=out/'queue-v2';private=Path(cfg['private'])/'queue-v2';sys.path.insert(0,cfg['source'])
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
original=json.loads((out/'pipeline-plan.json').read_text());tmrl=Path(original['dependency_report_roots'][0]);state=json.loads((tmrl/'queue-v2/state.json').read_text())
if state['phase']!='WAITING_FOR_EXISTING_QUALIFICATION':raise RuntimeError('Replacement tmrl queue not waiting')
identity=state['parent'];live=process_identity(identity['pid'])
if live is None or any(live[k]!=identity[k] for k in ('pid','birth','ppid','uid','sid','pgid')):raise RuntimeError('tmrl replacement identity mismatch')
plan=dict(original);plan['dependency_identities']=[i for i in original['dependency_identities'] if i['pid']!=2606]+[identity]
plan['dependency_completion_files']=[original['dependency_completion_files'][0],str(tmrl/'queue-v2/completion.json')]
plan['windows_dependencies']=[i for i in original['windows_dependencies'] if i['pid']!=47412]+[json.loads((tmrl/'queue-v2/windows-launch.json').read_text())]
(q/'pipeline-plan.json').write_text(json.dumps(plan,indent=2))
target=private/'pipeline.py';source=target.read_text();source=source.replace("(out/'pipeline-plan.json').read_text()","(queue/'pipeline-plan.json').read_text()")
# Each individual closure must also demonstrate successful process exits.
source=source.replace("data=json.loads(path.read_text());closure=data.get('closure',{})","data=json.loads(path.read_text());closure=data.get('closure',{})\n     if not data.get('exits') or any(c!=0 for c in data['exits'].values()):raise RuntimeError(f'Dependency phase exit not0: {path}')")
ast.parse(source);target.write_text(source)
paths=[target,q/'pipeline-plan.json']
(q/'source-manifest.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],indent=2))
print(json.dumps(dict(status='SEALED_REPLACEMENT_QUEUE_DEPENDENCIES',tmrl_parent=live)))
