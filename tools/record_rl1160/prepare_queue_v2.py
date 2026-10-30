"""Fresh waiting-controller snapshot; original learner and training sources stay pinned."""
from pathlib import Path
import ast,hashlib,json,sys
out=Path(sys.argv[1]);base=Path('/home/kuba/tmrl-full-qr-20261022-011');private=base/out.name;sys.path.insert(0,str(private/'source'))
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
completion=json.loads((out/'completion.json').read_text())
if completion.get('status')!='FAILED' or completion.get('error')!="RuntimeError('Own pipeline STOP')":raise RuntimeError('Expected explicit unstarted queue STOP')
if same_identity(completion['parent'],process_identity(completion['parent']['pid'])):raise RuntimeError('Original exact parent still alive')
if (out/'collect-supervision.json').exists():raise RuntimeError('Cannot migrate a started campaign')
if json.loads((out/'power-guard-state.json').read_text())['phase']!='released':raise RuntimeError('Original guard not released')
q=out/'queue-v2';q.mkdir(exist_ok=False);targetdir=private/'queue-v2';targetdir.mkdir(exist_ok=False)
source=(private/'pipeline.py').read_text()
old="capture_output=True, timeout=20)"
if source.count(old)!=1:raise RuntimeError('Interop preimage changed')
source=source.replace(old,"cwd='/mnt/c/Windows/System32', capture_output=True, timeout=20)")
source=source.replace("with (OUT / 'launch-marker').open('x')","with (OUT / 'queue-v2' / 'launch-marker').open('x')")
source=source.replace("save(OUT / 'completion.json', report)","save(OUT / 'queue-v2' / 'completion.json', report)")
source=source.replace("save(OUT / 'state.json',", "save(OUT / 'queue-v2' / 'state.json',")
ast.parse(source);target=targetdir/'pipeline.py';target.write_text(source)
(q/'source-manifest.json').write_text(json.dumps([dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())],indent=2))
(q/'repair-receipt.json').write_text(json.dumps(dict(status='REPAIRED_NEVER_STARTED_QUEUE',old_completion=completion,reason='Windows interop inspection needs a Windows cwd; learner/training code and budgets unchanged'),indent=2))
(out/'STOP').rename(out/'queue-v1-stop-request.txt')
print(json.dumps(dict(script=str(target),queue=str(q))))
