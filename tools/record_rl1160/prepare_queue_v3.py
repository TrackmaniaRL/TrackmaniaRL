from pathlib import Path
import ast,hashlib,json,runpy,sys
out=Path(sys.argv[1]);private=Path('/home/kuba/tmrl-full-qr-20261022-011')/out.name
q=out/'queue-v3';q.mkdir(exist_ok=False);d=private/'queue-v3';d.mkdir(exist_ok=False)
source=(private/'queue-v2/pipeline.py').read_text().replace("from contracts import clean_dependency, meets_goal","sys.path.insert(0, str(PRIVATE))\nfrom contracts import clean_dependency, meets_goal").replace("'queue-v2'","'queue-v3'")
ast.parse(source);target=d/'pipeline.py';target.write_text(source)
sys.argv=[str(target),str(out)];runpy.run_path(str(target),run_name='queue_import_preflight')
(q/'source-manifest.json').write_text(json.dumps([dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())],indent=2))
(q/'repair-receipt.json').write_text(json.dumps(dict(status='PASS_IMPORT_PREFLIGHT',reason='Preserved v2 import failure; contracts module needs explicit private path',training_started=False),indent=2))
print('PASS_IMPORT_PREFLIGHT_QUEUE_V3')
