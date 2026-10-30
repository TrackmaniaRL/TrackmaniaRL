"""Pin one fresh Fall launch to current full dependency identities and files."""
from pathlib import Path
import hashlib,json,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'environment-independent.json').read_text());source=Path(cfg['source']);sys.path.insert(0,str(source))
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
base=Path('/home/kuba/tmrl-full-qr-20261022-011')
study=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/pilot-optimized-qr-7h690/pr-review778/joint-optimization846/target18-setup1019/autonomous1032/retained-rl1068/multi-demonstration1088')
tmrl=study/'record-rl1160-20261029-103800'
prior=json.loads((tmrl/'plan.json').read_text())['dependency']
expected=prior['identities']+[dict(pid=2606,birth=262571,uid=0,ppid=2604,pgid=2606,sid=2606)]
actual=[]
for identity in expected:
 current=process_identity(identity['pid'])
 if current is None or any(current[k]!=identity[k] for k in ('pid','birth','uid','ppid','pgid','sid')):raise RuntimeError(f'Dependency changed; inspect instead of resealing: {identity}')
 actual.append(current)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot!=prior['boot']:raise RuntimeError('Unexpected boot')
plan=dict(schema='fall05-causal-native17-ppo-plan/v1',boot=boot,dependency_identities=actual,
 dependency_completion_files=[str(Path(prior['directory'])/'parent-result.json'),str(tmrl/'completion.json')],dependency_report_roots=[str(tmrl)],
 windows_dependencies=json.loads((out/'windows-dependencies.json').read_text()),
 map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',target_goal_ms=21757,final_frozen_trials=100,
 limits=dict(dependency_wait_seconds=43200,BC_seconds=1200,RL_seconds=14400,evaluation_seconds=3600,worker_envelope_seconds=21600,own_quota_GiB=8),
 training='Measured native demonstration BC, 3 independent greedy21758 qualification laps, fresh on-policy PPO from actual native returns, then100 frozen trials',
 success='At least1 of100 actual NativeGoal<=21757; public WR acceptance and Windows TICK deployment are separate and unverified')
with (out/'pipeline-plan.json').open('x') as f:json.dump(plan,f,indent=2)
assets=json.loads(Path(cfg['manifest']).read_text())
paths=list(source.rglob('*.py'))+[Path(cfg['runtime_root'])/p for p in assets['assets']]
paths += [Path(cfg['manifest']),out/'environment-independent.json',out/'environment-admission.json',out/'pipeline-plan.json',out/'qualification-v6/actual-rollout.npz']
rows=[]
for p in sorted(set(paths)):
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size))
with (out/'launch-source-manifest.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(dict(status='SEALED',files=len(rows),dependencies=actual)))
