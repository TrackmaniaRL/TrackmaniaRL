"""User-authorized direct record RL; preserve, but do not pass, stopped qualification."""
from pathlib import Path
import ast,hashlib,json,shutil,sys,time
BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
STUDY=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/pilot-optimized-qr-7h690/pr-review778/joint-optimization846/target18-setup1019/autonomous1032/retained-rl1068/multi-demonstration1088')
OLD=STUDY/'qualification-recovery1162-20261029-134230'
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
 plan=read(OLD/'plan.json');private=Path(plan['private']);sys.path.insert(0,plan['source'])
 from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
 from trackmaniarl.core.builtins import TorchCheckpointCodec
 from trackmaniarl.core.checkpoints import validate_checkpoint_v2
 import torch
 torch.set_num_threads(1)
 if not (OLD/'operator-transition1163.json').exists():raise RuntimeError('Explicit operator transition required')
 completed=read(OLD/'completion.json');qualification=read(private/'parent-result.json');supervision=read(OLD/'qualification-supervision.json')
 for closure in (qualification['closure'],supervision['closure']):
  if any(closure[k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Previous processes not cleanly closed')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 identities=[completed['parent']]+[r[k] for r in qualification['processes'] for k in ('worker','supervisor')]
 if boot==plan['boot'] and any(same_identity(r,process_identity(r['pid'])) for r in identities):raise RuntimeError('An exact previous process remains alive')
 if read(OLD/'power-guard-state.json')['phase']!='released':raise RuntimeError('Previous power request not released')
 tmrl=Path(plan['tmrl']);tp=Path(plan['tmrl_private'])
 if (tmrl/'launch-marker').exists():raise RuntimeError('tmrl pipeline already started; never duplicate')
 name='direct-record1163-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime());out=STUDY/name;newprivate=BASE/name;out.mkdir();newprivate.mkdir(mode=0o700)
 checkpoints=sorted((OLD/'artifacts'/OLD.name/'checkpoints').glob('distributed-update-*.pt'))
 if not checkpoints:raise RuntimeError('Controlled qualification stop did not preserve a checkpoint')
 checkpoint=checkpoints[-1];codec=TorchCheckpointCodec();state=codec.load(checkpoint);validate_checkpoint_v2(state['learner'])
 tensors=[]
 def visit(value):
  if isinstance(value,dict):
   for v in value.values():visit(v)
  elif isinstance(value,(tuple,list)):
   for v in value:visit(v)
  elif isinstance(value,torch.Tensor):tensors.append(value)
 visit(state['learner'])
 if not all(torch.isfinite(v).all() for v in tensors):raise RuntimeError('Nonfinite retained qualification checkpoint')
 shutil.copy2(checkpoint,out/'retained-qualification-checkpoint.pt')
 receipt=dict(status='PASS_OPERATOR_STOP_CLOSED',original_phase=qualification['phase'],qualification_completed=False,operator_request=read(OLD/'operator-transition1163.json'),previous_main_parent=completed['parent'],previous_boot=plan['boot'],closures=[qualification['closure'],supervision['closure']],checkpoint=dict(path=str(out/'retained-qualification-checkpoint.pt'),sha256=sha(checkpoint),counters=state['distributed'],finite_tensors=len(tensors)),native_reset_smoke=read(OLD/'smoke-result.json')['status'])
 save(out/'operator-stop-closure.json',receipt)
 text=(tp/'pipeline.py').read_text();a=text.index('def wait_for_dependency():');b=text.index('\ndef collect():',a)
 text=text[:a]+f'''def wait_for_dependency():
    receipt=read(Path({str(out/'operator-stop-closure.json')!r}))
    if receipt.get('status')!='PASS_OPERATOR_STOP_CLOSED' or receipt.get('qualification_completed') is not False:raise RuntimeError('Explicit controlled qualification bypass evidence required')
    for closure in receipt['closures']:
        if any(closure[k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Prior owned process remains')
    save(OUT/'dependency-closure.json',receipt)

'''+text[b:]
 start=text.index("        phase('bc', 1260)");end=text.index('        prepare_rl(selected)',start)
 text=text[:start]+'''        selected = OUT / 'baseline-policy.pt'
        save(OUT/'bc-skipped.json',dict(reason='User requests actual record RL now; the already qualified17705 policy initializes QR-DQN, with17601 demonstrations in replay and the releasing expert objective',standalone_BC_updates=0))
'''+text[end:]
 # One deterministic trial plus99 reproducible stochastic trials are useful for
 # the user's one-in100 criterion. Policy weights remain frozen throughout.
 needle='            policy.reset_episode()\n            started = time.monotonic()'
 replacement="            policy.reset_episode()\n            torch.manual_seed(1163000+trial)\n            stochastic = destination == 'final-100-evaluation' and trial > 0\n            policy.set_exploration_epsilon(.002 if stochastic else 0.)\n            started = time.monotonic()"
 if text.count(needle)!=1:raise RuntimeError('Evaluation seed preimage differs')
 text=text.replace(needle,replacement)
 text=text.replace('action = policy.act(observation, PolicyMode.EVALUATION)','action = policy.act(observation, PolicyMode.ONLINE if stochastic else PolicyMode.EVALUATION)')
 text=text.replace('weights_unchanged=True, greedy=True)','weights_unchanged=True, greedy=destination != \'final-100-evaluation\', final_trial_policy=\'trial0 greedy; trials1-99 epsilon0.002 fixed before run\')')
 ast.parse(text);target=tp/'pipeline-direct1163.py';target.write_text(text)
 rows=read(tmrl/'source-manifest.json');rows.append(dict(path=str(target.relative_to(tp)),sha256=sha(target)));save(tmrl/'source-manifest.json',rows)
 inputs=read(tmrl/'input-manifest.json');inputs.append(dict(path=str(out/'operator-stop-closure.json'),sha256=sha(out/'operator-stop-closure.json')));save(tmrl/'input-manifest.json',inputs)
 save(out/'plan.json',dict(private=str(newprivate),source=str(tp/'source'),tmrl=str(tmrl),tmrl_private=str(tp),tmrl_entry=str(target),fall=plan['fall'],fall_private=plan['fall_private'],old_qualification=str(OLD),boot=boot,qualification_skipped_by_user=True,RL_seconds=14400,standalone_BC=False,maximum_native_goal_ms=17600,final_trials=100,final_policy='1 greedy +99 stochastic epsilon0.002; no weight updates or teacher reads',public_WR_claim=False))
 shutil.copy2(HERE/'run.py',newprivate/'run.py')
 save(out/'source-manifest.json',[dict(path=str(p),sha256=sha(p)) for p in (target,newprivate/'run.py',out/'plan.json',out/'operator-stop-closure.json',tmrl/'source-manifest.json',tmrl/'input-manifest.json')])
 print(json.dumps(dict(report=str(out),private=str(newprivate),retained_qualification_updates=state['distributed']['updates'])))

if __name__=='__main__':main()
