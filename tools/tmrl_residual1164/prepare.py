"""Pinned10ms physics environment for autonomous neural-base residual PPO."""
from pathlib import Path
import ast,hashlib,json,shutil,sys,time
BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
STUDY=Path('/mnt/f/AITrackmania/reports/headless-full-qr-20261022-r11/pilot-optimized-qr-7h690/pr-review778/joint-optimization846/target18-setup1019/autonomous1032/retained-rl1068/multi-demonstration1088')
OLD=BASE/'fall-rl1161-20261029-110735'
REPO=Path('/mnt/c/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania')
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
 name='tmrl-residual1164-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime());out=STUDY/name;private=BASE/name;out.mkdir();private.mkdir(mode=0o700)
 shutil.copytree(OLD/'source',private/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(OLD/'runtime',private/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for p in HERE.glob('*.py'):
  ast.parse(p.read_text());shutil.copy2(p,private/'source'/p.name)
 source=private/'source';runtime=private/'runtime'
 cfg=json.loads((Path('/mnt/f/AITrackmania/experiments/fall2026-05-ghosts-0749e38ff0f6/fall-rl1161-20261029-110735')/'environment-independent.json').read_text());assets=json.loads(Path(cfg['manifest']).read_text())
 assets['runtime_root']=str(runtime);assets['map_uid']='oqIJ5rQDRrNwLPTh9H2p_W4tLof'
 original=json.loads((BASE/'waypoint-runtime002/assets-terminal-diagnostic584.json').read_text())
 for role in ('map','tape'):
  relative=original['roles'][role];src=BASE/'waypoint-runtime002'/relative;dst=runtime/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  assets['roles'][role]=relative;assets['assets'][relative]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
 save(private/'assets.json',assets)
 cfg.update(runtime_root=str(runtime),manifest=str(private/'assets.json'),manifest_sha256=sha(private/'assets.json'),report=str(out),private=str(private),source=str(source),source_control_file=str(out/'source-controls-TRAINING-ONLY.json'),bootstrap='fixed neutral2190->2200 reset; policy first observes2200',map_uid=assets['map_uid'],carrier_note='Unrelated native17712 carrier; each actual SGB command replaced and audited before native input load')
 save(out/'environment.json',cfg)
 text=(OLD/'source/environment.py').read_text().replace('a5fa295009541f85ebffff1557081dd5942f442493ea9966da8fe7d8fbb0ce13','ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc').replace('v96wCr7mP5xEGxNLSnZC6DAp4se','oqIJ5rQDRrNwLPTh9H2p_W4tLof').replace('fall05-causal-native17/bootstrap2190/v1','tmrl-causal-native17/reset2200/v1')
 old="packet=self.session.reset_bootstrap();self.state=physical_state(packet)\n        if self.state['nativeT']!=2190:raise RuntimeError('Actual pre-race policy-decision bootstrap required')"
 new="packet=self.session.reset_bootstrap()\n        packet,eof,applied=self.session.advance_control((0.,0.,0.))\n        if eof or applied!=(0.,0.,0.):raise RuntimeError('Measured neutral reset interval required')\n        self.session.provenance.update(literalResetNativeT=2200,initialIntervalPolicyControlled=False,administrativeNeutralReset=True)\n        self.state=physical_state(packet)\n        if self.state['nativeT']!=2200:raise RuntimeError('Actual native2200 first policy observation required')"
 if text.count(old)!=1:raise RuntimeError('Native reset preimage differs')
 text=text.replace(old,new).replace('21.758-goal/1000.','17.601-goal/1000.')
 text=text.replace('def observation(state):','from orientation import rotation_matrix\n\ndef observation(state):').replace("*physical['quaternionWXYZ']","*rotation_matrix(physical['quaternionWXYZ'])").replace('result.shape!=(17,)','result.shape!=(22,)').replace('causal-native17/v1','causal-native22-rotation/v1').replace('tmrl-causal-native17/reset2200/v1','tmrl-causal-native22-rotation/reset2200/v2')
 ast.parse(text);(source/'environment.py').write_text(text)
 sys.path.insert(0,str(source));from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
 best=json.loads((STUDY/'line-search1136/best.json').read_text());motors=Path(best['motors']);ticks=json.loads(motors.read_text())
 if best['actual_native_goal_ms']!=17601:raise RuntimeError('Expected pinned17601 baseline')
 save(out/'source-controls-TRAINING-ONLY.json',dict(controls=[action_sgb(t)[0] for t in ticks],race_command_packet_offset=1,actual_native_goal_ms=17601,source=str(motors),source_sha256=sha(motors)))
 save(out/'plan.json',dict(private=str(private),source=str(source),map_uid=assets['map_uid'],map_sha256=assets['assets'][assets['roles']['map']]['sha256'],baseline_ms=17601,target_ms=17600,BC_seconds=1200,RL_seconds=14400,evaluation_seconds=3600,architecture='frozen neural imitation base + learned bounded steering residual Normal policy + independent critic',observation='22 actual physical features every10ms; orientation is a quaternion-sign-invariant rotation matrix; no camera, geometry or teacher lookup',fixed_motor_transform='steering clamp(base +gate(time>=16500ms)*0.01*tanh(latent),-1,1); gas/brake from frozen neural base',residual_after_ms=16500,latent_std=.1,actor_learning_rate=1e-6,critic_learning_rate=1e-4,GAE_lambda=.995,actor_mean_bound=.2,final_trials=100,public_WR_claim=False))
 extra=[]
 if len(sys.argv)>1:
  previous=Path(sys.argv[1]);old_cfg=json.loads((previous/'environment.json').read_text());old_source=Path(old_cfg['source']);proof=json.loads((previous/'qualification-source/result.json').read_text());proof_root=previous if (previous/'qualify-supervision.json').exists() else Path(json.loads((previous/'reused-native-and-BC-provenance.json').read_text())['previous']);supervision=json.loads((proof_root/'qualify-supervision.json').read_text())
  same_environment=sha(source/'environment.py')==sha(old_source/'environment.py')
  if proof.get('info',{}).get('actual_goal_ms')!=17601:raise RuntimeError('Prior native baseline differs')
  if supervision['exits']!={'qualify':0} or any(supervision['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Prior native proof did not close cleanly')
  if json.loads((previous/'power-guard-state.json').read_text())['phase']!='released':raise RuntimeError('Prior experiment still owns power request')
  for tag in ('qualify','train'):
   receipt=previous/(tag+'-supervision.json')
   if not receipt.exists():continue
   closed=json.loads(receipt.read_text())['closure']
   if any(closed[k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Prior experiment processes not closed')
  shutil.copy2(previous/'bc-policy.pt',out/'initial-bc-policy.pt')
  if same_environment:
   (out/'qualification-source').mkdir();shutil.copy2(previous/'qualification-source/actual-rollout.npz',out/'qualification-source/actual-rollout.npz');shutil.copy2(previous/'qualification-source/result.json',out/'qualification-source/result.json')
   extra += [out/'qualification-source/actual-rollout.npz',out/'qualification-source/result.json']
  save(out/'reused-native-and-BC-provenance.json',dict(previous=str(proof_root),BC_source_report=str(previous),native_goal_ms=17601,environment_code_unchanged=same_environment,new_actual_source_collection_required=not same_environment,model_code_unchanged=sha(source/'model.py')==sha(old_source/'model.py'),policy_change='quaternion-sign-invariant rotation features, smaller fixed residual variance and separate actor/critic optimizers',input_projection_migration='unchanged time/position/velocity/engine/control/Fourier columns copied; new9 rotation columns start zero and are learned from fresh measured data',previous_failure_preserved=True,initial_policy_sha256=sha(out/'initial-bc-policy.pt'),new_BC_optimizer=True))
  extra += [out/'initial-bc-policy.pt',out/'reused-native-and-BC-provenance.json']
 save(out/'source-manifest.json',[dict(path=str(p),sha256=sha(p)) for p in sorted(source.rglob('*.py'))]+[dict(path=str(p),sha256=sha(p)) for p in (private/'assets.json',out/'environment.json',out/'source-controls-TRAINING-ONLY.json',out/'plan.json',*extra)])
 print(json.dumps(dict(report=str(out),private=str(private))))

if __name__=='__main__':main()
