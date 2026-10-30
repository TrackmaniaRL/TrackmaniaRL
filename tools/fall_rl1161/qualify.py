from pathlib import Path
import json,sys,time,traceback
out=Path(sys.argv[1]);tag=sys.argv[2];config_path=out/('environment-independent.json' if (out/'environment-independent.json').exists() else 'environment-source.json' if (out/'environment-source.json').exists() else 'environment-multi.json' if (out/'environment-multi.json').exists() else 'environment.json');cfg=json.loads(config_path.read_text());private=Path(cfg['private']);sys.path.insert(0,cfg['source'])
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
if len(sys.argv)>3:
 from environment import qualify
 controls_path=Path(cfg['source_control_file']) if cfg.get('source_control_file') else out/('source-controls-aligned-TRAINING-ONLY.json' if (out/'source-controls-aligned-TRAINING-ONLY.json').exists() else 'source-controls-TRAINING-ONLY.json')
 qualify(config_path,out/tag,controls_path)
else:
 parent=dict(status='FAILED',parent=process_identity(__import__('os').getpid()),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 cohort=CampaignProcessCohort(private/(tag+'-processes'),lambda:(out/(tag+'-STOP')).touch(exist_ok=True));started=time.monotonic()
 try:
  parent['identity']=cohort.start('qualify',[sys.executable,'-B',str(Path(__file__).resolve()),str(out),tag,'worker'],cwd=Path(cfg['source']))
  while not cohort.exited():
   if time.monotonic()-started>180:raise RuntimeError('Bounded 180s native qualification')
   time.sleep(.5)
  parent['exits']=cohort.exited()
  if parent['exits']!={'qualify':0}:raise RuntimeError('Actual native qualification child failed; retain stderr')
  parent['status']='PASS_QUALIFICATION_PROCESS_CLOSED'
 except BaseException as e:
  parent.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  parent['closure']=cohort.close(grace_seconds=10)
  (out/(tag+'-supervision.json')).write_text(json.dumps(parent,indent=2)+'\n')
 print(json.dumps(parent))
