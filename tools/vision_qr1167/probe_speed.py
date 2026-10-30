"""Bounded real-physics timing admission; never trains or writes a policy."""
import argparse,json,sys,time,statistics,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('preparation',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 a.output.mkdir(exist_ok=False)
 plan=json.loads((a.preparation/'campaign-04/plan.json').read_text())
 sys.path.insert(0,plan['dependencies'])
 from tools.vision_qr1167.bridge import TickBridge
 report=dict(status='FAIL',training_started=False,runs=[],bridge_sha256=hashlib.sha256(Path(__file__).with_name('bridge.py').read_bytes()).hexdigest());bridge=None
 started=time.monotonic()
 try:
  for fast,count in ((False,6),(True,30),(True,30)):
   if bridge is None:
    bridge=TickBridge(plan['installation'],plan['calibration'],fast_steps=fast)
   bridge.fast_steps=fast
   _,initial=bridge.reset_prefix(0)
   run=dict(fast=fast,initial=initial,transitions=[]);report['runs'].append(run)
   for i in range(count):
    if time.monotonic()-started>180: raise TimeoutError('probe wall budget')
    # Both signs, throttle, coast and brake; never teacher playback.
    action=((-0.125,1,0),(0.125,1,0),(0,1,0),(0,0,0),(0,1,1))[i%5]
    before=bridge.last;begin=time.monotonic()
    frame,info=bridge.advance(action)
    duration=time.monotonic()-begin
    if info.get('goal'): raise RuntimeError('unexpected finish in short probe')
    clocks=[before['raceTick']]+[x['raceTick'] for x in info['native_steps']]
    assert all(b-a==10 for a,b in zip(clocks,clocks[1:]))
    assert info['rendered_ms']==clocks[-1]+10
    # command is applied with the known one-tick pipeline latency.
    for sample in info['native_steps'][1:]:
     actual=sample['inputs']
     assert actual['accel']==action[1] and actual['brake']==action[2], (action,actual)
     assert (actual['steer']>0)-(actual['steer']<0)==(action[0]>0)-(action[0]<0)
    run['transitions'].append(dict(wall_seconds=duration,evidence=info))
   run['median_transition_s']=statistics.median(x['wall_seconds'] for x in run['transitions'])

  report['speedup']=report['runs'][0]['median_transition_s']/statistics.median(r['median_transition_s'] for r in report['runs'][1:])
  report['physics_steps']=sum(len(r['transitions'])*5 for r in report['runs'])
  report['status']='PASS_EXACT_STEPS_CAMERA_AND_INPUTS'
 except Exception as error:
  report['error']=type(error).__name__+': '+str(error)
  if hasattr(error,'evidence'):report['failure_evidence']=error.evidence
 finally:
  if bridge:
   try: bridge.close()
   except Exception as error: report['cleanup_error']=str(error)
  report['wall_seconds']=time.monotonic()-started
  (a.output/'result.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='runs'}))
 if report['status'].startswith('FAIL') or report.get('cleanup_error'):raise SystemExit(1)
if __name__=='__main__':main()
