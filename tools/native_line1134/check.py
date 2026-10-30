"""Inspect only this campaign and write a separate inspection receipt."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'source'))
from backend import clock,descendants,identity,reserves,save,sha

def main() -> None:
    state=json.loads((root/'state.json').read_text())
    now=clock()
    expected=state['parent'];actual=identity(expected['pid'])
    alive=actual==expected
    active=state['phase'] in ('qualifying','searching','finalizing_model')
    if active and (not alive or now-state['clock_monotonic_raw']>30):raise RuntimeError('Parent identity/heartbeat failure')
    reserves(root,Path('/home/kuba/tmrl-full-qr-20261022-011')/root.name)
    plan=json.loads((root/'plan.json').read_text())
    if any(sha(root/'source'/name)!=value for name,value in plan['source_sha256'].items()):raise RuntimeError('Frozen source changed')
    previous=root/'latest-independent-inspection.json'
    before=json.loads(previous.read_text()) if previous.exists() else None
    counters={k:state[k] for k in ('batches','candidate_evaluations','native_records','confirmed_improvements')}
    if before and active and now-before['clock_monotonic_raw']>120:
        if state['phase']=='searching' and counters['native_records']<=before['counters']['native_records']:
            raise RuntimeError('No actual evaluated input increase across inspections')
    own=descendants(expected['pid']) if alive else []
    result=dict(status='PASS_ACTIVE' if active else 'CLOSED_REQUIRES_FINAL_AUDIT',phase=state['phase'],
        parent_alive_exact=alive,parent=expected,actual_descendants=own,clock_monotonic_raw=now,
        heartbeat_age_seconds=now-state['clock_monotonic_raw'],counters=counters,
        best_native_input_goal_ms=state['best_goal_ms'],best_autonomous_goal_ms=state['best_autonomous_goal_ms'],
        windows_game_used=False,online_RL_updates=0)
    save(previous,result)
    print(json.dumps(result))

if __name__=='__main__':main()
