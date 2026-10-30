"""Fresh native episodes with exact command receipts and honest prefix provenance."""
import json
from pathlib import Path
import time
import numpy as np
from tools.record_rl1168.features import RouteFeatures,shaped_reward
from tools.record_rl1168.model import ACTIONS


def episode(config,folder,geometry,act,*,deadline,training=False,prefix_controls=(),on_transition=None,stop_file=None):
    if not training and (len(prefix_controls) or on_transition is not None):
        raise ValueError('evaluation forbids teacher prefixes and training callbacks')
    from environment import FallEnvironment
    folder=Path(folder);env=None
    report=dict(status='FAILED',actual_goal_ms=None,prefix_ticks=len(prefix_controls),
                full_start=len(prefix_controls)==0,training=training,policy_steps=0)
    started=time.monotonic();decisions=[]
    try:
        env=FallEnvironment(config,folder)
        route=RouteFeatures(geometry)
        raw=env.reset();state,progress,lateral=route.observe(raw,reset=True)
        stalled=0;furthest=progress
        for step in range(3000):
            if time.monotonic()>=deadline or stop_file and Path(stop_file).exists():
                report['status']='STOPPED_BUDGET_OR_OPERATOR';break
            prefix=step<len(prefix_controls)
            action=None if prefix else act(state)
            command=prefix_controls[step] if prefix else ACTIONS[action]
            following,_,goal,truncated,info=env.step(command)
            next_state,next_progress,lateral=route.observe(following)
            if not np.isclose((following[0]-raw[0])*30000,10,atol=.005):
                raise RuntimeError('native10ms observation discontinuity')
            stalled=0 if next_progress>furthest+.005 else stalled+1
            furthest=max(furthest,next_progress)
            failed=not goal and (truncated or stalled>=200 or lateral>20)
            dt=.01
            if goal:
                dt=(info['actual_goal_ms']-round(float(raw[0])*30000))/1000
                if not 0<dt<=.01:raise RuntimeError('Goal outside measured physics interval')
            try:
                reward=shaped_reward(progress,next_progress,dt=dt,goal=goal,failure=failed)
            except ValueError as error:
                # Reject this measured attempt, never invent a reward or a finish.
                # Training retains its exception/discard path. Evaluation records
                # the failure and advances only after the normal closure checks.
                if training or str(error)!='unphysical progress jump; discard transition':
                    raise
                report.update(status='INVALID_PROGRESS',actual_goal_ms=None,error=repr(error),
                              race_ms=info['race_ms'],previous_progress_m=progress,
                              rejected_progress_m=next_progress,measured_dt=dt)
                break
            if not prefix:
                report['policy_steps']+=1
                decisions.append(dict(race_ms=info['race_ms'],action=action,reward=reward,
                                      goal=goal,failure=failed,progress_m=next_progress,lateral_m=lateral))
                if on_transition:on_transition(state,action,reward,next_state,goal or failed)
            state,raw,progress=next_state,following,next_progress
            if goal or failed:
                report.update(status='NATIVE_GOAL' if goal else 'DRIVING_FAILURE',actual_goal_ms=info['actual_goal_ms'],
                              race_ms=info['race_ms'],progress_m=progress,lateral_m=lateral)
                break
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        if env:
            env.close()
            close=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
            report['closure']=close;report['trace_dir']=str(env.settings.trace_dir)
            report['clean_closure']=not(close['returncode'] or close['remainingOwned'] or close['cleanupErrors'] or close.get('supervisorError'))
        report['elapsed_seconds']=time.monotonic()-started
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'decisions.json').write_text(json.dumps(decisions))
        (folder/'result.json').write_text(json.dumps(report,indent=2))
    if not report.get('clean_closure'):raise RuntimeError('native episode closure failed')
    return report
