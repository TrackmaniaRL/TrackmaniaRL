"""Exercise the campaign loop with a rejected reward followed by a valid episode."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from tools.vision_qr1167 import run
from tools.vision_qr1167.curriculum import ProgressReward, ProgressDiscontinuity

class RecoveryTests(unittest.TestCase):
    def test_bad_progress_never_enters_replay_and_next_episode_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root/'reference.json'
            reference.write_text(json.dumps(dict(status='PASS_REWARD_REFERENCE',records=[])))
            (root/'plan.json').write_text(json.dumps(dict(dependencies=directory,
                reference=str(reference),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                device='cpu',installation='',calibration='',max_seconds=10,max_transitions=1)))
            class Bridge:
                def __init__(self,*args,**kwargs): self.last=None
                def reset_prefix(self,prefix): return self.advance(None)
                def advance(self,*args):
                    return np.zeros((84,84),dtype=np.uint8), dict(telemetry=dict(raceTick=100,position={}),native_steps=[{}]*5)
                def close(self): pass
            class Learning:
                def __init__(self,**kwargs):
                    self.retention=None;self.transitions=0;self.replay=[];self.learner=types.SimpleNamespace(update_count=0)
                def action(self,*args): return 0
                def begin_episode(self): pass
                def add(self,*args): self.replay.append(args);self.transitions+=1
                def update(self): self.learner.update_count+=1;return dict(loss=0.)
                def checkpoint(self): return dict(transitions=self.transitions)
            class Reference:
                def __init__(self,*args): pass
                def reset(self,*args): return 0.,0.
                def project(self,*args): return 1.,0.
            calls=[]
            def injected_reward(self,*args,**kwargs):
                calls.append(1)
                if len(calls)==1: raise ProgressDiscontinuity('injected impossible jump')
                return .05
            fake_bridge=types.SimpleNamespace(TickBridge=Bridge,FramePairingError=type('FramePairingError',(RuntimeError,),{}))
            with patch.dict(sys.modules,{'tools.vision_qr1167.bridge':fake_bridge,
                    'tools.vision_qr1167.learning':types.SimpleNamespace(Learning=Learning),
                    'tools.vision_qr1167.reference':types.SimpleNamespace(Reference=Reference)}), \
                    patch.object(ProgressReward,'step',injected_reward), \
                    patch.object(sys,'argv',['run',str(root)]):
                # End the valid episode immediately after one admitted transition.
                with patch.object(Reference,'project',return_value=(1.,26.)):
                    run.main()
            result=json.loads((root/'completion.json').read_text())
            self.assertEqual(result['phase'],'completed_training_budget')
            self.assertEqual(result['transitions'],1)
            self.assertEqual(result['learner_updates'],1)
            self.assertEqual(result['episodes'],2)
            self.assertEqual(result['rejected_progress'],1)
            rejection=json.loads((root/'rejected-progress-1.json').read_text())
            self.assertFalse(rejection['admitted_to_replay'])
            self.assertEqual(len((root/'transitions.jsonl').read_text().splitlines()),1)

if __name__=='__main__': unittest.main()
