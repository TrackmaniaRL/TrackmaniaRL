import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from tools.record_rl1168.episodes import episode
from tools.record_rl1168.archive import archive_closed_trace


class Environment:
    def __init__(self,config,folder):
        self.settings=SimpleNamespace(trace_dir=Path(folder)/'trace');self.tick=0
    def reset(self):return self.state()
    def state(self):
        result=np.zeros(22,dtype=np.float32);result[1:10]=np.eye(3).reshape(-1)
        result[0]=self.tick*10/30000;result[10]=self.tick*.1/1000
        return result
    def step(self,command):
        self.tick+=1;goal=self.tick==3
        return self.state(),0,goal,False,dict(actual_goal_ms=21 if goal else None,race_ms=self.tick*10)
    def close(self):
        self.settings.trace_dir.mkdir(parents=True)
        (self.settings.trace_dir/'native-supervisor.json').write_text(json.dumps(dict(returncode=0,remainingOwned=[],cleanupErrors=[])))


class EpisodeTests(unittest.TestCase):
    def test_invalid_progress_counts_as_failed_trial_and_next_trial_runs(self):
        geometry=np.array([[x,0,0] for x in range(150)])
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,environment=SimpleNamespace(FallEnvironment=Environment)):
            first=Path(directory)/'first';second=Path(directory)/'second'
            with patch('tools.record_rl1168.episodes.shaped_reward',side_effect=ValueError('unphysical progress jump; discard transition')):
                failed=episode({},first,geometry,lambda state:0,deadline=time.monotonic()+5)
            success=episode({},second,geometry,lambda state:0,deadline=time.monotonic()+5)
            rows=[failed,success]
            self.assertEqual(len(rows),2)
            self.assertEqual(sum(r['status']=='NATIVE_GOAL' for r in rows),1)
            self.assertEqual(failed['status'],'INVALID_PROGRESS')
            self.assertIsNone(failed['actual_goal_ms'])
            self.assertTrue(failed['clean_closure'])
            self.assertEqual(json.loads((first/'result.json').read_text())['status'],'INVALID_PROGRESS')

    def test_invalid_progress_does_not_hide_bad_closure(self):
        class BadClosure(Environment):
            def close(self):
                super().close()
                p=self.settings.trace_dir/'native-supervisor.json'
                d=json.loads(p.read_text());d['cleanupErrors']=['test failure'];p.write_text(json.dumps(d))
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,environment=SimpleNamespace(FallEnvironment=BadClosure)),patch('tools.record_rl1168.episodes.shaped_reward',side_effect=ValueError('unphysical progress jump; discard transition')):
            with self.assertRaisesRegex(RuntimeError,'closure failed'):
                episode({},directory,np.array([[x,0,0] for x in range(150)]),lambda state:0,deadline=time.monotonic()+5)

    def test_unrelated_evaluation_error_still_raises(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,environment=SimpleNamespace(FallEnvironment=Environment)),patch('tools.record_rl1168.episodes.shaped_reward',side_effect=ValueError('nonfinite reward input')):
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                episode({},directory,np.array([[x,0,0] for x in range(150)]),lambda state:0,deadline=time.monotonic()+5)

    def test_goal_reward_uses_measured_partial_final_tick(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,environment=SimpleNamespace(FallEnvironment=Environment)):
            rows=[]
            result=episode({},directory,np.array([[x,0,0] for x in range(150)]),lambda state:0,
                deadline=time.monotonic()+5,training=True,on_transition=lambda *row:rows.append(row))
            self.assertEqual(result['actual_goal_ms'],21)
            self.assertAlmostEqual(sum(row[2] for row in rows),100-.021)
            self.assertTrue(rows[-1][-1]);self.assertTrue(result['clean_closure'])
    def test_evaluation_rejects_teacher_prefix(self):
        with self.assertRaises(ValueError):
            episode({},'unused',[],None,deadline=0,prefix_controls=[(0,1,0)])
    def test_archive_roundtrip_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);trace=root/'traces'/'owned';trace.mkdir(parents=True)
            (trace/'native-supervisor.json').write_text(json.dumps(dict(returncode=0,remainingOwned=[],cleanupErrors=[])))
            (trace/'actual.bin').write_bytes(bytes(range(256))*100)
            receipt=archive_closed_trace(root,trace,root/'archive.tar.gz')
            self.assertTrue(receipt['roundtrip_verified']);self.assertFalse(trace.exists())
            with self.assertRaises(ValueError):archive_closed_trace(root,root/'other',root/'wrong.tar.gz')


if __name__=='__main__':unittest.main()
