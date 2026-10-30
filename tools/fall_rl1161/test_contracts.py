from pathlib import Path
import json,struct,sys,unittest
import numpy as np
import torch
BASE=Path('/home/kuba/tmrl-full-qr-20261022-011');PRIVATE=BASE/'fall-rl1161-20261029-110735'
sys.path.insert(0,str(PRIVATE/'source'))
from multi_goal import decode_multi_goal_facts
from trackmaniarl.trackmania.native_headless.full_qr import goal_delivery as original
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext
from model import Policy,controls,advantages


class Contracts(unittest.TestCase):
    def test_actual_goal_capture_and_append_tampering(self):
        trace=PRIVATE/'traces/qualification-v6'
        metadata=json.loads(next(trace.glob('episode-*/metadata.json')).read_text())
        pid=metadata['server_pid'];raw=(trace/'posts'/f'goal-fact-{pid}.bin').read_bytes()
        context=NativeGoalContext(pid,metadata)
        original_binding=original.decode_goal_facts.__globals__['decode_writer_pairs']
        facts=decode_multi_goal_facts(raw,context)
        self.assertEqual([f.event_native_t-2200 for f in facts if f.endpoint_requested],[21758])
        self.assertGreater(len(facts),1)
        for offset in (16,80,768+240+480):
            bad=bytearray(raw);bad[offset]^=1
            with self.assertRaises(ValueError):decode_multi_goal_facts(bytes(bad),context)
        with self.assertRaises(ValueError):decode_multi_goal_facts(raw[:-1],context)
        self.assertIs(original.decode_goal_facts.__globals__['decode_writer_pairs'],original_binding)
        with self.assertRaises(ValueError):original.decode_goal_facts(raw,context)

    def test_motor_domain_and_real_demo_exact_representation(self):
        out=Path('/mnt/f/AITrackmania/experiments/fall2026-05-ghosts-0749e38ff0f6')/PRIVATE.name
        with np.load(out/'qualification-v6/actual-rollout.npz',allow_pickle=False) as z:applied=torch.from_numpy(z['controls']).float()
        action=torch.stack(((applied[:,0]*127).round().long()+127,applied[:,2].long()),-1)
        torch.testing.assert_close(controls(action),applied,rtol=0,atol=0)
        with self.assertRaises(ValueError):controls(torch.tensor([[255,0]]))

    def test_gae_never_bootstraps_across_terminal(self):
        advantage,returns=advantages(torch.tensor([1.,2.]),torch.tensor([0.,0.,999.]),torch.tensor([1.,1.]),gamma=1,lam=1)
        torch.testing.assert_close(returns,torch.tensor([1.,2.]))
        torch.testing.assert_close(advantage,returns)

    def test_ppo_uses_returns_and_changes_policy_parameters(self):
        from train import ppo_update
        torch.manual_seed(16);torch.set_num_threads(1)
        model=Policy();x=torch.randn(16,17)
        with torch.no_grad():
            s,b,v=model.distributions(x);a=torch.stack((s.sample(),b.sample()),-1);logp=s.log_prob(a[:,0])+b.log_prob(a[:,1])
        data=dict(observation=x,action=a,old_logp=logp,advantage=torch.arange(16).float(),returns=torch.arange(16).float())
        before=model.steering.weight.detach().clone();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        metrics=ppo_update(model,optimizer,data,torch.device('cpu'))
        self.assertTrue(metrics);self.assertFalse(torch.equal(before,model.steering.weight))
        self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))


if __name__=='__main__':unittest.main()
