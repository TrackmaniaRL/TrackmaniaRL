import unittest
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from model import Policy,decode,advantages,OBS_DIM
from orientation import rotation_matrix
from train import ppo_update
from atomic_json import save

class Contracts(unittest.TestCase):
    def test_all_motor_modes_and_residual_bounds(self):
        classes=torch.tensor([[0,0],[127,1],[254,2],[127,3]])
        torch.testing.assert_close(decode(classes),torch.tensor([[-1.,0,0],[0,0,1],[1,1,0],[0,1,1]]))
        model=Policy();controls,*_=model.act(torch.randn(20,OBS_DIM),stochastic=True)
        self.assertTrue(bool((controls[:,0].abs()<=1).all()))

    def test_critic_gradient_is_separate_from_actor(self):
        model=Policy();_,value=model.residual_distribution(torch.randn(8,OBS_DIM));value.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.residual.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_actual_ppo_update_changes_residual_but_not_neural_base(self):
        torch.manual_seed(1164);model=Policy();model.freeze_base();x=torch.randn(32,OBS_DIM)
        with torch.no_grad():dist,_=model.residual_distribution(x);z=dist.sample();logp=dist.log_prob(z)
        base={k:v.clone() for k,v in model.encoder.state_dict().items()};before={k:v.clone() for k,v in model.residual.state_dict().items()}
        optimizer=(torch.optim.Adam(model.residual.parameters(),lr=1e-6),torch.optim.Adam(model.critic.parameters(),lr=1e-4))
        data=dict(x=x,z=z,logp=logp,advantage=z.clone(),returns=torch.linspace(-1,1,32))
        ppo_update(model,optimizer,data,torch.device('cpu'))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.residual.state_dict().items()))
        self.assertTrue(all(torch.equal(base[k],v) for k,v in model.encoder.state_dict().items()))

    def test_terminal_value_does_not_leak(self):
        a,_=advantages(torch.tensor([1.]),torch.tensor([0.,999.]),torch.tensor([1.]))
        self.assertEqual(float(a[0]),1.)

    def test_residual_cannot_alter_early_critical_section(self):
        model=Policy();x=torch.zeros(2,OBS_DIM);x[:,0]=torch.tensor([.5,.6])
        with torch.no_grad():
            model.steering.weight.zero_();model.steering.bias.zero_();model.steering.bias[127]=10;model.residual[-1].bias.fill_(1.)
            control,*_=model.act(x,stochastic=False)
        self.assertEqual(float(control[0,0]),0.)
        self.assertGreater(float(control[1,0]),0.)

    def test_quaternion_sign_cannot_change_orientation_features(self):
        q=np.asarray([.206,-.106,-.221,-.107],dtype=np.float32)
        np.testing.assert_array_equal(rotation_matrix(q),rotation_matrix(-q))
        matrix=rotation_matrix(q).reshape(3,3)
        np.testing.assert_allclose(matrix.T@matrix,np.eye(3),atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)),1.,places=6)

    def test_transient_windows_reader_does_not_abort_status_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'state.json';target.write_text('{"old":true}')
            original=Path.replace;attempts=[]
            def blocked(stage,destination):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('reader lacks delete sharing')
                return original(stage,destination)
            with patch.object(Path,'replace',blocked),patch('atomic_json.time.sleep'):
                save(target,dict(status='RUNNING'))
            self.assertEqual(json.loads(target.read_text()),dict(status='RUNNING'))
            self.assertEqual(len(attempts),3)

    def test_permanent_lock_is_bounded_and_keeps_original_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'state.json';target.write_text('{"old":true}')
            with patch.object(Path,'replace',side_effect=PermissionError('persistent lock')) as replacement,patch('atomic_json.time.sleep'):
                with self.assertRaises(PermissionError):save(target,dict(status='RUNNING'))
            self.assertEqual(replacement.call_count,101)
            self.assertEqual(json.loads(target.read_text()),dict(old=True))

if __name__=='__main__':unittest.main()
