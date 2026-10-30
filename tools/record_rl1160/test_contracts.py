import unittest
import torch
from contracts import clean_dependency, meets_goal, remap_head


class Contracts(unittest.TestCase):
    def test_one_actual_goal_suffices_but_declared_invalid_and_equal_do_not(self):
        self.assertTrue(meets_goal([{'actual_goal_ms': None}]*99 + [{'actual_goal_ms':17600}], 17600))
        for value in (None, True, 17600.0, 17601, 0, -1):
            self.assertFalse(meets_goal([{'actual_goal_ms':value, 'DeclaredResult':17000, 'IsValid':True}],17600))

    def test_dependency_requires_complete_receipt_and_every_clean_exit(self):
        row=dict(phase='completed',closure=dict(remainingOwned=[],cleanupErrors=[],forced=[]),exits=dict(learner=0,actor=0))
        self.assertTrue(clean_dependency(row))
        for key in ('remainingOwned','cleanupErrors','forced'):
            altered={**row,'closure':{**row['closure'],key:[1]}}
            self.assertFalse(clean_dependency(altered))
        self.assertFalse(clean_dependency({**row,'exits':dict(learner=0,actor=1)}))
        self.assertFalse(clean_dependency({'phase':'completed'}))

    def test_dueling_catalog_expansion_preserves_all_existing_action_values(self):
        torch.manual_seed(11)
        q, n, d = 3, 2, 4
        prior={'advantage.weight':torch.randn(q*n,d),'advantage.bias':torch.randn(q*n),
            'value.weight':torch.randn(q,d),'value.bias':torch.randn(q)}
        old=[(0,)*5,(2,)*5]
        new=[(0,)*5,(1,)*5,(2,)*5]
        remapped=remap_head(prior,old,new,q)
        x=torch.randn(7,d)
        def values(state,count):
            adv=(x @ state['advantage.weight'].T+state['advantage.bias']).reshape(7,q,count)
            return (x @ state['value.weight'].T+state['value.bias']).unsqueeze(-1)+adv-adv.mean(-1,keepdim=True)
        before,after=values(prior,2),values(remapped,3)
        torch.testing.assert_close(before,after[:,:,[0,2]],atol=3e-6,rtol=3e-6)
        self.assertTrue((after[:,:,1] < after[:,:,0]).all())
        with self.assertRaises(ValueError):
            remap_head(prior,old,[(0,)*5],q)


if __name__=='__main__':
    unittest.main()
