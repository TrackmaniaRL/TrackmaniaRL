import unittest
import torch
from tools.vision_qr1167.model import build_model
from tools.vision_qr1167.quantile_initialization import initialize_point_mass
from trackmaniarl.models.contracts import ValuePhase


class QuantileInitializationTests(unittest.TestCase):
    def test_all_mean_scores_and_greedy_actions_are_preserved(self):
        torch.manual_seed(17)
        model=build_model()
        features=torch.randn(31,256)*20
        support=model.support(features,ValuePhase.EVALUATE)
        before=model.head.evaluate_all(features,support).detach()
        initialize_point_mass(model.head)
        after=model.head.evaluate_all(features,support).detach()
        torch.testing.assert_close(before.mean(-2),after.mean(-2),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.equal(before.mean(-2).argmax(-1),after.mean(-2).argmax(-1)))
        torch.testing.assert_close(after,after[...,:1,:].expand_as(after),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
