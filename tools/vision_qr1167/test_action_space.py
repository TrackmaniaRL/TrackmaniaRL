import unittest
import types
from unittest.mock import patch

import numpy as np
import torch

from tools.vision_qr1167.learning import Learning
from trackmaniarl.algorithms.value_based.updater import ValueUpdater


class ActionSpaceTests(unittest.TestCase):
    def test_new_demonstrated_action_requires_verified_parent(self):
        learning=Learning(device='cpu')
        learning.configure_action_ids((0,8,16))
        learning.learner.update_count=123
        learning.pending_retention={'dataset_sha256':'base'}
        learning.retention=types.SimpleNamespace(actions=torch.tensor([0,8,16,34]),parent_sha256='different')
        with self.assertRaisesRegex(ValueError,'verified extension'):learning.extend_actions_from_retention()
        learning.retention.parent_sha256='base'
        self.assertEqual(learning.extend_actions_from_retention(),[34])
        self.assertEqual(learning.action_ids,(0,8,16,34))
        self.assertEqual(learning.learner.policy_action_ids,(0,8,16,34))
        self.assertEqual(learning.learner.update_count,123)
    def test_actor_and_double_q_target_ignore_unavailable_high_value(self):
        learning = Learning(device='cpu')
        learning.configure_action_ids(tuple(range(17)))
        values = torch.zeros(1,68)
        values[0,8] = 1
        values[0,61] = 999
        with patch.object(learning.learner.model,'expected_all_actions',return_value=values):
            self.assertEqual(learning.action(np.zeros((4,84,84),np.uint8),0),8)
            next_action = ValueUpdater._next_actions(learning.learner,torch.zeros(1,256))
            self.assertEqual(next_action.item(),8)
        self.assertTrue(all(learning.action(np.zeros((4,84,84),np.uint8),1)<17 for _ in range(100)))

    def test_checkpoint_preserves_actor_and_bootstrap_action_space(self):
        source = Learning(device='cpu')
        source.configure_action_ids(tuple(range(17)))
        restored = Learning(device='cpu')
        restored.restore(source.checkpoint())
        self.assertEqual(restored.action_ids,tuple(range(17)))
        self.assertEqual(restored.learner.policy_action_ids,tuple(range(17)))

    def test_cannot_silently_change_existing_rl_action_space(self):
        learning = Learning(device='cpu')
        learning.learner.update_count = 1
        with self.assertRaisesRegex(ValueError,'fresh learner'):
            learning.configure_action_ids(tuple(range(17)))

    def test_noncontiguous_action_ids_keep_native_catalog_mapping(self):
        learning = Learning(device='cpu')
        learning.configure_action_ids((0,8,17,50,51))
        values = torch.zeros(1,68)
        values[0,51] = 2
        values[0,61] = 999
        with patch.object(learning.learner.model,'expected_all_actions',return_value=values):
            self.assertEqual(learning.action(np.zeros((4,84,84),np.uint8),0),51)
            self.assertEqual(ValueUpdater._next_actions(learning.learner,torch.zeros(1,256)).item(),51)


if __name__=='__main__':
    unittest.main()
