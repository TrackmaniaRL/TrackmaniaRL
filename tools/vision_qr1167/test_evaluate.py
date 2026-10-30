import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools.vision_qr1167.evaluate import main, checkpoint_source


class EvaluationGates(unittest.TestCase):
    def test_bc_is_reported_as_supervised_only_and_requires_matching_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'initialization.pt').write_bytes(b'checkpoint')
            (root/'result.json').write_text(json.dumps(dict(
                status='BC_INITIALIZATION_ONLY', rl_updates=0, bc_updates=300,
                policy_sha256=hashlib.sha256(b'checkpoint').hexdigest())))
            _, _, provenance = checkpoint_source(root, {}, root)
            self.assertEqual(provenance['policy_kind'], 'BC_ONLY')
            self.assertEqual(provenance['training_updates'], 0)
            (root/'initialization.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                checkpoint_source(root, {}, root)

    def test_incomplete_cleanup_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'completion.json').write_text(json.dumps(dict(cleanupErrors=['owned worker'])))
            with patch.object(sys, 'argv', ['evaluate', folder, str(root/'out')]), self.assertRaisesRegex(RuntimeError, 'cleanup'):
                main()
            self.assertFalse((root/'out').exists())

    def test_exact_live_owner_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'completion.json').write_text(json.dumps(dict(cleanupErrors=[])))
            (root/'windows-launch.json').write_text(json.dumps(dict(pid=123, birthUtcTicks='456')))
            with patch.object(sys, 'argv', ['evaluate', folder, str(root/'out')]), patch(
                    'subprocess.run', side_effect=subprocess.CalledProcessError(9, 'owner-check')), self.assertRaises(subprocess.CalledProcessError):
                main()
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
