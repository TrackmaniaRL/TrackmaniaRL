import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from tools.vision_qr1167 import run


class EnvironmentStopTests(unittest.TestCase):
    def test_environment_stop_preserves_checkpoint_and_unexpected_errors_still_fail(self):
        for error, expected in [(EnvironmentPausedError('hidden'), 'stopped_environment_unavailable'),
                                (RuntimeError('broken physics'), 'failed')]:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reference = root/'reference.json'
                reference.write_text(json.dumps(dict(status='PASS_REWARD_REFERENCE', records=[])))
                (root/'plan.json').write_text(json.dumps(dict(dependencies=directory,
                    reference=str(reference), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    device='cpu', installation='', calibration='', max_seconds=10, max_transitions=1)))
                learning = types.SimpleNamespace(retention=None, transitions=7,
                    learner=types.SimpleNamespace(update_count=12), checkpoint=lambda: {'retained': True})
                fake_bridge = types.SimpleNamespace(TickBridge=Mock(side_effect=error),
                                                    FramePairingError=type('FramePairingError', (RuntimeError,), {}))
                with patch.dict(sys.modules, {'tools.vision_qr1167.bridge': fake_bridge,
                        'tools.vision_qr1167.learning': types.SimpleNamespace(Learning=lambda **kw: learning)}), \
                        patch.object(sys, 'argv', ['run', str(root)]):
                    if expected == 'failed':
                        with self.assertRaises(SystemExit):
                            run.main()
                    else:
                        run.main()
                report = json.loads((root/'completion.json').read_text())
                self.assertEqual(report['phase'], expected)
                self.assertEqual(report['learner_updates'], 12)
                self.assertEqual(report['transitions'], 7)
                self.assertEqual(report['cleanupErrors'], [])
                self.assertEqual(report['checkpoint_sha256'], hashlib.sha256((root/'latest.pt').read_bytes()).hexdigest())
                if expected != 'failed':
                    self.assertFalse((root/'failure.txt').exists())
                    self.assertFalse(json.loads((root/'environment-unavailable.json').read_text())['admitted_to_replay'])


if __name__ == '__main__':
    unittest.main()
