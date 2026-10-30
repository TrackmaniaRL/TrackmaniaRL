import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.vision_qr1167.run import atomic_json


class AtomicTests(unittest.TestCase):
    def test_transient_windows_sharing_failure_preserves_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            path.write_text('{"old":true}')
            import os
            replace = os.replace
            calls = []
            def transient(source, target):
                calls.append(1)
                if len(calls) == 1:
                    self.assertEqual(json.loads(path.read_text()), {'old': True})
                    raise PermissionError('sharing violation')
                replace(source, target)
            with patch('tools.vision_qr1167.run.os.replace', transient):
                atomic_json(path, {'new': True})
            self.assertEqual(json.loads(path.read_text()), {'new': True})

    def test_persistent_denial_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('tools.vision_qr1167.run.os.replace', side_effect=PermissionError), patch(
                    'tools.vision_qr1167.run.time.monotonic', side_effect=[0, 3]), self.assertRaises(PermissionError):
                atomic_json(Path(directory)/'state.json', {})


if __name__ == '__main__': unittest.main()
