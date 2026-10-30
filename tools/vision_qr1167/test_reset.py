import types
import unittest
from unittest.mock import Mock, patch
from tools.vision_qr1167.bridge import TickBridge


class ResetBoundaryTests(unittest.TestCase):
    def make_bridge(self):
        b = TickBridge.__new__(TickBridge)
        b.original = {'activeInputCollectionId': 'teacher'}
        b.revision = 1
        b.pad = Mock()
        b.vg = types.SimpleNamespace(XUSB_BUTTON=types.SimpleNamespace(XUSB_GAMEPAD_B=2))
        b.client = Mock(received_events=50)
        b.client.events = [dict(eventType='vehicle-state', _receive_index=1,
                                payload={'raceTick': 0})]
        b.client.request.return_value = {'revision': 2}
        b.client.require_map.return_value = {'inputExecutionEffective': False}
        b.recover_finish = Mock()
        b.newest = Mock(return_value={'raceTick': 130})
        b.tick = Mock()
        b.observe = Mock(return_value=(None, {}))
        return b

    def test_zero_boundary_survives_newer_nonzero_sample(self):
        b = self.make_bridge()
        def sleep(seconds):
            if seconds == 1.2:
                b.client.events.extend([
                    dict(eventType='vehicle-state', _receive_index=51, payload={'raceTick': 0}),
                    dict(eventType='vehicle-state', _receive_index=52, payload={'raceTick': 130})])
        with patch('tools.vision_qr1167.bridge.time.sleep', side_effect=sleep):
            b.reset_prefix(0)
        b.tick.assert_called_once_with((0, 0, 0))
        b.pad.release_button.assert_called_once()

    def test_old_zero_cannot_certify_a_new_reset(self):
        b = self.make_bridge()
        with patch('tools.vision_qr1167.bridge.time.sleep'), patch(
                'tools.vision_qr1167.bridge.time.monotonic', side_effect=[0, 0, 26]):
            with self.assertRaisesRegex(TimeoutError, 'reset/prefix'):
                b.reset_prefix(0)
        b.tick.assert_not_called()
        b.pad.release_button.assert_called_once()

    def test_interrupted_hold_releases_reset_button(self):
        b = self.make_bridge()
        with patch('tools.vision_qr1167.bridge.time.sleep', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                b.reset_prefix(0)
        b.pad.release_button.assert_called_once()


if __name__ == '__main__':
    unittest.main()
