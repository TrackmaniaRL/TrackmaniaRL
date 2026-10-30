import unittest
from unittest.mock import Mock, patch
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from tools.vision_qr1167.bridge import TickBridge, FramePairingError


class PausedVisibilityTests(unittest.TestCase):
    def bridge(self):
        b = TickBridge.__new__(TickBridge)
        b.last = dict(raceTick=500, simulationTick=1500)
        b.newest = Mock(return_value=dict(b.last))
        b.client = Mock()
        b.client.require_map.return_value = dict(effectiveGameSpeed=0, inputExecutionEffective=False)
        b.camera = Mock()
        return b

    def test_visibility_returns_same_state_without_an_extra_physics_step(self):
        b = self.bridge()
        frame = object()
        b.camera.capture.side_effect = [EnvironmentPausedError('hidden'), frame]
        with patch('tools.vision_qr1167.bridge.time.sleep'):
            actual, waited = b.capture_paused_frame()
        self.assertIs(actual, frame)
        self.assertGreaterEqual(waited, 0)
        b.client.speed.assert_not_called()
        self.assertEqual(b.client.require_map.call_count, 2)

    def test_changed_state_is_rejected_after_focus_returns(self):
        b = self.bridge()
        b.camera.capture.side_effect = [EnvironmentPausedError('hidden'), object()]
        b.newest.side_effect = [dict(b.last), dict(raceTick=510, simulationTick=1510)]
        with patch('tools.vision_qr1167.bridge.time.sleep'), self.assertRaises(FramePairingError):
            b.capture_paused_frame()

    def test_loss_of_pause_is_not_recoverable(self):
        b = self.bridge()
        b.camera.capture.side_effect = EnvironmentPausedError('hidden')
        b.client.require_map.return_value['effectiveGameSpeed'] = 1
        with self.assertRaises(FramePairingError):
            b.capture_paused_frame()

    def test_wait_has_a_deadline(self):
        b = self.bridge()
        b.camera.capture.side_effect = EnvironmentPausedError('hidden')
        with patch('tools.vision_qr1167.bridge.time.monotonic', side_effect=[0, 16]), self.assertRaises(EnvironmentPausedError):
            b.capture_paused_frame(timeout=15)
        self.assertEqual(b.camera.capture.call_count, 1)


if __name__ == '__main__':
    unittest.main()
