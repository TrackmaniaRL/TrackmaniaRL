import types
import unittest
from unittest.mock import patch
from tools.vision_qr1167.bridge import TickBridge
from tools.vision_qr1167.tick_client import UID


class FinishTests(unittest.TestCase):
    def bridge(self, fresh_after=2, hud_after=2):
        b=TickBridge.__new__(TickBridge)
        goal=dict(raceTick=20639,currentCheckpointNumber=1,totalCheckpointCount=1)
        b.last=goal
        events=[dict(_receive_index=0,eventType='vehicle-state',payload=goal)]
        speeds=[]
        b.client=types.SimpleNamespace(events=events,request=lambda endpoint:dict(currentMapUid=UID),speed=speeds.append)
        b.vg=types.SimpleNamespace(XUSB_BUTTON=types.SimpleNamespace(XUSB_GAMEPAD_A=1))
        class Pad:
            presses=0
            def update(self):pass
            def press_button(self,**kw):
                self.presses+=1
                value=dict(raceTick=0,currentCheckpointNumber=0,totalCheckpointCount=1) if self.presses>=fresh_after else goal
                events.append(dict(_receive_index=len(events),eventType='vehicle-state',payload=value))
            def release_button(self,**kw):pass
        b.pad=Pad()
        b.camera=types.SimpleNamespace(validate=lambda:None,capture=lambda:None)
        def read(frame):
            if b.pad.presses<hud_after:raise ValueError('finish replay has no racing clock')
            return 0,[]
        b.clock=types.SimpleNamespace(read=read)
        return b,speeds

    def test_map_uid_during_terminal_transition_does_not_mean_ready(self):
        b,speeds=self.bridge()
        with patch('tools.vision_qr1167.bridge.time.sleep'):b.recover_finish()
        self.assertEqual(b.pad.presses,2)
        self.assertEqual(speeds,[1])
        self.assertIsNone(b.last)

    def test_replay_vehicle_without_racing_hud_is_not_admitted(self):
        b,_=self.bridge(fresh_after=1,hud_after=3)
        with patch('tools.vision_qr1167.bridge.time.sleep'):b.recover_finish()
        self.assertEqual(b.pad.presses,3)

    def test_never_ready_stops_at_bounded_acknowledgements(self):
        b,_=self.bridge(fresh_after=99,hud_after=99)
        with patch('tools.vision_qr1167.bridge.time.sleep'),self.assertRaises(TimeoutError):b.recover_finish()
        self.assertEqual(b.pad.presses,8)


if __name__=='__main__':unittest.main()
