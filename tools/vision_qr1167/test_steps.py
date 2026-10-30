import types
import unittest
from unittest.mock import patch
from tools.vision_qr1167.bridge import TickBridge

class StepTests(unittest.TestCase):
 def bridge(self,count=1,unsolicited=False):
  b=TickBridge.__new__(TickBridge);b.fast_steps=True;b.step_timings=[]
  b.last=dict(sequence=1,raceTick=10)
  events=[dict(_receive_index=1,eventType='vehicle-state',payload=dict(sequence=1,raceTick=20 if unsolicited else 10))]
  b.client=types.SimpleNamespace(events=events,require_map=lambda:dict(effectiveGameSpeed=0,inputExecutionEffective=False))
  b.vg=types.SimpleNamespace(XUSB_BUTTON=types.SimpleNamespace(XUSB_GAMEPAD_DPAD_RIGHT=1))
  class Pad:
   released=False
   def left_joystick_float(self,**kw):pass
   def right_trigger_float(self,*a):pass
   def left_trigger_float(self,*a):pass
   def update(self):pass
   def press_button(self,**kw):
    for i in range(count):events.append(dict(_receive_index=events[-1]['_receive_index']+1,eventType='vehicle-state',payload=dict(sequence=i+2,raceTick=20+10*i)))
   def release_button(self,**kw):self.released=True
  b.pad=Pad();return b
 def test_exact_ack_and_release(self):
  b=self.bridge()
  with patch('tools.vision_qr1167.bridge.time.sleep'):
   self.assertEqual(b.tick((0,1,0))['raceTick'],20)
  self.assertTrue(b.pad.released)
 def test_duplicate_never_admitted(self):
  b=self.bridge(2)
  with patch('tools.vision_qr1167.bridge.time.sleep'),self.assertRaisesRegex(RuntimeError,'exactly one'):
   b.tick((0,1,0))
  self.assertTrue(b.pad.released)
  self.assertEqual(b.last['raceTick'],10)
 def test_unsolicited_step_never_admitted(self):
  b=self.bridge(unsolicited=True)
  with self.assertRaisesRegex(RuntimeError,'unsolicited'):b.tick((0,1,0))
 def test_transient_speed_waits_for_pause(self):
  b=self.bridge()
  states=iter([0,.1,0])
  b.client.require_map=lambda:dict(effectiveGameSpeed=next(states),inputExecutionEffective=False)
  with patch('tools.vision_qr1167.bridge.time.sleep'):
   self.assertEqual(b.tick((0,1,0))['raceTick'],20)
 def test_extra_step_while_settling_is_rejected(self):
  b=self.bridge()
  calls=[]
  def status():
   calls.append(1)
   if len(calls)==2:
    b.client.events.append(dict(_receive_index=3,eventType='vehicle-state',payload=dict(sequence=3,raceTick=30)))
   return dict(effectiveGameSpeed=0,inputExecutionEffective=False)
  b.client.require_map=status
  with patch('tools.vision_qr1167.bridge.time.sleep'),self.assertRaisesRegex(RuntimeError,'exactly one'):
   b.tick((0,1,0))
 def test_missing_ack_releases_button(self):
  b=self.bridge(0)
  with patch('tools.vision_qr1167.bridge.time.sleep'),patch('tools.vision_qr1167.bridge.time.monotonic',side_effect=[0,0,0,3,3]),self.assertRaises(TimeoutError):
   b.tick((0,1,0))
  self.assertTrue(b.pad.released)

 def test_previous_race_high_sequences_do_not_count_as_new_steps(self):
  b=self.bridge()
  b.client.events.insert(0,dict(_receive_index=0,eventType='vehicle-state',payload=dict(sequence=9000,raceTick=17708)))
  with patch('tools.vision_qr1167.bridge.time.sleep'):
   self.assertEqual(b.tick((0,1,0))['raceTick'],20)

 def test_new_extra_step_with_restarted_sequence_is_rejected(self):
  b=self.bridge(2)
  b.client.events[0]['payload']['sequence']=9000
  with patch('tools.vision_qr1167.bridge.time.sleep'),self.assertRaisesRegex(RuntimeError,'exactly one'):
   b.tick((0,1,0))

if __name__=='__main__':unittest.main()
