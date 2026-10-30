from __future__ import annotations
import random
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.native_line1134.core import ACTIVE, UID, mutate, native_goal, neighbourhood, validate


class ScoringTests(unittest.TestCase):
    def row(self, actual: object) -> dict[str, object]:
        return {'MapUid': UID, 'IsValid': False, 'DeclaredResult': {'Time': 1},
                'ValidatedResult': actual}

    def test_dnf_does_not_inherit_declared_time(self) -> None:
        self.assertIsNone(native_goal(self.row(None)))

    def test_computed_goal_is_used_even_for_worse_declaration(self) -> None:
        self.assertEqual(native_goal(self.row(dict(Time=17715, NbCheckpoints=1, NbRespawns=0))), 17715)

    def test_invalid_goal_records_are_rejected(self) -> None:
        for time in (True, 0, -1, 17.7, 150000):
            with self.subTest(time=time), self.assertRaises(ValueError):
                native_goal(self.row(dict(Time=time, NbCheckpoints=1, NbRespawns=0)))

    def test_checkpoint_and_respawn_guards(self) -> None:
        for cp, respawns in ((0, 0), (2, 0), (1, 1)):
            with self.subTest(cp=cp, respawns=respawns), self.assertRaises(ValueError):
                native_goal(self.row(dict(Time=17712, NbCheckpoints=cp, NbRespawns=respawns)))

    def test_map_binding(self) -> None:
        with self.assertRaises(ValueError):
            native_goal({'MapUid': 'other', 'ValidatedResult': None})


class MutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (39,) * 100 + (75,) * 3 + (39,) * (ACTIVE - 103) + (36,) * 4

    def test_edges_never_cross_second_boundary(self) -> None:
        values = list(neighbourhood(self.source, random.Random(17)))
        self.assertTrue(values)
        for _, edited in values:
            validate(edited)
            changed = [i for i, (a, b) in enumerate(zip(self.source, edited)) if a != b]
            self.assertEqual(len(set(self.source[i] for i in changed)), 1)

    def test_combined_exploration_keeps_clock_and_padding(self) -> None:
        rng = random.Random(29)
        changed = 0
        for _ in range(500):
            edited = mutate(self.source, rng, 3, 2)
            validate(edited)
            changed += edited != self.source
        self.assertGreater(changed, 490)

    def test_analog_and_pulse_aliases_are_rejected(self) -> None:
        for invalid in (42, 38, 78):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate((invalid,) + self.source[1:])


class FinalizationLifecycleTests(unittest.TestCase):
    def test_closed_collection_does_not_stop_training_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sys,'argv',['finalize.py',str(root)]),patch.dict(os.environ,{},clear=False):
                helpers=runpy.run_path(str(Path(__file__).with_name('finalize.py')),run_name='lifecycle_test')
            out=helpers['OUT'];(out/'collection').mkdir(parents=True)
            (out/'collection/STOP').touch()
            self.assertTrue(helpers['stopped']('collection'))
            self.assertFalse(helpers['stopped']('training'))
            self.assertFalse(helpers['stopped']('evaluation'))
            (out/'STOP').touch()
            self.assertTrue(helpers['stopped']('training'))
            self.assertTrue(helpers['stopped']('evaluation'))


if __name__ == '__main__':
    unittest.main()
