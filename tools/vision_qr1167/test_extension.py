import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.vision_qr1167.retention import validated_parent


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        base=dict(observations=np.zeros((2,4,84,84),np.uint8),actions=np.array([8,16]),race_ms=np.array([220,560]))
        added=dict(observations=np.ones((1,4,84,84),np.uint8),actions=np.array([0]),race_ms=np.array([1000]))
        np.savez(self.root/'base.npz',**base)
        np.savez(self.root/'added.npz',**added)
        self.combined={key:np.concatenate((base[key],added[key])) for key in base}
        digest=lambda name:hashlib.sha256((self.root/name).read_bytes()).hexdigest()
        admission=dict(status='ADMITTED_TEACHER_WINDOWS',teacher_goal_ms=17601,dataset_sha256=digest('added.npz'))
        (self.root/'added.json').write_text(json.dumps(admission))
        self.receipt=dict(parent_dataset_sha256=digest('base.npz'),parent_archive=str(self.root/'base.npz'),parent_rows=2,
            addition_receipt=str(self.root/'added.json'),addition_receipt_sha256=digest('added.json'),
            addition_archive=str(self.root/'added.npz'),addition_dataset_sha256=digest('added.npz'))

    def validate(self):
        np.savez(self.root/'combined.npz',**self.combined)
        with np.load(self.root/'combined.npz',allow_pickle=False) as data:
            return validated_parent(self.receipt,data)

    def test_exact_base_and_admitted_addition_are_accepted(self):
        self.assertEqual(self.validate(),self.receipt['parent_dataset_sha256'])

    def test_replacing_one_base_pixel_is_rejected(self):
        self.combined['observations'][0,0,0,0]=9
        with self.assertRaisesRegex(ValueError,'base rows'):self.validate()

    def test_appending_unadmitted_row_is_rejected(self):
        for key in self.combined:
            self.combined[key]=np.concatenate((self.combined[key],self.combined[key][-1:]))
        with self.assertRaisesRegex(ValueError,'unadmitted rows'):self.validate()

    def test_changed_parent_archive_is_rejected(self):
        (self.root/'base.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'parent archive'):self.validate()


if __name__=='__main__':unittest.main()
