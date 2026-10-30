"""The shared trace multiplexes physical POST, committed input and read audits."""
import struct
import unittest
from tools.native_ghost1140.extract_autonomous import physical_posts

def row(magic: bytes,version: int,pid: int,phase: int) -> bytes:
    data=bytearray(256);data[:8]=magic
    struct.pack_into('<4I',data,8,256,version,pid,pid)
    struct.pack_into('<I',data,148,phase)
    return bytes(data)

class TraceLayoutTests(unittest.TestCase):
    def test_multiplexed_records_are_not_misread_as_physics(self) -> None:
        source=row(b'TMPOST02',2,41,0)+row(b'TMAUD003',2,41,1065353216)+row(b'TMREAD01',1,41,447)+row(b'TMPOST03',3,41,75)
        self.assertEqual(list(physical_posts(source,41)),[source[-256:]])

    def test_rejects_foreign_owner_unknown_layout_and_truncation(self) -> None:
        for source in (row(b'TMPOST03',3,42,75),row(b'UNKNOWN!',3,41,75),row(b'TMPOST03',3,41,75)[:-1]):
            with self.subTest(length=len(source)),self.assertRaises(ValueError):list(physical_posts(source,41))

if __name__=='__main__':unittest.main()
