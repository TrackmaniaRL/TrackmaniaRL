"""Bound compressed task assets before NumPy decompresses their members."""

from __future__ import annotations

import io
import math
import zipfile

import numpy as np


def validate_npz_size(raw: bytes, *, max_uncompressed: int) -> None:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 512 or sum(member.file_size for member in members) > max_uncompressed:
            raise ValueError("decompressed task asset exceeds size limit")
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate task asset entries")
        for member in members:
            if member.is_dir() or not member.filename.endswith(".npy"):
                raise ValueError("task assets must contain only NPY arrays")
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError("unsupported task asset NPY version")
                count = math.prod(shape)
                if (
                    dtype.hasobject
                    or any(dimension < 0 for dimension in shape)
                    or count > max_uncompressed
                ):
                    raise ValueError("task asset array exceeds size limit or has object dtype")
                payload = count * dtype.itemsize
                if payload > max_uncompressed or source.tell() + payload != member.file_size:
                    raise ValueError("task asset array size does not match NPY payload")
