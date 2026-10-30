"""Protocol-2 checkpoint arrays backed by Torch's streamed tensor storages.

NumPy's protocol-2 reduction otherwise embeds the complete array in a Python
bytes/latin-1 pickle value. Large replay arrays then hold the GIL during copies.
Tensor storages use Torch's separate archive records instead; loading still
returns NumPy arrays and uses the restricted weights-only unpickler.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch


def restore_numpy_array(tensor: Any) -> Any:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError("checkpoint array requires a CPU tensor")
    if tensor.layout != torch.strided:
        raise ValueError("checkpoint array requires strided storage")
    return tensor.numpy()


class Pickler(pickle.Pickler):
    def reducer_override(self, value: Any) -> Any:
        if type(value) is not np.ndarray or not value.dtype.isnative:
            return NotImplemented
        # Unsupported structured/object/string dtypes keep their original
        # serialization and the existing restricted-load validation.
        if value.dtype.kind not in "biufc":
            return NotImplemented
        array = value
        if not array.flags.writeable or any(stride < 0 for stride in array.strides):
            array = array.copy()
        try:
            # NumPy pickle stores distinct views independently. Give each
            # reduction its own storage as well: Torch rejects shared pointers
            # interpreted as different dtypes, including mixed Tensor/ndarray
            # payloads. The pickler memo still preserves repeated array objects.
            tensor = torch.from_numpy(array).clone(memory_format=torch.preserve_format)
        except (TypeError, ValueError):
            return NotImplemented
        return restore_numpy_array, (tensor,)


# torch.save expects a pickle-module interface, rather than just a Pickler.
Unpickler = pickle.Unpickler
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
