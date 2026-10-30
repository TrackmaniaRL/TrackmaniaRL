"""Protocol-2 checkpoint arrays backed by Torch's streamed tensor storages.

NumPy's protocol-2 reduction otherwise embeds the complete array in a Python
bytes/latin-1 pickle value. Large replay arrays then hold the GIL during copies.
Tensor storages use Torch's separate archive records instead; loading still
returns NumPy arrays and uses the restricted weights-only unpickler.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
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
    def dump(self, value: Any) -> None:
        # Torch installs its storage persistent_id before calling dump. Inspect
        # the complete state first: a later Tensor can alias an earlier NumPy
        # array using another dtype, irrespective of dictionary insertion order.
        self._copy_array_ids = _conflicting_arrays(value)
        super().dump(value)

    def reducer_override(self, value: Any) -> Any:
        if type(value) is not np.ndarray or not value.dtype.isnative:
            return NotImplemented
        # Unsupported structured/object/string dtypes keep their original
        # serialization and the existing restricted-load validation.
        if value.dtype.kind not in "biufc":
            return NotImplemented
        array = value
        if _needs_array_copy(array):
            array = array.copy()
        try:
            tensor = torch.from_numpy(array)
            if self._copy_array_ids is None or id(value) in self._copy_array_ids:
                # Torch rejects one storage address interpreted as two dtypes.
                # Ordinary independent replay columns can be streamed directly;
                # only conflicting arrays need another full storage allocation.
                tensor = tensor.clone(memory_format=torch.preserve_format)
        except (TypeError, ValueError):
            return NotImplemented
        return restore_numpy_array, (tensor,)


def _needs_array_copy(array: np.ndarray[Any, Any]) -> bool:
    # A gapped view can expose storage outside its logical elements when Torch
    # streams its backing storage. Only dense C/F arrays can bypass this copy.
    return (
        not array.flags.writeable
        or not (array.flags.c_contiguous or array.flags.f_contiguous)
        or any(stride < 0 for stride in array.strides)
    )


def _conflicting_arrays(value: Any) -> set[int] | None:
    """Find address/dtype conflicts without inspecting array contents.

    Opaque objects may hide storages in custom pickle reductions. Preserve the
    defensive copy path for the entire payload when their storage is unknown.
    """
    pending = [value]
    seen: set[int] = set()
    addresses: dict[int, set[torch.dtype]] = {}
    arrays: dict[int, int] = {}
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is torch.Tensor:
            if vars(item):
                return None
            if item.layout != torch.strided:
                return None
            try:
                pointer = item.untyped_storage().data_ptr()
            except (RuntimeError, NotImplementedError):
                return None
            addresses.setdefault(pointer, set()).add(item.dtype)
        elif type(item) is np.ndarray:
            if not item.dtype.isnative or item.dtype.kind not in "biufc":
                return None
            # These arrays already receive their own copy in the reducer.
            if _needs_array_copy(item):
                continue
            try:
                tensor = torch.from_numpy(item)
            except (TypeError, ValueError):
                return None
            pointer = tensor.untyped_storage().data_ptr()
            addresses.setdefault(pointer, set()).add(tensor.dtype)
            arrays[id(item)] = pointer
        elif type(item) in (dict, OrderedDict):
            pending.extend(item.keys())
            pending.extend(item.values())
            if isinstance(item, OrderedDict):
                pending.extend(vars(item).values())
        elif type(item) in (tuple, list):
            pending.extend(item)
        elif isinstance(item, np.generic):
            if type(item) is not item.dtype.type or item.dtype.hasobject:
                return None
        elif isinstance(item, np.dtype):
            continue
        elif type(item) not in (
            str,
            bytes,
            int,
            float,
            complex,
            bool,
            type(None),
            torch.device,
            torch.dtype,
        ):
            return None
    return {key for key, pointer in arrays.items() if len(addresses[pointer]) > 1}


# torch.save expects a pickle-module interface, rather than just a Pickler.
Unpickler = pickle.Unpickler
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
