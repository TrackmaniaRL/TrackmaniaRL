"""Small, explicit PyTree helpers used at the collection/training boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from numbers import Number
from types import MappingProxyType, MemberDescriptorType
from typing import Any, Literal

import numpy as np
import torch

type PyTree = Any
type DeviceTransferMode = Literal["blocking", "non_blocking"]


def tree_map(function: Callable[[Any], Any], value: PyTree) -> PyTree:
    """Apply ``function`` to every leaf while preserving tuple/list/dict structure."""

    if isinstance(value, tuple):
        return tuple(tree_map(function, item) for item in value)
    if isinstance(value, list):
        return [tree_map(function, item) for item in value]
    if isinstance(value, Mapping):
        return {key: tree_map(function, item) for key, item in value.items()}
    return function(value)


def tree_to_device(
    value: PyTree,
    device: torch.device | str,
    *,
    mode: DeviceTransferMode = "blocking",
) -> PyTree:
    """Move a tensor PyTree to ``device`` without silently coercing unsupported leaves."""

    def move(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device, non_blocking=mode == "non_blocking")
        if isinstance(leaf, (bool, int, float, str, type(None))):
            return leaf
        raise TypeError(
            "PyTree inference inputs must contain tensors or scalar metadata; "
            f"got unsupported leaf {type(leaf).__name__}"
        )

    return tree_map(move, value)


def tree_collate(values: Sequence[PyTree]) -> PyTree:
    """Stack homogeneous tensor/array/scalar PyTrees into a batch."""

    if not values:
        raise ValueError("Cannot collate an empty PyTree sequence")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return _collate_tensors(values)
    if isinstance(first, np.ndarray):
        return _collate_arrays(values)
    if isinstance(first, tuple):
        return _collate_sequence(values, first, tuple)
    if isinstance(first, list):
        return _collate_sequence(values, first, list)
    if isinstance(first, Mapping):
        return _collate_mapping(values, first)
    if any(type(item) is np.uint64 for item in values) and all(
        type(item) in (int, np.uint64) and 0 <= item < 2**64 for item in values
    ):
        # NumPy scalars mixed with Python ints otherwise overflow Torch's
        # signed inference for values above int64's maximum.
        return torch.as_tensor(np.asarray(values, dtype=np.uint64))
    if isinstance(first, np.bool_) and all(type(item) is type(first) for item in values):
        return torch.as_tensor(np.asarray(values))
    if isinstance(first, (Number, np.bool_)):
        return torch.as_tensor(values)
    raise TypeError(f"Cannot collate unsupported PyTree leaf {type(first).__name__}")


def _collate_tensors(values: Sequence[PyTree]) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Cannot collate mixed tensor and non-tensor leaves")
    return torch.stack(list(values))


def _collate_arrays(values: Sequence[PyTree]) -> torch.Tensor:
    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError("Cannot collate mixed ndarray and non-ndarray leaves")
    return torch.as_tensor(np.stack(list(values)))


def _collate_sequence(
    values: Sequence[PyTree],
    first: Sequence[PyTree],
    container: type[list[Any]] | type[tuple[Any, ...]],
) -> PyTree:
    if not all(isinstance(value, container) and len(value) == len(first) for value in values):
        raise TypeError(f"Cannot collate {container.__name__}s with different structures")
    collated = [tree_collate([value[index] for value in values]) for index in range(len(first))]
    return tuple(collated) if container is tuple else collated


def _collate_mapping(values: Sequence[PyTree], first: Mapping[Any, Any]) -> PyTree:
    keys = tuple(first)
    if not all(
        isinstance(value, Mapping)
        and len(value) == len(first)
        and all(key in value for key in keys)
        for value in values
    ):
        raise TypeError("Cannot collate mappings with different keys")
    return {key: tree_collate([value[key] for value in values]) for key in keys}


def tree_snapshot(value: PyTree) -> PyTree:
    """Take ownership of mutable PyTree leaves at an asynchronous boundary."""

    def snapshot(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.detach().clone()
        if isinstance(leaf, np.ndarray):
            # An object array's buffer contains references; copying only the
            # buffer would leave its mutable elements owned by the caller.
            if leaf.dtype.hasobject:
                return policy_observation_snapshot(leaf)
            return np.array(leaf, copy=True, order="C")
        return deepcopy(leaf)

    return tree_map(snapshot, value)


def policy_observation_snapshot(value: PyTree) -> PyTree:
    """Isolate a policy's input while retaining its Python container types.

    PyTorch cannot deepcopy non-leaf tensors. Seed deepcopy's memo with owned
    tensor snapshots, then let deepcopy preserve custom mappings, namedtuples,
    and shared references without converting them to generic containers.
    """
    memo: dict[int, Any] = {}
    visited: set[int] = set()
    owned_arrays: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []

    def visit_array_elements(array: np.ndarray[Any, Any]) -> None:
        if array.dtype.names:
            for name in array.dtype.names:
                visit_array_elements(array[name])
        elif array.dtype.hasobject:
            for nested in array.flat:
                visit(nested)

    def copy_array_elements(source: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> None:
        if source.dtype.names:
            for name in source.dtype.names:
                copy_array_elements(source[name], target[name])
        elif source.dtype.hasobject:
            for index in np.ndindex(source.shape):
                target[index] = deepcopy(source[index], memo)

    def visit(item: Any) -> None:
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, torch.Tensor):
            memo[identity] = item.detach().clone()
        elif isinstance(item, np.void) and item.dtype.names:
            # Structured scalar records can be nested inside object arrays.
            for name in item.dtype.names:
                visit(item[name])
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject or type(item) is not np.ndarray:
                # NumPy's ndarray deepcopy cannot resolve a self-referential
                # object array. Register its copy before traversing elements.
                copied_array = np.ndarray.copy(item, order="K")
                memo[identity] = copied_array
                owned_arrays.append((item, copied_array))
                visit_array_elements(item)
        elif isinstance(item, memoryview):
            # memoryview does not support deepcopy. Keep element format and
            # dimensions while giving the policy its own writable backing
            # store (or a readonly view if the source is readonly).
            copied = memoryview(np.array(item, copy=True, order="C"))
            memo[identity] = copied.toreadonly() if item.readonly else copied
        elif isinstance(item, MappingProxyType):
            copied_mapping: dict[Any, Any] = {}
            memo[identity] = MappingProxyType(copied_mapping)
            for key, nested in item.items():
                visit(key)
                visit(nested)
                copied_mapping[deepcopy(key, memo)] = deepcopy(nested, memo)
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for nested in item:
                visit(nested)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        if not isinstance(item, (torch.Tensor, memoryview)) and (
            hasattr(item, "__dict__") or hasattr(type(item), "__slots__")
        ):
            # Seed tensors nested in ordinary objects before deepcopy reaches
            # them; non-leaf tensors otherwise reject deepcopy. Avoid walking
            # arbitrary properties by looking only at real slot descriptors.
            for nested in getattr(item, "__dict__", {}).values():
                visit(nested)
            for cls in type(item).__mro__:
                for descriptor in vars(cls).values():
                    if isinstance(descriptor, MemberDescriptorType):
                        try:
                            nested = descriptor.__get__(item, type(item))
                        except AttributeError:
                            continue
                        visit(nested)

    visit(value)
    for source, target in owned_arrays:
        copy_array_elements(source, target)
        if hasattr(source, "__dict__"):
            target.__dict__.clear()
            target.__dict__.update(deepcopy(source.__dict__, memo))
        for cls in type(source).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, MemberDescriptorType):
                    try:
                        nested = getattr(source, name)
                    except AttributeError:
                        continue
                    setattr(target, name, deepcopy(nested, memo))
    return deepcopy(value, memo)


def sanitize_finite(value: PyTree) -> PyTree:
    """Sanitize floating tensor, array, and NumPy scalar values at model boundaries."""

    def sanitize(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return torch.nan_to_num(leaf) if leaf.is_floating_point() else leaf
        if isinstance(leaf, np.ndarray):
            return np.nan_to_num(leaf) if np.issubdtype(leaf.dtype, np.floating) else leaf
        if isinstance(leaf, np.generic) and leaf.dtype.kind in "biufc":
            return np.nan_to_num(leaf) if leaf.dtype.kind == "f" else leaf
        if isinstance(leaf, (bool, int, float, str, type(None))):
            return leaf
        raise TypeError(f"Cannot sanitize unsupported PyTree leaf {type(leaf).__name__}")

    return tree_map(sanitize, value)
