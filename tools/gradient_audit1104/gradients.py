"""Compare two loss gradients on the same frozen model and forward graph."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch

type GradientMap = dict[str, torch.Tensor | None]


def gradients(loss: torch.Tensor, parameters: Mapping[str, torch.nn.Parameter]) -> GradientMap:
    values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True,
                                 allow_unused=True)
    result: GradientMap = dict(zip(parameters, values, strict=True))
    if any(value is not None and not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("nonfinite diagnostic gradient")
    return result


def compare_gradients(td: GradientMap, bc: GradientMap) -> dict[str, dict[str, float | None]]:
    if td.keys() != bc.keys():
        raise ValueError("gradient parameter names differ")
    groups = {"all": "", "encoder": "encoder.", "head": "head."}
    output: dict[str, dict[str, float | None]] = {}
    for group, prefix in groups.items():
        td_squared = bc_squared = dot = 0.0
        for name in td:
            if not name.startswith(prefix):
                continue
            left, right = td[name], bc[name]
            if left is not None:
                td_squared += float(left.detach().double().square().sum())
            if right is not None:
                bc_squared += float(right.detach().double().square().sum())
            if left is not None and right is not None:
                dot += float((left.detach().double() * right.detach().double()).sum())
        td_norm, bc_norm = math.sqrt(td_squared), math.sqrt(bc_squared)
        output[group] = {
            "td_norm": td_norm, "bc_weighted_norm": bc_norm,
            "weighted_bc_to_td_ratio": bc_norm / td_norm if td_norm else None,
            "cosine": max(-1.0, min(1.0, dot / (td_norm * bc_norm)))
            if td_norm and bc_norm else None,
            "combined_norm": math.sqrt(max(0.0, td_squared + bc_squared + 2 * dot)),
        }
    return output


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
