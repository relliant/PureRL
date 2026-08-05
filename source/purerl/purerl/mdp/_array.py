"""Small NumPy/PyTorch compatibility layer used by pure MDP functions."""

from __future__ import annotations

from typing import Any, Sequence


def is_torch(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def abs_value(value: Any) -> Any:
    if is_torch(value):
        return value.abs()
    import numpy as np

    return np.abs(value)


def arccos(value: Any) -> Any:
    if is_torch(value):
        return value.arccos()
    import numpy as np

    return np.arccos(value)


def clip(value: Any, minimum: float, maximum: float) -> Any:
    if is_torch(value):
        return value.clamp(min=minimum, max=maximum)
    import numpy as np

    return np.clip(value, minimum, maximum)


def concatenate(values: Sequence[Any], axis: int = -1) -> Any:
    if not values:
        raise ValueError("At least one tensor is required for concatenation")
    if is_torch(values[0]):
        import torch

        return torch.cat(tuple(values), dim=axis)
    import numpy as np

    return np.concatenate(tuple(values), axis=axis)


def exp(value: Any) -> Any:
    if is_torch(value):
        return value.exp()
    import numpy as np

    return np.exp(value)


def maximum(value: Any, minimum: float) -> Any:
    if is_torch(value):
        return value.clamp_min(minimum)
    import numpy as np

    return np.maximum(value, minimum)


def max_axis(value: Any, axis: int) -> Any:
    if is_torch(value):
        return value.max(dim=axis).values
    return value.max(axis=axis)


def min_axis(value: Any, axis: int) -> Any:
    if is_torch(value):
        return value.min(dim=axis).values
    return value.min(axis=axis)


def norm(value: Any, axis: int = -1) -> Any:
    if is_torch(value):
        return value.norm(dim=axis)
    import numpy as np

    return np.linalg.norm(value, axis=axis)


def sum_axis(value: Any, axis: int = -1) -> Any:
    if is_torch(value):
        return value.sum(dim=axis)
    return value.sum(axis=axis)


def where(condition: Any, true_value: Any, false_value: Any) -> Any:
    if is_torch(condition):
        import torch

        return torch.where(condition, true_value, false_value)
    import numpy as np

    return np.where(condition, true_value, false_value)


def zeros_like(value: Any) -> Any:
    if is_torch(value):
        return value.new_zeros(value.shape)
    import numpy as np

    return np.zeros_like(value)
