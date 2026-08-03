"""Simulator-independent scene layout helpers."""

from __future__ import annotations

import math


def make_grid_origins(num_envs: int, spacing: float):
    """Create centered XY origins for flat cloned environments."""

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    import numpy as np

    columns = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / columns)
    indices = np.arange(num_envs)
    x = (indices // columns - (rows - 1) / 2.0) * spacing
    y = (indices % columns - (columns - 1) / 2.0) * spacing
    return np.stack((x, y, np.zeros(num_envs)), axis=-1)
