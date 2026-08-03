"""Height-scanner pattern and observation conversion."""

from __future__ import annotations

from typing import Any

from purerl.mdp._array import clip


class HeightFieldSampler:
    """Sample batched generated terrain using the mesh's triangle split."""

    def __init__(
        self,
        height_fields: Any,
        tile_origins: Any,
        env_tile_indices: Any,
        *,
        size: tuple[float, float],
        horizontal_scale: float,
        device: str,
    ):
        import torch

        heights = torch.as_tensor(height_fields, dtype=torch.float32, device=device)
        origins = torch.as_tensor(tile_origins, dtype=torch.float32, device=device)
        indices = torch.as_tensor(env_tile_indices, dtype=torch.long, device=device)
        if heights.ndim != 3 or min(heights.shape[1:]) < 2:
            raise ValueError("height_fields must have shape (tiles, rows, columns)")
        if origins.shape != (heights.shape[0], 3):
            raise ValueError("tile_origins must have shape (tiles, 3)")
        if horizontal_scale <= 0:
            raise ValueError("horizontal_scale must be positive")
        if indices.numel() == 0 or indices.min() < 0 or indices.max() >= heights.shape[0]:
            raise ValueError("env_tile_indices contains an invalid tile index")

        self._torch = torch
        self._height_fields = heights
        self._flat_heights = heights.reshape(-1)
        self._tile_origins = origins
        self.env_tile_indices = indices
        self.size = size
        self.horizontal_scale = horizontal_scale

    def sample(self, world_points_xy: Any) -> Any:
        """Return terrain surface Z for points shaped ``(num_envs, points, 2)``."""

        if world_points_xy.ndim != 3 or world_points_xy.shape[-1] != 2:
            raise ValueError("world_points_xy must have shape (num_envs, points, 2)")
        if world_points_xy.shape[0] != self.env_tile_indices.shape[0]:
            raise ValueError("world_points_xy must contain every configured environment")

        rows, columns = self._height_fields.shape[1:]
        env_tiles = self.env_tile_indices[:, None]
        tile_centers = self._tile_origins[self.env_tile_indices, :2]
        lower_corner = tile_centers - world_points_xy.new_tensor(self.size) * 0.5
        local = (world_points_xy - lower_corner[:, None, :]) / self.horizontal_scale
        local_x = local[..., 0].clamp(0.0, rows - 1 - 1.0e-5)
        local_y = local[..., 1].clamp(0.0, columns - 1 - 1.0e-5)
        index_x = self._torch.floor(local_x).to(dtype=self._torch.long)
        index_y = self._torch.floor(local_y).to(dtype=self._torch.long)
        fraction_x = local_x - index_x
        fraction_y = local_y - index_y

        field_offset = env_tiles * (rows * columns)

        def gather(offset_x: int, offset_y: int) -> Any:
            flat_indices = field_offset + (index_x + offset_x) * columns + index_y + offset_y
            return self._flat_heights[flat_indices]

        lower_left = gather(0, 0)
        lower_right = gather(1, 0)
        upper_left = gather(0, 1)
        upper_right = gather(1, 1)
        lower_triangle = (
            (1.0 - fraction_x) * lower_left
            + (fraction_x - fraction_y) * lower_right
            + fraction_y * upper_right
        )
        upper_triangle = (
            (1.0 - fraction_y) * lower_left
            + fraction_x * upper_right
            + (fraction_y - fraction_x) * upper_left
        )
        return self._torch.where(fraction_y <= fraction_x, lower_triangle, upper_triangle)

    def update_env_tiles(self, env_ids: Any, tile_indices: Any) -> None:
        env_ids = self._torch.as_tensor(
            env_ids, dtype=self._torch.long, device=self.env_tile_indices.device
        ).flatten()
        tile_indices = self._torch.as_tensor(
            tile_indices, dtype=self._torch.long, device=self.env_tile_indices.device
        ).flatten()
        if env_ids.shape != tile_indices.shape:
            raise ValueError("env_ids and tile_indices must have the same shape")
        if tile_indices.numel() and (
            tile_indices.min() < 0 or tile_indices.max() >= self._height_fields.shape[0]
        ):
            raise ValueError("tile_indices contains an invalid tile index")
        self.env_tile_indices[env_ids] = tile_indices


def make_grid_pattern(size: tuple[float, float] = (1.6, 1.0), resolution: float = 0.1):
    """Return the pelvis-frame ray starts used by the 187-point policy scan."""

    if resolution <= 0 or size[0] < 0 or size[1] < 0:
        raise ValueError("Height scan size must be non-negative and resolution must be positive")

    import numpy as np

    count_x = round(size[0] / resolution) + 1
    count_y = round(size[1] / resolution) + 1
    x = np.linspace(-size[0] / 2.0, size[0] / 2.0, count_x)
    y = np.linspace(-size[1] / 2.0, size[1] / 2.0, count_y)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(count_x * count_y)), axis=-1)


def height_observation(
    root_height: Any,
    ray_hit_height: Any,
    *,
    offset: float = 0.5,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> Any:
    """Convert world ray hits to the relative height convention used by the policy."""

    if ray_hit_height.shape[-1] != 187:
        raise ValueError(f"Expected 187 ray hits, got {ray_hit_height.shape[-1]}")
    relative = root_height[..., None] - offset - ray_hit_height
    return clip(relative, minimum, maximum)
