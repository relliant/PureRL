"""Task-specific terrain generation with no Isaac Lab dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from purerl.config.env import TerrainCfg, TerrainPatchCfg


@dataclass(frozen=True)
class TerrainTile:
    row: int
    column: int
    difficulty: float
    patch_name: str
    height_field: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    origin: np.ndarray


@dataclass(frozen=True)
class TerrainAssignment:
    levels: np.ndarray
    columns: np.ndarray
    tile_indices: np.ndarray
    origins: np.ndarray


def generate_terrain_tiles(cfg: TerrainCfg, *, seed: int) -> tuple[TerrainTile, ...]:
    """Generate every curriculum tile using deterministic child RNG streams."""

    if cfg.terrain_type != "generator":
        raise ValueError("Terrain tiles are only available for generated terrain")
    column_patches = _allocate_columns(cfg.num_cols, cfg.patches)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(cfg.num_rows * cfg.num_cols))
    tiles = []
    for row in range(cfg.num_rows):
        difficulty = row / max(cfg.num_rows - 1, 1)
        for column, patch in enumerate(column_patches):
            rng = np.random.default_rng(next(child_seeds))
            heights = generate_height_field(cfg, patch, difficulty=difficulty, rng=rng)
            vertices, faces = height_field_to_mesh(heights, cfg.horizontal_scale)
            local_origin_z = float(heights[heights.shape[0] // 2, heights.shape[1] // 2])
            origin = np.asarray(
                [
                    (row + 0.5) * cfg.size[0],
                    (column + 0.5) * cfg.size[1],
                    local_origin_z,
                ],
                dtype=np.float32,
            )
            vertices = vertices + np.asarray([origin[0] - cfg.size[0] / 2, origin[1] - cfg.size[1] / 2, 0])
            tiles.append(
                TerrainTile(
                    row=row,
                    column=column,
                    difficulty=difficulty,
                    patch_name=patch.name,
                    height_field=heights,
                    vertices=vertices,
                    faces=faces,
                    origin=origin,
                )
            )
    return tuple(tiles)


def assign_terrain_tiles(
    tiles: tuple[TerrainTile, ...],
    cfg: TerrainCfg,
    *,
    num_envs: int,
    seed: int,
) -> TerrainAssignment:
    """Assign environments to deterministic terrain levels and type columns."""

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if len(tiles) != cfg.num_rows * cfg.num_cols:
        raise ValueError("Terrain tile count does not match the configured grid")

    rng = np.random.default_rng(seed)
    maximum_level = cfg.num_rows - 1
    if cfg.max_initial_level is not None:
        maximum_level = min(maximum_level, cfg.max_initial_level)
    levels = rng.integers(0, maximum_level + 1, size=num_envs, dtype=np.int64)
    columns = np.arange(num_envs, dtype=np.int64) % cfg.num_cols
    tile_indices = levels * cfg.num_cols + columns
    origins = np.stack([tiles[index].origin for index in tile_indices]).astype(np.float32)
    return TerrainAssignment(levels, columns, tile_indices, origins)


def combine_terrain_meshes(
    tiles: tuple[TerrainTile, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine generated tiles into one static triangle mesh."""

    if not tiles:
        raise ValueError("At least one terrain tile is required")
    vertices = np.concatenate([tile.vertices for tile in tiles], axis=0).astype(np.float32)
    faces = []
    vertex_offset = 0
    for tile in tiles:
        faces.append(tile.faces + vertex_offset)
        vertex_offset += len(tile.vertices)
    return vertices, np.concatenate(faces, axis=0).astype(np.int32)


def generate_height_field(
    cfg: TerrainCfg,
    patch: TerrainPatchCfg,
    *,
    difficulty: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one terrain tile height field in meters."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    rows = round(cfg.size[0] / cfg.horizontal_scale) + 1
    columns = round(cfg.size[1] / cfg.horizontal_scale) + 1
    x = np.linspace(-cfg.size[0] / 2.0, cfg.size[0] / 2.0, rows)
    y = np.linspace(-cfg.size[1] / 2.0, cfg.size[1] / 2.0, columns)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    edge_distance = np.minimum(cfg.size[0] / 2.0 - np.abs(grid_x), cfg.size[1] / 2.0 - np.abs(grid_y))
    platform_radius = patch.platform_width / 2.0
    climb_distance = np.maximum(edge_distance - patch.patch_border_width, 0.0)

    if patch.name == "flat":
        heights = np.zeros((rows, columns), dtype=np.float32)
    elif patch.name == "random_rough":
        amplitude = _interpolate_required(patch.noise_range, difficulty, "noise_range")
        step = _required(patch.noise_step, "noise_step")
        levels = max(round(amplitude / step), 1)
        heights = rng.integers(-levels, levels + 1, size=(rows, columns)).astype(np.float32) * step
    elif patch.name in {"slope_up", "slope_down"}:
        slope = _interpolate_required(patch.slope_range, difficulty, "slope_range")
        heights = np.minimum(climb_distance, max(edge_distance.max() - platform_radius, 0.0)) * slope
        if patch.name == "slope_down":
            heights *= -1.0
    elif patch.name in {"stairs_up", "stairs_down"}:
        step_height = _interpolate_required(patch.step_height_range, difficulty, "step_height_range")
        step_width = _required(patch.step_width, "step_width")
        heights = np.floor(climb_distance / step_width) * step_height
        maximum_step = max((min(cfg.size) / 2.0 - platform_radius) // step_width, 0.0)
        heights = np.minimum(heights, maximum_step * step_height)
        if patch.name == "stairs_down":
            heights *= -1.0
    elif patch.name == "random_blocks":
        grid_width = _required(patch.grid_width, "grid_width")
        low, high = _required(patch.grid_height_range, "grid_height_range")
        maximum_height = low + difficulty * (high - low)
        block_rows = max(round(cfg.size[0] / grid_width), 1)
        block_columns = max(round(cfg.size[1] / grid_width), 1)
        block_heights = rng.uniform(-maximum_height, maximum_height, (block_rows, block_columns))
        x_index = np.minimum((grid_x + cfg.size[0] / 2.0) / grid_width, block_rows - 1).astype(int)
        y_index = np.minimum((grid_y + cfg.size[1] / 2.0) / grid_width, block_columns - 1).astype(int)
        heights = block_heights[x_index, y_index]
    else:
        raise ValueError(f"Unsupported terrain patch: {patch.name}")

    platform = (np.abs(grid_x) <= platform_radius) & (np.abs(grid_y) <= platform_radius)
    platform_height = heights[rows // 2, columns // 2]
    heights[platform] = platform_height
    return np.asarray(heights, dtype=np.float32)


def height_field_to_mesh(height_field: np.ndarray, horizontal_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a regular height field for an Isaac Sim USD mesh."""

    if height_field.ndim != 2 or min(height_field.shape) < 2:
        raise ValueError("Height field must be a two-dimensional grid of at least 2x2")
    if horizontal_scale <= 0:
        raise ValueError("horizontal_scale must be positive")

    rows, columns = height_field.shape
    x, y = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    vertices = np.stack(
        (x.reshape(-1) * horizontal_scale, y.reshape(-1) * horizontal_scale, height_field.reshape(-1)),
        axis=-1,
    ).astype(np.float32)
    cell_x, cell_y = np.meshgrid(np.arange(rows - 1), np.arange(columns - 1), indexing="ij")
    lower_left = (cell_x * columns + cell_y).reshape(-1)
    lower_right = lower_left + columns
    upper_left = lower_left + 1
    upper_right = lower_right + 1
    faces = np.concatenate(
        (
            np.stack((lower_left, lower_right, upper_right), axis=-1),
            np.stack((lower_left, upper_right, upper_left), axis=-1),
        ),
        axis=0,
    ).astype(np.int32)
    return vertices, faces


def _allocate_columns(num_columns: int, patches: tuple[TerrainPatchCfg, ...]) -> tuple[TerrainPatchCfg, ...]:
    if not patches:
        raise ValueError("At least one terrain patch is required")
    expected = np.asarray([patch.proportion * num_columns for patch in patches])
    counts = np.floor(expected).astype(int)
    remainder = num_columns - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    allocation = []
    for patch, count in zip(patches, counts):
        allocation.extend([patch] * int(count))
    return tuple(allocation)


def _interpolate_required(value, difficulty: float, name: str) -> float:
    low, high = _required(value, name)
    return float(low + difficulty * (high - low))


def _required(value, name: str):
    if value is None:
        raise ValueError(f"Terrain patch is missing {name}")
    return value
