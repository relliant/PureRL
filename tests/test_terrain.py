import numpy as np
from purerl.config.env import make_rough_env_cfg
from purerl.terrain import (
    TerrainCurriculum,
    assign_terrain_tiles,
    combine_terrain_meshes,
    generate_height_field,
    generate_terrain_tiles,
    height_field_to_mesh,
)


def test_terrain_grid_is_deterministic_and_allocates_all_patch_types():
    cfg = make_rough_env_cfg().terrain
    first = generate_terrain_tiles(cfg, seed=42)
    second = generate_terrain_tiles(cfg, seed=42)

    assert len(first) == cfg.num_rows * cfg.num_cols
    assert {tile.patch_name for tile in first} == {patch.name for patch in cfg.patches}
    assert all(np.array_equal(left.height_field, right.height_field) for left, right in zip(first, second))
    assert all(np.array_equal(left.origin, right.origin) for left, right in zip(first, second))


def test_all_patch_generators_preserve_a_center_platform():
    cfg = make_rough_env_cfg().terrain
    for patch in cfg.patches:
        heights = generate_height_field(cfg, patch, difficulty=1.0, rng=np.random.default_rng(7))
        center = heights.shape[0] // 2, heights.shape[1] // 2
        radius = round((patch.platform_width / 2.0) / cfg.horizontal_scale)
        platform = heights[
            center[0] - radius : center[0] + radius + 1,
            center[1] - radius : center[1] + radius + 1,
        ]
        assert np.allclose(platform, platform[0, 0]), patch.name


def test_height_field_mesh_has_expected_topology():
    height_field = np.zeros((3, 4), dtype=np.float32)
    vertices, faces = height_field_to_mesh(height_field, 0.1)
    assert vertices.shape == (12, 3)
    assert faces.shape == (12, 3)
    assert faces.min() == 0
    assert faces.max() == 11


def test_terrain_assignment_is_seeded_balanced_and_respects_initial_level():
    cfg = make_rough_env_cfg().terrain
    tiles = generate_terrain_tiles(cfg, seed=42)
    first = assign_terrain_tiles(tiles, cfg, num_envs=40, seed=7)
    second = assign_terrain_tiles(tiles, cfg, num_envs=40, seed=7)

    assert np.array_equal(first.levels, second.levels)
    assert first.levels.max() <= cfg.max_initial_level
    assert np.bincount(first.columns, minlength=cfg.num_cols).tolist() == [2] * cfg.num_cols
    assert np.allclose(first.origins, np.stack([tiles[index].origin for index in first.tile_indices]))


def test_combined_terrain_mesh_offsets_tile_face_indices():
    cfg = make_rough_env_cfg().terrain
    tiles = generate_terrain_tiles(cfg, seed=42)[:2]
    vertices, faces = combine_terrain_meshes(tiles)

    first_vertex_count = len(tiles[0].vertices)
    first_face_count = len(tiles[0].faces)
    assert len(vertices) == sum(len(tile.vertices) for tile in tiles)
    assert faces[first_face_count:].min() >= first_vertex_count


def test_curriculum_moves_levels_and_wraps_overflow():
    curriculum = TerrainCurriculum(3, 4, max_initial_level=0, seed=1)
    levels = curriculum.update(
        np.asarray([0, 1, 2]),
        distance=np.asarray([5.0, 0.1, 5.0]),
        commanded_speed=np.asarray([1.0, 1.0, 1.0]),
        episode_length_s=10.0,
        terrain_length=8.0,
    )
    assert levels.tolist() == [1, 0, 1]

    curriculum.levels[0] = 3
    wrapped = curriculum.update(
        np.asarray([0]),
        distance=np.asarray([5.0]),
        commanded_speed=np.asarray([1.0]),
        episode_length_s=10.0,
        terrain_length=8.0,
    )
    assert 0 <= wrapped[0] < 4


def test_curriculum_accepts_per_environment_episode_lengths():
    curriculum = TerrainCurriculum(2, 3, max_initial_level=0, seed=1)
    curriculum.levels[:] = 1
    levels = curriculum.update(
        np.asarray([0, 1]),
        distance=np.asarray([0.1, 0.1]),
        commanded_speed=np.asarray([1.0, 1.0]),
        episode_length_s=np.asarray([0.1, 10.0]),
        terrain_length=8.0,
    )
    assert levels.tolist() == [1, 0]
