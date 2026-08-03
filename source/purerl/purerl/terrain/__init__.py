"""Deterministic terrain generation and curriculum."""

from .curriculum import TerrainCurriculum
from .generator import (
    TerrainAssignment,
    TerrainTile,
    assign_terrain_tiles,
    combine_terrain_meshes,
    generate_height_field,
    generate_terrain_tiles,
    height_field_to_mesh,
)

__all__ = [
    "TerrainCurriculum",
    "TerrainAssignment",
    "TerrainTile",
    "assign_terrain_tiles",
    "combine_terrain_meshes",
    "generate_height_field",
    "generate_terrain_tiles",
    "height_field_to_mesh",
]
