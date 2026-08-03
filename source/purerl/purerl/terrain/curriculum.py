"""Terrain-level curriculum state."""

from __future__ import annotations

import numpy as np


class TerrainCurriculum:
    def __init__(self, num_envs: int, num_levels: int, *, max_initial_level: int | None, seed: int):
        if num_envs <= 0 or num_levels <= 0:
            raise ValueError("Terrain curriculum dimensions must be positive")
        self.num_levels = num_levels
        self._rng = np.random.default_rng(seed)
        initial_max = num_levels - 1 if max_initial_level is None else min(max_initial_level, num_levels - 1)
        self.levels = self._rng.integers(0, initial_max + 1, size=num_envs, dtype=np.int64)

    def update(
        self,
        env_ids,
        *,
        distance: np.ndarray,
        commanded_speed: np.ndarray,
        episode_length_s: float | np.ndarray,
        terrain_length: float,
    ) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if distance.shape != commanded_speed.shape or distance.shape != env_ids.shape:
            raise ValueError("Curriculum inputs must have one value per completed environment")
        episode_length_s = np.asarray(episode_length_s)
        if episode_length_s.ndim > 0 and episode_length_s.shape != env_ids.shape:
            raise ValueError("episode_length_s must be scalar or contain one value per environment")
        move_up = distance > terrain_length / 2.0
        move_down = (distance < commanded_speed * episode_length_s * 0.5) & ~move_up
        self.levels[env_ids] += move_up.astype(np.int64) - move_down.astype(np.int64)
        overflow = self.levels[env_ids] >= self.num_levels
        if overflow.any():
            overflow_ids = env_ids[overflow]
            self.levels[overflow_ids] = self._rng.integers(0, self.num_levels, size=len(overflow_ids))
        self.levels[env_ids] = np.maximum(self.levels[env_ids], 0)
        return self.levels[env_ids].copy()
