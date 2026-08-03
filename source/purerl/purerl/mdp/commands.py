"""Batched velocity command sampling for locomotion tasks."""

from __future__ import annotations

from typing import Any

from purerl.config.env import CommandCfg


class VelocityCommandManager:
    """Sample and periodically replace planar velocity commands."""

    def __init__(
        self,
        cfg: CommandCfg,
        *,
        num_envs: int,
        device: str,
        step_dt: float,
        seed: int,
    ):
        import torch

        if num_envs <= 0 or step_dt <= 0:
            raise ValueError("Command manager dimensions and step_dt must be positive")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self._torch = torch
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self.command = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.time_left = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def update(self) -> None:
        self.time_left -= self.step_dt
        self.resample(self._torch.nonzero(self.time_left <= 0.0, as_tuple=False).flatten())

    def reset(self, env_ids: Any) -> None:
        self.resample(env_ids)

    def set_seed(self, seed: int) -> None:
        self._generator.manual_seed(seed)

    def resample(self, env_ids: Any) -> None:
        env_ids = self._normalize_ids(env_ids)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        ranges = self.cfg.ranges
        self.command[env_ids, 0] = self._uniform(ranges.lin_vel_x, count)
        self.command[env_ids, 1] = self._uniform(ranges.lin_vel_y, count)
        self.command[env_ids, 2] = self._uniform(ranges.ang_vel_z, count)
        standing = self._torch.rand(
            count, generator=self._generator, device=self.device
        ) < self.cfg.standing_env_ratio
        self.command[env_ids[standing]] = 0.0
        self.time_left[env_ids] = self._uniform(self.cfg.resampling_time_range, count)

    def _uniform(self, value_range: tuple[float, float], count: int) -> Any:
        low, high = value_range
        if low > high:
            raise ValueError(f"Invalid command range: {value_range}")
        values = self._torch.rand(count, generator=self._generator, device=self.device)
        return low + (high - low) * values

    def _normalize_ids(self, env_ids: Any) -> Any:
        if isinstance(env_ids, self._torch.Tensor):
            return env_ids.to(device=self.device, dtype=self._torch.long).flatten()
        return self._torch.as_tensor(env_ids, dtype=self._torch.long, device=self.device).flatten()

