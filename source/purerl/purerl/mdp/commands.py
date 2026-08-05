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
        self.heading_target = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.is_heading_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, current_heading: Any) -> None:
        self.time_left -= self.step_dt
        self.resample(self._torch.nonzero(self.time_left <= 0.0, as_tuple=False).flatten())
        self._update_command(current_heading)

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
        self.time_left[env_ids] = self._uniform(self.cfg.resampling_time_range, count)
        self.command[env_ids, 0] = self._uniform(ranges.lin_vel_x, count)
        self.command[env_ids, 1] = self._uniform(ranges.lin_vel_y, count)
        self.command[env_ids, 2] = self._uniform(ranges.ang_vel_z, count)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = self._uniform(ranges.heading, count)
            self.is_heading_env[env_ids] = self._torch.rand(
                count, generator=self._generator, device=self.device
            ) <= self.cfg.heading_env_ratio
        else:
            self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = self._torch.rand(
            count, generator=self._generator, device=self.device
        ) <= self.cfg.standing_env_ratio

    def _update_command(self, current_heading: Any) -> None:
        current_heading = self._torch.as_tensor(
            current_heading, dtype=self.command.dtype, device=self.device
        ).flatten()
        if current_heading.shape != (self.num_envs,):
            raise ValueError(
                f"Current heading must have shape ({self.num_envs},), "
                f"got {tuple(current_heading.shape)}"
            )
        if self.cfg.heading_command:
            heading_ids = self._torch.nonzero(
                self.is_heading_env, as_tuple=False
            ).flatten()
            heading_error = _wrap_to_pi(
                self.heading_target[heading_ids] - current_heading[heading_ids]
            )
            low, high = self.cfg.ranges.ang_vel_z
            self.command[heading_ids, 2] = self._torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=low,
                max=high,
            )
        self.command[self.is_standing_env] = 0.0

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


def _wrap_to_pi(angle: Any) -> Any:
    import math

    return (angle + math.pi) % (2.0 * math.pi) - math.pi
