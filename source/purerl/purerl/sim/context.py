"""Thin wrapper around Isaac Sim's direct simulation context."""

from __future__ import annotations

from typing import Any

from purerl.config.env import SimCfg


class IsaacSimContext:
    """Configure and step Isaac Sim without importing Isaac Lab."""

    def __init__(self, cfg: SimCfg):
        try:
            from isaacsim.core.api.simulation_context import SimulationContext
        except ImportError as exc:
            raise RuntimeError("Isaac Sim must be launched before creating the simulation context") from exc

        self.cfg = cfg
        self._context: Any = SimulationContext(
            physics_dt=cfg.dt,
            rendering_dt=cfg.dt * cfg.render_interval,
            backend="torch",
            device=cfg.device,
        )

    @property
    def raw(self) -> Any:
        return self._context

    def initialize(self) -> None:
        self._context.initialize_physics()
        self._context.play()

    def reset(self) -> None:
        self._context.reset()

    def step(self, *, render: bool = False) -> None:
        self._context.step(render=render)

    def close(self) -> None:
        self._context.stop()
        self._context.clear()
