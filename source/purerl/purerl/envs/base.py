"""Simulator-independent vectorized environment lifecycle."""

from __future__ import annotations

from typing import Any

from purerl.config.env import EnvCfg
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.managers import EventManager, ObservationManager, RewardManager, TerminationManager
from purerl.sim.backend import SimulationBackend


class BaseVecEnv:
    """Coordinate batched simulation and MDP managers in a fixed order."""

    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(
        self,
        cfg: EnvCfg,
        backend: SimulationBackend,
        *,
        action_manager: JointPositionActionManager,
        observation_manager: ObservationManager,
        reward_manager: RewardManager,
        termination_manager: TerminationManager,
        event_manager: EventManager | None = None,
        render_mode: str | None = None,
    ):
        cfg.validate()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.cfg = cfg
        self.backend = backend
        self.action_manager = action_manager
        self.observation_manager = observation_manager
        self.reward_manager = reward_manager
        self.termination_manager = termination_manager
        self.event_manager = event_manager or EventManager(())
        self.render_mode = render_mode
        self.metadata = {**type(self).metadata, "render_fps": round(1.0 / cfg.sim.step_dt)}
        self.num_envs = cfg.scene.num_envs
        self.num_actions = cfg.actions.dimension
        self.max_episode_length = cfg.max_episode_steps
        self.device = cfg.sim.device
        self.step_dt = cfg.sim.step_dt
        self._sim_step_counter = 0
        self.closed = False

        backend.initialize(cfg)
        if backend.num_envs != self.num_envs:
            raise ValueError(
                f"Backend initialized {backend.num_envs} environments, expected {self.num_envs}"
            )
        if render_mode is not None:
            backend.configure_viewer()
        self.episode_length_buf = backend.zeros((self.num_envs,), dtype="long")
        self.event_manager.run("startup", self)
        self._reset_idx(backend.all_env_ids)

    @property
    def state(self) -> Any:
        return self.backend.state

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._set_seed(seed)
        self._reset_idx(self.backend.all_env_ids)
        return self.get_observations(), {}

    def step(self, action: Any):
        if self.closed:
            raise RuntimeError("Cannot step a closed environment")

        targets = self.action_manager.process(action)
        for _ in range(self.cfg.sim.decimation):
            self._sim_step_counter += 1
            self.backend.set_joint_position_targets(targets)
            self.backend.simulate(
                render=(
                    (self.render_mode is not None or self.cfg.sensors.lidar.enabled)
                    and self._sim_step_counter % self.cfg.sim.render_interval == 0
                )
            )
            self._update_physics_step_sensors()

        self.backend.refresh()
        self._update_sensors()
        self.episode_length_buf += 1
        terminated, truncated = self.termination_manager.compute(self)
        reward = self.reward_manager.compute(self)
        self._clear_step_events()

        done = terminated | truncated
        done_env_ids = self.backend.nonzero(done)
        terminal_observation = self.observation_manager.compute(self)[done_env_ids]
        episode = self.reward_manager.reset(done_env_ids)
        episode.update(self._collect_episode_metrics(done_env_ids))
        for name, values in self.termination_manager.last_values.items():
            episode[f"Termination/{name}"] = values[done_env_ids] * 1.0
        self._apply_curriculum(done_env_ids)
        self._reset_idx(done_env_ids, reset_rewards=False)
        self._update_commands()
        self.event_manager.run("interval", self)

        info = {
            "time_outs": truncated,
            "terminal_env_ids": self.backend.clone(done_env_ids),
            "terminal_observation": self.backend.clone(terminal_observation),
            "episode": episode,
        }
        return self.get_observations(), reward, terminated, truncated, info

    def get_observations(self) -> dict[str, Any]:
        return {"policy": self.observation_manager.compute(self)}

    def render(self) -> Any | None:
        if self.render_mode != "rgb_array":
            return None
        return self.backend.render_rgb()

    def get_lidar_point_cloud(self) -> Any:
        """Return the latest head LiDAR point cloud without changing policy observations."""

        return self.backend.get_lidar_point_cloud()

    def close(self) -> None:
        if not self.closed:
            self.backend.close()
            self.closed = True

    def _reset_idx(self, env_ids: Any, *, reset_rewards: bool = True) -> None:
        if reset_rewards:
            self.reward_manager.reset(env_ids)
        self.backend.reset(env_ids)
        self.action_manager.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._reset_commands(env_ids)
        self.event_manager.run("reset", self, env_ids=env_ids)
        self.backend.refresh()
        self._reset_sensors(env_ids)

    def _set_seed(self, seed: int) -> None:
        self.cfg = self.cfg.replace(seed=seed)

    def _update_commands(self) -> None:
        """Update task commands. Concrete environments override this hook."""

    def _reset_commands(self, env_ids: Any) -> None:
        """Reset task commands. Concrete environments override this hook."""

    def _apply_curriculum(self, env_ids: Any) -> None:
        """Update curriculum state before resetting completed environments."""

    def _update_sensors(self) -> None:
        """Refresh derived sensor history after simulator state is current."""

    def _update_physics_step_sensors(self) -> None:
        """Refresh sensors whose update period is the physics time step."""

    def _reset_sensors(self, env_ids: Any) -> None:
        """Reset derived sensor history for selected environments."""

    def _clear_step_events(self) -> None:
        """Clear latched sensor events after reward terms consume them."""

    def _collect_episode_metrics(self, env_ids: Any) -> dict[str, Any]:
        """Return task metrics before completed episodes are reset."""
        return {}
