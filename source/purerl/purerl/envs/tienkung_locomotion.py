"""Standalone TienKung velocity locomotion environment."""

from __future__ import annotations

from typing import Any

import torch

from purerl.config.env import EnvCfg
from purerl.contracts import OBSERVATION_DIM
from purerl.mdp import events as event_terms
from purerl.mdp import rewards as reward_terms
from purerl.mdp import terminations as termination_terms
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.commands import VelocityCommandManager
from purerl.mdp.managers import (
    EventManager,
    EventTermSpec,
    ObservationManager,
    ObservationTermSpec,
    RewardManager,
    RewardTermSpec,
    TerminationManager,
    TerminationTermSpec,
)
from purerl.mdp.observations import relative_joint_positions
from purerl.registry import get_task_spec
from purerl.sensors import BipedContactHistory, height_observation, make_grid_pattern
from purerl.sim import IsaacSimBackend, SimulationBackend
from purerl.terrain import TerrainCurriculum

from .base import BaseVecEnv


class TienKungLocomotionEnv(BaseVecEnv):
    """Gymnasium-compatible vector environment backed directly by Isaac Sim."""

    def __init__(
        self,
        task_id: str = "PureRL-Velocity-Flat-TienKung-v0",
        *,
        cfg: EnvCfg | None = None,
        backend: SimulationBackend | None = None,
        render_mode: str | None = None,
    ):
        cfg = cfg or get_task_spec(task_id).make_env_cfg()
        backend = backend or IsaacSimBackend()
        defaults = torch.as_tensor(
            cfg.robot.default_joint_positions,
            dtype=torch.float32,
            device=cfg.sim.device,
        ).repeat(cfg.scene.num_envs, 1)
        self.default_joint_positions = defaults
        self.command_manager = VelocityCommandManager(
            cfg.commands,
            num_envs=cfg.scene.num_envs,
            device=cfg.sim.device,
            step_dt=cfg.sim.step_dt,
            seed=cfg.seed,
        )
        foot_template = torch.zeros(
            (cfg.scene.num_envs, len(cfg.robot.foot_body_names)),
            dtype=torch.float32,
            device=cfg.sim.device,
        )
        self.contact_history = BipedContactHistory(
            foot_template,
            force_threshold=cfg.gait.contact_threshold,
            release_time=cfg.gait.contact_release_time,
            support_time=cfg.gait.min_phase_time,
            min_air_time=cfg.gait.air_time_threshold,
            flight_grace_time=cfg.gait.flight_grace_time,
        )
        self.foot_heights = foot_template.clone()
        self.ground_height = foot_template[:, 0].clone()
        self._gait_metric_sums = {
            name: foot_template[:, 0].clone()
            for name in (
                "moving_time",
                "single_support_time",
                "flight_time",
                "phase_match_time",
                "swing_height_sum",
                "swing_time",
                "valid_landings",
                "landing_events",
                "velocity_error_sq_time",
            )
        }
        self.contact_force_history: torch.Tensor | None = None
        self._contact_force_history_index = 0
        self.height_scan = torch.zeros(
            (cfg.scene.num_envs, cfg.observations.height_scan_points),
            dtype=torch.float32,
            device=cfg.sim.device,
        )
        self._height_pattern = torch.as_tensor(
            make_grid_pattern(
                size=cfg.sensors.height_scan_size,
                resolution=cfg.sensors.height_scan_resolution,
            ),
            dtype=torch.float32,
            device=cfg.sim.device,
        )
        self._foot_body_indices: torch.Tensor | None = None
        self._non_foot_body_indices: torch.Tensor | None = None
        self._root_body_index: int | None = None
        self._joint_limits: torch.Tensor | None = None
        self._event_generator = torch.Generator(device=cfg.sim.device)
        self._event_generator.manual_seed(cfg.seed + 1)
        self._push_time_left = torch.zeros(
            cfg.scene.num_envs, dtype=torch.float32, device=cfg.sim.device
        )
        self.terrain_curriculum = (
            TerrainCurriculum(
                cfg.scene.num_envs,
                cfg.terrain.num_rows,
                max_initial_level=cfg.terrain.max_initial_level,
                seed=cfg.seed,
            )
            if cfg.task_kind == "rough" and cfg.terrain.curriculum
            else None
        )

        action_manager = JointPositionActionManager(
            defaults,
            scale=cfg.actions.scale,
            action_clip=cfg.actions.clip,
        )
        observation_manager = self._make_observation_manager(cfg)
        termination_manager = self._make_termination_manager()
        reward_manager = self._make_reward_manager(cfg)
        event_manager = self._make_event_manager(cfg)
        super().__init__(
            cfg,
            backend,
            action_manager=action_manager,
            observation_manager=observation_manager,
            reward_manager=reward_manager,
            termination_manager=termination_manager,
            event_manager=event_manager,
            render_mode=render_mode,
        )

        index_map = backend.index_map
        self._foot_body_indices = torch.as_tensor(
            index_map.foot_body_indices, dtype=torch.long, device=self.device
        )
        self._root_body_index = index_map.root_body_index
        self._non_foot_body_indices = torch.as_tensor(
            [index for index in range(len(backend.body_names)) if index not in index_map.foot_body_indices],
            dtype=torch.long,
            device=self.device,
        )
        self._joint_limits = backend.joint_limits
        self.contact_force_history = torch.zeros(
            (
                self.num_envs,
                cfg.sensors.contact_history_length,
                len(backend.body_names),
                3,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if self.terrain_curriculum is not None:
            self.terrain_curriculum.levels = backend.terrain_levels.detach().cpu().numpy().copy()
        self._reset_sensors(backend.all_env_ids)
        self._configure_spaces()

    @property
    def commands(self) -> torch.Tensor:
        return self.command_manager.command

    @property
    def projected_gravity(self) -> torch.Tensor:
        gravity = self.state.root_position.new_zeros((self.num_envs, 3))
        gravity[:, 2] = -1.0
        return _quat_rotate_inverse(self.state.root_quaternion, gravity)

    @property
    def base_linear_velocity_body(self) -> torch.Tensor:
        return _quat_rotate_inverse(
            self.state.root_quaternion, self.state.root_linear_velocity
        )

    @property
    def base_angular_velocity_body(self) -> torch.Tensor:
        return _quat_rotate_inverse(
            self.state.root_quaternion, self.state.root_angular_velocity
        )

    @property
    def base_linear_velocity_yaw(self) -> torch.Tensor:
        yaw = _yaw_from_quaternion(self.state.root_quaternion)
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        velocity = self.state.root_linear_velocity
        return torch.stack(
            (
                cosine * velocity[:, 0] + sine * velocity[:, 1],
                -sine * velocity[:, 0] + cosine * velocity[:, 1],
                velocity[:, 2],
            ),
            dim=-1,
        )

    def _make_observation_manager(self, cfg: EnvCfg) -> ObservationManager:
        noise = cfg.observations.noise
        return ObservationManager(
            (
                ObservationTermSpec(
                    "base_linear_velocity",
                    lambda env: env.base_linear_velocity_body,
                    noise=noise.base_linear_velocity,
                ),
                ObservationTermSpec(
                    "base_angular_velocity",
                    lambda env: env.base_angular_velocity_body,
                    noise=noise.base_angular_velocity,
                ),
                ObservationTermSpec(
                    "projected_gravity",
                    lambda env: env.projected_gravity,
                    noise=noise.projected_gravity,
                ),
                ObservationTermSpec("velocity_command", lambda env: env.commands),
                ObservationTermSpec(
                    "relative_joint_positions",
                    lambda env: relative_joint_positions(
                        env.state.joint_positions, env.default_joint_positions
                    ),
                    noise=noise.relative_joint_positions,
                ),
                ObservationTermSpec(
                    "joint_velocities",
                    lambda env: env.state.joint_velocities,
                    noise=noise.joint_velocities,
                ),
                ObservationTermSpec("previous_action", lambda env: env.action_manager.action),
                ObservationTermSpec(
                    "terrain_height_scan",
                    lambda env: env.height_scan,
                    noise=noise.terrain_height_scan,
                    clip=cfg.observations.height_scan_clip,
                ),
                ObservationTermSpec("gait_phase", lambda env: env.gait_phase),
            ),
            expected_dimension=OBSERVATION_DIM,
            enable_corruption=cfg.observations.enable_corruption,
            seed=cfg.seed + 2,
        )

    def _make_termination_manager(self) -> TerminationManager:
        return TerminationManager(
            (
                TerminationTermSpec(
                    "time_out",
                    lambda env: termination_terms.time_out(
                        env.episode_length_buf, env.max_episode_length
                    ),
                    time_out=True,
                ),
                TerminationTermSpec(
                    "base_contact",
                    lambda env: termination_terms.illegal_contact(
                        env.contact_force_history[:, :, [env._root_body_index]], threshold=1.0
                    ),
                ),
                TerminationTermSpec(
                    "bad_orientation",
                    lambda env: termination_terms.bad_orientation(
                        env.projected_gravity, limit_angle=0.8
                    ),
                ),
            )
        )

    @property
    def gait_phase(self) -> torch.Tensor:
        """Observable sin/cos clock shared by the actor, critic and rewards."""
        phase = self.episode_length_buf * self.step_dt / self.cfg.gait.cycle_time
        angle = 2 * torch.pi * phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _get_gait_phase(self) -> Any:
        return self._gait_targets()[0]

    def _gait_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Alternating swing arcs with an explicit double-support transition."""
        phase = self.episode_length_buf * self.step_dt / self.cfg.gait.cycle_time
        local_phase = torch.stack(((phase + 0.5) % 1.0, phase % 1.0), dim=-1)
        margin = self.cfg.gait.double_support_fraction / 4.0
        swing_progress = (local_phase - margin) / (0.5 - 2.0 * margin)
        swing = (swing_progress > 0.0) & (swing_progress < 1.0)
        target = self.cfg.gait.target_feet_height * torch.sin(torch.pi * swing_progress.clamp(0.0, 1.0))
        return ~swing, target.clamp_min(0.0) * swing

    def _foot_contact_mask(self) -> Any:
        return self.contact_history.in_contact

    def _make_reward_manager(self, cfg: EnvCfg) -> RewardManager:
        leg_joint_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("hip_", "knee_", "ankle_"))
        )
        ankle_joint_indices = tuple(
            index for index, name in enumerate(cfg.robot.joint_names) if name.startswith("ankle_")
        )
        hip_deviation_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("hip_roll_", "hip_yaw_"))
        )
        arm_joint_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("shoulder_", "elbow_"))
        )
        functions = {
            "termination_penalty": lambda env: reward_terms.termination_penalty(env._terminated()),
            "track_lin_vel_xy_exp": lambda env: reward_terms.track_lin_vel_xy_exp(
                env.base_linear_velocity_yaw, env.commands, std=env.cfg.commands.lin_vel_tracking_std
            ),
            "track_ang_vel_z_exp": lambda env: reward_terms.track_ang_vel_z_exp(
                env.state.root_angular_velocity, env.commands, std=0.5
            ),
            "lin_vel_z_l2": lambda env: reward_terms.lin_vel_z_l2(
                env.base_linear_velocity_body
            ),
            "ang_vel_xy_l2": lambda env: reward_terms.ang_vel_xy_l2(
                env.base_angular_velocity_body
            ),
            "flat_orientation_l2": lambda env: reward_terms.flat_orientation_l2(
                env.projected_gravity
            ),
            "dof_torques_l2": lambda env: reward_terms.joint_torques_l2(
                env.state.joint_torques[:, leg_joint_indices]
            ),
            "dof_acc_l2": lambda env: reward_terms.joint_acc_l2(
                env.state.joint_accelerations
            ),
            "action_rate_l2": lambda env: reward_terms.action_rate_l2(
                env.action_manager.action, env.action_manager.previous_action
            ),
            "feet_air_time": lambda env: reward_terms.feet_air_time_on_contact(
                env.contact_history.valid_landing_air_time,
                env.contact_history.contact_events,
                env.commands,
                supported_landing=env.contact_history.valid_landing_events,
                threshold=env.cfg.gait.air_time_threshold,
                command_threshold=env.cfg.gait.command_threshold,
            ),
            "feet_slide": lambda env: reward_terms.feet_slide(
                env.state.body_linear_velocities[:, env._foot_body_indices],
                env.contact_force_history[:, :, env._foot_body_indices],
            ),
            "undesired_contacts": lambda env: reward_terms.undesired_contacts(
                env.contact_force_history[:, :, env._non_foot_body_indices], threshold=1.0
            ),
            "dof_pos_limits": lambda env: reward_terms.joint_pos_limits(
                env.state.joint_positions[:, ankle_joint_indices],
                env._joint_limits[:, ankle_joint_indices],
            ),
            "joint_deviation_hip": lambda env: reward_terms.joint_deviation_l1(
                env.state.joint_positions[:, hip_deviation_indices],
                env.default_joint_positions[:, hip_deviation_indices],
            ),
            "joint_deviation_arms": lambda env: reward_terms.joint_deviation_l1(
                env.state.joint_positions[:, arm_joint_indices],
                env.default_joint_positions[:, arm_joint_indices],
            ),
            "stand_still": lambda env: reward_terms.stand_still_joint_deviation_l1(
                env.state.joint_positions[:, leg_joint_indices],
                env.default_joint_positions[:, leg_joint_indices],
                env.commands,
            ),
            "feet_contact_number": lambda env: reward_terms.feet_contact_number(
                env._foot_contact_mask(),
                env._get_gait_phase(),
                env.commands,
                current_air_time=env.contact_history.current_air_time,
                current_contact_time=env.contact_history.current_contact_time,
                min_phase_time=env.cfg.gait.min_phase_time,
                command_threshold=env.cfg.gait.command_threshold,
            ),
            "feet_flight": lambda env: env.contact_history.step_unsupported_time / env.step_dt,
            "feet_distance": lambda env: reward_terms.feet_distance(
                env.state.body_positions[:, env._foot_body_indices],
                min_dist=env.cfg.gait.foot_min_dist,
                max_dist=env.cfg.gait.foot_max_dist,
            ),
            "base_height": lambda env: reward_terms.base_height(
                env.state.root_position[:, 2],
                env.ground_height,
                target=env.cfg.robot.default_root_height,
                sigma=env.cfg.gait.base_height_sigma,
            ),
            "feet_clearance": lambda env: reward_terms.feet_clearance(
                env.foot_heights,
                env._gait_targets()[1],
                env.commands,
                contact=env._foot_contact_mask(),
                current_contact_time=env.contact_history.current_contact_time,
                support_time=env.cfg.gait.min_phase_time,
                min_clearance=env.cfg.gait.min_clearance,
                sigma=env.cfg.gait.clearance_sigma,
                command_threshold=env.cfg.gait.command_threshold,
            ),
        }
        unknown = set(cfg.reward_weights()) - set(functions)
        if unknown:
            raise ValueError("Unsupported reward terms: " + ", ".join(sorted(unknown)))
        specs = tuple(
            RewardTermSpec(term.name, functions[term.name], term.weight, is_event=term.is_event)
            for term in cfg.rewards
        )
        return RewardManager(specs, dt=cfg.sim.step_dt)

    def _make_event_manager(self, cfg: EnvCfg) -> EventManager:
        terms = [
            EventTermSpec(
                "startup_domain_randomization",
                event_terms.startup_domain_randomization,
                "startup",
            ),
            EventTermSpec(
                "reset_state_randomization",
                event_terms.reset_state_randomization,
                "reset",
            ),
        ]
        if cfg.randomization.apply_periodic_push:
            terms.append(
                EventTermSpec(
                    "periodic_velocity_push",
                    event_terms.periodic_velocity_push,
                    "interval",
                )
            )
        return EventManager(tuple(terms))

    def _terminated(self) -> torch.Tensor:
        values = self.termination_manager.last_values
        return values["base_contact"] | values["bad_orientation"]

    def _update_commands(self) -> None:
        self.command_manager.update(_yaw_from_quaternion(self.state.root_quaternion))

    def _reset_commands(self, env_ids: Any) -> None:
        self.command_manager.reset(env_ids)

    def randomize_startup_domain(self) -> None:
        cfg = self.cfg.randomization
        env_ids = self.backend.all_env_ids
        mass_delta = self._uniform_random(cfg.pelvis_mass_delta, (self.num_envs,))
        com_offset = torch.stack(
            (
                self._uniform_random(cfg.pelvis_com_x, (self.num_envs,)),
                self._uniform_random(cfg.pelvis_com_y, (self.num_envs,)),
                self._uniform_random(cfg.pelvis_com_z, (self.num_envs,)),
            ),
            dim=-1,
        )
        self.backend.randomize_body_properties(env_ids, mass_delta, com_offset)
        if cfg.randomize_actuator_gains:
            shape = (self.num_envs, self.num_actions)
            self.backend.set_actuator_gains(
                env_ids,
                self._uniform_random(cfg.actuator_gain_scale, shape),
                self._uniform_random(cfg.actuator_gain_scale, shape),
            )
        static_buckets = self._uniform_random(
            cfg.static_friction_range, (cfg.friction_buckets,)
        )
        dynamic_buckets = self._uniform_random(
            cfg.dynamic_friction_range, (cfg.friction_buckets,)
        )
        dynamic_buckets = torch.minimum(dynamic_buckets, static_buckets)
        bucket_ids = torch.randint(
            cfg.friction_buckets,
            (self.num_envs,),
            generator=self._event_generator,
            device=self.device,
        )
        self.backend.set_contact_material_friction(
            env_ids,
            static_buckets[bucket_ids],
            dynamic_buckets[bucket_ids],
        )

    def randomize_reset_state(self, env_ids: Any) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        cfg = self.cfg.randomization
        count = env_ids.numel()
        root_xy = torch.stack(
            (
                self._uniform_random(cfg.root_x, (count,)),
                self._uniform_random(cfg.root_y, (count,)),
            ),
            dim=-1,
        )
        root_yaw = self._uniform_random(cfg.root_yaw, (count,))
        joint_scale = self._uniform_random(cfg.joint_position_scale, (count, self.num_actions))
        self.backend.randomize_reset_state(env_ids, root_xy, root_yaw, joint_scale)
        self._push_time_left[env_ids] = self._uniform_random(cfg.push_interval_s, (count,))
        if cfg.apply_external_force:
            self.backend.apply_root_wrench(
                env_ids,
                self._uniform_random(cfg.external_force, (count, 3)),
                self._uniform_random(cfg.external_torque, (count, 3)),
            )

    def apply_periodic_pushes(self) -> None:
        self._push_time_left -= self.step_dt
        push_ids = torch.nonzero(self._push_time_left <= 0.0, as_tuple=False).flatten()
        if push_ids.numel() == 0:
            return
        cfg = self.cfg.randomization
        count = push_ids.numel()
        velocity = torch.stack(
            (
                self._uniform_random(cfg.push_velocity_x, (count,)),
                self._uniform_random(cfg.push_velocity_y, (count,)),
            ),
            dim=-1,
        )
        self.backend.add_root_velocity(push_ids, velocity)
        self._push_time_left[push_ids] = self._uniform_random(cfg.push_interval_s, (count,))

    def _apply_curriculum(self, env_ids: Any) -> None:
        if self.terrain_curriculum is None or env_ids.numel() == 0:
            return
        distance = torch.linalg.vector_norm(
            self.state.root_position[env_ids, :2] - self.backend.env_origins[env_ids, :2],
            dim=-1,
        )
        commanded_speed = torch.linalg.vector_norm(self.commands[env_ids, :2], dim=-1)
        episode_length_s = self.episode_length_buf[env_ids] * self.step_dt
        cpu_env_ids = env_ids.detach().cpu().numpy()
        levels = self.terrain_curriculum.update(
            cpu_env_ids,
            distance=distance.detach().cpu().numpy(),
            commanded_speed=commanded_speed.detach().cpu().numpy(),
            episode_length_s=episode_length_s.detach().cpu().numpy(),
            terrain_length=self.cfg.terrain.size[0],
        )
        self.backend.set_terrain_levels(env_ids, levels)

    def _update_sensors(self) -> None:
        if self._foot_body_indices is None:
            return
        terrain_heights = self.backend.sample_terrain_heights(self._height_scan_world_points())
        self.height_scan.copy_(
            height_observation(
                self.state.root_position[:, 2],
                terrain_heights,
                offset=self.cfg.sensors.height_scan_offset,
            )
        )
        self._update_terrain_relative_heights()

    def _update_terrain_relative_heights(self, env_ids: Any = None) -> None:
        feet = self.state.body_positions[:, self._foot_body_indices]
        points = torch.cat((self.state.root_position[:, None, :2], feet[..., :2]), dim=1)
        terrain = self.backend.sample_terrain_heights(points)
        ids = slice(None) if env_ids is None else env_ids
        self.ground_height[ids] = terrain[ids, 0]
        self.foot_heights[ids] = feet[ids, :, 2] - terrain[ids, 1:] - self.cfg.gait.foot_height_offset

    def _update_physics_step_sensors(self) -> None:
        if self._foot_body_indices is None:
            return
        contact_forces = self.backend.get_net_contact_forces()
        if self.contact_force_history is not None:
            self.contact_force_history[:, self._contact_force_history_index].copy_(contact_forces)
            self._contact_force_history_index = (
                self._contact_force_history_index + 1
            ) % self.contact_force_history.shape[1]
        self.contact_history.update(
            contact_forces[:, self._foot_body_indices], self.cfg.sim.dt
        )

    def _reset_sensors(self, env_ids: Any) -> None:
        self.contact_history.reset(env_ids)
        for values in self._gait_metric_sums.values():
            values[env_ids] = 0.0
        if self.contact_force_history is not None:
            self.contact_force_history[env_ids] = 0.0
        self.height_scan[env_ids] = 0.0
        if self._foot_body_indices is not None:
            terrain_heights = self.backend.sample_terrain_heights(
                self._height_scan_world_points()
            )[env_ids]
            self.height_scan[env_ids] = height_observation(
                self.state.root_position[env_ids, 2],
                terrain_heights,
                offset=self.cfg.sensors.height_scan_offset,
            )
            self._update_terrain_relative_heights(env_ids)

    def _clear_step_events(self) -> None:
        moving = torch.linalg.vector_norm(self.commands[:, :2], dim=-1) > self.cfg.gait.command_threshold
        moving_dt = moving * self.step_dt
        stance = self._get_gait_phase()
        contact = self._foot_contact_mask()
        history = self.contact_history
        mode_time = torch.where(stance, history.current_contact_time, history.current_air_time)
        matched = (contact == stance).all(dim=-1) & (mode_time.amin(dim=-1) + 1e-7 >= self.cfg.gait.min_phase_time)
        swing = ~stance & moving[:, None]
        valid_landings = (history.valid_landing_events & history.contact_events).sum(dim=-1)
        valid_landings *= history.contact_events.sum(dim=-1) == 1
        values = {
            "moving_time": moving_dt,
            "single_support_time": history.step_single_support_time * moving,
            "flight_time": history.step_flight_time * moving,
            "phase_match_time": matched * moving_dt,
            "swing_height_sum": (self.foot_heights.clamp_min(0.0) * swing).sum(dim=-1) * self.step_dt,
            "swing_time": swing.sum(dim=-1) * self.step_dt,
            "valid_landings": valid_landings * moving,
            "landing_events": history.contact_events.sum(dim=-1) * moving,
            "velocity_error_sq_time": (
                (self.base_linear_velocity_yaw[:, :2] - self.commands[:, :2]).square().sum(dim=-1) * moving_dt
            ),
        }
        for key, value in values.items():
            self._gait_metric_sums[key] += value
        self.contact_history.clear_events()

    def _collect_episode_metrics(self, env_ids: Any) -> dict[str, Any]:
        values = {name: sums[env_ids] for name, sums in self._gait_metric_sums.items()}
        duration = values["moving_time"].clamp_min(1e-6)
        return {
            "Gait/single_support_fraction": values["single_support_time"] / duration,
            "Gait/flight_fraction": values["flight_time"] / duration,
            "Gait/phase_match_fraction": values["phase_match_time"] / duration,
            "Gait/swing_clearance_m": values["swing_height_sum"] / values["swing_time"].clamp_min(1e-6),
            "Gait/valid_landing_rate_hz": values["valid_landings"] / duration,
            "Gait/valid_landing_fraction": values["valid_landings"] / values["landing_events"].clamp_min(1.0),
            "Gait/xy_velocity_rmse": (values["velocity_error_sq_time"] / duration).sqrt(),
        }

    def _height_scan_world_points(self) -> torch.Tensor:
        yaw = _yaw_from_quaternion(self.state.root_quaternion)
        cosine = torch.cos(yaw)[:, None]
        sine = torch.sin(yaw)[:, None]
        local_x = self._height_pattern[None, :, 0]
        local_y = self._height_pattern[None, :, 1]
        world_x = self.state.root_position[:, None, 0] + cosine * local_x - sine * local_y
        world_y = self.state.root_position[:, None, 1] + sine * local_x + cosine * local_y
        return torch.stack((world_x, world_y), dim=-1)

    def _set_seed(self, seed: int) -> None:
        super()._set_seed(seed)
        self.command_manager.set_seed(seed)
        self._event_generator.manual_seed(seed + 1)
        self.observation_manager.set_seed(seed + 2)

    def _uniform_random(self, value_range: tuple[float, float], shape: tuple[int, ...]):
        low, high = value_range
        values = torch.rand(shape, generator=self._event_generator, device=self.device)
        return low + (high - low) * values

    def _configure_spaces(self) -> None:
        try:
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Gymnasium is required to construct task spaces") from exc
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(self.num_actions,), dtype=float)
        self.single_observation_space = spaces.Dict(
            {"policy": spaces.Box(-float("inf"), float("inf"), shape=(OBSERVATION_DIM,), dtype=float)}
        )
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    axis = conjugate[..., 1:]
    twice_cross = 2.0 * torch.cross(axis, vector, dim=-1)
    return vector + conjugate[..., :1] * twice_cross + torch.cross(axis, twice_cross, dim=-1)


def _yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
