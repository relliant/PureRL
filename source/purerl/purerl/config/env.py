"""Explicit flat and rough locomotion environment configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from purerl.contracts import ACTION_DIM, DECIMATION, OBSERVATION_DIM, PHYSICS_DT

from .base import ConfigMixin
from .robot import RobotCfg, make_tienkung_robot_cfg

PRESET_DIR = Path(__file__).resolve().parent / "presets"
FLAT_ENV_PRESET = PRESET_DIR / "flat_env.yaml"
FLAT_PLAY_ENV_PRESET = PRESET_DIR / "flat_play_env.yaml"
ROUGH_ENV_PRESET = PRESET_DIR / "rough_env.yaml"
ROUGH_PLAY_ENV_PRESET = PRESET_DIR / "rough_play_env.yaml"


@dataclass(frozen=True)
class SimCfg(ConfigMixin):
    dt: float = PHYSICS_DT
    decimation: int = DECIMATION
    device: str = "cuda:0"
    render_interval: int = DECIMATION
    gpu_max_rigid_contact_count: int = 2**23
    gpu_max_rigid_patch_count: int = 2**22
    gpu_found_lost_pairs_capacity: int = 2**21
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    gpu_total_aggregate_pairs_capacity: int = 2**21
    gpu_heap_capacity: int = 2**26
    gpu_temp_buffer_capacity: int = 2**24
    gpu_max_num_partitions: int = 8
    gpu_collision_stack_size: int = 2**26

    @property
    def step_dt(self) -> float:
        return self.dt * self.decimation


@dataclass(frozen=True)
class SceneCfg(ConfigMixin):
    num_envs: int = 4096
    env_spacing: float = 2.5


@dataclass(frozen=True)
class ActionCfg(ConfigMixin):
    dimension: int = ACTION_DIM
    scale: float = 0.25
    clip: float | None = None
    use_default_offset: bool = True


@dataclass(frozen=True)
class ObservationNoiseCfg(ConfigMixin):
    base_linear_velocity: tuple[float, float] = (-0.03, 0.03)
    base_angular_velocity: tuple[float, float] = (-0.06, 0.06)
    projected_gravity: tuple[float, float] = (-0.018, 0.018)
    relative_joint_positions: tuple[float, float] = (-0.03, 0.03)
    joint_velocities: tuple[float, float] = (-0.3, 0.3)
    terrain_height_scan: tuple[float, float] = (-0.06, 0.06)


@dataclass(frozen=True)
class ObservationCfg(ConfigMixin):
    dimension: int = OBSERVATION_DIM
    height_scan_points: int = 187
    enable_corruption: bool = True
    noise: ObservationNoiseCfg = field(default_factory=ObservationNoiseCfg)
    height_scan_clip: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class LidarCfg(ConfigMixin):
    enabled: bool = False
    env_index: int = 0
    mount_body: str = "head"
    fallback_body: str = "pelvis"
    mount_translation: tuple[float, float, float] = (0.08, 0.0, 0.03)
    fallback_translation: tuple[float, float, float] = (0.08, 0.0, 0.63)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    config_file_name: str = "OS1"
    variant: str = "OS1_REV6_32ch10hz512res"
    prim_name: str = "HeadLidar"
    collect_point_cloud: bool = True
    visualize: bool = False


@dataclass(frozen=True)
class SensorCfg(ConfigMixin):
    contact_update_period: float = PHYSICS_DT
    height_scan_update_period: float = PHYSICS_DT * DECIMATION
    contact_history_length: int = 3
    height_scan_size: tuple[float, float] = (1.6, 1.0)
    height_scan_resolution: float = 0.1
    height_scan_offset: float = 0.5
    lidar: LidarCfg = field(default_factory=LidarCfg)


@dataclass(frozen=True)
class ViewerCfg(ConfigMixin):
    eye: tuple[float, float, float] = (3.0, 3.0, 2.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.8)
    camera_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)


@dataclass(frozen=True)
class EnvironmentVisualCfg(ConfigMixin):
    sky_color: tuple[float, float, float] = (0.53, 0.69, 0.90)
    sky_intensity: float = 850.0
    ground_color: tuple[float, float, float] = (0.18, 0.28, 0.16)
    terrain_color: tuple[float, float, float] = (0.32, 0.30, 0.23)


@dataclass(frozen=True)
class VelocityRangesCfg(ConfigMixin):
    lin_vel_x: tuple[float, float] = (-0.3, 0.6)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-0.3, 0.3)
    heading: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class CommandCfg(ConfigMixin):
    resampling_time_range: tuple[float, float] = (8.0, 8.0)
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    heading_env_ratio: float = 1.0
    standing_env_ratio: float = 0.1
    lin_vel_tracking_std: float = 0.35
    ranges: VelocityRangesCfg = field(default_factory=VelocityRangesCfg)


@dataclass(frozen=True)
class RandomizationCfg(ConfigMixin):
    static_friction_range: tuple[float, float] = (0.1, 2.0)
    dynamic_friction_range: tuple[float, float] = (0.1, 2.0)
    friction_buckets: int = 64
    pelvis_mass_delta: tuple[float, float] = (-5.0, 5.0)
    pelvis_com_x: tuple[float, float] = (-0.03, 0.03)
    pelvis_com_y: tuple[float, float] = (-0.03, 0.03)
    pelvis_com_z: tuple[float, float] = (-0.02, 0.02)
    actuator_gain_scale: tuple[float, float] = (0.9, 1.1)
    joint_position_scale: tuple[float, float] = (0.9, 1.1)
    root_x: tuple[float, float] = (-0.5, 0.5)
    root_y: tuple[float, float] = (-0.5, 0.5)
    root_yaw: tuple[float, float] = (-math.pi, math.pi)
    external_force: tuple[float, float] = (-5.0, 5.0)
    external_torque: tuple[float, float] = (-5.0, 5.0)
    push_interval_s: tuple[float, float] = (4.0, 4.0)
    push_velocity_x: tuple[float, float] = (-0.2, 0.2)
    push_velocity_y: tuple[float, float] = (-0.2, 0.2)
    randomize_actuator_gains: bool = True
    apply_external_force: bool = True
    apply_periodic_push: bool = True


@dataclass(frozen=True)
class RewardTermCfg(ConfigMixin):
    name: str
    weight: float
    is_event: bool = False


@dataclass(frozen=True)
class GaitCfg(ConfigMixin):
    cycle_time: float = 0.5           # 步态周期 [s]（天工腿长 0.8m，步频 ~2Hz）
    contact_threshold: float = 1.0    # 脚接触力判定阈值 [N]
    command_threshold: float = 0.1    # 低于该速度时不强制交替步态 [m/s]
    air_time_threshold: float = 0.12  # 最小摆动时间 [s]
    foot_min_dist: float = 0.20       # 步宽下限 [m]
    foot_max_dist: float = 0.50       # 步宽上限 [m]
    target_feet_height: float = 0.06  # 摆动脚目标离地高度 [m]
    foot_height_offset: float = 0.0569  # 脚 body(ankle_roll) 原点到脚底距离 [m]
    clearance_sigma: float = 0.025  # 摆脚高度奖励的平滑尺度 [m]
    base_height_sigma: float = 0.05  # 躯干高度奖励的平滑尺度 [m]
    double_support_fraction: float = 0.2  # 双支撑占完整周期的比例
    contact_release_time: float = 0.01  # 接触力短暂丢失的容忍时间 [s]
    min_phase_time: float = 0.04  # 有效支撑/摆动的最小持续时间 [s]
    flight_grace_time: float = 0.01  # 无支撑惩罚的容忍时间 [s]
    min_clearance: float = 0.02  # 有效抬脚的高度尺度 [m]


ROUGH_REWARD_TERMS = (
    RewardTermCfg("termination_penalty", -200.0),
    RewardTermCfg("track_lin_vel_xy_exp", 1.5),
    RewardTermCfg("track_ang_vel_z_exp", 0.75),
    RewardTermCfg("lin_vel_z_l2", -1.5),
    RewardTermCfg("ang_vel_xy_l2", -0.1),
    RewardTermCfg("flat_orientation_l2", -1.0),
    RewardTermCfg("dof_torques_l2", -1.0e-6),
    RewardTermCfg("dof_acc_l2", -2.5e-7),
    RewardTermCfg("action_rate_l2", -0.01),
    RewardTermCfg("feet_air_time", 0.5, is_event=True),
    RewardTermCfg("feet_slide", -0.25),
    RewardTermCfg("undesired_contacts", -1.0),
    RewardTermCfg("dof_pos_limits", -1.0),
    RewardTermCfg("joint_deviation_hip", -0.1),
    RewardTermCfg("joint_deviation_arms", -0.05),
    RewardTermCfg("stand_still", -0.2),
    RewardTermCfg("feet_contact_number", 1.0),
    RewardTermCfg("feet_flight", -1.0),
    RewardTermCfg("feet_distance", 0.1),
    RewardTermCfg("base_height", 0.5),
    RewardTermCfg("feet_clearance", 0.5),
)


@dataclass(frozen=True)
class TerrainPatchCfg(ConfigMixin):
    name: str
    proportion: float
    noise_range: tuple[float, float] | None = None
    noise_step: float | None = None
    slope_range: tuple[float, float] | None = None
    step_height_range: tuple[float, float] | None = None
    step_width: float | None = None
    grid_width: float | None = None
    grid_height_range: tuple[float, float] | None = None
    platform_width: float = 2.5
    patch_border_width: float = 0.25


ROUGH_TERRAIN_PATCHES = (
    TerrainPatchCfg("flat", 0.10),
    TerrainPatchCfg("random_rough", 0.20, noise_range=(0.01, 0.08), noise_step=0.01),
    TerrainPatchCfg("slope_up", 0.10, slope_range=(0.0, 0.30)),
    TerrainPatchCfg("slope_down", 0.10, slope_range=(0.0, 0.30)),
    TerrainPatchCfg(
        "stairs_up",
        0.15,
        step_height_range=(0.02, 0.16),
        step_width=0.35,
        patch_border_width=1.0,
    ),
    TerrainPatchCfg(
        "stairs_down",
        0.15,
        step_height_range=(0.02, 0.16),
        step_width=0.35,
        patch_border_width=1.0,
    ),
    TerrainPatchCfg(
        "random_blocks",
        0.20,
        grid_width=0.45,
        grid_height_range=(0.02, 0.12),
    ),
)


@dataclass(frozen=True)
class TerrainCfg(ConfigMixin):
    terrain_type: str = "generator"
    size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    num_rows: int = 10
    num_cols: int = 20
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    max_initial_level: int | None = 5
    curriculum: bool = True
    selected_patch: str | None = None
    selected_level: int | None = None
    patches: tuple[TerrainPatchCfg, ...] = ROUGH_TERRAIN_PATCHES


@dataclass(frozen=True)
class EnvCfg(ConfigMixin):
    task_kind: str
    play: bool
    seed: int = 5
    episode_length_s: float = 24.0
    sim: SimCfg = field(default_factory=SimCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    robot: RobotCfg = field(default_factory=make_tienkung_robot_cfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)
    viewer: ViewerCfg = field(default_factory=ViewerCfg)
    visuals: EnvironmentVisualCfg = field(default_factory=EnvironmentVisualCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    rewards: tuple[RewardTermCfg, ...] = ROUGH_REWARD_TERMS
    gait: GaitCfg = field(default_factory=GaitCfg)

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_length_s / self.sim.step_dt)

    def reward_weights(self) -> dict[str, float]:
        return {term.name: term.weight for term in self.rewards}

    def validate(self, *, require_assets: bool = True) -> None:
        if self.task_kind not in {"flat", "rough"}:
            raise ValueError(f"Unsupported task kind: {self.task_kind}")
        if self.sim.dt <= 0 or self.sim.decimation <= 0 or self.sim.render_interval <= 0:
            raise ValueError("Simulation dt, decimation, and render_interval must be positive")
        gpu_capacities = (
            self.sim.gpu_max_rigid_contact_count,
            self.sim.gpu_max_rigid_patch_count,
            self.sim.gpu_found_lost_pairs_capacity,
            self.sim.gpu_found_lost_aggregate_pairs_capacity,
            self.sim.gpu_total_aggregate_pairs_capacity,
            self.sim.gpu_heap_capacity,
            self.sim.gpu_temp_buffer_capacity,
            self.sim.gpu_max_num_partitions,
            self.sim.gpu_collision_stack_size,
        )
        if any(value <= 0 for value in gpu_capacities):
            raise ValueError("GPU PhysX capacities must be positive")
        if self.scene.num_envs <= 0:
            raise ValueError("scene.num_envs must be positive")
        if self.terrain.num_rows <= 0 or self.terrain.num_cols <= 0:
            raise ValueError("Terrain rows and columns must be positive")
        if self.actions.dimension != len(self.robot.joint_names):
            raise ValueError("Action dimension must match the robot joint count")
        if self.actions.clip is not None and self.actions.clip <= 0.0:
            raise ValueError("Action clip must be positive when enabled")
        if self.sensors.contact_history_length <= 0:
            raise ValueError("sensors.contact_history_length must be positive")
        lidar = self.sensors.lidar
        if lidar.env_index < 0 or (lidar.enabled and lidar.env_index >= self.scene.num_envs):
            raise ValueError("sensors.lidar.env_index must select an existing environment")
        if not lidar.mount_body or not lidar.fallback_body:
            raise ValueError("LiDAR mount and fallback body names must be non-empty")
        if not lidar.config_file_name:
            raise ValueError("LiDAR config_file_name must be non-empty")
        if not lidar.variant:
            raise ValueError("LiDAR variant must be non-empty")
        if not lidar.prim_name or "/" in lidar.prim_name:
            raise ValueError("LiDAR prim_name must be a single non-empty USD path component")
        orientation_norm = math.sqrt(sum(value * value for value in lidar.orientation))
        if not math.isclose(orientation_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("LiDAR orientation must be a normalized scalar-first quaternion")
        if self.observations.dimension != OBSERVATION_DIM:
            raise ValueError(f"Policy observation dimension must remain {OBSERVATION_DIM}")
        if len(self.viewer.resolution) != 2 or any(value <= 0 for value in self.viewer.resolution):
            raise ValueError("Viewer resolution must contain two positive values")
        if not self.viewer.camera_prim_path.startswith("/"):
            raise ValueError("Viewer camera_prim_path must be an absolute USD path")
        colors = (
            self.visuals.sky_color,
            self.visuals.ground_color,
            self.visuals.terrain_color,
        )
        if any(not 0.0 <= channel <= 1.0 for color in colors for channel in color):
            raise ValueError("Environment visual colors must use values between zero and one")
        if self.visuals.sky_intensity <= 0.0:
            raise ValueError("Environment sky intensity must be positive")
        command_ranges = (
            self.commands.resampling_time_range,
            self.commands.ranges.lin_vel_x,
            self.commands.ranges.lin_vel_y,
            self.commands.ranges.ang_vel_z,
            self.commands.ranges.heading,
        )
        if any(low > high for low, high in command_ranges):
            raise ValueError("Command ranges must be ordered from low to high")
        if self.commands.resampling_time_range[0] <= 0.0:
            raise ValueError("Command resampling times must be positive")
        if self.commands.heading_control_stiffness < 0.0:
            raise ValueError("Command heading_control_stiffness must be non-negative")
        if not 0.0 <= self.commands.heading_env_ratio <= 1.0:
            raise ValueError("Command heading_env_ratio must be between zero and one")
        if not 0.0 <= self.commands.standing_env_ratio <= 1.0:
            raise ValueError("Command standing_env_ratio must be between zero and one")
        if not math.isfinite(self.commands.lin_vel_tracking_std) or self.commands.lin_vel_tracking_std <= 0.0:
            raise ValueError("Command lin_vel_tracking_std must be finite and positive")
        gait = self.gait
        if any(not math.isfinite(value) for value in gait.to_dict().values()):
            raise ValueError("Gait parameters must be finite")
        if gait.cycle_time <= 0.0 or gait.contact_threshold < 0.0:
            raise ValueError("Gait cycle time and contact threshold must be non-negative/positive")
        if gait.command_threshold < 0.0 or gait.air_time_threshold < 0.0:
            raise ValueError("Gait command and air-time thresholds must be non-negative")
        if gait.foot_min_dist < 0.0 or gait.foot_max_dist < gait.foot_min_dist:
            raise ValueError("Gait foot distance bounds are invalid")
        if gait.target_feet_height < 0.0 or gait.foot_height_offset < 0.0:
            raise ValueError("Gait foot heights must be non-negative")
        if gait.clearance_sigma <= 0.0 or gait.base_height_sigma <= 0.0:
            raise ValueError("Gait reward smoothing scales must be positive")
        if not 0.0 <= gait.double_support_fraction < 1.0:
            raise ValueError("Gait double_support_fraction must be in [0, 1)")
        swing_time = 0.5 * gait.cycle_time * (1.0 - gait.double_support_fraction)
        if not 0.0 <= gait.contact_release_time < gait.min_phase_time <= gait.air_time_threshold < swing_time:
            raise ValueError("Gait times must satisfy release < min_phase <= air_time_threshold < swing duration")
        if gait.flight_grace_time < 0.0 or not 0.0 < gait.min_clearance <= gait.target_feet_height:
            raise ValueError("Gait flight grace/clearance bounds are invalid")
        scan_x, scan_y = self.sensors.height_scan_size
        scan_points = (round(scan_x / self.sensors.height_scan_resolution) + 1) * (
            round(scan_y / self.sensors.height_scan_resolution) + 1
        )
        if scan_points != self.observations.height_scan_points:
            raise ValueError(
                f"Height scanner creates {scan_points} points, expected {self.observations.height_scan_points}"
            )
        if self.terrain.terrain_type == "generator":
            total_proportion = sum(patch.proportion for patch in self.terrain.patches)
            if abs(total_proportion - 1.0) > 1.0e-6:
                raise ValueError("Terrain patch proportions must sum to one")
            patch_names = {patch.name for patch in self.terrain.patches}
            if self.terrain.selected_patch is not None and self.terrain.selected_patch not in patch_names:
                raise ValueError(f"Selected terrain patch does not exist: {self.terrain.selected_patch}")
            if self.terrain.selected_level is not None and not (
                0 <= self.terrain.selected_level < self.terrain.num_rows
            ):
                raise ValueError("Selected terrain level is outside the configured rows")
        elif self.terrain.selected_patch is not None or self.terrain.selected_level is not None:
            raise ValueError("Selected terrain patch/level requires generated terrain")
        if len(self.reward_weights()) != len(self.rewards):
            raise ValueError("Reward term names must be unique")
        randomization_ranges = (
            self.observations.noise.base_linear_velocity,
            self.observations.noise.base_angular_velocity,
            self.observations.noise.projected_gravity,
            self.observations.noise.relative_joint_positions,
            self.observations.noise.joint_velocities,
            self.observations.noise.terrain_height_scan,
            self.observations.height_scan_clip,
            self.randomization.static_friction_range,
            self.randomization.dynamic_friction_range,
            self.randomization.pelvis_mass_delta,
            self.randomization.pelvis_com_x,
            self.randomization.pelvis_com_y,
            self.randomization.pelvis_com_z,
            self.randomization.actuator_gain_scale,
            self.randomization.joint_position_scale,
            self.randomization.root_x,
            self.randomization.root_y,
            self.randomization.root_yaw,
            self.randomization.external_force,
            self.randomization.external_torque,
            self.randomization.push_interval_s,
            self.randomization.push_velocity_x,
            self.randomization.push_velocity_y,
        )
        if any(low > high for low, high in randomization_ranges):
            raise ValueError("Randomization ranges must be ordered from low to high")
        if self.randomization.friction_buckets <= 0:
            raise ValueError("randomization.friction_buckets must be positive")
        self.robot.validate(require_assets=require_assets)


def load_env_cfg(path: str | Path) -> EnvCfg:
    cfg = EnvCfg.from_yaml(path)
    cfg.validate()
    return cfg


def make_rough_env_cfg() -> EnvCfg:
    return load_env_cfg(ROUGH_ENV_PRESET)


def make_flat_env_cfg() -> EnvCfg:
    return load_env_cfg(FLAT_ENV_PRESET)


def make_flat_play_env_cfg() -> EnvCfg:
    return load_env_cfg(FLAT_PLAY_ENV_PRESET)


def make_rough_play_env_cfg() -> EnvCfg:
    return load_env_cfg(ROUGH_PLAY_ENV_PRESET)
