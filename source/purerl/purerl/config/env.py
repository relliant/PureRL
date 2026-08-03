"""Explicit flat and rough locomotion environment configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from purerl.contracts import ACTION_DIM, DECIMATION, OBSERVATION_DIM, PHYSICS_DT

from .base import ConfigMixin
from .robot import RobotCfg, make_tienkung_robot_cfg


@dataclass(frozen=True)
class SimCfg(ConfigMixin):
    dt: float = PHYSICS_DT
    decimation: int = DECIMATION
    device: str = "cuda:0"
    render_interval: int = DECIMATION

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
    scale: float = 0.5
    clip: float = 1.0
    use_default_offset: bool = True


@dataclass(frozen=True)
class ObservationCfg(ConfigMixin):
    dimension: int = OBSERVATION_DIM
    height_scan_points: int = 187
    enable_corruption: bool = True


@dataclass(frozen=True)
class SensorCfg(ConfigMixin):
    contact_update_period: float = PHYSICS_DT
    height_scan_update_period: float = PHYSICS_DT * DECIMATION
    contact_history_length: int = 3
    height_scan_size: tuple[float, float] = (1.6, 1.0)
    height_scan_resolution: float = 0.1
    height_scan_offset: float = 0.5


@dataclass(frozen=True)
class VelocityRangesCfg(ConfigMixin):
    lin_vel_x: tuple[float, float] = (-0.5, 1.2)
    lin_vel_y: tuple[float, float] = (-0.4, 0.4)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    heading: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class CommandCfg(ConfigMixin):
    resampling_time_range: tuple[float, float] = (4.0, 8.0)
    standing_env_ratio: float = 0.1
    ranges: VelocityRangesCfg = field(default_factory=VelocityRangesCfg)


@dataclass(frozen=True)
class RandomizationCfg(ConfigMixin):
    static_friction_range: tuple[float, float] = (0.6, 1.2)
    dynamic_friction_range: tuple[float, float] = (0.5, 1.0)
    pelvis_mass_delta: tuple[float, float] = (-3.0, 3.0)
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
    push_interval_s: tuple[float, float] = (10.0, 15.0)
    push_velocity_x: tuple[float, float] = (-0.5, 0.5)
    push_velocity_y: tuple[float, float] = (-0.5, 0.5)
    randomize_actuator_gains: bool = True
    apply_external_force: bool = True
    apply_periodic_push: bool = True


@dataclass(frozen=True)
class RewardTermCfg(ConfigMixin):
    name: str
    weight: float


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
    RewardTermCfg("feet_air_time", 0.5),
    RewardTermCfg("feet_slide", -0.25),
    RewardTermCfg("undesired_contacts", -1.0),
    RewardTermCfg("dof_pos_limits", -1.0),
    RewardTermCfg("joint_deviation_hip", -0.1),
    RewardTermCfg("joint_deviation_arms", -0.05),
    RewardTermCfg("stand_still", -0.2),
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
    max_initial_level: int | None = 2
    curriculum: bool = True
    patches: tuple[TerrainPatchCfg, ...] = ROUGH_TERRAIN_PATCHES


@dataclass(frozen=True)
class EnvCfg(ConfigMixin):
    task_kind: str
    play: bool
    seed: int = 42
    episode_length_s: float = 20.0
    sim: SimCfg = field(default_factory=SimCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    robot: RobotCfg = field(default_factory=make_tienkung_robot_cfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    rewards: tuple[RewardTermCfg, ...] = ROUGH_REWARD_TERMS

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_length_s / self.sim.step_dt)

    def reward_weights(self) -> dict[str, float]:
        return {term.name: term.weight for term in self.rewards}

    def validate(self, *, require_assets: bool = True) -> None:
        if self.task_kind not in {"flat", "rough"}:
            raise ValueError(f"Unsupported task kind: {self.task_kind}")
        if self.sim.dt <= 0 or self.sim.decimation <= 0:
            raise ValueError("Simulation dt and decimation must be positive")
        if self.scene.num_envs <= 0:
            raise ValueError("scene.num_envs must be positive")
        if self.terrain.num_rows <= 0 or self.terrain.num_cols <= 0:
            raise ValueError("Terrain rows and columns must be positive")
        if self.actions.dimension != len(self.robot.joint_names):
            raise ValueError("Action dimension must match the robot joint count")
        if self.observations.dimension != OBSERVATION_DIM:
            raise ValueError(f"Policy observation dimension must remain {OBSERVATION_DIM}")
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
        if len(self.reward_weights()) != len(self.rewards):
            raise ValueError("Reward term names must be unique")
        randomization_ranges = (
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
        self.robot.validate(require_assets=require_assets)


def _replace_reward_weights(
    terms: tuple[RewardTermCfg, ...], replacements: dict[str, float]
) -> tuple[RewardTermCfg, ...]:
    return tuple(replace(term, weight=replacements.get(term.name, term.weight)) for term in terms)


def make_rough_env_cfg() -> EnvCfg:
    cfg = EnvCfg(task_kind="rough", play=False)
    cfg.validate()
    return cfg


def make_rough_play_env_cfg() -> EnvCfg:
    train = make_rough_env_cfg()
    cfg = replace(
        train,
        play=True,
        scene=replace(train.scene, num_envs=16),
        observations=replace(train.observations, enable_corruption=False),
        terrain=replace(
            train.terrain,
            num_rows=5,
            num_cols=5,
            max_initial_level=None,
            curriculum=False,
        ),
        randomization=replace(
            train.randomization,
            randomize_actuator_gains=False,
            apply_external_force=False,
            apply_periodic_push=False,
        ),
    )
    cfg.validate()
    return cfg


def make_flat_env_cfg() -> EnvCfg:
    rough = make_rough_env_cfg()
    cfg = replace(
        rough,
        task_kind="flat",
        terrain=TerrainCfg(
            terrain_type="plane",
            max_initial_level=None,
            curriculum=False,
            patches=(),
        ),
        rewards=_replace_reward_weights(
            rough.rewards,
            {
                "lin_vel_z_l2": -0.5,
                "flat_orientation_l2": -1.5,
                "feet_air_time": 0.75,
                "action_rate_l2": -0.005,
            },
        ),
        randomization=replace(rough.randomization, apply_periodic_push=False),
    )
    cfg.validate()
    return cfg


def make_flat_play_env_cfg() -> EnvCfg:
    train = make_flat_env_cfg()
    cfg = replace(
        train,
        play=True,
        scene=replace(train.scene, num_envs=16),
        observations=replace(train.observations, enable_corruption=False),
        randomization=replace(
            train.randomization,
            randomize_actuator_gains=False,
            apply_external_force=False,
            apply_periodic_push=False,
        ),
    )
    cfg.validate()
    return cfg
