"""Manager-based multi-terrain velocity task for TienKung2 Lite."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from purerl.assets import TIENKUNG_CFG
from purerl.assets.tienkung import TIENKUNG_JOINT_NAMES

from .terrain_cfg import TIENKUNG_ROUGH_TERRAINS_CFG


FEET_BODY_PATTERN = "ankle_roll_.*_link"
NON_FEET_BODY_PATTERN = "^(?!ankle_roll_[lr]_link$).+"
LEG_JOINT_PATTERN = ["hip_.*_joint", "knee_pitch_.*_joint", "ankle_.*_joint"]
ARM_JOINT_PATTERN = ["shoulder_.*_joint", "elbow_pitch_.*_joint"]


@configclass
class TienKungActionsCfg(ActionsCfg):
    """Twenty joint-position residuals around the WBC-SONIC stand pose."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=TIENKUNG_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class TienKungEventCfg(EventCfg):
    """Physics and actuator randomization for sim-to-real robustness."""

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


@configclass
class TienKungRewardsCfg(RewardsCfg):
    """Velocity tracking, gait quality, posture, and energy objectives."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TIENKUNG_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_FEET_BODY_PATTERN),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="ankle_.*_joint")},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_roll_.*", "hip_yaw_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_PATTERN)},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
        },
    )


@configclass
class TienKungTerminationsCfg(TerminationsCfg):
    """Terminate on pelvis contact or excessive tilt."""

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class TienKungRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Training configuration for mixed rough terrain."""

    actions: TienKungActionsCfg = TienKungActionsCfg()
    events: TienKungEventCfg = TienKungEventCfg()
    rewards: TienKungRewardsCfg = TienKungRewardsCfg()
    terminations: TienKungTerminationsCfg = TienKungTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Robot and sensors.
        self.scene.robot = TIENKUNG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.scene.terrain.terrain_generator = TIENKUNG_ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # 200 Hz physics and 50 Hz policy, matching WBC-SONIC.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.episode_length_s = 20.0

        # Domain randomization around the 27.8 kg pelvis assembly.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.02, 0.02),
        }
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Command curriculum starts conservatively but covers omnidirectional walking.
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)


@configclass
class TienKungRoughEnvCfg_PLAY(TienKungRoughEnvCfg):
    """Deterministic, lightweight rough-terrain evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None

