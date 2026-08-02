"""Flat-ground variants used to bootstrap TienKung locomotion training."""

from isaaclab.utils import configclass

from .rough_env_cfg import TienKungRoughEnvCfg


@configclass
class TienKungFlatEnvCfg(TienKungRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # Preserve the height scan so flat and rough checkpoints have the same
        # observation shape and can be used for staged training.

        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.action_rate_l2.weight = -0.005
        self.events.push_robot = None


@configclass
class TienKungFlatEnvCfg_PLAY(TienKungFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.actuator_gains = None
