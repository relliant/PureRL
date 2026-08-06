from types import SimpleNamespace

import torch
from purerl.config import make_flat_env_cfg
from purerl.envs import TienKungLocomotionEnv
from purerl.sim import ArticulationIndexMap


class FakeTienKungBackend:
    def __init__(self):
        self.device = "cpu"
        self.num_envs = 0
        self.state = None
        self.body_names = ()
        self.index_map = None
        self.joint_limits = None

    def initialize(self, cfg):
        self.num_envs = cfg.scene.num_envs
        self.all_env_ids = torch.arange(self.num_envs)
        link_names = tuple(name.removesuffix("_joint") + "_link" for name in cfg.robot.joint_names)
        self.body_names = (cfg.robot.root_body_name, *link_names)
        self.index_map = ArticulationIndexMap.resolve(
            cfg.robot, cfg.robot.joint_names, self.body_names
        )
        self.joint_limits = torch.empty((self.num_envs, 20, 2))
        self.joint_limits[..., 0] = -3.0
        self.joint_limits[..., 1] = 3.0
        defaults = torch.tensor(cfg.robot.default_joint_positions).repeat(self.num_envs, 1)
        self.state = SimpleNamespace(
            root_position=torch.zeros((self.num_envs, 3)),
            root_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(self.num_envs, 1),
            root_linear_velocity=torch.zeros((self.num_envs, 3)),
            root_angular_velocity=torch.zeros((self.num_envs, 3)),
            joint_positions=defaults.clone(),
            joint_velocities=torch.zeros((self.num_envs, 20)),
            joint_accelerations=torch.zeros((self.num_envs, 20)),
            joint_torques=torch.zeros((self.num_envs, 20)),
            body_positions=torch.zeros((self.num_envs, len(self.body_names), 3)),
            body_linear_velocities=torch.zeros((self.num_envs, len(self.body_names), 3)),
            body_angular_velocities=torch.zeros((self.num_envs, len(self.body_names), 3)),
            net_contact_forces=torch.zeros((self.num_envs, len(self.body_names), 3)),
        )
        self._default_height = cfg.robot.default_root_height
        self._defaults = defaults

    def zeros(self, shape, *, dtype):
        types = {"bool": torch.bool, "float": torch.float32, "long": torch.long}
        return torch.zeros(shape, dtype=types[dtype])

    def set_joint_position_targets(self, targets):
        self._targets = targets

    def simulate(self, *, render):
        del render

    def configure_viewer(self):
        pass

    def render_rgb(self):
        return torch.zeros((4, 4, 3), dtype=torch.uint8).numpy()

    def refresh(self):
        pass

    def get_net_contact_forces(self):
        return self.state.net_contact_forces

    def sample_terrain_heights(self, world_points_xy):
        return torch.zeros(world_points_xy.shape[:-1])

    def set_contact_material_friction(self, env_ids, static_friction, dynamic_friction):
        self.static_friction = static_friction.clone()
        self.dynamic_friction = dynamic_friction.clone()

    def randomize_body_properties(self, env_ids, mass_delta, com_offset):
        self.mass_delta = mass_delta.clone()
        self.com_offset = com_offset.clone()

    def set_actuator_gains(self, env_ids, stiffness_scale, damping_scale):
        self.stiffness_scale = stiffness_scale.clone()
        self.damping_scale = damping_scale.clone()

    def randomize_reset_state(self, env_ids, root_xy, root_yaw, joint_position_scale):
        self.state.root_position[env_ids, :2] = root_xy
        self.state.root_quaternion[env_ids] = 0.0
        self.state.root_quaternion[env_ids, 0] = torch.cos(root_yaw * 0.5)
        self.state.root_quaternion[env_ids, 3] = torch.sin(root_yaw * 0.5)
        self.state.joint_positions[env_ids] = self._defaults[env_ids] * joint_position_scale

    def apply_root_wrench(self, env_ids, forces, torques):
        self.root_forces = forces.clone()
        self.root_torques = torques.clone()

    def add_root_velocity(self, env_ids, linear_velocity_xy):
        self.state.root_linear_velocity[env_ids, :2] += linear_velocity_xy

    def reset(self, env_ids):
        self.state.root_position[env_ids] = 0.0
        self.state.root_position[env_ids, 2] = self._default_height
        self.state.root_quaternion[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.state.root_linear_velocity[env_ids] = 0.0
        self.state.root_angular_velocity[env_ids] = 0.0
        self.state.joint_positions[env_ids] = self._defaults[env_ids]
        self.state.joint_velocities[env_ids] = 0.0
        self.state.joint_accelerations[env_ids] = 0.0
        self.state.joint_torques[env_ids] = 0.0
        self.state.net_contact_forces[env_ids] = 0.0

    def nonzero(self, mask):
        return torch.nonzero(mask, as_tuple=False).flatten()

    def clone(self, value):
        return value.clone()

    def close(self):
        pass


def make_env(*, play=False):
    base = make_flat_env_cfg()
    if play:
        base = base.replace(
            play=True,
            observations=base.observations.replace(enable_corruption=False),
        )
    cfg = base.replace(
        scene=base.scene.replace(num_envs=2),
        sim=base.sim.replace(device="cpu"),
    )
    return TienKungLocomotionEnv(cfg=cfg, backend=FakeTienKungBackend())


def test_flat_environment_produces_stable_policy_contract():
    env = make_env(play=True)
    observations, _ = env.reset(seed=7)

    assert observations["policy"].shape == (2, 259)
    assert torch.allclose(observations["policy"][:, 72:], torch.full((2, 187), 0.39))
    assert env.action_space.shape == (20,)
    assert env.observation_space["policy"].shape == (259,)
    assert env.backend.mass_delta.shape == (2,)
    assert env.backend.stiffness_scale.shape == (2, 20)
    assert env.backend.static_friction.shape == (2,)
    assert (env.backend.static_friction >= env.cfg.randomization.static_friction_range[0]).all()
    assert (env.backend.static_friction <= env.cfg.randomization.static_friction_range[1]).all()
    assert (env.backend.dynamic_friction >= env.cfg.randomization.dynamic_friction_range[0]).all()
    assert (env.backend.dynamic_friction <= env.cfg.randomization.dynamic_friction_range[1]).all()
    assert (env.backend.dynamic_friction <= env.backend.static_friction).all()


def test_training_observation_corruption_is_seeded_by_reset():
    env = make_env()
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)

    assert torch.equal(first["policy"], second["policy"])
    assert not torch.allclose(first["policy"][:, 72:], torch.full((2, 187), 0.39))
    assert (first["policy"][:, 72:] >= 0.33).all()
    assert (first["policy"][:, 72:] <= 0.45).all()


def test_flat_environment_runs_manager_lifecycle():
    env = make_env()
    observations, rewards, terminated, truncated, info = env.step(torch.zeros((2, 20)))

    assert observations["policy"].shape == (2, 259)
    assert rewards.shape == terminated.shape == truncated.shape == (2,)
    assert torch.isfinite(rewards).all()
    assert not terminated.any()
    assert not truncated.any()
    assert info["terminal_observation"].shape == (0, 259)
    assert torch.allclose(
        env.contact_history.current_air_time,
        torch.full((2, 2), env.cfg.sim.dt * env.cfg.sim.decimation),
    )


def test_reset_domain_randomization_is_seeded_and_batched():
    env = make_env()
    env.reset(seed=7)
    first_root_pose = torch.cat((env.state.root_position, env.state.root_quaternion), dim=-1)
    first_joints = env.state.joint_positions.clone()

    env.reset(seed=7)

    assert torch.allclose(
        torch.cat((env.state.root_position, env.state.root_quaternion), dim=-1),
        first_root_pose,
    )
    assert torch.allclose(env.state.joint_positions, first_joints)
    assert (env.state.root_position[:, :2].abs() <= 0.5).all()
