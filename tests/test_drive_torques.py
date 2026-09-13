"""PD reward estimates use the same targets, gains and limits as the drive."""

import torch
from purerl.config import make_flat_env_cfg
from purerl.sim import IsaacSimBackend


class RecordingArticulation:
    def set_gains(self, **kwargs):
        self.gains = kwargs

    def set_joint_position_targets(self, targets, **kwargs):
        self.targets = targets.clone()

    def set_world_poses(self, *args, **kwargs):
        pass

    def set_velocities(self, *args, **kwargs):
        pass

    def set_joint_positions(self, *args, **kwargs):
        pass

    def set_joint_velocities(self, *args, **kwargs):
        pass


def make_backend():
    backend = IsaacSimBackend()
    backend._torch = torch
    base = make_flat_env_cfg()
    backend._cfg = base.replace(scene=base.scene.replace(num_envs=2), sim=base.sim.replace(device="cpu"))
    backend.num_envs = 2
    backend.device = "cpu"
    backend._world = object()
    backend._body_view = object()
    backend._contact_view = object()
    backend._articulation = RecordingArticulation()
    backend._joint_indices = torch.arange(19, -1, -1)
    backend.env_origins = torch.zeros(2, 3)
    backend._default_joint_positions = torch.tensor(base.robot.default_joint_positions).repeat(2, 1)
    backend._previous_joint_velocities = torch.zeros(2, 20)
    backend._joint_position_targets = backend._default_joint_positions.clone()
    backend._joint_stiffness = torch.tensor(base.robot.stiffness).repeat(2, 1)
    backend._joint_damping = torch.tensor(base.robot.damping).repeat(2, 1)
    backend._joint_effort_limits = torch.tensor(base.robot.effort_limits).repeat(2, 1)
    return backend


def test_drive_estimate_responds_to_targets_damping_and_effort_limits():
    backend = make_backend()
    positions = backend._default_joint_positions.clone()
    targets = positions.clone()
    targets[:, 0] += 0.1  # hip roll: 700 * 0.1 = 70 Nm
    targets[:, 1] -= 100  # saturates at -180 Nm
    targets[:, 2] += 100  # shoulder pitch saturates at 52.5 Nm
    backend.set_joint_position_targets(targets)
    targets.zero_()  # Caller mutation must not change the cached drive target.
    velocities = torch.zeros_like(positions)
    velocities[:, 0] = 2  # hip roll damping subtracts 20 Nm
    torques = backend._estimate_drive_torques(positions, velocities)
    assert torch.allclose(torques[:, :3], torch.tensor([[50.0, -180.0, 52.5]]).repeat(2, 1))
    assert torch.count_nonzero(torques[:, 3:]) == 0


def test_gain_randomization_matches_actuator_and_keeps_other_environment():
    backend = make_backend()
    backend.set_actuator_gains(torch.tensor([0]), torch.full((1, 20), 0.5), torch.full((1, 20), 2.0))
    assert torch.equal(backend._joint_stiffness[:1], backend._articulation.gains["kps"])
    assert torch.equal(backend._joint_damping[:1], backend._articulation.gains["kds"])
    assert torch.equal(backend._articulation.gains["joint_indices"], backend._joint_indices)
    positions = backend._default_joint_positions.clone()
    backend.set_joint_position_targets(positions + 0.1)
    torques = backend._estimate_drive_torques(positions, torch.ones_like(positions))
    assert torch.allclose(torques[:, 0], torch.tensor([15.0, 60.0]), atol=1e-4)


def test_selective_reset_and_randomized_reset_update_only_selected_targets():
    backend = make_backend()
    backend.set_joint_position_targets(backend._default_joint_positions + 0.3)
    unselected = backend._joint_position_targets[1].clone()
    backend.reset(torch.tensor([0]))
    assert torch.equal(backend._joint_position_targets[0], backend._default_joint_positions[0])
    assert torch.equal(backend._joint_position_targets[1], unselected)
    scale = torch.full((1, 20), 1.05)
    backend.randomize_reset_state(torch.tensor([0]), torch.zeros(1, 2), torch.zeros(1), scale)
    assert torch.equal(backend._joint_position_targets[:1], backend._default_joint_positions[:1] * scale)
    assert torch.equal(backend._joint_position_targets[1], unselected)
