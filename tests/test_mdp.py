from types import SimpleNamespace

import numpy as np
import pytest
from purerl.config.env import CommandCfg, VelocityRangesCfg
from purerl.config.robot import DEFAULT_JOINT_POSITIONS
from purerl.contracts import ACTION_DIM, OBSERVATION_DIM
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.commands import VelocityCommandManager
from purerl.mdp.managers import (
    ObservationManager,
    ObservationTermSpec,
    RewardManager,
    RewardTermSpec,
    TerminationManager,
    TerminationTermSpec,
)
from purerl.mdp.observations import build_policy_observation
from purerl.mdp.rewards import (
    action_rate_l2,
    feet_air_time_on_contact,
    feet_air_time_positive_biped,
    feet_clearance,
    feet_contact_number,
    feet_slide,
    track_lin_vel_xy_exp,
    undesired_contacts,
)
from purerl.mdp.terminations import bad_orientation, illegal_contact, time_out


def test_joint_position_action_clips_scales_and_tracks_history():
    defaults = np.broadcast_to(np.asarray(DEFAULT_JOINT_POSITIONS), (2, ACTION_DIM)).copy()
    manager = JointPositionActionManager(defaults, scale=0.5, action_clip=1.0)

    target = manager.process(np.full_like(defaults, 2.0))
    assert np.allclose(target, defaults + 0.5)
    manager.process(np.full_like(defaults, -0.25))
    assert np.allclose(manager.previous_action, 1.0)

    manager.reset(np.asarray([1]))
    assert np.allclose(manager.action[1], 0.0)
    assert np.allclose(manager.previous_action[1], 0.0)


def test_joint_position_action_is_unclipped_by_default():
    defaults = np.broadcast_to(np.asarray(DEFAULT_JOINT_POSITIONS), (1, ACTION_DIM)).copy()
    manager = JointPositionActionManager(defaults, scale=0.5)

    target = manager.process(np.full_like(defaults, 2.0))

    assert np.allclose(target, defaults + 1.0)


def test_velocity_command_heading_controller_wraps_clips_and_stands():
    cfg = CommandCfg(
        resampling_time_range=(2.0, 2.0),
        heading_control_stiffness=0.5,
        heading_env_ratio=1.0,
        standing_env_ratio=0.0,
        ranges=VelocityRangesCfg(ang_vel_z=(-1.0, 1.0)),
    )
    manager = VelocityCommandManager(
        cfg, num_envs=3, device="cpu", step_dt=0.02, seed=7
    )
    manager.reset(np.arange(3))
    sampled_yaw_rate = manager.command[:, 2].clone()
    manager.heading_target.copy_(
        manager._torch.tensor([-3.0, 3.0, 2.5], dtype=manager.command.dtype)
    )
    manager.is_standing_env[2] = True

    assert manager.command[:, 2].equal(sampled_yaw_rate)
    manager.update(manager._torch.tensor([3.0, -3.0, 0.0]))

    expected = manager._torch.tensor([0.1415927, -0.1415927, 0.0])
    assert manager._torch.allclose(manager.command[:, 2], expected, atol=1.0e-6)
    assert manager._torch.count_nonzero(manager.command[2]) == 0


def test_velocity_command_can_mix_direct_yaw_and_heading_control():
    cfg = CommandCfg(
        resampling_time_range=(1.0, 1.0),
        heading_env_ratio=0.0,
        standing_env_ratio=0.0,
    )
    manager = VelocityCommandManager(
        cfg, num_envs=2, device="cpu", step_dt=0.02, seed=11
    )
    manager.reset(np.arange(2))
    sampled_yaw_rate = manager.command[:, 2].clone()

    manager.update(manager._torch.tensor([1.0, -1.0]))

    assert manager.command[:, 2].equal(sampled_yaw_rate)


def test_policy_observation_order_and_dimension():
    batch = 2
    values = [
        np.full((batch, 3), 1.0),
        np.full((batch, 3), 2.0),
        np.full((batch, 3), 3.0),
        np.full((batch, 3), 4.0),
        np.full((batch, 20), 5.0),
        np.full((batch, 20), 6.0),
        np.full((batch, 20), 7.0),
        np.full((batch, 187), 8.0),
    ]
    observation = build_policy_observation(
        base_linear_velocity=values[0],
        base_angular_velocity=values[1],
        projected_gravity=values[2],
        velocity_command=values[3],
        relative_joint_position=values[4],
        joint_velocity=values[5],
        previous_action=values[6],
        terrain_height_scan=values[7],
    )

    assert observation.shape == (batch, OBSERVATION_DIM)
    assert np.all(observation[:, 72:] == 8.0)


def test_observation_manager_applies_seeded_noise_and_clipping():
    context = SimpleNamespace(
        noisy=np.zeros((4, 2), dtype=np.float32),
        exact=np.full((4, 1), 2.0, dtype=np.float32),
    )
    manager = ObservationManager(
        (
            ObservationTermSpec("noisy", lambda ctx: ctx.noisy, noise=(-0.25, 0.25)),
            ObservationTermSpec("exact", lambda ctx: ctx.exact, clip=(-1.0, 1.0)),
        ),
        expected_dimension=3,
        enable_corruption=True,
        seed=17,
    )

    first = manager.compute(context)
    assert np.all(first[:, :2] >= -0.25)
    assert np.all(first[:, :2] <= 0.25)
    assert np.all(first[:, 2] == 1.0)

    manager.set_seed(17)
    assert np.array_equal(manager.compute(context), first)


def test_observation_manager_can_disable_corruption():
    context = SimpleNamespace(value=np.zeros((2, 1), dtype=np.float32))
    manager = ObservationManager(
        (ObservationTermSpec("value", lambda ctx: ctx.value, noise=(-1.0, 1.0)),),
        expected_dimension=1,
        enable_corruption=False,
    )

    assert np.array_equal(manager.compute(context), context.value)


def test_reward_manager_integrates_weights_and_resets_selected_envs():
    context = SimpleNamespace(value=np.asarray([1.0, 2.0]))
    manager = RewardManager(
        (RewardTermSpec("value", lambda ctx: ctx.value, weight=2.0),),
        dt=0.02,
    )

    assert np.allclose(manager.compute(context), [0.04, 0.08])
    assert np.allclose(manager.compute(context), [0.04, 0.08])
    completed = manager.reset(np.asarray([1]))
    assert np.allclose(completed["value"], [0.16])
    assert np.allclose(manager.episode_sums["value"], [0.08, 0.0])


def test_termination_manager_keeps_timeouts_separate():
    context = SimpleNamespace(
        episode_length=np.asarray([5, 10]),
        gravity=np.asarray([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
    )
    manager = TerminationManager(
        (
            TerminationTermSpec(
                "time_out", lambda ctx: time_out(ctx.episode_length, 10), time_out=True
            ),
            TerminationTermSpec("bad_orientation", lambda ctx: bad_orientation(ctx.gravity)),
        )
    )

    terminated, truncated = manager.compute(context)
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [False, True]


def test_velocity_tracking_reward_and_action_rate():
    velocity = np.asarray([[1.0, 0.0, 0.0]])
    command = np.asarray([[1.0, 0.0, 0.0]])

    assert track_lin_vel_xy_exp(velocity, command) == pytest.approx([1.0])
    assert action_rate_l2(np.ones((1, 3)), np.zeros((1, 3))) == pytest.approx([3.0])


def test_positive_biped_air_time_uses_single_stance_mode_time():
    air_time = np.asarray([[0.3, 0.0], [0.3, 0.3]])
    contact_time = np.asarray([[0.0, 0.2], [0.0, 0.0]])
    command = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    reward = feet_air_time_positive_biped(air_time, contact_time, command, threshold=0.4)

    assert reward == pytest.approx([0.2, 0.0])


def test_air_time_reward_pays_only_on_latched_landing_events():
    last_air_time = np.asarray([[0.4, 0.1], [0.4, 0.4]])
    contact_events = np.asarray([[True, False], [False, False]])
    command = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    reward = feet_air_time_on_contact(
        last_air_time, contact_events, command, threshold=0.25
    )

    assert reward == pytest.approx([0.15, 0.0])


def test_gait_contact_reward_ignores_standing_commands():
    contact = np.asarray([[True, True], [True, False]])
    stance = np.asarray([[True, True], [True, False]])
    command = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])

    reward = feet_contact_number(contact, stance, command)

    assert reward == pytest.approx([0.0, 1.0])


def test_swing_clearance_is_smooth_and_zero_for_stance():
    foot_positions = np.asarray([[[0.1169, 0.0, 0.1169], [0.0569, 0.0, 0.0569]]])
    swing = np.asarray([[1.0, 0.0]])

    reward = feet_clearance(
        foot_positions,
        swing,
        np.asarray([[0.2, 0.0, 0.0]]),
        target=0.06,
        foot_offset=0.0569,
        sigma=0.025,
    )

    assert reward == pytest.approx([1.0], abs=1.0e-5)


def test_swing_clearance_ignores_standing_commands():
    foot_positions = np.asarray([[[0.1169, 0.0, 0.1169], [0.0569, 0.0, 0.0569]]])
    swing = np.asarray([[1.0, 0.0]])

    reward = feet_clearance(
        foot_positions, swing, np.asarray([[0.0, 0.0, 0.0]]), target=0.06
    )

    assert reward == pytest.approx([0.0])


def test_contact_terms_reduce_sensor_history_before_body_dimension():
    forces = np.zeros((2, 3, 2, 3))
    forces[0, 0, 1, 2] = 2.0
    foot_velocity = np.ones((2, 2, 3))

    assert illegal_contact(forces).tolist() == [True, False]
    assert undesired_contacts(forces).tolist() == [1, 0]
    assert feet_slide(foot_velocity, forces) == pytest.approx([np.sqrt(2.0), 0.0])
