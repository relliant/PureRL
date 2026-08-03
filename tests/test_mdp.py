from types import SimpleNamespace

import numpy as np
import pytest
from purerl.config.robot import DEFAULT_JOINT_POSITIONS
from purerl.contracts import ACTION_DIM, OBSERVATION_DIM
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.managers import RewardManager, RewardTermSpec, TerminationManager, TerminationTermSpec
from purerl.mdp.observations import build_policy_observation
from purerl.mdp.rewards import action_rate_l2, track_lin_vel_xy_exp
from purerl.mdp.terminations import bad_orientation, time_out


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
