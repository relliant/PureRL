"""Vectorized locomotion reward terms."""

from __future__ import annotations

from typing import Any

from ._array import abs_value, clip, exp, max_axis, maximum, min_axis, norm, sum_axis, where


def termination_penalty(terminated: Any) -> Any:
    return terminated * 1.0


def track_lin_vel_xy_exp(base_linear_velocity_yaw: Any, command: Any, *, std: float = 0.5) -> Any:
    error = command[..., :2] - base_linear_velocity_yaw[..., :2]
    return exp(-sum_axis(error * error) / (std * std))


def track_ang_vel_z_exp(base_angular_velocity_world: Any, command: Any, *, std: float = 0.5) -> Any:
    error = command[..., 2] - base_angular_velocity_world[..., 2]
    return exp(-(error * error) / (std * std))


def lin_vel_z_l2(base_linear_velocity_body: Any) -> Any:
    return base_linear_velocity_body[..., 2] ** 2


def ang_vel_xy_l2(base_angular_velocity_body: Any) -> Any:
    return sum_axis(base_angular_velocity_body[..., :2] ** 2)


def flat_orientation_l2(projected_gravity: Any) -> Any:
    return sum_axis(projected_gravity[..., :2] ** 2)


def joint_torques_l2(joint_torques: Any) -> Any:
    return sum_axis(joint_torques**2)


def joint_acc_l2(joint_accelerations: Any) -> Any:
    return sum_axis(joint_accelerations**2)


def action_rate_l2(action: Any, previous_action: Any) -> Any:
    return sum_axis((action - previous_action) ** 2)


def feet_air_time_positive_biped(
    current_air_time: Any,
    current_contact_time: Any,
    command: Any,
    *,
    threshold: float = 0.4,
    command_threshold: float = 0.1,
) -> Any:
    in_contact = current_contact_time > 0.0
    in_mode_time = where(in_contact, current_contact_time, current_air_time)
    single_stance = sum_axis(in_contact) == 1
    reward = min_axis(
        where(single_stance[..., None], in_mode_time, in_mode_time * 0.0),
        axis=-1,
    )
    moving = norm(command[..., :2]) > command_threshold
    return clip(reward, 0.0, threshold) * moving


def feet_slide(foot_linear_velocity: Any, contact_forces: Any, *, threshold: float = 1.0) -> Any:
    magnitudes = norm(contact_forces)
    if len(magnitudes.shape) == 3:
        magnitudes = max_axis(magnitudes, axis=1)
    in_contact = magnitudes > threshold
    return sum_axis(norm(foot_linear_velocity[..., :2]) * in_contact)


def undesired_contacts(contact_forces: Any, *, threshold: float = 1.0) -> Any:
    magnitudes = norm(contact_forces)
    if len(magnitudes.shape) == 3:
        magnitudes = max_axis(magnitudes, axis=1)
    return sum_axis(magnitudes > threshold)


def joint_pos_limits(joint_positions: Any, soft_limits: Any) -> Any:
    below = maximum(soft_limits[..., 0] - joint_positions, 0.0)
    above = maximum(joint_positions - soft_limits[..., 1], 0.0)
    return sum_axis(below + above)


def joint_deviation_l1(joint_positions: Any, default_joint_positions: Any) -> Any:
    return sum_axis(abs_value(joint_positions - default_joint_positions))


def stand_still_joint_deviation_l1(
    joint_positions: Any,
    default_joint_positions: Any,
    command: Any,
    *,
    command_threshold: float = 0.06,
) -> Any:
    standing = norm(command[..., :2]) < command_threshold
    return joint_deviation_l1(joint_positions, default_joint_positions) * standing
