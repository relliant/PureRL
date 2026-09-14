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


def feet_air_time_on_contact(
    last_air_time: Any,
    first_contact: Any,
    command: Any,
    *,
    supported_landing: Any,
    threshold: float = 0.25,
    command_threshold: float = 0.1,
) -> Any:
    """Pay for a supported, alternating landing once; reject simultaneous landings."""

    moving = norm(command[..., :2]) > command_threshold
    single_event = sum_axis(first_contact) == 1
    valid = first_contact & supported_landing
    return sum_axis(clip(last_air_time - threshold, 0.0, threshold) * valid) * single_event * moving


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


def feet_contact_number(
    contact: Any,
    stance_mask: Any,
    command: Any | None = None,
    *,
    current_air_time: Any,
    current_contact_time: Any,
    min_phase_time: float = 0.04,
    command_threshold: float = 0.1,
) -> Any:
    """Score the complete support pattern, requiring sustained support and swing."""

    matched = sum_axis(contact == stance_mask) == 2
    mode_time = where(stance_mask, current_contact_time, current_air_time)
    sustained = min_axis(mode_time, axis=-1) + 1e-7 >= min_phase_time
    value = where(matched, where(sustained, 1.0, 0.0), -1.0)
    if command is None:
        return value
    moving = norm(command[..., :2]) > command_threshold
    return value * moving


def feet_distance(body_positions: Any, *, min_dist: float = 0.2, max_dist: float = 0.5) -> Any:
    """惩罚两脚靠太近（交叉步）或分太开（螃蟹步）。"""
    foot_dist = norm(body_positions[:, 0, :2] - body_positions[:, 1, :2], axis=-1)
    d_min = clip(foot_dist - min_dist, -0.5, 0.0)
    d_max = clip(foot_dist - max_dist, 0.0, 0.5)
    return (exp(-abs_value(d_min) * 100.0) + exp(-abs_value(d_max) * 100.0)) / 2.0


def base_height(
    root_position_z: Any,
    ground_height: Any,
    *,
    target: float = 0.9,
    sigma: float = 0.05,
) -> Any:
    """Track pelvis height above the terrain, independent of swinging feet."""
    height = root_position_z - ground_height
    return exp(-((height - target) / sigma) ** 2)


def feet_clearance(
    foot_heights: Any,
    target_heights: Any,
    command: Any | None = None,
    *,
    contact: Any,
    current_contact_time: Any,
    support_time: float = 0.04,
    min_clearance: float = 0.02,
    sigma: float = 0.025,
    command_threshold: float = 0.1,
) -> Any:
    """Follow a swing arc while the opposite foot provides sustained support."""

    supported = contact[:, [1, 0]] & (current_contact_time[:, [1, 0]] + 1e-7 >= support_time)
    swing = (target_heights > 0.0) & ~contact & supported
    # A foot near the floor must not collect full credit at the endpoints of
    # the arc; smoothly increase credit as actual clearance develops.
    lift = clip(foot_heights / maximum(target_heights, min_clearance), 0.0, 1.0)
    reward = exp(-((foot_heights - target_heights) / sigma) ** 2) * lift
    value = sum_axis(reward * swing, axis=-1)
    if command is None:
        return value
    moving = norm(command[..., :2]) > command_threshold
    return value * moving
