"""Policy observation assembly and validation."""

from __future__ import annotations

from typing import Any

from purerl.contracts import OBSERVATION_DIM

from ._array import concatenate


def relative_joint_positions(joint_positions: Any, default_joint_positions: Any) -> Any:
    return joint_positions - default_joint_positions


def build_policy_observation(
    *,
    base_linear_velocity: Any,
    base_angular_velocity: Any,
    projected_gravity: Any,
    velocity_command: Any,
    relative_joint_position: Any,
    joint_velocity: Any,
    previous_action: Any,
    terrain_height_scan: Any,
) -> Any:
    observation = concatenate(
        (
            base_linear_velocity,
            base_angular_velocity,
            projected_gravity,
            velocity_command,
            relative_joint_position,
            joint_velocity,
            previous_action,
            terrain_height_scan,
        ),
        axis=-1,
    )
    if observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(
            f"Policy observation has dimension {observation.shape[-1]}, expected {OBSERVATION_DIM}"
        )
    return observation
