"""Stable public contracts shared by environments, policies, and tests."""

from __future__ import annotations

from dataclasses import dataclass

ACTION_DIM = 20
PHYSICS_DT = 0.005
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION

ROOT_BODY_NAME = "pelvis"
FOOT_BODY_NAMES = ("ankle_roll_l_link", "ankle_roll_r_link")

TASK_IDS = (
    "PureRL-Velocity-Flat-TienKung-v0",
    "PureRL-Velocity-Flat-TienKung-Play-v0",
    "PureRL-Velocity-Rough-TienKung-v0",
    "PureRL-Velocity-Rough-TienKung-Play-v0",
)


@dataclass(frozen=True)
class ObservationTerm:
    """One contiguous term in the policy observation tensor."""

    name: str
    dimension: int


OBSERVATION_TERMS = (
    ObservationTerm("base_linear_velocity", 3),
    ObservationTerm("base_angular_velocity", 3),
    ObservationTerm("projected_gravity", 3),
    ObservationTerm("velocity_command", 3),
    ObservationTerm("relative_joint_positions", 20),
    ObservationTerm("joint_velocities", 20),
    ObservationTerm("previous_action", 20),
    ObservationTerm("terrain_height_scan", 187),
    ObservationTerm("gait_phase", 2),  # sin/cos of the reward clock, without observation noise
)
OBSERVATION_DIM = sum(term.dimension for term in OBSERVATION_TERMS)


def observation_slices() -> dict[str, slice]:
    """Return the immutable policy layout as named slices."""

    offset = 0
    layout: dict[str, slice] = {}
    for term in OBSERVATION_TERMS:
        layout[term.name] = slice(offset, offset + term.dimension)
        offset += term.dimension
    return layout
