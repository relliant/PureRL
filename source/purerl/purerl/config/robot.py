"""Explicit TienKung robot configuration independent of Isaac Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from purerl.contracts import ACTION_DIM, FOOT_BODY_NAMES, ROOT_BODY_NAME

from .base import ConfigMixin

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = (
    PACKAGE_ROOT / "assets" / "robot_description" / "tienkung" / "tienkung2_lite.urdf"
)

_DECLARED_JOINT_NAMES = (
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
)

_DECLARED_DEFAULT_JOINT_POSITIONS = (
    0.0,
    -0.5,
    0.0,
    1.0,
    -0.5,
    0.0,
    0.0,
    -0.5,
    0.0,
    1.0,
    -0.5,
    0.0,
    0.0,
    0.1,
    0.0,
    -0.3,
    0.0,
    -0.1,
    0.0,
    -0.3,
)

_DECLARED_STIFFNESS = (
    700.0,
    700.0,
    500.0,
    700.0,
    30.0,
    16.8,
    700.0,
    700.0,
    500.0,
    700.0,
    30.0,
    16.8,
    60.0,
    20.0,
    10.0,
    10.0,
    60.0,
    20.0,
    10.0,
    10.0,
)

_DECLARED_DAMPING = (
    10.0,
    10.0,
    5.0,
    10.0,
    2.5,
    1.4,
    10.0,
    10.0,
    5.0,
    10.0,
    2.5,
    1.4,
    3.0,
    1.5,
    1.0,
    1.0,
    3.0,
    1.5,
    1.0,
    1.0,
)

_DECLARED_EFFORT_LIMITS = (
    180.0,
    300.0,
    180.0,
    300.0,
    60.0,
    30.0,
    180.0,
    300.0,
    180.0,
    300.0,
    60.0,
    30.0,
    52.5,
    52.5,
    52.5,
    52.5,
    52.5,
    52.5,
    52.5,
    52.5,
)

_DECLARED_VELOCITY_LIMITS = (
    15.6,
    15.6,
    15.6,
    15.6,
    12.8,
    7.8,
    15.6,
    15.6,
    15.6,
    15.6,
    12.8,
    7.8,
    14.1,
    14.1,
    14.1,
    14.1,
    14.1,
    14.1,
    14.1,
    14.1,
)

_DECLARED_ARMATURE = (
    0.0103,
    0.0251,
    0.0103,
    0.0251,
    0.003597,
    0.003597,
    0.0103,
    0.0251,
    0.0103,
    0.0251,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
    0.003597,
)

# The legacy action configuration listed joints in ``_DECLARED_JOINT_NAMES``
# but left ``preserve_order`` disabled. Isaac Lab therefore exposed actions and
# joint observations in this articulation order. Existing checkpoints depend on
# the resolved order, not the declaration order.
JOINT_NAMES = (
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
)

_DECLARED_INDEX = {name: index for index, name in enumerate(_DECLARED_JOINT_NAMES)}


def _resolve_articulation_order(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(values[_DECLARED_INDEX[name]] for name in JOINT_NAMES)


DEFAULT_JOINT_POSITIONS = _resolve_articulation_order(_DECLARED_DEFAULT_JOINT_POSITIONS)
STIFFNESS = _resolve_articulation_order(_DECLARED_STIFFNESS)
DAMPING = _resolve_articulation_order(_DECLARED_DAMPING)
EFFORT_LIMITS = _resolve_articulation_order(_DECLARED_EFFORT_LIMITS)
VELOCITY_LIMITS = _resolve_articulation_order(_DECLARED_VELOCITY_LIMITS)
ARMATURE = _resolve_articulation_order(_DECLARED_ARMATURE)


@dataclass(frozen=True)
class RobotCfg(ConfigMixin):
    urdf_path: Path = field(default_factory=lambda: DEFAULT_URDF_PATH)
    root_body_name: str = ROOT_BODY_NAME
    foot_body_names: tuple[str, str] = FOOT_BODY_NAMES
    joint_names: tuple[str, ...] = JOINT_NAMES
    default_root_height: float = 0.89
    default_joint_positions: tuple[float, ...] = DEFAULT_JOINT_POSITIONS
    stiffness: tuple[float, ...] = STIFFNESS
    damping: tuple[float, ...] = DAMPING
    effort_limits: tuple[float, ...] = EFFORT_LIMITS
    velocity_limits: tuple[float, ...] = VELOCITY_LIMITS
    armature: tuple[float, ...] = ARMATURE
    merge_fixed_joints: bool = True
    self_collisions: bool = False
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 4

    def validate(self, *, require_assets: bool = True) -> None:
        vector_fields = (
            self.default_joint_positions,
            self.stiffness,
            self.damping,
            self.effort_limits,
            self.velocity_limits,
            self.armature,
        )
        if len(self.joint_names) != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} joints, got {len(self.joint_names)}")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint names must be unique")
        if any(len(values) != len(self.joint_names) for values in vector_fields):
            raise ValueError("Every joint parameter vector must match joint_names")
        if require_assets and not self.urdf_path.is_file():
            raise FileNotFoundError(f"Robot URDF does not exist: {self.urdf_path}")


def make_tienkung_robot_cfg() -> RobotCfg:
    cfg = RobotCfg()
    cfg.validate()
    return cfg
