"""TienKung2 Lite articulation configuration.

The actuator parameters and default pose are kept in sync with WBC-SONIC's
``gear_sonic/envs/manager_env/robots/tienkung.py``.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TIENKUNG_URDF_PATH = PROJECT_ROOT / "assets" / "robot_description" / "tienkung" / "tienkung2_lite.urdf"

TIENKUNG_JOINT_NAMES = [
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
]

TIENKUNG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(TIENKUNG_URDF_PATH),
        fix_base=False,
        # The fixed head/hand marker links are only needed by motion tracking.
        # Locomotion uses pelvis and foot bodies, so merging them avoids empty-link USD warnings.
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            "hip_roll_.*_joint": 0.0,
            "hip_pitch_.*_joint": -0.5,
            "hip_yaw_.*_joint": 0.0,
            "knee_pitch_.*_joint": 1.0,
            "ankle_pitch_.*_joint": -0.5,
            "ankle_roll_.*_joint": 0.0,
            "shoulder_pitch_.*_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_.*_joint": 0.0,
            "elbow_pitch_.*_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 180.0,
                "hip_pitch_.*_joint": 300.0,
                "hip_yaw_.*_joint": 180.0,
                "knee_pitch_.*_joint": 300.0,
            },
            velocity_limit_sim={".*": 15.6},
            stiffness={
                "hip_roll_.*_joint": 700.0,
                "hip_pitch_.*_joint": 700.0,
                "hip_yaw_.*_joint": 500.0,
                "knee_pitch_.*_joint": 700.0,
            },
            damping={
                "hip_roll_.*_joint": 10.0,
                "hip_pitch_.*_joint": 10.0,
                "hip_yaw_.*_joint": 5.0,
                "knee_pitch_.*_joint": 10.0,
            },
            armature={
                "hip_roll_.*_joint": 0.0103,
                "hip_pitch_.*_joint": 0.0251,
                "hip_yaw_.*_joint": 0.0103,
                "knee_pitch_.*_joint": 0.0251,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            effort_limit_sim={"ankle_pitch_.*_joint": 60.0, "ankle_roll_.*_joint": 30.0},
            velocity_limit_sim={"ankle_pitch_.*_joint": 12.8, "ankle_roll_.*_joint": 7.8},
            stiffness={"ankle_pitch_.*_joint": 30.0, "ankle_roll_.*_joint": 16.8},
            damping={"ankle_pitch_.*_joint": 2.5, "ankle_roll_.*_joint": 1.4},
            armature={".*": 0.003597},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={".*": 52.5},
            velocity_limit_sim={".*": 14.1},
            stiffness={
                "shoulder_pitch_.*_joint": 60.0,
                "shoulder_roll_.*_joint": 20.0,
                "shoulder_yaw_.*_joint": 10.0,
                "elbow_pitch_.*_joint": 10.0,
            },
            damping={
                "shoulder_pitch_.*_joint": 3.0,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1.0,
                "elbow_pitch_.*_joint": 1.0,
            },
            armature={".*": 0.003597},
        ),
    },
)
