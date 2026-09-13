import json
from pathlib import Path

import numpy as np
import torch
from purerl.config import JOINT_NAMES, make_flat_env_cfg, make_rough_env_cfg
from purerl.contracts import ACTION_DIM, OBSERVATION_TERMS, STEP_DT, observation_slices
from purerl.envs.tienkung_locomotion import _quat_rotate_inverse, _yaw_from_quaternion
from purerl.mdp import rewards

FIXTURE_DIR = Path(__file__).parent / "fixtures"
METADATA_PATH = FIXTURE_DIR / "isaaclab_legacy_reference.json"
ARRAYS_PATH = FIXTURE_DIR / "isaaclab_legacy_reference.npz"
# The frozen Isaac Lab trajectory predates the appended sin/cos clock.
LEGACY_OBSERVATION_DIM = observation_slices()["gait_phase"].start


def test_legacy_fixture_identifies_exact_source_revisions():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["baseline_kind"] == "isaaclab_simulator_reference"
    assert metadata["isaaclab_revision"] == "2ed331acfc"
    assert metadata["purerl_revision"] == "b6e5f36"
    assert metadata["isaac_sim_version"] == "5.1.0"
    assert metadata["terrain_mesh"]["vertices"] > 0
    assert metadata["terrain_mesh"]["faces"] > 0
    assert len(metadata["terrain_mesh"]["sha256_vertices_faces"]) == 64
    assert tuple(metadata["resolved_joint_names"]) == JOINT_NAMES


def test_legacy_trajectory_contains_term_level_numeric_samples():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(ARRAYS_PATH) as fixture:
        assert fixture["flat_action"].shape == (16, 4, ACTION_DIM)
        assert fixture["flat_observation"].shape == (17, 4, LEGACY_OBSERVATION_DIM)
        assert fixture["flat_heading_target"].shape == (17, 4)
        np.testing.assert_array_equal(
            fixture["flat_heading_target"],
            np.broadcast_to(fixture["flat_heading_target"][0], (17, 4)),
        )
        assert fixture["flat_reward"].shape == (16, 4)
        assert fixture["flat_reward_terms_per_second"].shape == (
            16,
            4,
            len(metadata["reward_terms"]),
        )
        assert fixture["flat_termination_terms"].shape == (
            16,
            4,
            len(metadata["termination_terms"]),
        )
        expected_reward = fixture["flat_reward_terms_per_second"].sum(axis=-1) * STEP_DT
        np.testing.assert_allclose(fixture["flat_reward"], expected_reward, atol=1.0e-6)


def test_legacy_schema_and_weights_match_current_contract():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    flat = make_flat_env_cfg()
    rough = make_rough_env_cfg()

    assert metadata["observation_term_shapes"] == [
        [term.dimension] for term in OBSERVATION_TERMS if term.name != "gait_phase"
    ]
    # The frozen simulator fixture predates the PureRL gait terms. Preserve
    # its baseline contract while allowing current configs to add rewards.
    flat_weights = flat.reward_weights()
    rough_weights = rough.reward_weights()
    assert set(metadata["reward_terms"]).issubset(flat_weights)
    assert {
        name: flat_weights[name] for name in metadata["reward_terms"]
    } == metadata["reward_weights"]
    assert set(flat_weights) == set(rough_weights)
    assert metadata["termination_terms"] == ["time_out", "base_contact", "bad_orientation"]


def test_legacy_selective_reset_preserved_unselected_environments():
    with np.load(ARRAYS_PATH) as fixture:
        selected = set(fixture["selective_reset_env_ids"].tolist())
        unselected = [index for index in range(4) if index not in selected]
        for field in (
            "root_position",
            "root_quaternion",
            "root_linear_velocity",
            "root_angular_velocity",
            "joint_position",
            "joint_velocity",
            "command",
            "heading_target",
        ):
            np.testing.assert_array_equal(
                fixture[f"selective_reset_{field}_before"][unselected],
                fixture[f"selective_reset_{field}_after"][unselected],
            )


def test_legacy_rough_fixture_covers_origins_and_assignment():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    override = metadata["rough_assignment_override"]
    with np.load(ARRAYS_PATH) as fixture:
        assert fixture["rough_observation"].shape == (override["num_envs"], LEGACY_OBSERVATION_DIM)
        assert fixture["rough_terrain_origins"].shape == (
            override["num_rows"],
            override["num_cols"],
            3,
        )
        assert fixture["terrain_origins"].shape == (10, 20, 3)
        assert fixture["rough_terrain_levels"].shape == (override["num_envs"],)
        assert fixture["rough_terrain_types"].shape == (override["num_envs"],)


def test_current_reward_math_matches_legacy_numeric_fixture():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    cfg = make_flat_env_cfg()
    weights = cfg.reward_weights()
    with np.load(ARRAYS_PATH) as fixture:
        quaternion = torch.from_numpy(fixture["flat_root_quaternion"][1:])
        linear_world = torch.from_numpy(fixture["flat_root_linear_velocity"][1:])
        angular_world = torch.from_numpy(fixture["flat_root_angular_velocity"][1:])
        command = torch.from_numpy(fixture["flat_command"][:-1])
        joint_position = torch.from_numpy(fixture["flat_joint_position"][1:])
        action = torch.from_numpy(fixture["flat_action"])
        previous_action = torch.cat((torch.zeros_like(action[:1]), action[:-1]), dim=0)
        default = torch.tensor(cfg.robot.default_joint_positions).expand_as(joint_position)
        hip_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("hip_roll_", "hip_yaw_"))
        )
        arm_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("shoulder_", "elbow_"))
        )
        leg_indices = tuple(
            index
            for index, name in enumerate(cfg.robot.joint_names)
            if name.startswith(("hip_", "knee_", "ankle_"))
        )
        gravity = torch.zeros_like(linear_world)
        gravity[..., 2] = -1.0
        linear_body = _quat_rotate_inverse(quaternion, linear_world)
        angular_body = _quat_rotate_inverse(quaternion, angular_world)
        projected_gravity = _quat_rotate_inverse(quaternion, gravity)
        yaw = _yaw_from_quaternion(quaternion)
        linear_yaw = torch.stack(
            (
                torch.cos(yaw) * linear_world[..., 0] + torch.sin(yaw) * linear_world[..., 1],
                -torch.sin(yaw) * linear_world[..., 0] + torch.cos(yaw) * linear_world[..., 1],
                linear_world[..., 2],
            ),
            dim=-1,
        )
        termination = torch.from_numpy(fixture["flat_termination_terms"])[..., 1:].any(dim=-1)
        actual_raw = {
            "termination_penalty": rewards.termination_penalty(termination),
            "track_lin_vel_xy_exp": rewards.track_lin_vel_xy_exp(linear_yaw, command, std=0.5),
            "track_ang_vel_z_exp": rewards.track_ang_vel_z_exp(angular_world, command, std=0.5),
            "lin_vel_z_l2": rewards.lin_vel_z_l2(linear_body),
            "ang_vel_xy_l2": rewards.ang_vel_xy_l2(angular_body),
            "flat_orientation_l2": rewards.flat_orientation_l2(projected_gravity),
            "action_rate_l2": rewards.action_rate_l2(action, previous_action),
            "joint_deviation_hip": rewards.joint_deviation_l1(
                joint_position[..., hip_indices], default[..., hip_indices]
            ),
            "joint_deviation_arms": rewards.joint_deviation_l1(
                joint_position[..., arm_indices], default[..., arm_indices]
            ),
            "stand_still": rewards.stand_still_joint_deviation_l1(
                joint_position[..., leg_indices], default[..., leg_indices], command
            ),
        }
        legacy = fixture["flat_reward_terms_per_second"]
        for name, raw in actual_raw.items():
            term_index = metadata["reward_terms"].index(name)
            np.testing.assert_allclose(
                raw.numpy() * weights[name],
                legacy[..., term_index],
                rtol=2.0e-5,
                atol=2.0e-5,
                err_msg=name,
            )
