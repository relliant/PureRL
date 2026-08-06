from dataclasses import replace
from pathlib import Path

import pytest
from purerl.config import (
    EnvCfg,
    OnPolicyRunnerCfg,
    make_flat_env_cfg,
    make_flat_play_env_cfg,
    make_flat_runner_cfg,
    make_rough_env_cfg,
    make_rough_play_env_cfg,
    make_rough_runner_cfg,
)
from purerl.contracts import OBSERVATION_DIM


@pytest.mark.parametrize(
    ("filename", "config_type"),
    (
        ("flat_env.yaml", EnvCfg),
        ("flat_play_env.yaml", EnvCfg),
        ("rough_env.yaml", EnvCfg),
        ("rough_play_env.yaml", EnvCfg),
        ("flat_runner.yaml", OnPolicyRunnerCfg),
        ("rough_runner.yaml", OnPolicyRunnerCfg),
    ),
)
def test_complete_yaml_presets_load_and_validate(filename, config_type):
    preset = Path(__file__).parents[1] / "source" / "purerl" / "purerl" / "config" / "presets" / filename
    cfg = config_type.from_yaml(preset)

    cfg.validate()


def test_yaml_loader_restores_nested_types_and_relative_paths():
    cfg = make_rough_env_cfg()

    assert isinstance(cfg.robot.urdf_path, Path)
    assert cfg.robot.urdf_path.is_file()
    assert isinstance(cfg.robot.joint_names, tuple)
    assert isinstance(cfg.terrain.patches, tuple)
    assert cfg.terrain.patches[0].noise_range is None


def test_yaml_round_trip_preserves_complete_config(tmp_path):
    original = make_rough_env_cfg()
    output = tmp_path / "env.yaml"

    original.to_yaml(output)
    restored = EnvCfg.from_yaml(output)

    assert restored == original


def test_yaml_loader_rejects_unknown_nested_fields():
    data = make_flat_env_cfg().to_dict()
    data["scene"]["num_environment"] = data["scene"].pop("num_envs")

    with pytest.raises(ValueError, match=r"Unknown field.*EnvCfg\.scene.*num_environment"):
        EnvCfg.from_dict(data)


def test_yaml_loader_rejects_missing_fields():
    data = make_flat_runner_cfg().to_dict()
    del data["algorithm"]["learning_rate"]

    with pytest.raises(ValueError, match=r"Missing field.*algorithm.*learning_rate"):
        OnPolicyRunnerCfg.from_dict(data)


def test_environment_variants_are_explicit_and_checkpoint_compatible():
    flat = make_flat_env_cfg()
    rough = make_rough_env_cfg()
    flat_play = make_flat_play_env_cfg()
    rough_play = make_rough_play_env_cfg()

    assert flat.terrain.terrain_type == "plane"
    assert rough.terrain.terrain_type == "generator"
    assert flat.observations.dimension == rough.observations.dimension == OBSERVATION_DIM
    assert flat.scene.num_envs == rough.scene.num_envs == 4096
    assert flat_play.scene.num_envs == rough_play.scene.num_envs == 1
    assert not flat.sensors.lidar.enabled
    assert not rough.sensors.lidar.enabled
    assert flat_play.sensors.lidar.enabled
    assert rough_play.sensors.lidar.enabled
    assert rough_play.sensors.lidar.config_file_name == "OS1"
    assert rough_play.sensors.lidar.variant == "OS1_REV6_32ch10hz512res"
    assert rough_play.visuals.sky_color == (0.53, 0.69, 0.9)
    assert not flat_play.observations.enable_corruption
    assert not rough_play.terrain.curriculum
    assert rough_play.terrain.max_initial_level is None
    assert rough_play.terrain.selected_patch == "random_rough"
    assert rough_play.terrain.selected_level == 4
    assert rough_play.commands.ranges.lin_vel_x == (0.5, 0.5)
    assert not rough_play.commands.heading_command
    assert flat.actions.clip is None
    assert rough.actions.clip is None


def test_training_presets_use_humanoid_gym_inspired_locomotion_settings():
    flat = make_flat_env_cfg().reward_weights()
    rough_cfg = make_rough_env_cfg()
    rough = rough_cfg.reward_weights()

    assert flat["lin_vel_z_l2"] == -0.5
    assert rough["lin_vel_z_l2"] == -1.5
    assert flat["feet_air_time"] == 0.75
    assert rough["feet_air_time"] == 0.5
    assert rough_cfg.actions.scale == 0.25
    assert rough_cfg.episode_length_s == 24.0
    assert rough_cfg.commands.resampling_time_range == (8.0, 8.0)
    assert rough_cfg.commands.ranges.lin_vel_x == (-0.3, 0.6)
    assert rough_cfg.commands.ranges.lin_vel_y == (-0.3, 0.3)
    assert rough_cfg.commands.ranges.ang_vel_z == (-0.3, 0.3)
    assert rough_cfg.randomization.pelvis_mass_delta == (-5.0, 5.0)
    assert rough_cfg.randomization.push_interval_s == (4.0, 4.0)
    assert rough_cfg.terrain.max_initial_level == 5


def test_invalid_observation_dimension_is_rejected():
    cfg = make_rough_env_cfg()
    invalid = replace(cfg, observations=replace(cfg.observations, dimension=1))

    with pytest.raises(ValueError, match="observation dimension"):
        invalid.validate()


def test_heading_command_matches_legacy_locomotion_contract():
    command = make_rough_env_cfg().commands

    assert command.heading_command
    assert command.heading_control_stiffness == 0.5
    assert command.heading_env_ratio == 1.0
    assert command.standing_env_ratio == 0.1


def test_command_probabilities_are_validated():
    cfg = make_flat_env_cfg()
    invalid = replace(
        cfg,
        commands=replace(cfg.commands, heading_env_ratio=1.1),
    )

    with pytest.raises(ValueError, match="heading_env_ratio"):
        invalid.validate(require_assets=False)


def test_friction_bucket_count_must_be_positive():
    cfg = make_flat_env_cfg()
    invalid = replace(
        cfg,
        randomization=replace(cfg.randomization, friction_buckets=0),
    )

    with pytest.raises(ValueError, match="friction_buckets"):
        invalid.validate(require_assets=False)


def test_gpu_physx_capacities_must_be_positive():
    cfg = make_flat_env_cfg()
    invalid = replace(
        cfg,
        sim=replace(cfg.sim, gpu_total_aggregate_pairs_capacity=0),
    )

    with pytest.raises(ValueError, match="GPU PhysX capacities"):
        invalid.validate(require_assets=False)


def test_enabled_lidar_must_select_an_existing_environment():
    cfg = make_flat_play_env_cfg()
    invalid = cfg.replace(
        sensors=cfg.sensors.replace(
            lidar=cfg.sensors.lidar.replace(env_index=cfg.scene.num_envs)
        )
    )

    with pytest.raises(ValueError, match="lidar.env_index"):
        invalid.validate(require_assets=False)


def test_environment_visual_colors_are_validated():
    cfg = make_flat_env_cfg()
    invalid = cfg.replace(visuals=cfg.visuals.replace(sky_color=(1.1, 0.5, 0.5)))

    with pytest.raises(ValueError, match="visual colors"):
        invalid.validate(require_assets=False)


def test_runner_configs_expose_rsl_rl_dictionary_contract():
    flat = make_flat_runner_cfg()
    rough = make_rough_runner_cfg()
    serialized = rough.to_dict()

    assert flat.max_iterations > 0
    assert rough.max_iterations > 0
    assert flat.experiment_name == "tienkung_flat"
    assert rough.experiment_name == "tienkung_rough"
    assert serialized["class_name"] == "OnPolicyRunner"
    assert serialized["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert serialized["policy"]["critic_hidden_dims"] == [768, 256, 128]
    assert serialized["algorithm"]["class_name"] == "PPO"
    assert serialized["algorithm"]["entropy_coef"] == 0.001
    assert serialized["algorithm"]["learning_rate"] == 1.0e-5
    assert serialized["algorithm"]["num_learning_epochs"] == 2
    assert serialized["algorithm"]["gamma"] == 0.994
    assert serialized["algorithm"]["lam"] == 0.9
    assert serialized["num_steps_per_env"] == 60
    assert flat.max_iterations == 3001
    assert serialized["max_iterations"] == 10001
    assert serialized["min_action_noise_std"] == 0.05
    assert serialized["max_action_noise_std"] == 3.0
    assert serialized["max_checkpoint_noise_std"] == 3.0
    assert serialized["wandb_project"] == "purerl"
    assert serialized["wandb_mode"] == "online"


def test_runner_rejects_invalid_wandb_mode():
    cfg = make_flat_runner_cfg().replace(wandb_mode="invalid")

    with pytest.raises(ValueError, match="W&B mode"):
        cfg.validate()
