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
    assert flat_play.scene.num_envs == rough_play.scene.num_envs == 16
    assert not flat_play.observations.enable_corruption
    assert not rough_play.terrain.curriculum
    assert rough_play.terrain.max_initial_level is None


def test_flat_reward_overrides_do_not_mutate_rough_configuration():
    flat = make_flat_env_cfg().reward_weights()
    rough = make_rough_env_cfg().reward_weights()

    assert flat["lin_vel_z_l2"] == -0.5
    assert rough["lin_vel_z_l2"] == -1.5
    assert flat["feet_air_time"] == 0.75
    assert rough["feet_air_time"] == 0.5


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
    assert serialized["algorithm"]["class_name"] == "PPO"
    assert serialized["wandb_project"] == "purerl"
    assert serialized["wandb_mode"] == "online"


def test_runner_rejects_invalid_wandb_mode():
    cfg = make_flat_runner_cfg().replace(wandb_mode="invalid")

    with pytest.raises(ValueError, match="W&B mode"):
        cfg.validate()
