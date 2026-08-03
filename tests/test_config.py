from dataclasses import replace

import pytest
from purerl.config import (
    make_flat_env_cfg,
    make_flat_play_env_cfg,
    make_flat_runner_cfg,
    make_rough_env_cfg,
    make_rough_play_env_cfg,
    make_rough_runner_cfg,
)
from purerl.contracts import OBSERVATION_DIM


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


def test_runner_configs_expose_rsl_rl_dictionary_contract():
    flat = make_flat_runner_cfg()
    rough = make_rough_runner_cfg()
    serialized = rough.to_dict()

    assert flat.max_iterations == 2500
    assert rough.max_iterations == 5000
    assert serialized["class_name"] == "OnPolicyRunner"
    assert serialized["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert serialized["algorithm"]["class_name"] == "PPO"
