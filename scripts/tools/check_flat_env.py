#!/usr/bin/env python3
"""Run a short end-to-end flat locomotion environment smoke test."""

from __future__ import annotations

import argparse
import sys
import traceback

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.config import make_flat_env_cfg
from purerl.envs import TienKungLocomotionEnv
from purerl.rl import RslRlVecEnvWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--check-selective-reset", action="store_true")
    parser.add_argument("--check-periodic-push", action="store_true")
    args = parser.parse_args()

    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    env = None
    try:
        _stage("launching")
        launcher.launch()
        _stage("launched")
        import torch

        base = make_flat_env_cfg()
        cfg = base.replace(
            scene=base.scene.replace(num_envs=args.num_envs),
            sim=base.sim.replace(device=args.device),
        )
        env = TienKungLocomotionEnv(cfg=cfg)
        _stage("environment-created")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        observations = wrapper.reset()
        _stage("wrapper-reset")
        assert tuple(observations["policy"].shape) == (args.num_envs, 259)

        for _ in range(args.steps):
            if args.random_actions:
                actions = 2.0 * torch.rand((args.num_envs, 20), device=args.device) - 1.0
            else:
                actions = torch.zeros((args.num_envs, 20), device=args.device)
            observations, rewards, dones, extras = wrapper.step(actions)
            assert tuple(observations["policy"].shape) == (args.num_envs, 259)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert tuple(dones.shape) == (args.num_envs,)
            assert "time_outs" in extras

        if args.check_selective_reset:
            _check_selective_reset(env, torch)
            _stage("selective-reset-checked")
        if args.check_periodic_push:
            _check_periodic_push(env, torch)
            _stage("periodic-push-checked")

        print(
            f"FLAT_ENV_OK envs={args.num_envs} steps={args.steps} "
            f"observation_dim={observations['policy'].shape[-1]}",
            flush=True,
        )
    except BaseException as exc:
        print(f"FLAT_ENV_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.close()


def _check_selective_reset(env: TienKungLocomotionEnv, torch_module) -> None:
    if env.num_envs < 2:
        raise ValueError("Selective reset verification requires at least two environments")

    reset_ids = torch_module.tensor([0], dtype=torch_module.long, device=env.device)
    root_before = env.state.root_position.clone()
    joints_before = env.state.joint_positions.clone()
    env._reset_idx(reset_ids)

    assert torch_module.equal(env.state.root_position[1:], root_before[1:])
    assert torch_module.equal(env.state.joint_positions[1:], joints_before[1:])
    origin = env.backend.env_origins[0]
    offset = env.state.root_position[0, :2] - origin[:2]
    assert env.cfg.randomization.root_x[0] <= offset[0] <= env.cfg.randomization.root_x[1]
    assert env.cfg.randomization.root_y[0] <= offset[1] <= env.cfg.randomization.root_y[1]
    expected_height = origin[2] + env.cfg.robot.default_root_height
    assert torch_module.isclose(env.state.root_position[0, 2], expected_height, atol=1.0e-4)
    low, high = env.cfg.randomization.joint_position_scale
    lower = torch_module.minimum(
        env.default_joint_positions[0] * low, env.default_joint_positions[0] * high
    )
    upper = torch_module.maximum(
        env.default_joint_positions[0] * low, env.default_joint_positions[0] * high
    )
    assert torch_module.all(env.state.joint_positions[0] >= lower - 1.0e-4)
    assert torch_module.all(env.state.joint_positions[0] <= upper + 1.0e-4)


def _check_periodic_push(env: TienKungLocomotionEnv, torch_module) -> None:
    before = env.state.root_linear_velocity[:, :2].clone()
    env._push_time_left.zero_()
    env.apply_periodic_pushes()
    velocity_delta = env.state.root_linear_velocity[:, :2] - before
    assert torch_module.any(velocity_delta != 0.0)
    low_x, high_x = env.cfg.randomization.push_velocity_x
    low_y, high_y = env.cfg.randomization.push_velocity_y
    assert torch_module.all((velocity_delta[:, 0] >= low_x) & (velocity_delta[:, 0] <= high_x))
    assert torch_module.all((velocity_delta[:, 1] >= low_y) & (velocity_delta[:, 1] <= high_y))


def _stage(name: str) -> None:
    print(f"FLAT_ENV_STAGE {name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
