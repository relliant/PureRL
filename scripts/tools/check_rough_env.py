#!/usr/bin/env python3
"""Run a short end-to-end rough locomotion environment smoke test."""

from __future__ import annotations

import argparse
import sys
import traceback

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.config import make_rough_env_cfg
from purerl.envs import TienKungLocomotionEnv
from purerl.rl import RslRlVecEnvWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--terrain-rows", type=int, default=2)
    parser.add_argument("--terrain-cols", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    env = None
    try:
        _stage("launching")
        launcher.launch()
        _stage("launched")
        import torch

        base = make_rough_env_cfg()
        cfg = base.replace(
            scene=base.scene.replace(num_envs=args.num_envs),
            sim=base.sim.replace(device=args.device),
            terrain=base.terrain.replace(
                num_rows=args.terrain_rows,
                num_cols=args.terrain_cols,
                max_initial_level=min(1, args.terrain_rows - 1),
            ),
        )
        env = TienKungLocomotionEnv(cfg=cfg)
        _stage("environment-created")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        observations = wrapper.reset()
        _stage("wrapper-reset")

        assert tuple(observations["policy"].shape) == (args.num_envs, 259)
        assert tuple(env.backend.env_origins.shape) == (args.num_envs, 3)
        assert tuple(env.backend.terrain_levels.shape) == (args.num_envs,)
        assert torch.isfinite(env.height_scan).all()
        assert torch.equal(
            env.backend.terrain_tile_indices,
            env.backend.terrain_levels * args.terrain_cols + env.backend.terrain_columns,
        )

        for _ in range(args.steps):
            actions = torch.zeros((args.num_envs, 20), device=args.device)
            observations, rewards, dones, extras = wrapper.step(actions)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert tuple(dones.shape) == (args.num_envs,)
            assert "time_outs" in extras

        print(
            f"ROUGH_ENV_OK envs={args.num_envs} steps={args.steps} "
            f"terrain={args.terrain_rows}x{args.terrain_cols} "
            f"observation_dim={observations['policy'].shape[-1]}",
            flush=True,
        )
    except BaseException as exc:
        print(f"ROUGH_ENV_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.close()


def _stage(name: str) -> None:
    print(f"ROUGH_ENV_STAGE {name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
