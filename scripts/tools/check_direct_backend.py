#!/usr/bin/env python3
"""Exercise the direct Isaac Sim backend without importing Isaac Lab."""

from __future__ import annotations

import argparse

import torch
from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.config import JOINT_NAMES, make_flat_env_cfg
from purerl.sim import IsaacSimBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    backend = None
    try:
        launcher.launch()
        cfg = make_flat_env_cfg().replace(
            scene=make_flat_env_cfg().scene.replace(num_envs=args.num_envs),
            sim=make_flat_env_cfg().sim.replace(device=args.device),
        )
        backend = IsaacSimBackend()
        backend.initialize(cfg)

        assert backend.num_envs == args.num_envs
        assert backend.joint_names == JOINT_NAMES
        assert cfg.robot.root_body_name in backend.body_names
        assert set(cfg.robot.foot_body_names).issubset(backend.body_names)
        assert tuple(backend.state.joint_positions.shape) == (args.num_envs, 20)
        assert tuple(backend.joint_limits.shape) == (args.num_envs, 20, 2)

        targets = torch.as_tensor(
            cfg.robot.default_joint_positions, dtype=torch.float32, device=args.device
        ).repeat(args.num_envs, 1)
        for step in range(args.steps):
            targets[:, 1] += 0.05 if step % 2 == 0 else -0.05
            backend.set_joint_position_targets(targets)
            backend.simulate(render=False)
        backend.refresh()

        tensors = (
            backend.state.root_position,
            backend.state.root_quaternion,
            backend.state.root_linear_velocity,
            backend.state.root_angular_velocity,
            backend.state.joint_positions,
            backend.state.joint_velocities,
            backend.state.joint_torques,
            backend.state.body_positions,
            backend.state.body_linear_velocities,
            backend.state.net_contact_forces,
        )
        assert all(torch.isfinite(value).all() for value in tensors)
        assert tuple(backend.state.body_positions.shape) == (
            args.num_envs,
            len(backend.body_names),
            3,
        )

        before = backend.state.root_position.clone()
        backend.reset(torch.tensor([0], device=args.device))
        backend.refresh()
        if args.num_envs > 1:
            assert torch.equal(backend.state.root_position[1:], before[1:])
        expected_height = backend.env_origins[0, 2] + cfg.robot.default_root_height
        assert torch.isclose(backend.state.root_position[0, 2], expected_height, atol=1.0e-4)

        print(
            f"DIRECT_BACKEND_OK envs={args.num_envs} "
            f"dofs={len(backend.joint_names)} bodies={len(backend.body_names)}"
        )
        print("JOINT_NAMES", backend.joint_names)
        print("BODY_NAMES", backend.body_names)
    finally:
        if backend is not None:
            backend.close()
        launcher.close()


if __name__ == "__main__":
    main()
