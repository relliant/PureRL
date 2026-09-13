#!/usr/bin/env python3
"""Check overlapping-clone isolation and nonzero bounded PD reward torques."""

from __future__ import annotations

import argparse
import json
import math

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.config import make_flat_env_cfg
from purerl.sim import IsaacSimBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 32:
        parser.error("--steps must be at least 32 to exercise ground contact")

    import torch

    # Fail before starting Kit if the host driver cannot create a CUDA context.
    torch.zeros(1, device=args.device)
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    backend = None
    try:
        launcher.launch()
        base = make_flat_env_cfg()
        cfg = base.replace(scene=base.scene.replace(num_envs=3), sim=base.sim.replace(device=args.device))
        backend = IsaacSimBackend()
        backend.initialize(cfg)
        assert backend.collision_filtering_mode == "collision_groups"

        # Clones 0/1 occupy exactly the same world pose; clone 2 is an isolated
        # control. Without filtering the overlapping clones contact each other.
        poses, orientations = backend._articulation.get_world_poses(clone=True)
        poses[1] = poses[0]
        poses[2] = poses[0]
        poses[2, 0] += 20.0
        reference_positions = poses.clone()
        backend._articulation.set_world_poses(poses, orientations)
        defaults = torch.tensor(cfg.robot.default_joint_positions, device=args.device).repeat(3, 1)
        max_pose_difference = 0.0
        peak_contact = 0.0
        peak_torque = 0.0
        for step in range(args.steps):
            targets = defaults.clone()
            targets[:, 0] += 0.02 * math.sin(step * 0.1)
            backend.set_joint_position_targets(targets)
            backend.simulate(render=False)
            backend.refresh()
            state = backend.state
            displacement = state.root_position - reference_positions
            difference = float((displacement[:2] - displacement[2:]).abs().max())
            max_pose_difference = max(max_pose_difference, difference)
            peak_contact = max(peak_contact, float(state.net_contact_forces.abs().max()))
            peak_torque = max(peak_torque, float(state.joint_torques.abs().max()))
            assert torch.isfinite(state.root_position).all()
            assert torch.isfinite(state.joint_torques).all()
            limits = torch.tensor(cfg.robot.effort_limits, device=args.device)
            assert (state.joint_torques.abs() <= limits + 1e-4).all()

        assert max_pose_difference < 0.02, f"Overlapping clones diverged from control: {max_pose_difference} m"
        assert peak_contact > 1.0, "Shared ground contact was lost"
        assert peak_torque > 1.0, "Position drives produced no reward torque estimate"
        print(
            "TRAINING_PHYSICS_OK "
            + json.dumps(
                {
                    "steps": args.steps,
                    "max_pose_difference_m": max_pose_difference,
                    "peak_contact_N": peak_contact,
                    "peak_estimated_torque_Nm": peak_torque,
                }
            ),
            flush=True,
        )
    finally:
        if backend is not None:
            backend.close()
        launcher.close()


if __name__ == "__main__":
    main()
