#!/usr/bin/env python3
"""Start Isaac Sim and validate PureRL task configuration construction."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym

import purerl  # noqa: F401, E402
from purerl.tasks.locomotion.agents.rsl_rl_ppo_cfg import TienKungRoughPPORunnerCfg  # noqa: E402
from purerl.tasks.locomotion.flat_env_cfg import TienKungFlatEnvCfg  # noqa: E402
from purerl.tasks.locomotion.rough_env_cfg import TienKungRoughEnvCfg  # noqa: E402


def main():
    tasks = sorted(task_id for task_id in gym.registry if task_id.startswith("PureRL-"))
    rough = TienKungRoughEnvCfg()
    flat = TienKungFlatEnvCfg()
    runner = TienKungRoughPPORunnerCfg()

    assert len(tasks) == 4
    assert len(rough.actions.joint_pos.joint_names) == 20
    assert rough.sim.dt == 0.005 and rough.decimation == 4
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.scene.height_scanner is not None

    print(f"Registered tasks: {tasks}", flush=True)
    print(
        f"Rough task: {rough.scene.num_envs} envs, "
        f"{len(rough.scene.terrain.terrain_generator.sub_terrains)} terrain types, "
        f"{len(rough.actions.joint_pos.joint_names)} actions",
        flush=True,
    )
    print(f"PPO: {runner.experiment_name}, {runner.max_iterations} iterations", flush=True)
    print("Configuration validation: OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
