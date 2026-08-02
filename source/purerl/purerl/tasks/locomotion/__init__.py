"""Gym registrations for TienKung velocity locomotion."""

import gymnasium as gym

from . import agents


_TASKS = {
    "PureRL-Velocity-Rough-TienKung-v0": ("rough_env_cfg:TienKungRoughEnvCfg", "TienKungRoughPPORunnerCfg"),
    "PureRL-Velocity-Rough-TienKung-Play-v0": (
        "rough_env_cfg:TienKungRoughEnvCfg_PLAY",
        "TienKungRoughPPORunnerCfg",
    ),
    "PureRL-Velocity-Flat-TienKung-v0": ("flat_env_cfg:TienKungFlatEnvCfg", "TienKungFlatPPORunnerCfg"),
    "PureRL-Velocity-Flat-TienKung-Play-v0": (
        "flat_env_cfg:TienKungFlatEnvCfg_PLAY",
        "TienKungFlatPPORunnerCfg",
    ),
}

for task_id, (env_cfg, runner_cfg) in _TASKS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )

