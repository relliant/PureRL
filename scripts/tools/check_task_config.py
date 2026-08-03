#!/usr/bin/env python3
"""Validate local PureRL task registrations and configuration contracts."""

from __future__ import annotations

import gymnasium as gym
from purerl import (
    ACTION_DIM,
    OBSERVATION_DIM,
    TASK_IDS,
    get_task_spec,
    register_gymnasium_tasks,
)


def main() -> None:
    register_gymnasium_tasks()
    registered = tuple(task_id for task_id in TASK_IDS if task_id in gym.registry)
    assert registered == TASK_IDS

    for task_id in TASK_IDS:
        spec = get_task_spec(task_id)
        env_cfg = spec.make_env_cfg()
        runner_cfg = spec.make_runner_cfg()
        env_cfg.validate()
        runner_cfg.validate()
        assert env_cfg.actions.dimension == ACTION_DIM
        assert env_cfg.observations.dimension == OBSERVATION_DIM
        assert env_cfg.sim.dt == 0.005
        assert env_cfg.sim.decimation == 4
        assert gym.spec(task_id).entry_point == spec.env_entry_point

    print(f"TASK_CONFIG_OK tasks={len(registered)} actions={ACTION_DIM} observations={OBSERVATION_DIM}")


if __name__ == "__main__":
    main()
