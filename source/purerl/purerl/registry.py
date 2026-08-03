"""Local task registry independent of Isaac Lab and Hydra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config.env import (
    EnvCfg,
    make_flat_env_cfg,
    make_flat_play_env_cfg,
    make_rough_env_cfg,
    make_rough_play_env_cfg,
)
from .config.runner import OnPolicyRunnerCfg, make_flat_runner_cfg, make_rough_runner_cfg
from .contracts import TASK_IDS

EnvCfgFactory = Callable[[], EnvCfg]
RunnerCfgFactory = Callable[[], OnPolicyRunnerCfg]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_cfg_factory: EnvCfgFactory
    runner_cfg_factory: RunnerCfgFactory
    env_entry_point: str = "purerl.envs.tienkung_locomotion:TienKungLocomotionEnv"

    def make_env_cfg(self) -> EnvCfg:
        return self.env_cfg_factory()

    def make_runner_cfg(self) -> OnPolicyRunnerCfg:
        return self.runner_cfg_factory()


_TASK_SPECS = {
    TASK_IDS[0]: TaskSpec(TASK_IDS[0], make_flat_env_cfg, make_flat_runner_cfg),
    TASK_IDS[1]: TaskSpec(TASK_IDS[1], make_flat_play_env_cfg, make_flat_runner_cfg),
    TASK_IDS[2]: TaskSpec(TASK_IDS[2], make_rough_env_cfg, make_rough_runner_cfg),
    TASK_IDS[3]: TaskSpec(TASK_IDS[3], make_rough_play_env_cfg, make_rough_runner_cfg),
}


def list_task_ids() -> tuple[str, ...]:
    return tuple(_TASK_SPECS)


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return _TASK_SPECS[task_id]
    except KeyError as exc:
        available = ", ".join(list_task_ids())
        raise KeyError(f"Unknown PureRL task {task_id!r}. Available tasks: {available}") from exc


def register_gymnasium_tasks() -> None:
    """Register PureRL tasks when Gymnasium is available."""

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Gymnasium is required to register PureRL environments") from exc

    for spec in _TASK_SPECS.values():
        if spec.task_id in gym.registry:
            continue
        gym.register(
            id=spec.task_id,
            entry_point=spec.env_entry_point,
            disable_env_checker=True,
            kwargs={"task_id": spec.task_id},
        )
