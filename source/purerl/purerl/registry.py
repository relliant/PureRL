"""Local task registry independent of Isaac Lab and Hydra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config.env import (
    FLAT_ENV_PRESET,
    FLAT_PLAY_ENV_PRESET,
    ROUGH_ENV_PRESET,
    ROUGH_PLAY_ENV_PRESET,
    EnvCfg,
    load_env_cfg,
)
from .config.runner import (
    FLAT_RUNNER_PRESET,
    ROUGH_RUNNER_PRESET,
    OnPolicyRunnerCfg,
    load_runner_cfg,
)
from .contracts import TASK_IDS


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_config_path: Path
    runner_config_path: Path
    env_entry_point: str = "purerl.envs.tienkung_locomotion:TienKungLocomotionEnv"

    def make_env_cfg(self, config_path: str | Path | None = None) -> EnvCfg:
        return load_env_cfg(self.env_config_path if config_path is None else config_path)

    def make_runner_cfg(self, config_path: str | Path | None = None) -> OnPolicyRunnerCfg:
        return load_runner_cfg(self.runner_config_path if config_path is None else config_path)


_TASK_SPECS = {
    TASK_IDS[0]: TaskSpec(TASK_IDS[0], FLAT_ENV_PRESET, FLAT_RUNNER_PRESET),
    TASK_IDS[1]: TaskSpec(TASK_IDS[1], FLAT_PLAY_ENV_PRESET, FLAT_RUNNER_PRESET),
    TASK_IDS[2]: TaskSpec(TASK_IDS[2], ROUGH_ENV_PRESET, ROUGH_RUNNER_PRESET),
    TASK_IDS[3]: TaskSpec(TASK_IDS[3], ROUGH_PLAY_ENV_PRESET, ROUGH_RUNNER_PRESET),
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
