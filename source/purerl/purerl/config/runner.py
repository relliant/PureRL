"""Local RSL-RL runner configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import ConfigMixin

PRESET_DIR = Path(__file__).resolve().parent / "presets"
FLAT_RUNNER_PRESET = PRESET_DIR / "flat_runner.yaml"
ROUGH_RUNNER_PRESET = PRESET_DIR / "rough_runner.yaml"


@dataclass(frozen=True)
class ActorCriticCfg(ConfigMixin):
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (768, 256, 128)
    activation: str = "elu"


@dataclass(frozen=True)
class PpoAlgorithmCfg(ConfigMixin):
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    num_learning_epochs: int = 2
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-5
    schedule: str = "adaptive"
    gamma: float = 0.994
    lam: float = 0.9
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class OnPolicyRunnerCfg(ConfigMixin):
    class_name: str = "OnPolicyRunner"
    seed: int = 5
    device: str = "cuda:0"
    num_steps_per_env: int = 60
    max_iterations: int = 3001
    save_interval: int = 100
    experiment_name: str = "tienkung_rough"
    run_name: str = ""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"policy": ("policy",), "critic": ("policy",)}
    )
    clip_actions: float | None = None
    min_action_noise_std: float = 0.05
    max_action_noise_std: float = 3.0
    max_checkpoint_noise_std: float = 3.0
    logger: str = "wandb"
    wandb_project: str = "purerl"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()
    wandb_run_id: str | None = None
    wandb_resume: str = "never"
    neptune_project: str = "purerl"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    policy: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    algorithm: PpoAlgorithmCfg = field(default_factory=PpoAlgorithmCfg)

    def validate(self) -> None:
        if self.num_steps_per_env <= 0 or self.max_iterations <= 0:
            raise ValueError("Runner step and iteration counts must be positive")
        if self.class_name != "OnPolicyRunner":
            raise ValueError(f"Unsupported runner: {self.class_name}")
        if self.max_checkpoint_noise_std <= 0.0:
            raise ValueError("max_checkpoint_noise_std must be positive")
        if not 0.0 < self.min_action_noise_std <= self.max_action_noise_std:
            raise ValueError("Action noise bounds must satisfy 0 < minimum <= maximum")
        if "policy" not in self.obs_groups or "critic" not in self.obs_groups:
            raise ValueError("RSL-RL requires policy and critic observation groups")
        if self.logger not in {"tensorboard", "wandb", "neptune"}:
            raise ValueError(f"Unsupported logger: {self.logger}")
        if not self.wandb_project:
            raise ValueError("wandb_project cannot be empty")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported W&B mode: {self.wandb_mode}")
        if self.wandb_resume not in {"allow", "must", "never", "auto"}:
            raise ValueError(f"Unsupported W&B resume mode: {self.wandb_resume}")


def load_runner_cfg(path: str | Path) -> OnPolicyRunnerCfg:
    cfg = OnPolicyRunnerCfg.from_yaml(path)
    cfg.validate()
    return cfg


def make_rough_runner_cfg() -> OnPolicyRunnerCfg:
    return load_runner_cfg(ROUGH_RUNNER_PRESET)


def make_flat_runner_cfg() -> OnPolicyRunnerCfg:
    return load_runner_cfg(FLAT_RUNNER_PRESET)
