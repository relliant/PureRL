"""Local RSL-RL runner configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ConfigMixin


@dataclass(frozen=True)
class ActorCriticCfg(ConfigMixin):
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"


@dataclass(frozen=True)
class PpoAlgorithmCfg(ConfigMixin):
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class OnPolicyRunnerCfg(ConfigMixin):
    class_name: str = "OnPolicyRunner"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 100
    experiment_name: str = "tienkung_rough"
    run_name: str = ""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"policy": ("policy",), "critic": ("policy",)}
    )
    clip_actions: float | None = None
    logger: str = "tensorboard"
    wandb_project: str = "purerl"
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
        if "policy" not in self.obs_groups or "critic" not in self.obs_groups:
            raise ValueError("RSL-RL requires policy and critic observation groups")


def make_rough_runner_cfg() -> OnPolicyRunnerCfg:
    cfg = OnPolicyRunnerCfg()
    cfg.validate()
    return cfg


def make_flat_runner_cfg() -> OnPolicyRunnerCfg:
    cfg = OnPolicyRunnerCfg(max_iterations=2500, experiment_name="tienkung_flat")
    cfg.validate()
    return cfg
