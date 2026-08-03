from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict

from purerl.config import make_flat_runner_cfg  # noqa: E402
from purerl.rl import RslRlVecEnvWrapper  # noqa: E402


class TorchEnv:
    def __init__(self):
        self.num_envs = 2
        self.num_actions = 20
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.device = "cpu"
        self.cfg = SimpleNamespace(name="test")

    def get_observations(self):
        return {"policy": torch.zeros((self.num_envs, 259))}

    def reset(self):
        return self.get_observations(), {}

    def step(self, actions):
        del actions
        return (
            self.get_observations(),
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def close(self):
        pass


def test_wrapper_returns_rsl_rl_3_tensor_dict():
    wrapper = RslRlVecEnvWrapper(TorchEnv())
    observations = wrapper.get_observations()
    assert isinstance(observations, TensorDict)
    assert observations.batch_size == torch.Size([2])
    assert observations["policy"].shape == (2, 259)


def test_local_runner_config_constructs_rsl_rl_3_runner():
    from rsl_rl.runners import OnPolicyRunner

    wrapper = RslRlVecEnvWrapper(TorchEnv())
    cfg = make_flat_runner_cfg().replace(device="cpu").to_dict()
    runner = OnPolicyRunner(wrapper, cfg, log_dir=None, device="cpu")

    assert runner.env is wrapper
    assert runner.alg.policy.actor[0].in_features == 259
    assert runner.alg.policy.actor[-1].out_features == 20
