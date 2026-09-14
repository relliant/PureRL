from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict

from purerl.config import make_flat_runner_cfg  # noqa: E402
from purerl.contracts import OBSERVATION_DIM  # noqa: E402
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
        return {"policy": torch.zeros((self.num_envs, OBSERVATION_DIM))}

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
    assert observations["policy"].shape == (2, OBSERVATION_DIM)


def test_local_runner_config_constructs_rsl_rl_3_runner():
    from rsl_rl.runners import OnPolicyRunner

    wrapper = RslRlVecEnvWrapper(TorchEnv())
    cfg = make_flat_runner_cfg().replace(device="cpu").to_dict()
    runner = OnPolicyRunner(wrapper, cfg, log_dir=None, device="cpu")

    assert runner.env is wrapper
    assert runner.alg.policy.actor[0].in_features == OBSERVATION_DIM
    assert runner.alg.policy.actor[-1].out_features == 20


def test_real_runner_omits_empty_episode_metrics_and_logs_completed_episodes(tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from test_tienkung_env import make_env

    env = make_env(play=True)
    env.max_episode_length = 5
    wrapped = RslRlVecEnvWrapper(env)
    cfg = make_flat_runner_cfg()
    cfg = cfg.replace(
        device="cpu", logger="tensorboard", num_steps_per_env=2,
        policy=cfg.policy.replace(actor_hidden_dims=(16,), critic_hidden_dims=(16,)),
        algorithm=cfg.algorithm.replace(num_mini_batches=1, num_learning_epochs=1),
    )
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=str(tmp_path), device="cpu")
    try:
        # No episode completes in iterations 0/1. Both finish in iteration 2.
        runner.learn(num_learning_iterations=3, init_at_random_ep_len=False)
        runner.writer.flush()
        accumulator = EventAccumulator(str(tmp_path)).Reload()
        for tag in ("Gait/flight_fraction", "Episode/feet_contact_number", "Termination/time_out"):
            events = accumulator.Scalars(tag)
            assert [event.step for event in events] == [2]
            assert all(torch.isfinite(torch.tensor(event.value)) for event in events)
    finally:
        if runner.writer is not None:
            runner.writer.close()
        wrapped.close()


@pytest.mark.parametrize("reason", ["terminated", "truncated"])
def test_auto_reset_preserves_ppo_actions_and_old_log_prob(reason):
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from test_tienkung_env import make_env

    env = make_env(play=True)
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    torch.manual_seed(5)
    policy = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        20,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    ppo = PPO(policy, device="cpu")
    ppo.init_storage("rl", env.num_envs, 1, obs, [20])
    if reason == "terminated":
        env.state.net_contact_forces[0, env._root_body_index, 2] = 20.0
    else:
        env.episode_length_buf[0] = env.max_episode_length - 1

    with torch.inference_mode():
        actions = ppo.act(obs)
        executed = actions.clone()
        next_obs, reward, done, extras = wrapped.step(actions)
        assert done.tolist() == [True, False]
        assert bool(extras["time_outs"][0]) == (reason == "truncated")
        ppo.process_env_step(next_obs, reward, done, extras)
        assert torch.equal(actions, executed)
        assert torch.equal(ppo.storage.actions[0], executed)
        policy.act(obs)
        log_prob = policy.get_actions_log_prob(ppo.storage.actions[0])
        ratio = (log_prob - ppo.storage.actions_log_prob[0, :, 0]).exp()
        assert torch.equal(ratio, torch.ones(env.num_envs))
