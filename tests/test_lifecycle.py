from types import SimpleNamespace

import numpy as np
from purerl.config.env import make_flat_env_cfg
from purerl.envs.base import BaseVecEnv
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.managers import (
    EventManager,
    EventTermSpec,
    ObservationManager,
    ObservationTermSpec,
    RewardManager,
    RewardTermSpec,
    TerminationManager,
    TerminationTermSpec,
)
from purerl.rl import RslRlVecEnvWrapper


class FakeBackend:
    def __init__(self, num_envs, log):
        self.num_envs = num_envs
        self.device = "cpu"
        self.log = log
        self.all_env_ids = np.arange(num_envs)
        self.state = SimpleNamespace(value=np.zeros(num_envs))

    def initialize(self, cfg):
        self.log.append("initialize")

    def zeros(self, shape, *, dtype):
        return np.zeros(shape, dtype=np.int64 if dtype == "long" else np.float32)

    def set_joint_position_targets(self, targets):
        self.log.append("target")

    def simulate(self, *, render):
        self.log.append(("simulate", render))
        self.state.value += 1.0

    def configure_viewer(self):
        self.log.append("configure_viewer")

    def render_rgb(self):
        self.log.append("render_rgb")
        return np.full((3, 4, 3), 127, dtype=np.uint8)

    def refresh(self):
        self.log.append("refresh")

    def reset(self, env_ids):
        self.log.append("reset")
        self.state.value[env_ids] = 0.0

    def nonzero(self, mask):
        return np.flatnonzero(mask)

    def clone(self, value):
        return value.copy()

    def close(self):
        self.log.append("close")


def make_env(num_envs=2, *, render_mode=None):
    log = []
    cfg = make_flat_env_cfg().replace(
        scene=make_flat_env_cfg().scene.replace(num_envs=num_envs),
        sim=make_flat_env_cfg().sim.replace(device="cpu"),
    )
    backend = FakeBackend(num_envs, log)
    defaults = np.zeros((num_envs, 20))
    action_manager = JointPositionActionManager(defaults)
    observations = ObservationManager(
        (ObservationTermSpec("value", lambda env: env.state.value[:, None]),),
        expected_dimension=1,
    )
    rewards = RewardManager(
        (RewardTermSpec("value", lambda env: env.state.value, weight=1.0),),
        dt=cfg.sim.step_dt,
    )
    terminations = TerminationManager(
        (
            TerminationTermSpec("done", lambda env: env.state.value >= 4.0),
            TerminationTermSpec(
                "timeout",
                lambda env: env.episode_length_buf >= 10,
                time_out=True,
            ),
        )
    )
    events = EventManager(
        (
            EventTermSpec("startup", lambda env, env_ids: log.append("startup"), "startup"),
            EventTermSpec("on_reset", lambda env, env_ids: log.append("reset_event"), "reset"),
            EventTermSpec("interval", lambda env, env_ids: log.append("interval"), "interval"),
        )
    )
    env = BaseVecEnv(
        cfg,
        backend,
        action_manager=action_manager,
        observation_manager=observations,
        reward_manager=rewards,
        termination_manager=terminations,
        event_manager=events,
        render_mode=render_mode,
    )
    return env, log


def test_step_order_and_auto_reset_are_explicit():
    env, log = make_env()
    log.clear()

    observations, rewards, terminated, truncated, info = env.step(np.zeros((2, 20)))

    assert log[:8] == ["target", ("simulate", False)] * 4
    assert log[8:] == ["refresh", "reset", "reset_event", "refresh", "interval"]
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert rewards.tolist() == [0.08, 0.08]
    assert info["terminal_observation"].tolist() == [[4.0], [4.0]]
    assert observations["policy"].tolist() == [[0.0], [0.0]]


def test_rsl_rl_wrapper_merges_termination_and_timeout():
    env, _ = make_env()
    wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)

    observations, rewards, dones, extras = wrapper.step(np.full((2, 20), 2.0))

    assert observations["policy"].shape == (2, 1)
    assert rewards.shape == dones.shape == (2,)
    assert dones.tolist() == [True, True]
    assert extras["time_outs"].tolist() == [False, False]


def test_close_is_idempotent():
    env, log = make_env()
    env.close()
    env.close()
    assert log.count("close") == 1


def test_rgb_array_render_delegates_to_backend():
    env, log = make_env(render_mode="rgb_array")

    frame = env.render()

    assert frame.shape == (3, 4, 3)
    assert frame.dtype == np.uint8
    assert env.metadata["render_fps"] == 50
    assert log[-1] == "render_rgb"


def test_human_render_mode_configures_viewer_and_submits_render_steps():
    env, log = make_env(render_mode="human")
    assert "configure_viewer" in log
    log.clear()

    env.step(np.zeros((2, 20)))

    simulate_calls = [entry for entry in log if isinstance(entry, tuple)]
    assert simulate_calls == [
        ("simulate", False),
        ("simulate", False),
        ("simulate", False),
        ("simulate", True),
    ]
    assert env.render() is None
