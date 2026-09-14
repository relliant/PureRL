"""Duck-typed RSL-RL vector environment adapter."""

from __future__ import annotations

from typing import Any

from purerl.mdp._array import clip


class RslRlVecEnvWrapper:
    """Expose PureRL's Gymnasium-style lifecycle using RSL-RL's VecEnv contract."""

    def __init__(self, env: Any, *, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env

    def get_observations(self) -> dict[str, Any]:
        return self._to_tensor_dict(self.env.get_observations())

    def reset(self) -> dict[str, Any]:
        observations, _ = self.env.reset()
        return self._to_tensor_dict(observations)

    def step(self, actions: Any) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = clip(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        extras["time_outs"] = truncated
        # RSL averages every collected episode tensor. Publishing only empty
        # tensors for an entire rollout produces NaN episode/gait metrics.
        if "terminal_env_ids" in extras and len(extras["terminal_env_ids"]) == 0:
            extras.pop("episode", None)
        return self._to_tensor_dict(observations), rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _to_tensor_dict(self, observations: Any) -> Any:
        if type(observations).__module__.startswith("tensordict"):
            return observations
        if not isinstance(observations, dict) or not observations:
            raise TypeError("RSL-RL observations must be a non-empty mapping")
        first_value = next(iter(observations.values()))
        if not type(first_value).__module__.startswith("torch"):
            return observations
        try:
            from tensordict import TensorDict
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("tensordict is required by rsl-rl-lib 3.1.2") from exc
        return TensorDict(observations, batch_size=[self.num_envs], device=self.device)
