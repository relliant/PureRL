"""Small typed managers for the PureRL locomotion MDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ._array import concatenate, zeros_like

TermCallable = Callable[..., Any]


@dataclass(frozen=True)
class ObservationTermSpec:
    name: str
    func: TermCallable
    params: Mapping[str, Any] = field(default_factory=dict)
    noise: tuple[float, float] | None = None
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for label, value_range in (("noise", self.noise), ("clip", self.clip)):
            if value_range is not None and value_range[0] > value_range[1]:
                raise ValueError(f"Observation term {self.name!r} has an invalid {label} range")


class ObservationManager:
    def __init__(
        self,
        terms: tuple[ObservationTermSpec, ...],
        *,
        expected_dimension: int,
        enable_corruption: bool = False,
        seed: int = 0,
    ):
        self.terms = terms
        self.expected_dimension = expected_dimension
        self.enable_corruption = enable_corruption
        self._seed = seed
        self._torch_generators: dict[str, Any] = {}
        self._numpy_generator: Any = None
        _validate_unique_names(term.name for term in terms)

    def compute(self, context: Any) -> Any:
        values = []
        for term in self.terms:
            value = term.func(context, **term.params)
            if self.enable_corruption and term.noise is not None:
                value = value + self._uniform_noise(value, term.noise)
            if term.clip is not None:
                value = _clip(value, term.clip)
            values.append(value)
        observation = concatenate(values)
        if observation.shape[-1] != self.expected_dimension:
            raise ValueError(
                f"Observation manager produced {observation.shape[-1]} values, "
                f"expected {self.expected_dimension}"
            )
        return observation

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._torch_generators.clear()
        self._numpy_generator = None

    def _uniform_noise(self, value: Any, value_range: tuple[float, float]) -> Any:
        low, high = value_range
        module = type(value).__module__
        if module.startswith("torch"):
            import torch

            device = str(value.device)
            generator = self._torch_generators.get(device)
            if generator is None:
                generator = torch.Generator(device=value.device)
                generator.manual_seed(self._seed)
                self._torch_generators[device] = generator
            samples = torch.rand(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                generator=generator,
            )
        else:
            import numpy as np

            if self._numpy_generator is None:
                self._numpy_generator = np.random.default_rng(self._seed)
            samples = self._numpy_generator.random(value.shape, dtype=value.dtype)
        return low + (high - low) * samples


@dataclass(frozen=True)
class RewardTermSpec:
    name: str
    func: TermCallable
    weight: float
    params: Mapping[str, Any] = field(default_factory=dict)


class RewardManager:
    def __init__(self, terms: tuple[RewardTermSpec, ...], *, dt: float):
        if dt <= 0:
            raise ValueError("Reward integration dt must be positive")
        self.terms = terms
        self.dt = dt
        self.last_raw: dict[str, Any] = {}
        self.last_weighted: dict[str, Any] = {}
        self.episode_sums: dict[str, Any] = {}
        _validate_unique_names(term.name for term in terms)

    def compute(self, context: Any) -> Any:
        total = None
        for term in self.terms:
            raw = term.func(context, **term.params)
            weighted = raw * term.weight * self.dt
            self.last_raw[term.name] = raw
            self.last_weighted[term.name] = weighted
            if term.name not in self.episode_sums:
                self.episode_sums[term.name] = zeros_like(weighted)
            self.episode_sums[term.name] += weighted
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("RewardManager requires at least one enabled term")
        return total

    def reset(self, env_ids: Any) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        for name, values in self.episode_sums.items():
            completed[name] = values[env_ids].copy() if hasattr(values[env_ids], "copy") else values[env_ids].clone()
            values[env_ids] = 0.0
        return completed


@dataclass(frozen=True)
class TerminationTermSpec:
    name: str
    func: TermCallable
    time_out: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)


class TerminationManager:
    def __init__(self, terms: tuple[TerminationTermSpec, ...]):
        self.terms = terms
        self.last_values: dict[str, Any] = {}
        _validate_unique_names(term.name for term in terms)

    def compute(self, context: Any) -> tuple[Any, Any]:
        terminated = None
        truncated = None
        for term in self.terms:
            value = term.func(context, **term.params)
            self.last_values[term.name] = value
            if term.time_out:
                truncated = value if truncated is None else truncated | value
            else:
                terminated = value if terminated is None else terminated | value
        template = next(iter(self.last_values.values()), None)
        if template is None:
            raise RuntimeError("TerminationManager requires at least one term")
        if terminated is None:
            terminated = template & False
        if truncated is None:
            truncated = template & False
        return terminated, truncated


@dataclass(frozen=True)
class EventTermSpec:
    name: str
    func: TermCallable
    mode: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"startup", "reset", "interval"}:
            raise ValueError(f"Unsupported event mode: {self.mode}")


class EventManager:
    def __init__(self, terms: tuple[EventTermSpec, ...]):
        self.terms = terms
        _validate_unique_names(term.name for term in terms)

    def run(self, mode: str, context: Any, *, env_ids: Any = None) -> None:
        for term in self.terms:
            if term.mode == mode:
                term.func(context, env_ids=env_ids, **term.params)


def _validate_unique_names(names: Any) -> None:
    names_tuple = tuple(names)
    if len(names_tuple) != len(set(names_tuple)):
        raise ValueError("Manager term names must be unique")


def _clip(value: Any, value_range: tuple[float, float]) -> Any:
    low, high = value_range
    if type(value).__module__.startswith("torch"):
        return value.clamp(min=low, max=high)

    import numpy as np

    return np.clip(value, low, high)
