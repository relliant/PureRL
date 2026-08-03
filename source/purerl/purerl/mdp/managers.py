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


class ObservationManager:
    def __init__(self, terms: tuple[ObservationTermSpec, ...], *, expected_dimension: int):
        self.terms = terms
        self.expected_dimension = expected_dimension
        _validate_unique_names(term.name for term in terms)

    def compute(self, context: Any) -> Any:
        values = [term.func(context, **term.params) for term in self.terms]
        observation = concatenate(values)
        if observation.shape[-1] != self.expected_dimension:
            raise ValueError(
                f"Observation manager produced {observation.shape[-1]} values, "
                f"expected {self.expected_dimension}"
            )
        return observation


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
