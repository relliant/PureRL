"""Contact history derived from batched net-force tensors."""

from __future__ import annotations

from typing import Any

from purerl.mdp._array import norm


class ContactHistory:
    """Track contacts and air time without simulator-specific objects."""

    def __init__(self, template: Any, *, force_threshold: float = 1.0):
        if len(template.shape) != 2:
            raise ValueError("Contact template must have shape (num_envs, num_bodies)")
        self.force_threshold = force_threshold
        self.in_contact = template > force_threshold
        self.first_contact = self.in_contact & False
        self.current_air_time = template * 0.0
        self.last_air_time = template * 0.0

    def update(self, net_forces: Any, dt: float) -> None:
        if net_forces.shape[:-1] != self.in_contact.shape or net_forces.shape[-1] != 3:
            raise ValueError("Net forces must have shape (num_envs, num_bodies, 3)")
        if dt <= 0:
            raise ValueError("Sensor update dt must be positive")

        contact = norm(net_forces) > self.force_threshold
        self.first_contact = contact & ~self.in_contact
        completed_air_time = self.current_air_time + dt
        self.last_air_time = _where(self.first_contact, completed_air_time, self.last_air_time)
        self.current_air_time = _where(contact, self.current_air_time * 0.0, completed_air_time)
        self.in_contact = contact

    def reset(self, env_ids: Any) -> None:
        self.in_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.current_air_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0


def _where(condition: Any, true_value: Any, false_value: Any) -> Any:
    if type(condition).__module__.startswith("torch"):
        import torch

        return torch.where(condition, true_value, false_value)
    import numpy as np

    return np.where(condition, true_value, false_value)
