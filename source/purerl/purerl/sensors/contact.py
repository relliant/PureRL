"""Contact history derived from batched net-force tensors."""

from __future__ import annotations

from typing import Any

from purerl.mdp._array import maximum, norm, sum_axis


class ContactHistory:
    """Track contacts and air time without simulator-specific objects."""

    def __init__(self, template: Any, *, force_threshold: float = 1.0, release_time: float = 0.0):
        if len(template.shape) != 2:
            raise ValueError("Contact template must have shape (num_envs, num_bodies)")
        self.force_threshold = force_threshold
        if release_time < 0.0:
            raise ValueError("Contact release time must be non-negative")
        self.release_time = release_time
        self.raw_contact = template > force_threshold
        self.raw_air_time = template * 0.0
        self.in_contact = template > force_threshold
        self.first_contact = self.in_contact & False
        # A policy step contains several physics steps. Keep contact events
        # latched until the environment has consumed the reward for the step.
        self.contact_events = (
            self.first_contact.clone()
            if type(self.first_contact).__module__.startswith("torch")
            else self.first_contact.copy()
        )
        self.current_air_time = template * 0.0
        self.current_contact_time = template * 0.0
        self.last_air_time = template * 0.0

    def update(self, net_forces: Any, dt: float) -> None:
        if net_forces.shape[:-1] != self.in_contact.shape or net_forces.shape[-1] != 3:
            raise ValueError("Net forces must have shape (num_envs, num_bodies, 3)")
        if dt <= 0:
            raise ValueError("Sensor update dt must be positive")

        self.raw_contact = norm(net_forces) > self.force_threshold
        self.raw_air_time = _where(self.raw_contact, self.raw_air_time * 0.0, self.raw_air_time + dt)
        # Bridge brief force dropouts, but retain their elapsed air time when
        # the loss of contact lasts long enough to count as a swing.
        contact = self.raw_contact | (self.in_contact & (self.raw_air_time + 1e-7 < self.release_time))
        self.first_contact = contact & ~self.in_contact
        self.contact_events |= self.first_contact
        completed_air_time = maximum(self.current_air_time + dt, self.raw_air_time)
        completed_contact_time = self.current_contact_time + dt
        self.last_air_time = _where(self.first_contact, completed_air_time, self.last_air_time)
        self.current_air_time = _where(contact, self.current_air_time * 0.0, completed_air_time)
        self.current_contact_time = _where(
            contact,
            completed_contact_time,
            self.current_contact_time * 0.0,
        )
        self.in_contact = contact

    def reset(self, env_ids: Any) -> None:
        self.raw_contact[env_ids] = False
        self.raw_air_time[env_ids] = 0.0
        self.in_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.contact_events[env_ids] = False
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0

    def clear_events(self) -> None:
        """Clear contact transitions after the current policy reward is computed."""

        self.first_contact[...] = False
        self.contact_events[...] = False


class BipedContactHistory(ContactHistory):
    """Latch supported, alternating landings and integrate flight at physics rate."""

    def __init__(
        self,
        template: Any,
        *,
        force_threshold: float = 1.0,
        release_time: float = 0.01,
        support_time: float = 0.04,
        min_air_time: float = 0.12,
        flight_grace_time: float = 0.01,
    ):
        if template.shape[-1] != 2:
            raise ValueError("Biped contact history requires exactly two feet")
        super().__init__(template, force_threshold=force_threshold, release_time=release_time)
        self.support_time = support_time
        self.min_air_time = min_air_time
        self.flight_grace_time = flight_grace_time
        self.valid_landing_events = self.in_contact & False
        self.last_landed_foot = self.in_contact & False
        self.valid_landing_air_time = template * 0.0
        self.flight_time = template[:, 0] * 0.0
        self.step_flight_time = template[:, 0] * 0.0
        self.step_unsupported_time = template[:, 0] * 0.0
        self.step_single_support_time = template[:, 0] * 0.0

    def update(self, net_forces: Any, dt: float) -> None:
        super().update(net_forces, dt)
        other_support_time = self.current_contact_time[:, [1, 0]]
        # The other foot must support the entire swing, not just touch down
        # slightly earlier after a hop. A repeated landing of the same foot
        # cannot keep earning rewards while the other foot stays planted.
        valid = (
            self.first_contact
            & self.raw_contact[:, [1, 0]]
            & (other_support_time + 1e-7 >= maximum(self.last_air_time, self.support_time))
            & (self.last_air_time > self.min_air_time)
            & ~self.last_landed_foot
        )
        valid &= (sum_axis(self.first_contact) == 1)[:, None]
        self.valid_landing_events |= valid
        self.valid_landing_air_time = maximum(
            self.valid_landing_air_time, _where(valid, self.last_air_time, self.last_air_time * 0.0)
        )
        self.last_landed_foot = _where((sum_axis(valid) > 0)[:, None], valid, self.last_landed_foot)
        count = sum_axis(self.raw_contact)
        self.flight_time = _where(count == 0, self.flight_time + dt, self.flight_time * 0.0)
        self.step_flight_time += (count == 0) * dt
        self.step_unsupported_time += (self.flight_time > self.flight_grace_time + 1e-7) * dt
        self.step_single_support_time += (count == 1) * dt

    def reset(self, env_ids: Any) -> None:
        super().reset(env_ids)
        self.valid_landing_events[env_ids] = False
        self.last_landed_foot[env_ids] = False
        self.valid_landing_air_time[env_ids] = 0.0
        self.flight_time[env_ids] = 0.0
        self.step_flight_time[env_ids] = 0.0
        self.step_unsupported_time[env_ids] = 0.0
        self.step_single_support_time[env_ids] = 0.0

    def clear_events(self) -> None:
        super().clear_events()
        self.valid_landing_events[...] = False
        self.valid_landing_air_time[...] = 0.0
        self.step_flight_time[...] = 0.0
        self.step_unsupported_time[...] = 0.0
        self.step_single_support_time[...] = 0.0


def _where(condition: Any, true_value: Any, false_value: Any) -> Any:
    if type(condition).__module__.startswith("torch"):
        import torch

        return torch.where(condition, true_value, false_value)
    import numpy as np

    return np.where(condition, true_value, false_value)
