"""Vectorized locomotion termination terms."""

from __future__ import annotations

from typing import Any

from ._array import arccos, clip, norm


def time_out(episode_length: Any, max_episode_length: int) -> Any:
    return episode_length >= max_episode_length


def illegal_contact(contact_forces: Any, *, threshold: float = 1.0) -> Any:
    violations = norm(contact_forces) > threshold
    if type(violations).__module__.startswith("torch"):
        return violations.reshape(violations.shape[0], -1).any(dim=-1)
    return violations.reshape(violations.shape[0], -1).any(axis=-1)


def bad_orientation(projected_gravity: Any, *, limit_angle: float = 0.8) -> Any:
    tilt = arccos(clip(-projected_gravity[..., 2], -1.0, 1.0))
    return tilt > limit_angle
