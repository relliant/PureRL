"""Locomotion lifecycle event terms."""

from __future__ import annotations

from typing import Any


def startup_domain_randomization(env: Any, *, env_ids: Any = None) -> None:
    del env_ids
    env.randomize_startup_domain()


def reset_state_randomization(env: Any, *, env_ids: Any) -> None:
    env.randomize_reset_state(env_ids)


def periodic_velocity_push(env: Any, *, env_ids: Any = None) -> None:
    del env_ids
    env.apply_periodic_pushes()
