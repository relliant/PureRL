"""Checkpoint discovery without Isaac Lab utilities."""

from __future__ import annotations

import re
from pathlib import Path


def find_checkpoint(
    log_root: str | Path,
    *,
    load_run: str = ".*",
    load_checkpoint: str = "model_.*.pt",
) -> Path:
    """Return the newest naturally sorted run/checkpoint matching both patterns."""

    root = Path(log_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")

    explicit_run = Path(load_run).expanduser()
    if explicit_run.is_dir():
        run = explicit_run.resolve()
    else:
        runs = sorted(
            (path for path in root.iterdir() if path.is_dir() and re.fullmatch(load_run, path.name)),
            key=lambda path: _natural_key(path.name),
        )
        if not runs:
            raise FileNotFoundError(f"No run matching {load_run!r} under {root}")
        run = runs[-1]

    explicit_checkpoint = Path(load_checkpoint).expanduser()
    if explicit_checkpoint.is_file():
        return explicit_checkpoint.resolve()
    checkpoints = sorted(
        (path for path in run.iterdir() if path.is_file() and re.fullmatch(load_checkpoint, path.name)),
        key=lambda path: _natural_key(path.name),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint matching {load_checkpoint!r} under {run}")
    return checkpoints[-1]


def read_checkpoint_mean_noise_std(checkpoint: str | Path) -> float | None:
    """Read the mean action-noise standard deviation from an RSL-RL checkpoint."""

    import torch

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", {})
    if "std" in state:
        std = state["std"].detach().float()
    elif "log_std" in state:
        std = state["log_std"].detach().float().exp()
    else:
        return None
    if not torch.isfinite(std).all() or (std <= 0.0).any():
        raise ValueError(f"Checkpoint contains invalid action noise: {path}")
    return float(std.mean())


def validate_checkpoint_noise_std(
    checkpoint: str | Path, *, maximum: float
) -> float | None:
    """Reject checkpoints whose exploration distribution is already degenerate."""

    if maximum <= 0.0:
        raise ValueError("maximum checkpoint noise must be positive")
    mean_std = read_checkpoint_mean_noise_std(checkpoint)
    if mean_std is not None and mean_std > maximum:
        path = Path(checkpoint).expanduser().resolve()
        raise ValueError(
            f"Checkpoint mean action noise std is {mean_std:.2f}, above the safe limit "
            f"{maximum:.2f}: {path}. Start from scratch or explicitly allow the unsafe checkpoint."
        )
    return mean_std


def validate_checkpoint_observation_dim(checkpoint: str | Path, *, expected: int) -> None:
    """Reject policies trained before the gait clock became observable."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=True).get("model_state_dict", {})
    for network in ("actor", "critic"):
        weight = state.get(f"{network}.0.weight")
        if weight is None or weight.ndim != 2:
            raise ValueError(f"Checkpoint has no supported feed-forward {network} input layer: {path}")
        if weight.shape[1] != expected:
            raise ValueError(
                f"Checkpoint {network} expects {weight.shape[1]} observations, but this environment "
                f"requires {expected}, including the gait sin/cos clock. "
                f"Start a fresh training run; legacy 259-dimensional checkpoints cannot be resumed: {path}"
            )


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", value)
        if part
    )
