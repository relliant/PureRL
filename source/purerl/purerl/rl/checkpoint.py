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


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", value)
        if part
    )
