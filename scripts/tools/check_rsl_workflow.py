#!/usr/bin/env python3
"""Check RSL-RL save, resume, transfer, play, export, and optional W&B logging."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "rsl_rl" / "train.py"
PLAY = ROOT / "scripts" / "rsl_rl" / "play.py"


def main() -> None:
    args = _parse_args()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="purerl-rsl-workflow-"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log_root = work_dir / "logs"
    initial_checkpoint = work_dir / "initial.pt"
    resumed_checkpoint = work_dir / "resumed.pt"
    rough_checkpoint = work_dir / "rough.pt"

    common = [
        sys.executable,
        str(TRAIN),
        "--num-envs",
        str(args.num_envs),
        "--max-iterations",
        "1",
        "--num-steps-per-env",
        str(args.num_steps_per_env),
        "--save-interval",
        "1",
        "--log-root",
        str(log_root),
        "--disable-random-episode-length",
    ]
    logger_args = []
    if args.wandb_offline:
        logger_args = [
            "--logger",
            "wandb",
            "--wandb-mode",
            "offline",
            "--wandb-project",
            "purerl-workflow-smoke",
            "--wandb-tags",
            "smoke",
            "resume",
        ]

    _run(
        common
        + [
            "--task",
            "PureRL-Velocity-Flat-TienKung-v0",
            "--experiment-name",
            "workflow_flat",
            "--run-name",
            "initial",
            "--output-checkpoint",
            str(initial_checkpoint),
        ]
        + logger_args
    )
    initial_run = _latest_run(log_root / "workflow_flat")

    _run(
        common
        + [
            "--task",
            "PureRL-Velocity-Flat-TienKung-v0",
            "--experiment-name",
            "workflow_flat",
            "--run-name",
            "resumed",
            "--resume",
            "--load-run",
            str(initial_run),
            "--load-checkpoint",
            str(initial_checkpoint),
            "--output-checkpoint",
            str(resumed_checkpoint),
        ]
    )

    _run(
        common
        + [
            "--task",
            "PureRL-Velocity-Rough-TienKung-Play-v0",
            "--experiment-name",
            "workflow_rough",
            "--run-name",
            "pretrained",
            "--pretrained-checkpoint",
            str(resumed_checkpoint),
            "--output-checkpoint",
            str(rough_checkpoint),
        ]
    )

    export_dir = work_dir / "exported"
    _run(
        [
            sys.executable,
            str(PLAY),
            "--task",
            "PureRL-Velocity-Rough-TienKung-Play-v0",
            "--num-envs",
            str(args.num_envs),
            "--steps",
            "4",
            "--checkpoint",
            str(rough_checkpoint),
            "--export-dir",
            str(export_dir),
        ]
    )

    import torch

    initial_iter = int(torch.load(initial_checkpoint, weights_only=False, map_location="cpu")["iter"])
    resumed_iter = int(torch.load(resumed_checkpoint, weights_only=False, map_location="cpu")["iter"])
    if initial_iter != 0 or resumed_iter != 1:
        raise RuntimeError(
            f"Resume iteration mismatch: initial={initial_iter}, resumed={resumed_iter}"
        )
    for path in (rough_checkpoint, export_dir / "policy.pt", export_dir / "policy.onnx"):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected workflow artifact is missing: {path}")
    if args.wandb_offline and not tuple(initial_run.rglob("offline-run-*")):
        raise RuntimeError(f"W&B offline run was not created under {initial_run}")
    print(
        f"RSL_WORKFLOW_OK initial_iter={initial_iter} resumed_iter={resumed_iter} "
        f"wandb_offline={args.wandb_offline} work_dir={work_dir}",
        flush=True,
    )


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _latest_run(root: Path) -> Path:
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        raise RuntimeError(f"No training run was created under {root}")
    return runs[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps-per-env", type=int, default=4)
    parser.add_argument("--work-dir")
    parser.add_argument("--wandb-offline", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_steps_per_env <= 0:
        parser.error("environment and rollout sizes must be positive")
    return args


if __name__ == "__main__":
    main()
