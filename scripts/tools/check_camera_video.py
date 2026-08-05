#!/usr/bin/env python3
"""Render the standalone environment to RGB frames and an MP4 file."""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.contracts import TASK_IDS
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec


def main() -> None:
    args = _parse_args()
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True, enable_cameras=True))
    env = None
    writer = None
    try:
        launcher.launch()
        import imageio.v2 as imageio
        import numpy as np
        import torch

        cfg = get_task_spec(args.task).make_env_cfg()
        cfg = cfg.replace(
            scene=cfg.scene.replace(num_envs=args.num_envs),
            sim=cfg.sim.replace(device=args.device),
            viewer=cfg.viewer.replace(resolution=(args.width, args.height)),
        )
        env = TienKungLocomotionEnv(
            task_id=args.task,
            cfg=cfg,
            render_mode="rgb_array",
        )
        output = _output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output, fps=args.fps, codec="libx264")
        actions = torch.zeros((args.num_envs, env.num_actions), device=env.device)
        nonblank_frames = 0
        for _ in range(args.frames):
            env.step(actions)
            frame = env.render()
            if frame.shape != (args.height, args.width, 3) or frame.dtype != np.uint8:
                raise RuntimeError(f"Invalid RGB frame: shape={frame.shape} dtype={frame.dtype}")
            nonblank_frames += int(float(frame.std()) > 1.0)
            writer.append_data(frame)
        writer.close()
        writer = None
        if nonblank_frames == 0:
            raise RuntimeError("All rendered frames are blank")
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Video was not written: {output}")
        print(
            f"VIDEO_OK frames={args.frames} nonblank={nonblank_frames} "
            f"resolution={args.width}x{args.height} output={output}",
            flush=True,
        )
    except BaseException as exc:
        print(f"VIDEO_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        launcher.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS, default="PureRL-Velocity-Flat-TienKung-Play-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.num_envs, args.frames, args.width, args.height, args.fps) <= 0:
        parser.error("numeric arguments must be positive")
    return args


def _output_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.gettempdir()) / f"purerl-camera-{timestamp}.mp4"


if __name__ == "__main__":
    main()
