#!/usr/bin/env python3
"""Run and optionally export a trained PureRL RSL-RL policy."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.contracts import OBSERVATION_DIM, TASK_IDS
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec
from purerl.rl import RslRlVecEnvWrapper, export_feedforward_policy, find_checkpoint


def main() -> None:
    args = _parse_args()
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=not args.show))
    env = None
    try:
        launcher.launch()
        import torch
        from rsl_rl.runners import OnPolicyRunner

        spec = get_task_spec(args.task)
        env_cfg = spec.make_env_cfg()
        runner_cfg = spec.make_runner_cfg()
        seed = runner_cfg.seed if args.seed is None else args.seed
        device = runner_cfg.device if args.device is None else args.device
        env_cfg = env_cfg.replace(
            seed=seed,
            scene=env_cfg.scene.replace(
                num_envs=env_cfg.scene.num_envs if args.num_envs is None else args.num_envs
            ),
            sim=env_cfg.sim.replace(device=device),
        )
        runner_cfg = runner_cfg.replace(seed=seed, device=device)
        checkpoint = _resolve_checkpoint(args, runner_cfg.experiment_name)

        env = TienKungLocomotionEnv(task_id=args.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=device)
        runner.disable_logs = True
        runner.logger_type = runner_cfg.logger
        runner.load(str(checkpoint), load_optimizer=False, map_location=device)
        policy = runner.get_inference_policy(device=device)

        if not args.no_export:
            export_dir = (
                Path(args.export_dir).expanduser().resolve()
                if args.export_dir
                else checkpoint.parent / "exported"
            )
            exported = export_feedforward_policy(
                runner.alg.policy,
                export_dir,
                observation_dim=OBSERVATION_DIM,
            )
            print("Exported " + ", ".join(f"{name}={path}" for name, path in exported.items()))

        observations = wrapped.get_observations()
        for _ in range(args.steps):
            start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = wrapped.step(actions)
                runner.alg.policy.reset(dones)
            if args.real_time:
                delay = env.step_dt - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)

        print(
            f"PLAY_OK task={args.task} envs={env_cfg.scene.num_envs} "
            f"steps={args.steps} checkpoint={checkpoint}",
            flush=True,
        )
    except BaseException as exc:
        print(f"PLAY_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_IDS[1])
    parser.add_argument("--checkpoint")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--load-run", default=".*")
    parser.add_argument("--load-checkpoint", default="model_.*.pt")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--export-dir")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _resolve_checkpoint(args: argparse.Namespace, experiment_name: str) -> Path:
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    return find_checkpoint(
        Path(args.log_root).expanduser().resolve() / experiment_name,
        load_run=args.load_run,
        load_checkpoint=args.load_checkpoint,
    )


if __name__ == "__main__":
    main()
