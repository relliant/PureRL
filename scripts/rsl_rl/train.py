#!/usr/bin/env python3
"""Train a PureRL locomotion task with RSL-RL 3.1."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.contracts import TASK_IDS
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec
from purerl.rl import RslRlVecEnvWrapper, find_checkpoint


def main() -> None:
    args = _parse_args()
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=not args.show))
    env = None
    runner = None
    try:
        launcher.launch()
        import torch
        from rsl_rl.runners import OnPolicyRunner

        spec = get_task_spec(args.task)
        env_cfg = spec.make_env_cfg(args.env_config)
        runner_cfg = spec.make_runner_cfg(args.runner_config)
        seed = runner_cfg.seed if args.seed is None else args.seed
        device = runner_cfg.device if args.device is None else args.device
        env_cfg = env_cfg.replace(
            seed=seed,
            scene=env_cfg.scene.replace(
                num_envs=env_cfg.scene.num_envs if args.num_envs is None else args.num_envs
            ),
            sim=env_cfg.sim.replace(device=device),
        )
        runner_cfg = runner_cfg.replace(
            seed=seed,
            device=device,
            max_iterations=(
                runner_cfg.max_iterations if args.max_iterations is None else args.max_iterations
            ),
            num_steps_per_env=(
                runner_cfg.num_steps_per_env
                if args.num_steps_per_env is None
                else args.num_steps_per_env
            ),
            save_interval=(
                runner_cfg.save_interval if args.save_interval is None else args.save_interval
            ),
            experiment_name=args.experiment_name or runner_cfg.experiment_name,
            run_name=runner_cfg.run_name if args.run_name is None else args.run_name,
            logger=runner_cfg.logger if args.logger is None else args.logger,
            wandb_project=(
                runner_cfg.wandb_project if args.wandb_project is None else args.wandb_project
            ),
            wandb_entity=(
                runner_cfg.wandb_entity if args.wandb_entity is None else args.wandb_entity
            ),
            wandb_mode=runner_cfg.wandb_mode if args.wandb_mode is None else args.wandb_mode,
            wandb_tags=(
                runner_cfg.wandb_tags if args.wandb_tags is None else tuple(args.wandb_tags)
            ),
            wandb_run_id=(
                runner_cfg.wandb_run_id if args.wandb_run_id is None else args.wandb_run_id
            ),
            wandb_resume=(
                runner_cfg.wandb_resume if args.wandb_resume is None else args.wandb_resume
            ),
            resume=runner_cfg.resume if args.resume is None else args.resume,
            load_run=runner_cfg.load_run if args.load_run is None else args.load_run,
            load_checkpoint=(
                runner_cfg.load_checkpoint
                if args.load_checkpoint is None
                else args.load_checkpoint
            ),
        )
        if runner_cfg.resume and args.pretrained_checkpoint:
            raise ValueError("resume and pretrained_checkpoint are mutually exclusive")
        if any(
            value is not None
            for value in (
                args.wandb_project,
                args.wandb_entity,
                args.wandb_mode,
                args.wandb_tags,
                args.wandb_run_id,
                args.wandb_resume,
            )
        ) and runner_cfg.logger != "wandb":
            raise ValueError("W&B options require the wandb logger")
        env_cfg.validate()
        runner_cfg.validate()
        torch.manual_seed(seed)

        log_root = Path(args.log_root).expanduser().resolve() / runner_cfg.experiment_name
        checkpoint = _resolve_checkpoint(args, log_root, runner_cfg)
        log_dir = None if args.no_logging else _make_log_dir(log_root, runner_cfg.run_name)
        if log_dir is not None:
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True, exist_ok=True)
            env_cfg.to_yaml(params_dir / "env.yaml")
            runner_cfg.to_yaml(params_dir / "agent.yaml")
            print(f"Logging run to {log_dir}", flush=True)
        _configure_wandb_environment(runner_cfg, log_dir)

        env = TienKungLocomotionEnv(task_id=args.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped,
            runner_cfg.to_dict(),
            log_dir=None if log_dir is None else str(log_dir),
            device=device,
        )
        if log_dir is None:
            runner.disable_logs = True
            runner.logger_type = runner_cfg.logger

        if checkpoint is not None:
            load_optimizer = args.pretrained_checkpoint is None
            runner.load(str(checkpoint), load_optimizer=load_optimizer, map_location=device)
            if load_optimizer:
                runner.current_learning_iteration += 1
            else:
                runner.current_learning_iteration = 0
            print(
                f"Loaded checkpoint {checkpoint} at iteration "
                f"{runner.current_learning_iteration}",
                flush=True,
            )

        start_iteration = runner.current_learning_iteration
        runner.learn(
            num_learning_iterations=runner_cfg.max_iterations,
            init_at_random_ep_len=not args.disable_random_episode_length,
        )
        if args.output_checkpoint is not None:
            output_checkpoint = Path(args.output_checkpoint).expanduser().resolve()
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            runner.save(str(output_checkpoint))
            print(f"Saved checkpoint {output_checkpoint}", flush=True)
        print(
            f"TRAIN_OK task={args.task} envs={env_cfg.scene.num_envs} "
            f"start_iteration={start_iteration} trained_iterations={runner_cfg.max_iterations} "
            f"last_iteration={runner.current_learning_iteration}",
            flush=True,
        )
    except BaseException as exc:
        print(f"TRAIN_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        _close_runner_writer(runner)
        if env is not None:
            env.close()
        launcher.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_IDS[0])
    parser.add_argument("--env-config", help="Path to a complete environment YAML config")
    parser.add_argument(
        "--runner-config",
        "--agent-config",
        dest="runner_config",
        help="Path to a complete RSL-RL runner YAML config",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-steps-per-env", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-resume", choices=("allow", "must", "never", "auto"))
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-run")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--no-logging", action="store_true")
    parser.add_argument("--disable-random-episode-length", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.resume and args.pretrained_checkpoint:
        parser.error("--resume and --pretrained-checkpoint are mutually exclusive")
    if args.logger not in {None, "wandb"} and any(
        value is not None
        for value in (
            args.wandb_project,
            args.wandb_entity,
            args.wandb_mode,
            args.wandb_tags,
            args.wandb_run_id,
            args.wandb_resume,
        )
    ):
        parser.error("W&B options require --logger wandb")
    return args


def _make_log_dir(log_root: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = timestamp if not run_name else f"{timestamp}_{run_name}"
    directory = log_root / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _resolve_checkpoint(args, log_root, runner_cfg):
    if args.pretrained_checkpoint:
        checkpoint = Path(args.pretrained_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint}")
        return checkpoint
    if runner_cfg.resume:
        return find_checkpoint(
            log_root,
            load_run=runner_cfg.load_run,
            load_checkpoint=runner_cfg.load_checkpoint,
        )
    return None


def _configure_wandb_environment(runner_cfg, log_dir: Path | None) -> None:
    if runner_cfg.logger != "wandb" or log_dir is None:
        return
    os.environ["WANDB_MODE"] = runner_cfg.wandb_mode
    os.environ["WANDB_DIR"] = str(log_dir)
    os.environ["WANDB_RESUME"] = runner_cfg.wandb_resume
    if runner_cfg.wandb_entity:
        # RSL-RL 3.1 reads WANDB_USERNAME while current W&B uses WANDB_ENTITY.
        os.environ["WANDB_USERNAME"] = runner_cfg.wandb_entity
        os.environ["WANDB_ENTITY"] = runner_cfg.wandb_entity
    if runner_cfg.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(runner_cfg.wandb_tags)
    if runner_cfg.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = runner_cfg.wandb_run_id


def _close_runner_writer(runner) -> None:
    writer = None if runner is None else getattr(runner, "writer", None)
    if writer is None:
        return
    stop = getattr(writer, "stop", None)
    if callable(stop):
        stop()
        return
    close = getattr(writer, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
