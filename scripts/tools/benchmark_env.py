#!/usr/bin/env python3
"""Benchmark throughput and long-run stability of a PureRL task."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.contracts import TASK_IDS
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec


def main() -> None:
    args = _parse_args()
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    env = None
    try:
        _stage("launching")
        launcher.launch()
        _stage("launched")
        import torch

        spec = get_task_spec(args.task)
        base = spec.make_env_cfg()
        cfg = base.replace(
            seed=args.seed,
            scene=base.scene.replace(num_envs=args.num_envs),
            sim=base.sim.replace(device=args.device),
        )
        env = TienKungLocomotionEnv(task_id=args.task, cfg=cfg)
        observations, _ = env.reset(seed=args.seed)
        actions = torch.zeros((args.num_envs, env.num_actions), device=args.device)
        _assert_finite(torch, env, observations["policy"], None)
        _stage("environment-created")

        for _ in range(args.warmup_steps):
            if args.random_actions:
                actions.uniform_(-1.0, 1.0)
            observations, rewards, _, _, _ = env.step(actions)
        _assert_finite(torch, env, observations["policy"], rewards)
        _synchronize(torch, args.device)
        _stage("warmup-complete")

        cuda_metrics = args.device.startswith("cuda") and torch.cuda.is_available()
        if cuda_metrics:
            torch.cuda.reset_peak_memory_stats(args.device)
        memory_baseline = _memory_snapshot(torch, args.device) if args.memory_baseline_step == 0 else None
        done_count = torch.zeros((), dtype=torch.long, device=args.device)
        termination_counts = {
            name: torch.zeros((), dtype=torch.long, device=args.device)
            for name in env.termination_manager.last_values
        }

        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            if args.random_actions:
                actions.uniform_(-1.0, 1.0)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            done_count.add_((terminated | truncated).sum())
            for name, values in env.termination_manager.last_values.items():
                termination_counts[name].add_(values.sum())

            if step % args.check_interval == 0 or step == args.steps:
                _assert_finite(torch, env, observations["policy"], rewards)
            if step == args.memory_baseline_step:
                _synchronize(torch, args.device)
                memory_baseline = _memory_snapshot(torch, args.device)
            if step % args.progress_interval == 0:
                _stage(f"progress-{step}/{args.steps}")

        _synchronize(torch, args.device)
        elapsed = time.perf_counter() - started
        memory_final = _memory_snapshot(torch, args.device)
        if memory_baseline is None:
            raise RuntimeError("Memory baseline was not captured")

        allocated_growth = memory_final["allocated_mib"] - memory_baseline["allocated_mib"]
        reserved_growth = memory_final["reserved_mib"] - memory_baseline["reserved_mib"]
        device_growth = memory_final["device_used_mib"] - memory_baseline["device_used_mib"]
        if allocated_growth > args.max_allocated_growth_mib:
            raise RuntimeError(
                f"CUDA allocated memory grew by {allocated_growth:.1f} MiB, "
                f"limit is {args.max_allocated_growth_mib:.1f} MiB"
            )
        if reserved_growth > args.max_reserved_growth_mib:
            raise RuntimeError(
                f"CUDA reserved memory grew by {reserved_growth:.1f} MiB, "
                f"limit is {args.max_reserved_growth_mib:.1f} MiB"
            )
        if device_growth > args.max_device_growth_mib:
            raise RuntimeError(
                f"Total CUDA device memory use grew by {device_growth:.1f} MiB, "
                f"limit is {args.max_device_growth_mib:.1f} MiB"
            )

        result = {
            "task": args.task,
            "num_envs": args.num_envs,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "elapsed_s": round(elapsed, 3),
            "policy_steps_per_s": round(args.steps / elapsed, 2),
            "transitions_per_s": round(args.steps * args.num_envs / elapsed, 2),
            "simulated_time_realtime_factor": round(args.steps * cfg.sim.step_dt / elapsed, 3),
            "completed_episodes": int(done_count.item()),
            "termination_counts": {
                name: int(count.item()) for name, count in termination_counts.items()
            },
            "memory_baseline": memory_baseline,
            "memory_final": memory_final,
            "allocated_growth_mib": round(allocated_growth, 3),
            "reserved_growth_mib": round(reserved_growth, 3),
            "device_growth_mib": round(device_growth, 3),
        }
        print("BENCHMARK_OK " + json.dumps(result, sort_keys=True), flush=True)
    except BaseException as exc:
        print(f"BENCHMARK_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_IDS[0])
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--memory-baseline-step", type=int, default=0)
    parser.add_argument("--max-allocated-growth-mib", type=float, default=128.0)
    parser.add_argument("--max-reserved-growth-mib", type=float, default=512.0)
    parser.add_argument("--max-device-growth-mib", type=float, default=512.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-actions", action="store_true")
    args = parser.parse_args()
    for name in ("num_envs", "steps", "check_interval", "progress_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if not 0 <= args.memory_baseline_step <= args.steps:
        parser.error("--memory-baseline-step must be between zero and --steps")
    return args


def _assert_finite(torch_module, env: TienKungLocomotionEnv, observation, reward) -> None:
    values = {
        "observation": observation,
        "root_position": env.state.root_position,
        "root_quaternion": env.state.root_quaternion,
        "joint_positions": env.state.joint_positions,
        "joint_velocities": env.state.joint_velocities,
        "contact_forces": env.state.net_contact_forces,
        "height_scan": env.height_scan,
    }
    if reward is not None:
        values["reward"] = reward
    invalid = [name for name, value in values.items() if not torch_module.isfinite(value).all()]
    if invalid:
        raise RuntimeError("Non-finite tensors detected: " + ", ".join(invalid))


def _synchronize(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _memory_snapshot(torch_module, device: str) -> dict[str, float]:
    if not device.startswith("cuda") or not torch_module.cuda.is_available():
        return {
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "peak_allocated_mib": 0.0,
            "device_used_mib": 0.0,
            "device_free_mib": 0.0,
            "device_total_mib": 0.0,
        }
    scale = 1024.0**2
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    return {
        "allocated_mib": round(torch_module.cuda.memory_allocated(device) / scale, 3),
        "reserved_mib": round(torch_module.cuda.memory_reserved(device) / scale, 3),
        "peak_allocated_mib": round(torch_module.cuda.max_memory_allocated(device) / scale, 3),
        "device_used_mib": round((total_bytes - free_bytes) / scale, 3),
        "device_free_mib": round(free_bytes / scale, 3),
        "device_total_mib": round(total_bytes / scale, 3),
    }


def _stage(name: str) -> None:
    print(f"BENCHMARK_STAGE {name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
