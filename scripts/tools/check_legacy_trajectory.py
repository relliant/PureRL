#!/usr/bin/env python3
"""Compare the standalone backend with the checked-in legacy trajectory fixture."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
METADATA_PATH = FIXTURE_DIR / "isaaclab_legacy_reference.json"
ARRAYS_PATH = FIXTURE_DIR / "isaaclab_legacy_reference.npz"


def main() -> None:
    args = _parse_args()
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    env = None
    try:
        launcher.launch()
        import numpy as np
        import torch

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        fixture = np.load(ARRAYS_PATH)
        num_envs = int(metadata["num_envs"])
        seed = int(metadata["seed"])
        cfg = get_task_spec("PureRL-Velocity-Flat-TienKung-Play-v0").make_env_cfg()
        cfg = cfg.replace(
            seed=seed,
            scene=cfg.scene.replace(num_envs=num_envs),
            sim=cfg.sim.replace(device=args.device),
        )
        env = TienKungLocomotionEnv(
            task_id="PureRL-Velocity-Flat-TienKung-Play-v0",
            cfg=cfg,
        )
        observation, _ = env.reset(seed=seed)
        if not args.no_align_initial_state:
            _align_initial_state(env, fixture, torch)
            observation = env.get_observations()

        actual = {name: [] for name in _STATE_FIELDS}
        actual["observation"] = [_cpu(observation["policy"])]
        _append_state(actual, env)
        actual["reward"] = []
        actual["reward_terms_per_second"] = []
        actual["termination_terms"] = []
        actual["terminated"] = []
        actual["truncated"] = []

        actions = fixture["flat_action"]
        for action in actions:
            observation, reward, terminated, truncated, _ = env.step(
                torch.as_tensor(action, device=args.device)
            )
            actual["observation"].append(_cpu(observation["policy"]))
            actual["reward"].append(_cpu(reward))
            actual["reward_terms_per_second"].append(
                np.stack(
                    [
                        _cpu(env.reward_manager.last_weighted[name]) / env.step_dt
                        for name in metadata["reward_terms"]
                    ],
                    axis=-1,
                )
            )
            actual["termination_terms"].append(
                np.stack(
                    [
                        _cpu(env.termination_manager.last_values[name])
                        for name in metadata["termination_terms"]
                    ],
                    axis=-1,
                )
            )
            actual["terminated"].append(_cpu(terminated))
            actual["truncated"].append(_cpu(truncated))
            _append_state(actual, env)

        stacked = {name: np.stack(values) for name, values in actual.items()}
        metrics = {}
        for name in (*_STATE_FIELDS, "observation", "reward", "reward_terms_per_second"):
            current = stacked[name]
            reference = fixture[f"flat_{name}"]
            if name == "root_quaternion":
                current = _align_quaternion_sign(current, reference)
            metrics[name] = _error_metrics(current, reference)

        termination_agreement = {
            name: float(np.mean(stacked[name] == fixture[f"flat_{name}"]))
            for name in ("termination_terms", "terminated", "truncated")
        }
        if not all(np.isfinite(value) for metric in metrics.values() for value in metric.values()):
            raise RuntimeError("Trajectory comparison produced a non-finite metric")
        for name, metric in metrics.items():
            print(
                f"TRAJECTORY_METRIC field={name} rmse={metric['rmse']:.8g} "
                f"max_abs={metric['max_abs']:.8g}",
                flush=True,
            )
        limits = {
            "root_position": args.max_root_position_error,
            "joint_position": args.max_joint_position_error,
            "command": args.max_command_error,
            "reward": args.max_reward_error,
        }
        for name, limit in limits.items():
            if metrics[name]["max_abs"] > limit:
                raise RuntimeError(
                    f"{name} migration error exceeds the configured tolerance: "
                    f"{metrics[name]['max_abs']:.6g} > {limit:.6g}"
                )
        if min(termination_agreement.values()) < 1.0:
            raise RuntimeError(f"Termination behavior differs from the fixture: {termination_agreement}")
        print(
            "LEGACY_TRAJECTORY_OK "
            f"steps={len(actions)} envs={num_envs} "
            f"termination_agreement={min(termination_agreement.values()):.3f}",
            flush=True,
        )
    except BaseException as exc:
        print(
            f"LEGACY_TRAJECTORY_FAILED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if env is not None:
            env.close()
        launcher.close()


_STATE_FIELDS = (
    "root_position",
    "root_quaternion",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_position",
    "joint_velocity",
    "command",
)


def _append_state(values, env) -> None:
    state = env.state
    values["root_position"].append(_cpu(state.root_position))
    values["root_quaternion"].append(_cpu(state.root_quaternion))
    values["root_linear_velocity"].append(_cpu(state.root_linear_velocity))
    values["root_angular_velocity"].append(_cpu(state.root_angular_velocity))
    values["joint_position"].append(_cpu(state.joint_positions))
    values["joint_velocity"].append(_cpu(state.joint_velocities))
    values["command"].append(_cpu(env.commands))


def _align_initial_state(env, fixture, torch_module) -> None:
    root_position = torch_module.as_tensor(
        fixture["flat_root_position"][0], device=env.device
    )
    root_quaternion = torch_module.as_tensor(
        fixture["flat_root_quaternion"][0], device=env.device
    )
    joint_position = torch_module.as_tensor(
        fixture["flat_joint_position"][0], device=env.device
    )
    default = env.default_joint_positions
    zero_default = default.abs() < 1.0e-8
    if torch_module.any(joint_position[zero_default].abs() > 1.0e-6):
        raise RuntimeError("Fixture cannot be represented by joint reset scaling")
    joint_scale = torch_module.ones_like(default)
    joint_scale[~zero_default] = joint_position[~zero_default] / default[~zero_default]
    root_yaw = torch_module.atan2(
        2.0
        * (
            root_quaternion[:, 0] * root_quaternion[:, 3]
            + root_quaternion[:, 1] * root_quaternion[:, 2]
        ),
        1.0
        - 2.0
        * (
            root_quaternion[:, 2] * root_quaternion[:, 2]
            + root_quaternion[:, 3] * root_quaternion[:, 3]
        ),
    )
    env.backend.randomize_reset_state(
        env.backend.all_env_ids,
        root_position[:, :2] - env.backend.env_origins[:, :2],
        root_yaw,
        joint_scale,
    )
    env.commands.copy_(torch_module.as_tensor(fixture["flat_command"][0], device=env.device))
    env.command_manager.heading_target.copy_(
        torch_module.as_tensor(fixture["flat_heading_target"][0], device=env.device)
    )
    env.command_manager.is_heading_env.fill_(True)
    env.command_manager.is_standing_env.fill_(False)
    env.command_manager.time_left.fill_(env.cfg.commands.resampling_time_range[0])
    env.episode_length_buf.zero_()
    env.action_manager.reset(env.backend.all_env_ids)
    env.backend.refresh()
    env._reset_sensors(env.backend.all_env_ids)


def _cpu(value):
    return value.detach().cpu().numpy().copy()


def _align_quaternion_sign(current, reference):
    import numpy as np

    sign = np.where(np.sum(current * reference, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return current * sign


def _error_metrics(current, reference) -> dict[str, float]:
    import numpy as np

    difference = current.astype(np.float64) - reference.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-joint-position-error", type=float, default=0.65)
    parser.add_argument("--max-root-position-error", type=float, default=0.18)
    parser.add_argument("--max-command-error", type=float, default=0.12)
    parser.add_argument("--max-reward-error", type=float, default=0.06)
    parser.add_argument("--no-align-initial-state", action="store_true")
    args = parser.parse_args()
    limits = (
        args.max_joint_position_error,
        args.max_root_position_error,
        args.max_command_error,
        args.max_reward_error,
    )
    if any(limit <= 0.0 for limit in limits):
        parser.error("migration error tolerances must be positive")
    return args


if __name__ == "__main__":
    main()
