#!/usr/bin/env python3
"""Measure contacts, swing clearance and rewards of a deterministic flat walking policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.contracts import TASK_IDS
from purerl.envs import TienKungLocomotionEnv
from purerl.registry import get_task_spec
from purerl.rl import RslRlVecEnvWrapper, validate_checkpoint_observation_dim


class MeasuredEnv(TienKungLocomotionEnv):
    def __init__(self, *args, **kwargs):
        self.samples = defaultdict(list)
        self.physics_samples = defaultdict(list)
        super().__init__(*args, **kwargs)

    def _update_physics_step_sensors(self):
        super()._update_physics_step_sensors()
        if self._foot_body_indices is not None:
            # ContactHistory has just consumed the current physics-step forces.
            self.physics_samples["contact"].append(self.contact_history.raw_contact.clone())
            self.physics_samples["landing"].append(self.contact_history.first_contact.clone())
            self.physics_samples["air_time"].append(self.contact_history.last_air_time.clone())
            self.physics_samples["age"].append(self.episode_length_buf.clone())

    def _clear_step_events(self):
        # Capture before episode resets and before the latched landing events clear.
        values = {
            "contact": self.contact_history.raw_contact,
            "filtered_contact": self._foot_contact_mask(),
            "foot_height": self.foot_heights,
            "target_height": self._gait_targets()[1],
            "stance": self._get_gait_phase(),
            "foot_position": self.state.body_positions[:, self._foot_body_indices],
            "root_position": self.state.root_position,
            "velocity_yaw": self.base_linear_velocity_yaw,
            "velocity_body": self.base_linear_velocity_body,
            "command": self.commands,
            "age": self.episode_length_buf,
            "landing": self.contact_history.contact_events,
            "valid_landing": self.contact_history.valid_landing_events
            & (self.contact_history.contact_events.sum(dim=-1) == 1)[:, None],
            "air_time": self.contact_history.last_air_time,
        }
        values.update({f"reward/{k}": v for k, v in self.reward_manager.last_weighted.items()})
        values.update({f"termination/{k}": v for k, v in self.termination_manager.last_values.items()})
        for key, value in values.items():
            self.samples[key].append(value.clone())
        super()._clear_step_events()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/gait_evaluation"))
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--reference-default-pose", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or not np.isfinite(args.vx):
        parser.error("num-envs and steps must be positive, and vx must be finite")
    import torch

    torch.zeros(1, device="cuda:0")
    torch.manual_seed(args.seed)
    spec = get_task_spec(TASK_IDS[1])
    cfg = spec.make_env_cfg()
    cfg = cfg.replace(
        seed=args.seed,
        scene=cfg.scene.replace(num_envs=args.num_envs),
        sensors=cfg.sensors.replace(lidar=cfg.sensors.lidar.replace(enabled=False)),
        commands=cfg.commands.replace(ranges=cfg.commands.ranges.replace(lin_vel_x=(args.vx, args.vx))),
    )
    runner_cfg = spec.make_runner_cfg(args.checkpoint.parent / "params/agent.yaml")
    validate_checkpoint_observation_dim(args.checkpoint, expected=cfg.observations.dimension)
    launcher = IsaacSimLauncher(AppLauncherCfg(headless=True))
    env = None
    try:
        launcher.launch()
        from rsl_rl.runners import OnPolicyRunner

        env = MeasuredEnv(task_id=TASK_IDS[1], cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = None
        if args.reference_default_pose:
            fixed_action = torch.zeros((cfg.scene.num_envs, cfg.actions.dimension), device=cfg.sim.device)

            def policy(observations):
                return fixed_action
        else:
            runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=cfg.sim.device)
            runner.disable_logs = True
            runner.logger_type = runner_cfg.logger
            runner.load(str(args.checkpoint), load_optimizer=False, map_location=cfg.sim.device)
            policy = runner.get_inference_policy(device=cfg.sim.device)
        observations = wrapped.get_observations()
        with torch.inference_mode():
            for step in range(args.steps):
                observations, _, dones, _ = wrapped.step(policy(observations))
                if runner is not None:
                    runner.alg.policy.reset(dones)
                if (step + 1) % 200 == 0:
                    print(f"GAIT_PROGRESS {step + 1}/{args.steps}", flush=True)
        data = {key: torch.stack(values).cpu().numpy() for key, values in env.samples.items()}
        physics = {key: torch.stack(values).cpu().numpy() for key, values in env.physics_samples.items()}
        args.output.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(args.output / "env.yaml")
        np.savez_compressed(args.output / "rollout.npz", **data)
        np.savez_compressed(args.output / "physics_contacts.npz", **physics)
        summary = summarize(data, physics, cfg)
        summary["checkpoint"] = str(args.checkpoint.resolve())
        summary["num_envs"] = args.num_envs
        summary["steps"] = args.steps
        summary["seed"] = args.seed
        summary["command"] = [args.vx, 0.0, 0.0]
        summary["setup"] = (
            "Flat-Play preset, deterministic policy, LiDAR disabled; startup/reset randomization retained"
        )
        summary["control"] = "fixed default joint targets" if args.reference_default_pose else "checkpoint mean action"
        (args.output / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("GAIT_RESULT " + json.dumps(summary), flush=True)
    finally:
        if env is not None:
            env.close()
        launcher.close()


def summarize(data, physics, cfg):
    dt = cfg.sim.step_dt
    valid = data["age"] * dt > 2.0
    substep_time = ((np.arange(len(physics["age"])) % cfg.sim.decimation) + 1)[:, None] * cfg.sim.dt
    valid_physics = physics["age"] * dt + substep_time > 2.0
    if not valid.any():
        return {
            "warmup_excluded_s": 2.0,
            "valid_env_seconds": 0.0,
            "note": "No episode survived the warmup; no steady-state gait comparison is possible.",
            "maximum_episode_age_s_per_env": (data["age"].max(axis=0) * dt).tolist(),
            "termination_counts": {
                key.removeprefix("termination/"): int(value.sum())
                for key, value in data.items()
                if key.startswith("termination/")
            },
        }

    def contact_stats(contact, mask):
        count = contact.sum(axis=-1)
        return {
            "double_support_fraction": float((count[mask] == 2).mean()),
            "single_support_fraction": float((count[mask] == 1).mean()),
            "flight_fraction": float((count[mask] == 0).mean()),
        }

    contact = data["contact"]
    stance = data["stance"]
    single_phase = stance.sum(axis=-1) == 1
    phase_valid = valid & single_phase
    foot_z = data["foot_height"]
    events = physics["landing"][valid_physics]
    landed_air_time = physics["air_time"][valid_physics][events]
    any_landing = events.any(axis=-1)
    return {
        "warmup_excluded_s": 2.0,
        "valid_env_seconds": float(valid.sum() * dt),
        "valid_landing_rate_hz": float(data["valid_landing"][valid].sum() / (valid.sum() * dt)),
        "contact_at_50hz": contact_stats(contact, valid),
        "filtered_contact_at_50hz": contact_stats(data["filtered_contact"], valid),
        "contact_at_200hz": contact_stats(physics["contact"], valid_physics),
        "correct_support_during_single_support_target_fraction": float(
            (contact == stance).all(axis=-1)[phase_valid].mean()
        ),
        "terrain_relative_foot_height_m_percentiles_5_50_95": np.percentile(
            foot_z[valid], [5, 50, 95], axis=0
        ).tolist(),
        "foot_height_above_3cm_fraction": float((foot_z[valid] > 0.03).mean()),
        "root_height_m_percentiles_5_50_95": np.percentile(data["root_position"][..., 2][valid], [5, 50, 95]).tolist(),
        "mean_velocity_yaw": data["velocity_yaw"][valid].mean(axis=0).tolist(),
        "xy_velocity_rmse_mps": float(
            np.sqrt(
                np.mean(np.sum((data["velocity_yaw"][valid][..., :2] - data["command"][valid][..., :2]) ** 2, axis=-1))
            )
        ),
        "body_vertical_velocity_rms_mps": float(np.sqrt(np.mean(data["velocity_body"][valid][..., 2] ** 2))),
        "landing_air_time_s_percentiles_50_95_99": np.percentile(landed_air_time, [50, 95, 99]).tolist()
        if landed_air_time.size
        else [],
        "landings_with_air_time_over_threshold_fraction": float((landed_air_time > cfg.gait.air_time_threshold).mean())
        if landed_air_time.size
        else 0.0,
        "simultaneous_5ms_landings_fraction": float(events.all(axis=-1)[any_landing].mean())
        if any_landing.any()
        else 0.0,
        "mean_weighted_reward_per_second": {
            key.removeprefix("reward/"): float(value[valid].mean() / dt)
            for key, value in data.items()
            if key.startswith("reward/")
        },
        "termination_counts": {
            key.removeprefix("termination/"): int(value.sum())
            for key, value in data.items()
            if key.startswith("termination/")
        },
    }


if __name__ == "__main__":
    main()
