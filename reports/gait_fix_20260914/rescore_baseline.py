"""Compare old and corrected rewards on the SAME recorded baseline trajectory."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from purerl.config import make_flat_play_env_cfg
from purerl.envs import TienKungLocomotionEnv
from purerl.mdp import rewards
from purerl.sensors import BipedContactHistory

directory = Path(__file__).parent
baseline = directory.parent / "flat_baseline_20260914"
data = {key: torch.from_numpy(value) for key, value in np.load(baseline / "rollout.npz").items()}
physics = {key: torch.from_numpy(value) for key, value in np.load(baseline / "physics_contacts.npz").items()}
cfg = make_flat_play_env_cfg()
gait = cfg.gait
history = BipedContactHistory(
    torch.zeros(data["contact"].shape[1:]),
    force_threshold=gait.contact_threshold,
    release_time=gait.contact_release_time,
    support_time=gait.min_phase_time,
    min_air_time=gait.air_time_threshold,
    flight_grace_time=gait.flight_grace_time,
)
values = []
previous_age = physics["age"][0]
for step in range(len(data["age"])):
    for substep in range(4):
        index = step * 4 + substep
        age = physics["age"][index]
        history.reset(torch.nonzero(age < previous_age).flatten())
        previous_age = age
        forces = torch.zeros((*history.in_contact.shape, 3))
        forces[..., 2] = physics["contact"][index] * 100.0
        history.update(forces, cfg.sim.dt)
    context = SimpleNamespace(cfg=cfg, episode_length_buf=data["age"][step], step_dt=cfg.sim.step_dt)
    stance, target = TienKungLocomotionEnv._gait_targets(context)
    raw = {
        "feet_contact_number": rewards.feet_contact_number(
            history.in_contact,
            stance,
            data["command"][step],
            current_air_time=history.current_air_time,
            current_contact_time=history.current_contact_time,
            min_phase_time=gait.min_phase_time,
            command_threshold=gait.command_threshold,
        ),
        "feet_clearance": rewards.feet_clearance(
            data["foot_position"][step, ..., 2] - gait.foot_height_offset,
            target,
            data["command"][step],
            contact=history.in_contact,
            current_contact_time=history.current_contact_time,
            support_time=gait.min_phase_time,
            min_clearance=gait.min_clearance,
            sigma=gait.clearance_sigma,
            command_threshold=gait.command_threshold,
        ),
        "feet_air_time": rewards.feet_air_time_on_contact(
            history.valid_landing_air_time,
            history.contact_events,
            data["command"][step],
            supported_landing=history.valid_landing_events,
            threshold=gait.air_time_threshold,
            command_threshold=gait.command_threshold,
        ),
        "base_height": rewards.base_height(
            data["root_position"][step, :, 2],
            0.0,
            target=cfg.robot.default_root_height,
            sigma=gait.base_height_sigma,
        ),
        "feet_flight": history.step_unsupported_time / cfg.sim.step_dt,
        "track_lin_vel_xy_exp": rewards.track_lin_vel_xy_exp(
            data["velocity_yaw"][step],
            data["command"][step],
            std=cfg.commands.lin_vel_tracking_std,
        ),
    }
    values.append(
        {
            term.name: raw[term.name] * term.weight * (1.0 if term.is_event else cfg.sim.step_dt)
            for term in cfg.rewards
            if term.name in raw
        }
    )
    history.clear_events()
valid = data["age"] * cfg.sim.step_dt > 2.0
comparison = {}
for key in values[0]:
    new = torch.stack([row[key] for row in values])
    old = data.get("reward/" + key, torch.zeros_like(new))
    comparison[key] = {
        "before_per_second": float(old[valid].mean() / cfg.sim.step_dt),
        "after_per_second": float(new[valid].mean() / cfg.sim.step_dt),
    }
result = {
    "method": "Re-score the same flat baseline trajectory; contact booleans encode the original 1 N threshold.",
    "valid_env_seconds": float(valid.sum() * cfg.sim.step_dt),
    "comparison": comparison,
}
(directory / "baseline_rescoring.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
