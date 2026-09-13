"""Read-only training audit; writes derived evidence beside this script."""
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
import csv
import json
import os

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "logs/rsl_rl/tienkung_flat/2026-09-12_23-29-34_flat_baseline_seed5"
CUTOFF = 5000
torch.set_num_threads(2)

events = EventAccumulator(str(RUN), size_guidance={"scalars": 0}).Reload()
data = {
    tag: np.array([[v.step, v.value, v.wall_time] for v in events.Scalars(tag) if v.step <= CUTOFF])
    for tag in events.Tags()["scalars"]
}
with (OUT / "scalars.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "iteration", "value", "wall_time"])
    for tag, rows in data.items():
        writer.writerows((tag, int(step), value, wall) for step, value, wall in rows)

windows = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 3000),
           (3000, 4000), (4000, 4500), (4500, 5001), (4901, 5001)]
summary = {"run": str(RUN), "cutoff_iteration": CUTOFF,
           "last_event_local_time": datetime.fromtimestamp(data["Train/mean_reward"][-1, 2]).isoformat(),
           "windows": {}, "outliers": {}, "checkpoints": {}}
for start, end in windows:
    metrics = {}
    for tag, rows in data.items():
        values = rows[(rows[:, 0] >= start) & (rows[:, 0] < end), 1]
        metrics[tag] = {"mean": float(values.mean()), "median": float(np.median(values)),
                        "min": float(values.min()), "max": float(values.max())}
    summary["windows"][f"{start}:{end}"] = metrics
for tag in ["Loss/value_function", "Episode/ang_vel_xy_l2", "Loss/surrogate"]:
    rows = data[tag]
    summary["outliers"][tag] = rows[np.argsort(np.abs(rows[:, 1]))[-8:][::-1], :2].tolist()

names = yaml.safe_load((RUN / "params/env.yaml").read_text())["robot"]["joint_names"]
for path in sorted(RUN.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
    if int(path.stem.split("_")[-1]) > CUTOFF:
        continue
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    summary["checkpoints"][path.name] = {
        "iteration": checkpoint["iter"],
        "learning_rates": [g["lr"] for g in checkpoint["optimizer_state_dict"]["param_groups"]],
        "all_model_tensors_finite": all(torch.isfinite(v).all().item() for v in state.values()),
        "noise_std": dict(zip(names, state["std"].tolist())),
    }

# Use the installed RSL-RL PPO/storage and the repository action manager.
# No simulator, optimizer updates, or observation normalization are involved.
from purerl.mdp.actions import JointPositionActionManager
from purerl.mdp.rewards import feet_contact_number
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

def reproduce_action_alias(clone_input):
    torch.manual_seed(5)
    obs = TensorDict({"policy": torch.randn(4, 259)}, batch_size=[4])
    with redirect_stdout(StringIO()):
        policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 20,
                             actor_hidden_dims=[16], critic_hidden_dims=[16],
                             actor_obs_normalization=False, critic_obs_normalization=False)
    ppo = PPO(policy, device="cpu")
    ppo.init_storage("rl", 4, 1, obs, [20])
    manager = JointPositionActionManager(torch.zeros(4, 20), scale=0.25, action_clip=None)
    with torch.inference_mode():
        actions = ppo.act(obs)
        executed = actions.clone()
        manager.process(actions.clone() if clone_input else actions)
        manager.reset(torch.tensor([0]))
        ppo.process_env_step(obs, torch.zeros(4), torch.tensor([True, False, False, False]), {})
        stored = ppo.storage.actions[0]
        policy.act(obs)
        log_ratio = policy.get_actions_log_prob(stored) - ppo.storage.actions_log_prob[0, :, 0]
        return {"stored_executed_max_error_per_env": (stored - executed).abs().amax(-1).tolist(),
                "log_ratio_without_policy_update": log_ratio.tolist(),
                "ratio_without_policy_update": log_ratio.exp().tolist()}

summary["action_alias_reproduction"] = {
    "current_behavior": reproduce_action_alias(False),
    "cloned_input_control": reproduce_action_alias(True),
}
contact = torch.tensor([[True, False]])
command = torch.tensor([[0.3, 0.0, 0.0]])
summary["hidden_phase_reward"] = {
    "left_stance": feet_contact_number(contact, torch.tensor([[True, False]]), command).item(),
    "right_stance": feet_contact_number(contact, torch.tensor([[False, True]]), command).item(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/purerl_training_audit_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)

def curve(ax, tag, label, scale=1, smooth=100):
    rows = data[tag]
    x, y = rows[:, 0], rows[:, 1] * scale
    ax.plot(x, y, alpha=0.15, linewidth=0.6)
    if smooth:
        ax.plot(x[smooth-1:], np.convolve(y, np.ones(smooth)/smooth, "valid"), label=label, linewidth=1.8)
    else:
        ax.plot(x, y, label=label, linewidth=1)

curve(axes[0, 0], "Train/mean_reward", "Return")
axes[0, 0].set_title("Early learning, then oscillation and regression")
axes[0, 0].set_ylabel("Training episode return")
curve(axes[0, 1], "Train/mean_episode_length", "Episode duration", scale=0.02)
axes[0, 1].axhline(24, color="black", linestyle="--", label="24 s limit")
axes[0, 1].set_title("Survival remains below episode limit")
axes[0, 1].set_ylabel("Seconds")
for tag, label in [("base_height", "Base height"), ("joint_deviation_arms", "Arm deviation"),
                   ("flat_orientation_l2", "Tilt penalty"), ("action_rate_l2", "Action rate")]:
    curve(axes[1, 0], "Episode/" + tag, label)
axes[1, 0].set_title("Integrated reward components")
axes[1, 0].set_ylabel("Weighted return per completed episode")
for tag, label in [("feet_air_time", "Air time"), ("feet_clearance", "Foot clearance")]:
    curve(axes[1, 1], "Episode/" + tag, label)
axes[1, 1].set_yscale("symlog", linthresh=1e-5)
axes[1, 1].set_title("Very small foot lift / air time rewards")
axes[1, 1].set_ylabel("Weighted return per completed episode")
curve(axes[2, 0], "Loss/value_function", "Value loss", smooth=0)
axes[2, 0].set_yscale("log")
axes[2, 0].annotate("iter 3514: 2.08e9", xy=(3514, 2.082192e9), xytext=(1800, 1e7),
                    arrowprops={"arrowstyle": "->"})
axes[2, 0].set_title("Isolated extreme value target / loss")
noise = summary["checkpoints"]["model_5000.pt"]["noise_std"]
colors = ["#d55e00" if noise[n] <= 0.051 else "#0072b2" for n in names]
axes[2, 1].bar(np.arange(20), [noise[n] for n in names], color=colors)
axes[2, 1].set_xticks(np.arange(20), [n.replace("_joint", "") for n in names], rotation=90, fontsize=8)
axes[2, 1].set_title("Checkpoint 5000: per-joint exploration std")
axes[2, 1].set_ylabel("Raw action standard deviation")
for ax in axes.flat:
    ax.grid(alpha=0.2)
    if ax is not axes[2, 1]:
        ax.set_xlabel("Training iteration")
        ax.legend(fontsize=8)
fig.suptitle("PureRL flat_baseline_seed5 | iterations 0-5000 | bold curves: trailing 100 iterations", fontsize=14)
fig.savefig(OUT / "training_diagnostics.png", dpi=170)
fig.savefig(OUT / "training_diagnostics.pdf")
print(json.dumps({"artifacts": str(OUT), "cutoff": CUTOFF,
                  "action_alias_reproduction": summary["action_alias_reproduction"]}, indent=2))
