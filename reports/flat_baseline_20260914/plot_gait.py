#!/usr/bin/env python3
"""Plot recorded contacts and motion; no simulator needed."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

directory = Path(__file__).parent
data = np.load(directory / "rollout.npz")
physics = np.load(directory / "physics_contacts.npz")
env_id = 0
start, end = 4.0, 5.0
time = (np.arange(data["age"].shape[0]) + 1) * 0.02
physics_time = (np.arange(physics["age"].shape[0]) + 1) * 0.005
selected = (time >= start) & (time < end)
physics_selected = (physics_time >= start) & (physics_time < end)
fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
fig.suptitle(
    "model_10000: low hops / shuffling instead of alternating steps\n"
    "Flat-Play, seed 5, command 0.5 m/s; example environment 0 of 8",
    fontsize=13,
)

# Repeat policy target samples to align with the four physics contact samples.
target = np.repeat(data["stance"][:, env_id], 4, axis=0)
contact_rows = np.column_stack((target, physics["contact"][:, env_id]))[physics_selected].T
axes[0].imshow(
    contact_rows,
    aspect="auto",
    interpolation="nearest",
    origin="upper",
    extent=[start, end, 3.5, -0.5],
    cmap=ListedColormap(["#e9edf2", "#247ba0"]),
    vmin=0,
    vmax=1,
)
axes[0].set_yticks(range(4), ["Target L", "Target R", "Actual L", "Actual R"])
axes[0].set_title("Dark = support contact. Actual contact sampled at 200 Hz.", loc="left", fontsize=10)

feet_mm = (data["foot_position"][selected, env_id, :, 2] - 0.0569) * 1000
axes[1].plot(time[selected], feet_mm[:, 0], label="Left", color="#247ba0")
axes[1].plot(time[selected], feet_mm[:, 1], label="Right", color="#e07a31")
axes[1].set_ylabel("Foot height* (mm)")
axes[1].set_ylim(-0.5, 8)
axes[1].set_title("Measured foot lift is a few millimetres; configured swing target is 60 mm.", loc="left", fontsize=10)
axes[1].legend(loc="upper right", ncol=2)

root_mm = data["root_position"][selected, env_id, 2] * 1000
axes[2].plot(time[selected], root_mm - root_mm.mean(), color="#854d9b", label="Checkpoint")
axes[2].set_ylabel("Pelvis height\nminus mean (mm)")
axes[2].set_title("Pelvis oscillation is approximately 12.5 Hz (configured gait cycle: 2 Hz).", loc="left", fontsize=10)
axes[2].legend(loc="upper right")

axes[3].plot(time[selected], data["velocity_yaw"][selected, env_id, 0], color="#247ba0", label="Actual")
axes[3].axhline(0.5, color="#e07a31", linestyle="--", label="Command")
axes[3].set_ylabel("Forward speed (m/s)")
axes[3].set_xlabel("Simulation time (s)")
axes[3].legend(loc="upper right", ncol=2)
for ax in axes[1:]:
    ax.grid(alpha=0.2)
axes[-1].set_xlim(start, end)
fig.supxlabel("* Foot body origin minus the configured sole offset; tilting is not compensated.", fontsize=9)
fig.savefig(directory / "gait_diagnostics.png", dpi=180)
fig.savefig(directory / "gait_diagnostics.pdf")
plt.close(fig)
print(directory / "gait_diagnostics.png")
