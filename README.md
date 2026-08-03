# PureRL: TienKung Multi-Terrain Locomotion

PureRL provides standalone Isaac Sim 5.1 locomotion environments for the
20-DOF TienKung2 Lite humanoid. It uses direct Isaac Sim tensor APIs,
Gymnasium, PyTorch, and RSL-RL 3.1. No Isaac Lab checkout or `isaaclab*`
Python package is required.

## Tasks

| Gym ID | Purpose |
| --- | --- |
| `PureRL-Velocity-Flat-TienKung-v0` | Flat-ground training |
| `PureRL-Velocity-Flat-TienKung-Play-v0` | Flat-ground evaluation |
| `PureRL-Velocity-Rough-TienKung-v0` | Generated-terrain curriculum training |
| `PureRL-Velocity-Rough-TienKung-Play-v0` | Generated-terrain evaluation |

All tasks use 20 joint-position residual actions and the same 259-dimensional
policy observation. The final 187 values are a pelvis-yaw-frame terrain height
scan, so flat checkpoints can be loaded directly into rough environments.

## Installation

Isaac Sim 5.1 requires Python 3.11. NVIDIA packages require acceptance of the
Isaac Sim EULA and access to NVIDIA's Python package index.

```bash
cd /home/vega/Project/Locomotion/PureRL
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install --extra-index-url https://pypi.nvidia.com -e "source/purerl[sim,dev]"
```

Validate the local package without starting Isaac Sim:

```bash
python scripts/tools/validate_urdf.py
python scripts/tools/check_task_config.py
pytest -q
```

## Simulator Checks

The direct backend and complete environments can be checked independently:

```bash
python scripts/tools/check_direct_backend.py --num-envs 2 --steps 16
python scripts/tools/check_flat_env.py --num-envs 32 --steps 1000 \
  --random-actions --check-selective-reset
python scripts/tools/check_rough_env.py --num-envs 7 --steps 64 \
  --terrain-rows 2 --terrain-cols 7
```

## Training

Run a short flat PPO smoke test:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --num-envs 32 --max-iterations 2 --num-steps-per-env 8
```

Run full flat training, then initialize rough training from its policy:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 --num-envs 4096

python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Rough-TienKung-v0 --num-envs 4096 \
  --pretrained-checkpoint /absolute/path/to/model_2499.pt
```

Checkpoints are written below `logs/rsl_rl/tienkung_flat` and
`logs/rsl_rl/tienkung_rough`. Start with 2048 environments if the default 4096
exceeds available GPU memory.

## Evaluation And Export

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --num-envs 16 --checkpoint /absolute/path/to/model.pt
```

The play command exports `policy.pt` and `policy.onnx` into an `exported`
directory next to the checkpoint. Use `--no-export` to run inference only.

## Runtime Contract

- Physics frequency: 200 Hz (`dt=0.005`).
- Policy frequency: 50 Hz (`decimation=4`).
- Action: 20 position residuals with a `0.5 rad` scale.
- Contact history update: every physics step.
- Height scan update: every policy step.
- Rough terrain: flat, random rough, slopes, stairs, and random blocks.
- Selective reset and terrain-level curriculum are batched by environment ID.
- Domain events: pelvis mass/COM, PD gains, joint/root reset pose, external
  wrench, and periodic planar velocity pushes.

The current direct backend still uses fixed contact-material friction and does
not yet apply policy-observation noise. These remaining alignment items are
tracked in `ISAACLAB_DECOUPLING_REFACTOR_PLAN.md`.
