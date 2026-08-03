# PureRL: TienKung Multi-Terrain Locomotion

Isaac Lab manager-based reinforcement learning environments for the 20-DOF
TienKung2 Lite humanoid. The task tracks planar velocity commands on flat and
procedurally generated rough terrain using RSL-RL PPO.

## Implemented Tasks

| Gym ID | Purpose |
| --- | --- |
| `PureRL-Velocity-Flat-TienKung-v0` | Flat-ground bootstrap training |
| `PureRL-Velocity-Flat-TienKung-Play-v0` | Flat-ground evaluation |
| `PureRL-Velocity-Rough-TienKung-v0` | Mixed-terrain curriculum training |
| `PureRL-Velocity-Rough-TienKung-Play-v0` | Mixed-terrain evaluation |

The rough task contains flat patches, random height fields, ascending and
descending slopes, ascending and descending stairs, and random blocks. Terrain
difficulty changes per environment based on commanded walking performance.

## Compatibility

This initial implementation targets the Isaac Lab checkout at
`/home/vega/IsaacLab` (`v2.2.1-143-g2ed331acfc`, framework extension `0.48.5`)
with the Isaac Sim 5.1 Python 3.11 runtime. Do not use the machine's Python 3.13
environment for simulation.

## Environment Setup

Choose either the existing Conda environment or a clean uv environment. Do not
activate both at the same time: `isaaclab.sh` gives an active Conda environment
priority over `VIRTUAL_ENV`.

### Existing Conda Environment

```bash
source /home/vega/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PURE_RL_ISAACLAB_ROOT=/home/vega/IsaacLab
cd /mnt/data/Project/Locomotion/PureRL
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p -m pip install -e source/purerl
```

### Clean uv Environment

The installed Isaac Sim 5.1 runtime requires Python 3.11. The commands below
create `.venv` inside PureRL and install the versions matched by the current
Isaac Lab checkout. Exit any active Conda environment before activating uv.

```bash
# Run `conda deactivate` first if CONDA_PREFIX is currently set.
export PURE_RL_ROOT=/mnt/data/Project/Locomotion/PureRL
export PURE_RL_ISAACLAB_ROOT=/home/vega/IsaacLab

cd ${PURE_RL_ROOT}
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate

# Isaac Sim is distributed from NVIDIA's Python package index.
uv pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

# Install the local Isaac Lab extensions, CUDA PyTorch, and RSL-RL 3.1.2.
cd ${PURE_RL_ISAACLAB_ROOT}
./isaaclab.sh -i rsl_rl

# Install this external project into the same uv environment.
cd ${PURE_RL_ROOT}
uv pip install -e source/purerl
```

Verify that the uv environment resolves the intended runtime:

```bash
python -c "import sys, isaaclab, isaacsim; print(sys.version); print(isaaclab.__file__)"
python scripts/tools/check_task_config.py --headless
```

After activation, either `python scripts/...` or
`${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/...` uses `.venv/bin/python`.
Leave the environment with `deactivate`.

Validate the robot files without starting Isaac Sim:

```bash
python3 scripts/tools/validate_urdf.py
```

Validate Gym registrations and construct both environment configurations inside
an Isaac Sim runtime:

```bash
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/tools/check_task_config.py --headless
```

## Smoke Test

Run a short flat-ground training job before launching a full experiment:

```bash
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --headless --num_envs 32 --max_iterations 2
```

## Training

Bootstrap locomotion on flat ground:

```bash
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --headless --num_envs 4096
```

Then initialize rough-terrain training from the flat checkpoint:

```bash
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --headless --num_envs 4096 \
  --pretrained_checkpoint /absolute/path/to/tienkung_flat/model_2500.pt
```

Flat and rough tasks both expose a 259-dimensional policy observation,
including the 187-point height scan, so their checkpoints are shape-compatible.

RSL-RL stores checkpoints under `logs/rsl_rl/tienkung_flat` and
`logs/rsl_rl/tienkung_rough`. On a 32 GB RTX 5090, begin with 2048 environments
if 4096 environments exceed available PhysX or policy memory.

## Evaluation And Export

```bash
${PURE_RL_ISAACLAB_ROOT}/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --num_envs 16 --checkpoint /absolute/path/to/model.pt
```

The play script exports `policy.pt` and `policy.onnx` next to the selected
checkpoint. The current policy observation order is inherited from Isaac Lab's
velocity task: base linear velocity, base angular velocity, projected gravity,
velocity command, relative joint positions, joint velocities, previous action,
and terrain height scan for the rough task.

## Control And Randomization

- Physics frequency: 200 Hz (`dt=0.005`).
- Policy frequency: 50 Hz (`decimation=4`).
- Action: 20 joint-position residuals with scale `0.5` rad.
- Foot contact bodies: `ankle_roll_l_link`, `ankle_roll_r_link`.
- Root body: `pelvis`; nominal root height: `0.89 m`.
- Randomization: friction, pelvis mass and center of mass, PD gains, joint reset
  pose, initial heading, and periodic lateral/forward pushes.

Actuator gains, effort/velocity limits, armature values, and the nominal stand
pose come from WBC-SONIC's TienKung configuration. The URDF and meshes are
vendored under `assets/robot_description/tienkung` so the project has no runtime
dependency on WBC-SONIC.
