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
export OMNI_KIT_ACCEPT_EULA=YES
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
python scripts/tools/check_camera_video.py --frames 12
python scripts/tools/check_legacy_trajectory.py
```

Measure throughput, tensor stability, termination causes, and CUDA memory at
training scale:

```bash
python scripts/tools/benchmark_env.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --num-envs 2048 --warmup-steps 32 --steps 100 --check-interval 25

python scripts/tools/benchmark_env.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --num-envs 32 --warmup-steps 32 --steps 10000 \
  --memory-baseline-step 2000 --progress-interval 1000 --random-actions
```

## Configuration

Environment and RSL-RL training values are stored as complete, directly
editable YAML presets under `source/purerl/purerl/config/presets`:

| Task variant | Environment config | Runner config |
| --- | --- | --- |
| Flat train | `flat_env.yaml` | `flat_runner.yaml` |
| Flat play | `flat_play_env.yaml` | `flat_runner.yaml` |
| Rough train | `rough_env.yaml` | `rough_runner.yaml` |
| Rough play | `rough_play_env.yaml` | `rough_runner.yaml` |

Each environment preset includes simulation, scene, robot, action,
observation, sensor, command, randomization, terrain, and reward values. Each
runner preset includes policy, PPO, checkpoint, and logger values. Presets do
not inherit from one another, so a file shows the complete configuration that
will be loaded for that variant. Relative filesystem paths, including
`robot.urdf_path`, are resolved relative to the YAML file that declares them.

Pass custom complete configs to either training or playback:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --env-config configs/my_flat_env.yaml \
  --runner-config configs/my_flat_runner.yaml

python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Flat-TienKung-Play-v0 \
  --env-config configs/my_flat_play_env.yaml \
  --runner-config configs/my_flat_runner.yaml \
  --checkpoint /absolute/path/to/model.pt
```

`--agent-config` remains available as an alias for `--runner-config`. Values
are resolved in this order: the selected YAML preset, dataclass type conversion
and validation, then explicit CLI overrides such as `--num-envs`, `--device`,
or `--max-iterations`. The final resolved configs are written to the run's
`params/env.yaml` and `params/agent.yaml` files.

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
  --pretrained-checkpoint /absolute/path/to/model_3000.pt
```

Checkpoints are written below `logs/rsl_rl/tienkung_flat` and
`logs/rsl_rl/tienkung_rough`. Start with 2048 environments if the default 4096
exceeds available GPU memory.

The default training presets are adapted from the XBot-L PPO configuration in
[roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym): 24-second
episodes, 60 policy steps per rollout, 3001 PPO iterations, a `1e-5` learning
rate, two learning epochs, `gamma=0.994`, `lambda=0.9`, and a wider
`[768, 256, 128]` critic. Compatible environment settings use a `0.25` joint
target action scale, 8-second command resampling, and command ranges of
`x=[-0.3, 0.6]`, `y=[-0.3, 0.3]`, and `yaw=[-0.3, 0.3]`. TienKung-specific PD
gains, 20-action observations, height scanning, reward functions and weights,
termination behavior, and the policy-noise guard remain local because the
XBot-L values are not transferable as configuration constants.

The action scale and PPO hyperparameters differ from earlier PureRL presets.
Start a new Flat run after this change; do not resume or transfer a checkpoint
trained with the old `0.5` action scale.

Resume from the latest matching checkpoint in a run, or pass explicit run and
checkpoint paths:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 --resume \
  --load-run /absolute/path/to/previous/run \
  --load-checkpoint /absolute/path/to/model_100.pt
```

Use W&B online or offline logging with the same run directory as checkpoints
and serialized configs:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --logger wandb --wandb-mode online --wandb-project purerl \
  --wandb-entity YOUR_ENTITY --wandb-tags flat baseline

python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --logger wandb --wandb-mode offline --wandb-project purerl
```

For an online W&B run continuation, also provide `--wandb-run-id` and set
`--wandb-resume allow`. Validate save, resume, flat-to-rough transfer, export,
and offline W&B logging together with:

```bash
python scripts/tools/check_rsl_workflow.py --wandb-offline
```

The policy exploration standard deviation is constrained by
`min_action_noise_std` and `max_action_noise_std` in the runner YAML. Loading a
checkpoint whose mean standard deviation exceeds `max_checkpoint_noise_std`
is rejected by default. `--allow-unsafe-checkpoint` exists for diagnostics,
but an unhealthy checkpoint should not be used to continue or transfer
training. Runs produced before the unclipped-action fix must be retrained from
scratch; in particular, do not use a checkpoint whose W&B
`Policy/mean_noise_std` has grown beyond the configured limit.

## Evaluation And Export

Use the play task that matches the checkpoint's training terrain. For example,
open an interactive Isaac Sim window for a rough-terrain policy and run it at
real-time speed with one robot:

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/healthy_model.pt \
  --num-envs 1 \
  --steps 1000 \
  --command 0.5 0.0 0.0 \
  --terrain-patch random_rough \
  --terrain-level 4 \
  --show \
  --real-time \
  --no-export
```

`--show` uses the interactive `human` render mode, submits viewport frames,
and aims the camera relative to the selected environment origin. Rough Play
defaults to one robot, a fixed `[0.5, 0.0, 0.0]` velocity command, and the
highest-level `random_rough` tile. Select another generated patch explicitly:

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/healthy_model.pt \
  --terrain-patch stairs_up \
  --terrain-level 4 \
  --command 0.5 0.0 0.0 \
  --show --real-time --no-export
```

Available patch names are `flat`, `random_rough`, `slope_up`, `slope_down`,
`stairs_up`, `stairs_down`, and `random_blocks`. Playback prints the resolved
command, terrain patch/level/column, and periodic robot positions. Set
`--log-interval 0` to disable progress lines.

Record 1000 policy steps, approximately 20 seconds at the default 50 Hz
policy frequency, without opening the interactive window:

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/healthy_model.pt \
  --num-envs 1 \
  --steps 1000 \
  --video \
  --video-length 1000 \
  --video-fps 50 \
  --video-path videos/rough_policy.mp4 \
  --no-export
```

The play command exports `policy.pt` and `policy.onnx` into an `exported`
directory next to the checkpoint unless `--no-export` is set. Pass
`--command VX VY WZ` for a different fixed command, or provide a complete
custom environment YAML through `--env-config` for randomized command tests.

## Runtime Contract

- Physics frequency: 200 Hz (`dt=0.005`).
- Policy frequency: 50 Hz (`decimation=4`).
- Action: 20 unclipped position residuals with a `0.25 rad` scale. Exploration
  noise is bounded separately in the runner configuration so PPO log
  probabilities and executed actions remain consistent.
- Velocity commands use a sampled world-heading target, proportional yaw
  control with gain `0.5`, and a 10% standing-environment ratio.
- Contact history update: every physics step.
- Height scan update: every policy step.
- Rough terrain: flat, random rough, slopes, stairs, and random blocks.
- Rough environments use explicit PhysX collision groups so robots assigned to
  the same terrain tile cannot collide with each other.
- Selective reset and terrain-level curriculum are batched by environment ID.
- Domain events: pelvis mass/COM, PD gains, joint/root reset pose, external
  wrench, and periodic planar velocity pushes.
- Robot contact friction is randomized per environment from 64 coefficient
  buckets and written to every robot collision shape.
- Training observations apply term-specific uniform corruption to base
  velocities, projected gravity, joint state, and height scan. Play tasks
  disable corruption; height scan values are always clipped to `[-1, 1]`.
- Legacy fixtures lock the historical Isaac Lab/PureRL revisions, articulation
  joint order, term-level reward math, reset behavior, terrain assignment, and
  a 16-step action trajectory. The direct backend is checked against documented
  migration tolerances rather than claimed to be bitwise physics-identical.
