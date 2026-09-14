# PureRL

English | [中文](README.zh-CN.md)

Train the TienKung 2 Lite humanoid to walk on flat ground, slopes, and stairs using reinforcement learning. Built on Isaac Sim 5.1 and RSL-RL (PPO). Isaac Lab is not required.

## Install

You need Linux, an NVIDIA GPU with a working driver, and Python 3.11. Run all commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export OMNI_KIT_ACCEPT_EULA=YES
python -m pip install --extra-index-url https://pypi.nvidia.com -e "source/purerl[sim,dev]"
python scripts/tools/check_task_config.py
```

The installation downloads Isaac Sim. Setting `OMNI_KIT_ACCEPT_EULA=YES` accepts its license agreement. Activate the virtual environment and set this variable again in each new terminal.

## Train

Training defaults to flat ground. Check the setup with a small run before starting full training:

```bash
# Check the setup: 32 environments, 5 iterations
python scripts/rsl_rl/train.py --num-envs 32 --max-iterations 5 \
  --run-name flat_check --logger tensorboard

# Full training: defaults to 4096 environments, 10001 iterations
python scripts/rsl_rl/train.py --run-name flat_seed5 --logger tensorboard
```

Reduce `--num-envs` if GPU memory is insufficient. Models and logs are saved in timestamped run directories under `logs/rsl_rl/tienkung_flat/` or `logs/rsl_rl/tienkung_rough/`.

Start rough-terrain training from a trained flat policy. Replace `/path/to/flat/model.pt` with your model file:

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --pretrained-checkpoint /path/to/flat/model.pt \
  --run-name rough_seed5 --logger tensorboard
```

- **View training curves:** run `tensorboard --logdir logs/rsl_rl` and open the address printed in the terminal.
- **Use W&B:** run `wandb login`, then omit `--logger tensorboard`. The default project is `purerl`.
- **Resume an interrupted run:** add `--resume --load-run /path/to/run --load-checkpoint model_3000.pt` to the original training command. `--max-iterations` counts additional iterations. Do not combine this with `--pretrained-checkpoint`, which starts a new optimizer.

## Evaluate and export

On a machine with a graphical desktop, view a trained flat policy:

```bash
python scripts/rsl_rl/play.py --checkpoint /path/to/model.pt \
  --show --real-time --no-lidar --no-export
```

Replace the model path with your file. By default, one robot walks forward at 0.5 m/s for about 20 seconds.

- **Set velocity:** add `--command VX VY WZ` for forward speed, lateral speed (m/s), and yaw rate (rad/s).
- **Evaluate a rough policy:** add `--task PureRL-Velocity-Rough-TienKung-Play-v0`. This displays seven terrain types by default.
- **Record without a desktop:** replace `--show --real-time` with `--video --video-path videos/demo.mp4`.
- **Export:** omit `--no-export` to generate `policy.pt` (TorchScript) and `policy.onnx` in an `exported/` directory next to the model.

## Configure

Configuration files are in [config/presets](source/purerl/purerl/config/presets/):

| Terrain | Training environment | Evaluation environment | PPO and logging |
| --- | --- | --- | --- |
| Flat | `flat_env.yaml` | `flat_play_env.yaml` | `flat_runner.yaml` |
| Rough | `rough_env.yaml` | `rough_play_env.yaml` | `rough_runner.yaml` |

Command-line options override YAML values. Use `--env-config` and `--runner-config` for custom, complete configuration files. Relative URDF paths are resolved from the YAML file's directory. Each run saves its resolved configuration in `params/`.

**Current policies use 261 observations. Legacy 259-dimensional models cannot be evaluated or resumed directly; start a new training run.**

For all options, run `python scripts/rsl_rl/train.py --help` or `python scripts/rsl_rl/play.py --help`. Run tests with `python -m pytest -q`.

Details: [Training fixes and validation](TRAINING_FIXES.md) · [Gait reward design](GAIT_REWARD_TUNING.md)
