# PureRL

[English](README.md) | 中文

用强化学习训练天工 2 Lite 人形机器人在平地、斜坡和楼梯上行走。基于 Isaac Sim 5.1 和 RSL-RL（PPO），无需安装 Isaac Lab。

## 安装

需要 Linux、可用的 NVIDIA 显卡及驱动、Python 3.11。以下命令均在仓库根目录执行。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export OMNI_KIT_ACCEPT_EULA=YES
python -m pip install --extra-index-url https://pypi.nvidia.com -e "source/purerl[sim,dev]"
python scripts/tools/check_task_config.py
```

安装命令会下载 Isaac Sim。`OMNI_KIT_ACCEPT_EULA=YES` 表示接受其许可协议；新终端中需重新激活虚拟环境并设置该变量。

## 训练

默认训练平地行走。先用小规模训练检查环境，再开始正式训练：

```bash
# 运行 32 个环境、5 轮训练
python scripts/rsl_rl/train.py --num-envs 32 --max-iterations 5 \
  --run-name flat_check --logger tensorboard

# 正式训练：默认 4096 个环境、10001 轮
python scripts/rsl_rl/train.py --run-name flat_seed5 --logger tensorboard
```

显存不足时，用 `--num-envs` 减少环境数量。模型和日志按运行时间分别保存在 `logs/rsl_rl/tienkung_flat/` 或 `logs/rsl_rl/tienkung_rough/` 下。

用训练好的平地模型开始崎岖地形训练（将 `/path/to/flat/model.pt` 替换为实际模型路径）：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --pretrained-checkpoint /path/to/flat/model.pt \
  --run-name rough_seed5 --logger tensorboard
```

- **查看曲线**：运行 `tensorboard --logdir logs/rsl_rl`，打开终端显示的地址。
- **使用 W&B**：先运行 `wandb login`，再去掉示例中的 `--logger tensorboard`。默认项目为 `purerl`。
- **中断后续训**：在原训练命令中添加 `--resume --load-run /path/to/run --load-checkpoint model_3000.pt`；`--max-iterations` 表示再训练多少轮。不要同时使用 `--pretrained-checkpoint`，它会重新初始化优化器。

## 评估与导出

在有图形桌面的机器上，查看训练好的平地模型：

```bash
python scripts/rsl_rl/play.py --checkpoint /path/to/model.pt \
  --show --real-time --no-lidar --no-export
```

将模型路径换成实际文件。默认展示一个机器人，以 0.5 m/s 前进，运行约 20 秒。

- **指定速度**：添加 `--command VX VY WZ`，依次为前进速度、侧向速度（m/s）和转向角速度（rad/s）。
- **评估崎岖模型**：添加 `--task PureRL-Velocity-Rough-TienKung-Play-v0`，默认展示 7 种地形。
- **无桌面录视频**：将 `--show --real-time` 换成 `--video --video-path videos/demo.mp4`。
- **导出模型**：去掉 `--no-export`，会在模型旁的 `exported/` 目录生成 `policy.pt`（TorchScript）和 `policy.onnx`。

## 修改配置

配置位于 [config/presets](source/purerl/purerl/config/presets/)：

| 地形 | 训练环境 | 评估环境 | PPO 与日志 |
| --- | --- | --- | --- |
| 平地 | `flat_env.yaml` | `flat_play_env.yaml` | `flat_runner.yaml` |
| 崎岖 | `rough_env.yaml` | `rough_play_env.yaml` | `rough_runner.yaml` |

命令行参数优先于 YAML。自定义配置用 `--env-config` 和 `--runner-config` 指定，须提供完整文件；其中的 URDF 相对路径以 YAML 所在目录为基准。每次训练的实际配置保存在运行目录的 `params/` 中。

**当前模型使用 261 维观测。旧 259 维模型不能直接评估或续训，需要重新训练。**

更多参数：`python scripts/rsl_rl/train.py --help`、`python scripts/rsl_rl/play.py --help`。运行测试：`python -m pytest -q`。

详细说明：[训练修复与验证](TRAINING_FIXES.md) · [步态奖励设计](GAIT_REWARD_TUNING.md)
