# PureRL：天工（TienKung）多地形运动控制

PureRL 为 20 自由度的天工2 Lite 人形机器人提供**独立于 Isaac Lab** 的 Isaac Sim 5.1 运动控制环境。它直接使用 Isaac Sim 的张量 API、Gymnasium、PyTorch 和 RSL-RL 3.1，无需检出 Isaac Lab 代码库，也不需要任何 `isaaclab*` Python 包。

## 任务

| Gym ID | 用途 |
| --- | --- |
| `PureRL-Velocity-Flat-TienKung-v0` | 平地训练 |
| `PureRL-Velocity-Flat-TienKung-Play-v0` | 平地评估 |
| `PureRL-Velocity-Rough-TienKung-v0` | 生成地形课程训练 |
| `PureRL-Velocity-Rough-TienKung-Play-v0` | 生成地形评估 |

所有任务都使用 20 维关节位置残差动作，以及相同的 259 维策略观测。最后 187 维是 pelvis-yaw 坐标系下的地形高度扫描，因此平地 checkpoint 可以直接加载到崎岖地形环境中。

## 安装

Isaac Sim 5.1 需要 Python 3.11。NVIDIA 相关包需要接受 Isaac Sim EULA，并能访问 NVIDIA 的 Python 包索引。

```bash
cd /home/vega/Project/Locomotion/PureRL
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install --extra-index-url https://pypi.nvidia.com -e "source/purerl[sim,dev]"
export OMNI_KIT_ACCEPT_EULA=YES
```

无需启动 Isaac Sim 即可验证本地包：

```bash
python scripts/tools/validate_urdf.py
python scripts/tools/check_task_config.py
pytest -q
```

## 仿真检查

直接后端（direct backend）和完整环境可以分别检查：

```bash
python scripts/tools/check_direct_backend.py --num-envs 2 --steps 16
python scripts/tools/check_flat_env.py --num-envs 32 --steps 1000 \
  --random-actions --check-selective-reset
python scripts/tools/check_rough_env.py --num-envs 7 --steps 64 \
  --terrain-rows 2 --terrain-cols 7
python scripts/tools/check_camera_video.py --frames 12
python scripts/tools/check_legacy_trajectory.py
```

在训练规模下测量吞吐量、张量稳定性、终止原因和 CUDA 显存：

```bash
python scripts/tools/benchmark_env.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --num-envs 2048 --warmup-steps 32 --steps 100 --check-interval 25

python scripts/tools/benchmark_env.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --num-envs 32 --warmup-steps 32 --steps 10000 \
  --memory-baseline-step 2000 --progress-interval 1000 --random-actions
```

## 配置

环境和 RSL-RL 训练参数都以**完整、可直接编辑的 YAML** 形式存放在 `source/purerl/purerl/config/presets` 下：

| 任务变体 | 环境配置 | Runner 配置 |
| --- | --- | --- |
| 平地训练 | `flat_env.yaml` | `flat_runner.yaml` |
| 平地评估 | `flat_play_env.yaml` | `flat_runner.yaml` |
| 崎岖训练 | `rough_env.yaml` | `rough_runner.yaml` |
| 崎岖评估 | `rough_play_env.yaml` | `rough_runner.yaml` |

每个环境 preset 包含仿真、场景、机器人、动作、观测、传感器、视觉、指令、随机化、地形和奖励的全部取值；每个 runner preset 包含策略、PPO、checkpoint 和日志的全部取值。各 preset **互不继承**，因此单个文件即完整展示了该变体加载后的配置。相对文件路径（包括 `robot.urdf_path`）都相对于声明它的 YAML 文件解析。

训练或评估时都可以传入自定义的完整配置：

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

`--agent-config` 仍可作为 `--runner-config` 的别名。配置解析顺序为：选定的 YAML preset → dataclass 类型转换与校验 → 显式 CLI 覆盖（如 `--num-envs`、`--device`、`--max-iterations`）。最终解析出的配置会写入本次运行的 `params/env.yaml` 和 `params/agent.yaml`。

Play preset 会启用一个头部安装的 RTX 3D LiDAR，使用本地 `OS1_REV6_32ch10hz512res` 配置。导入的机器人会合并固定关节，因此当 `head` prim 存在时后端挂载到 `head`，否则按配置的头部偏移挂载到 `pelvis`。LiDAR 字段位于 `sensors.lidar` 下；`visuals.sky_color`、`sky_intensity`、`ground_color`、`terrain_color` 控制仿真背景和地形外观。训练 preset 保持 LiDAR 关闭，避免为大规摸向量化运行创建 RTX 渲染产品。LiDAR 点云**有意地不**追加到 259 维运动控制观测中，以保留既有 checkpoint 的兼容性。

### 步态奖励

平地与崎岖运动 preset 在标准的速度跟踪和能量项之上，新增了四个参考 humanoid-gym 的步态奖励项：

| 奖励项 | 权重 | 作用 |
| --- | --- | --- |
| `feet_contact_number` | 1.2 | 惩罚与开环步态相位不一致的脚部接触 |
| `feet_distance` | 0.2 | 将步宽约束在 `[foot_min_dist, foot_max_dist]` 内 |
| `base_height` | 0.2 | 保持 pelvis 离地高度为 `default_root_height` |
| `feet_clearance` | 1.0 | 让摆动脚抬升到 `target_feet_height` |

步态参数位于每个环境 YAML 的 `gait:` 块中，且是天工专属的（腿长 `0.8 m`、脚底偏移 `0.0569 m`）。实测数值与调参流程见 [`GAIT_REWARD_TUNING.md`](GAIT_REWARD_TUNING.md)。

## 训练

跑一个简短的平地 PPO 冒烟测试：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --num-envs 32 --max-iterations 2 --num-steps-per-env 8
```

跑完整平地训练，然后用其策略初始化崎岖地形训练：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --run-name flat_baseline_seed5

python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Rough-TienKung-v0 \
  --run-name rough_transfer_seed5 \
  --pretrained-checkpoint /absolute/path/to/model_3000.pt
```

checkpoint 写入 `logs/rsl_rl/tienkung_flat` 和 `logs/rsl_rl/tienkung_rough`。两个训练 preset 默认 4096 个环境，并使用当前已认证账号在 `purerl` 项目下进行在线 W&B 记录。若 4096 超出可用显存，先用 `--num-envs 2048`。

默认训练 preset 改编自 [roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym) 的 XBot-L PPO 配置：24 秒 episode、每次 rollout 60 个策略步、学习率 `1e-5`、两个学习 epoch、`gamma=0.994`、`lambda=0.9`，以及更宽的 `[768, 256, 128]` critic。兼容的环境设置使用 `0.25` 的关节目标动作尺度、8 秒指令重采样，以及 `x=[-0.3, 0.6]`、`y=[-0.3, 0.3]`、`yaw=[-0.3, 0.3]` 的指令范围。天工专属的 PD 增益、20 维动作观测、高度扫描、奖励函数与权重、终止行为和策略噪声防护仍保留在本地，因为 XBot-L 的这些值无法作为配置常量直接迁移。平地与崎岖 preset 都使用 10001 次 PPO 迭代。

动作尺度和 PPO 超参数与早期 PureRL preset 不同。此变更后请**重新开始新的平地运行**；不要续训或迁移用旧 `0.5` 动作尺度训练的 checkpoint。

用 `--resume` 恢复策略、优化器状态和已存储的 RSL-RL 迭代。不带选择器时加载自然最新的匹配 run 和 checkpoint；存在多个实验时用显式选择器更安全：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --num-envs 4096 \
  --resume \
  --load-run /absolute/path/to/previous/run \
  --load-checkpoint model_3000.pt \
  --max-iterations 7000
```

对续训而言，`--max-iterations` 表示**额外**的 PPO 迭代次数，而非绝对最终迭代。上面的例子在 3001 迭代时续训 `model_3000.pt`，再训练 7000 次迭代，最终接近 `model_10000.pt`。此工作流**不要**使用 `--pretrained-checkpoint`：该选项只迁移策略权重，并刻意开启新的优化器和迭代计数。

在线 W&B 记录默认开启，因此命名训练运行只需：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --run-name flat_baseline_seed5
```

W&B 使用当前已认证账号和 runner preset 中的 `purerl` 项目。仅覆盖单次运行需要的值。例如，本地保留同样的记录但不上传：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --run-name flat_offline_debug \
  --wandb-mode offline
```

用 `--logger tensorboard` 可为某次运行禁用 W&B。自定义 W&B 项目、实体和标签仍可通过 `--wandb-project`、`--wandb-entity`、`--wandb-tags` 或对应的 runner YAML 字段设置。

`--run-name` 同时命名本地 checkpoint 目录和 W&B run。RSL-RL 会在显示名称前加上启动时间戳，因此 `--run-name flat_baseline_seed5` 显示为 `2026-08-08_12-30-00_flat_baseline_seed5`。使用能标识地形、实验变体和种子的名称。该值也可作为 `run_name` 存入 runner YAML；命令行选项会覆盖配置值。

要在续训本地 checkpoint 的同时把指标追加到原在线 W&B run，提供其 run ID 并使用 `--wandb-resume must`：

```bash
python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --resume \
  --load-run /absolute/path/to/previous/run \
  --load-checkpoint model_3000.pt \
  --max-iterations 7000 \
  --run-name flat_baseline_seed5 \
  --wandb-run-id EXISTING_RUN_ID \
  --wandb-resume must \
  --wandb-tags flat baseline
```

续训的 checkpoint 和本地 W&B 文件会写入一个新的带时间戳的运行目录；W&B run ID 保持远端图表连续。在此工作流中，PureRL 允许 RSL-RL 刷新既有 run 的 `log_dir` 和序列化训练配置，它们在续训后会有合法差异。若指定 ID 无法续训时允许新建 W&B run，则改用 `--wandb-resume allow`。保存、续训、平地转崎岖、导出和离线 W&B 记录可一起通过以下命令验证：

```bash
python scripts/tools/check_rsl_workflow.py --wandb-offline
```

策略探索标准差由 runner YAML 中的 `min_action_noise_std` 和 `max_action_noise_std` 约束。加载平均标准差超过 `max_checkpoint_noise_std` 的 checkpoint 默认会被拒绝。`--allow-unsafe-checkpoint` 仅供诊断，不健康的 checkpoint 不应继续或迁移训练。未裁剪动作修复之前产生的运行必须从头重训；尤其不要使用 W&B `Policy/mean_noise_std` 已增长超过配置上限的 checkpoint。

## 评估与导出

使用与 checkpoint 训练地形相匹配的 play 任务。

在交互式 Isaac Sim 窗口中评估平地 checkpoint（单机器人、固定前进速度指令）：

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Flat-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/flat_run/model_10000.pt \
  --command 0.5 0.0 0.0 \
  --show \
  --real-time \
  --no-export
```

Flat Play 已默认单机器人、1000 策略步和 `[0.5, 0.0, 0.0]` 指令，因此当这些默认值合适时 `--num-envs`、`--steps`、`--command` 都可以省略。省略 `--checkpoint` 则从 `logs/rsl_rl/tienkung_flat` 下最新 run 中加载最新的 `model_*.pt`，或用 `--load-run` 和 `--load-checkpoint` 选择某个 run 而不写绝对路径。修改 `--command VX VY WZ` 可评估前进、侧移和偏航指令跟踪。

对于崎岖 checkpoint，打开对应的生成地形 play 任务。Rough Play 现在默认**多地形评估**：七个环境，各自位于不同地形类型的最高难度等级（`selected_level: 4`），一次评估即可并排覆盖全部七种地形类型：

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/healthy_model.pt \
  --command 0.5 0.0 0.0 \
  --show \
  --real-time \
  --no-export
```

七个机器人分别置于 `flat`、`random_rough`、`slope_up`、`slope_down`、`stairs_up`、`stairs_down`、`random_blocks`。只想看单一地形时，传入 `--terrain-patch` 和 `--terrain-level`（它们会覆盖 YAML 中的 `selected_patch: null` / `selected_level: 4`）：

```bash
python scripts/rsl_rl/play.py \
  --task PureRL-Velocity-Rough-TienKung-Play-v0 \
  --checkpoint /absolute/path/to/healthy_model.pt \
  --num-envs 1 \
  --terrain-patch stairs_up \
  --terrain-level 4 \
  --command 0.5 0.0 0.0 \
  --show --real-time --no-export
```

`--show` 使用交互式 `human` 渲染模式，提交视口帧，并将相机对准所选环境原点。可用的 patch 名称为 `flat`、`random_rough`、`slope_up`、`slope_down`、`stairs_up`、`stairs_down`、`random_blocks`。评估会打印解析后的指令、地形 patch/等级/列、LiDAR 挂载/配置、周期性机器人位置和最新 LiDAR 点数。用 `--no-lidar` 关闭评估传感器，或用 `--lidar` 为自定义配置启用它；`--log-interval 0` 可关闭进度行。

录制 1000 个策略步（默认 50 Hz 策略频率下约 20 秒），不打开交互窗口：

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

除非设置了 `--no-export`，play 命令会把 `policy.pt` 和 `policy.onnx` 导出到 checkpoint 旁的 `exported` 目录。传入 `--command VX VY WZ` 可指定不同的固定指令，或通过 `--env-config` 提供完整的自定义环境 YAML 用于随机指令测试。

## 运行时契约

- 物理频率：200 Hz（`dt=0.005`）。
- 策略频率：50 Hz（`decimation=4`）。
- 动作：20 维未裁剪位置残差，`0.25 rad` 尺度。探索噪声在 runner 配置中单独约束，使 PPO 对数概率与实际执行的动作保持一致。
- 速度指令使用采样的世界朝向目标、增益为 `0.5` 的比例偏航控制，以及 10% 的站立环境比例。
- 接触历史更新：每个物理步。
- 高度扫描更新：每个策略步。
- 崎岖地形：平地、随机粗糙、斜坡、楼梯和随机方块。
- 崎岖环境使用显式 PhysX 碰撞组，使分配到同一地形 tile 的机器人之间不会相互碰撞。
- 选择性重置和地形等级课程按环境 ID 批处理。
- 域随机事件：pelvis 质量/质心、PD 增益、关节/根重置姿态、外部力旋，以及周期性平面速度推挤。
- 机器人接触摩擦按环境从 64 个系数桶中随机化，并写入每个机器人碰撞形状。
- 训练观测对基座速度、投影重力、关节状态和高度扫描施加逐项均匀扰动。Play 任务关闭扰动；高度扫描值始终裁剪到 `[-1, 1]`。
- Legacy fixture 锁定了历史 Isaac Lab/PureRL 修订版本、关节顺序、逐项奖励数值、重置行为、地形分配，以及一段 16 步动作轨迹。直接后端按文档化的迁移容差进行核对，而非声称与物理逐位一致。
