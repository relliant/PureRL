# PureRL 解除 Isaac Lab 依赖重构计划

## 1. 背景与目标

当前 PureRL 是一个 Isaac Lab 外部任务扩展。环境生命周期、场景和传感器、MDP term、地形、任务注册以及 RSL-RL 适配均直接依赖以下包：

- `isaaclab`
- `isaaclab_assets`
- `isaaclab_tasks`
- `isaaclab_rl`

本次重构的目标是保留 Isaac Sim 5.1、PyTorch、Gymnasium 和 `rsl-rl-lib`，彻底移除上述 Isaac Lab 包以及外部 `/home/vega/IsaacLab` 源码 checkout。重构完成后，PureRL 应能从干净 clone 独立安装、训练、评估和导出策略。

本计划不包含移除 Isaac Sim。如果后续要求同时移除 Isaac Sim，则需要迁移至 MuJoCo 等其他仿真后端，应作为独立项目规划。

## 2. 重构原则

1. 面向当前 TienKung locomotion 任务实现最小且完整的运行框架，不复制整个 Isaac Lab。
2. 保持现有四个 Gym task ID、259 维策略观测和 checkpoint 输入兼容性。
3. 所有隐式继承配置改为显式、可序列化的本地 dataclass 配置。
4. 仿真循环中使用批量 tensor API，不进行逐环境 USD 查询或 Python 循环。
5. Isaac Sim 相关 import 隔离在启动和仿真模块中，使配置、MDP 数学和地形生成可执行 CPU 单元测试。
6. 先建立行为基准，再替换实现；每个阶段都必须有独立验收条件。
7. 不直接复制 Isaac Lab 大段实现。若必须移植小段算法，需要记录来源并保留对应许可证声明。

## 3. 最终依赖边界

重构后的运行时依赖应限定为：

- Python 3.11
- Isaac Sim 5.1
- PyTorch/CUDA
- Gymnasium
- `rsl-rl-lib`
- NumPy
- Warp 或 Isaac Sim 提供的等价 GPU raycast API
- 用于配置序列化、mesh 构建和导出的少量显式依赖

以下内容不得再作为运行时或开发环境前置条件：

- 任意外部 IsaacLab 仓库目录
- `isaaclab.sh` 或 `isaaclab.bat`
- 任意 `isaaclab*` Python package
- Isaac Lab extension manifest 和 extension discovery
- `isaaclab_tasks` 的 Hydra task registry

## 4. 目标架构

```text
train.py / play.py
        |
        +-- PureRL task registry and explicit configs
        +-- PureRL RSL-RL adapter
                    |
             TienKungLocomotionEnv
                    |
       +------------+-------------+
       |            |             |
  lifecycle     MDP managers   scene/sensors
                                    |
                             Isaac Sim API
```

建议将 Python 包调整为：

```text
source/purerl/purerl/
├── app/
│   └── launcher.py
├── config/
│   ├── env.py
│   ├── robot.py
│   └── runner.py
├── sim/
│   ├── context.py
│   ├── scene.py
│   └── articulation.py
├── sensors/
│   ├── contact.py
│   └── height_scan.py
├── terrain/
│   ├── config.py
│   ├── generator.py
│   └── curriculum.py
├── mdp/
│   ├── actions.py
│   ├── commands.py
│   ├── observations.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── events.py
│   └── curriculum.py
├── envs/
│   ├── base.py
│   └── tienkung_locomotion.py
├── rl/
│   ├── rsl_vec_env.py
│   ├── runner_cfg.py
│   ├── checkpoint.py
│   └── export.py
└── registry.py
```

## 5. 必须保持的行为契约

### 5.1 动作

- 动作维度保持 20。
- joint 顺序保持稳定并显式写入配置。
- 动作为默认站立姿态附近的 joint-position residual。
- action scale 保持 `0.5 rad`。
- physics 频率保持 200 Hz，policy 频率保持 50 Hz，decimation 保持 4。

### 5.2 策略观测

策略观测总维度保持 259，顺序固定为：

1. base linear velocity：3
2. base angular velocity：3
3. projected gravity：3
4. velocity command：3
5. relative joint positions：20
6. joint velocities：20
7. previous action：20
8. terrain height scan：187

Flat 和 rough 环境都必须输出相同维度，使 flat checkpoint 能继续用于 rough 训练。

### 5.3 环境 step 顺序

环境 step 的处理顺序必须固定并通过测试锁定：

1. 裁剪并处理 action。
2. 将 action 转换为 PD joint target。
3. 执行 4 次 physics step。
4. 按更新周期刷新 articulation state 和传感器。
5. 更新 episode counter。
6. 计算 terminated 和 truncated。
7. 使用本步更新前的 velocity command 和 reset 前终态计算 reward 与 episode statistics。
8. 保存需要的 terminal information，并更新 terrain curriculum。
9. 仅 reset 已完成的环境并处理 reset event。
10. 更新 heading/velocity command，再处理 interval event。
11. 计算并返回 reset 后 observation。

### 5.4 Reset

- 支持按 env ID selective reset。
- reset 一个环境不得改变其他环境状态。
- reset pose 必须叠加对应 terrain origin。
- joint、root pose、root velocity、command、sensor history、action history 和 episode statistics 必须同步重置。
- startup、reset 和 interval domain randomization 必须区分执行时机。

## 6. 分阶段实施计划

### 阶段 0：冻结基准行为

预计工作量：2-3 人日。

任务：

- 临时恢复 README 指定的 Isaac Lab revision 和可运行环境。
- 导出解析后的完整环境配置，避免遗漏当前继承链中的默认值。
- 固定 joint/body 顺序、action schema 和 observation schema。
- 保存固定 seed 下的 observation、reward term、termination、reset 和 command 样本。
- 保存 terrain mesh、terrain origin、难度和类型分配结果。
- 保存一段固定 action sequence 对应的短轨迹。
- 使用现有 checkpoint 保存推理输入和输出样本。
- 将基准保存为仓库内 JSON/NPZ fixture，fixture 本身不得依赖 Isaac Lab。

验收条件：

- 可以在不 import Isaac Lab 的测试中读取全部行为契约。
- 259 维 observation 的每一段都有名称、offset、shape 和数值样本。
- 每个 reward term 都有独立参考值，而不只有总 reward。

实施结果：已恢复 Isaac Lab revision `2ed331acfc` 和 PureRL revision `b6e5f36` 的历史运行环境，
并将真实 simulator reference 保存为仓库内 JSON/NPZ fixture。当前运行时和测试读取 fixture 时均不
import Isaac Lab；历史源码仅用于一次性采集，不属于项目依赖。

### 阶段 1：独立配置、依赖和应用启动层

预计工作量：3-4 人日。

任务：

- 使用标准 dataclass 定义 `SimCfg`、`RobotCfg`、`SensorCfg`、`TerrainCfg`、`MDPCfg` 和 `RunnerCfg`。
- 展开所有当前从 Isaac Lab 父类继承的默认配置。
- 实现配置 validation、复制、覆盖和 YAML 序列化。
- 编写基于 Isaac Sim `SimulationApp` 的本地 launcher。
- 显式管理 headless、camera、device、rendering 和 extension 启用参数。
- 将依赖声明集中到根 `pyproject.toml`，生成可复现的 uv lock。
- 增加 Isaac Sim 版本检查和清晰的启动错误信息。

验收条件：

- `import purerl.config` 不启动 Isaac Sim。
- launcher 可在没有 IsaacLab checkout 的环境中启动和关闭 Isaac Sim。
- flat、rough、train 和 play 配置均可构造、校验并序列化。

### 阶段 2：环境生命周期和基础场景

预计工作量：5-7 人日。

任务：

- 实现 simulation context、physics scene、stage 和基础灯光。
- 从仓库内 URDF 创建 TienKung articulation。
- 创建一个模板环境并批量 clone，配置环境间 collision filtering。
- 建立批量 articulation tensor view。
- 解析并锁定 joint/body index，不依赖运行时发现顺序。
- 实现 root/joint state read-write、PD target 和 torque/state buffer。
- 实现 `BaseVecEnv.reset()`、`step()`、`render()` 和 `close()`。
- 实现 physics decimation、selected env reset 和 episode length buffer。
- 先完成 flat terrain 上的 1 env 和 32 env 垂直链路。

验收条件：

- 32 个环境可连续执行 1,000 个随机 action step，无 NaN 和 simulator error。
- selective reset 后只有指定 env 的 root/joint state 改变。
- action target、physics dt 和 policy dt 与旧实现一致。

### 阶段 3：接触传感器、高度扫描与地形

预计工作量：6-9 人日。

任务：

- 实现批量 contact force 读取和 body index 映射。
- 实现 contact history、首次落地、腾空时间和 undesired contact 状态。
- 实现 pelvis 局部坐标系下的 187 点 height scan。
- 使用 NumPy/Warp/USD mesh 实现以下 terrain：
  - flat
  - random rough height field
  - ascending/descending slope
  - ascending/descending stairs
  - random blocks
- 实现 terrain row/column、类型比例、难度、origin 和 deterministic seed。
- 实现基于环境表现的 terrain level curriculum。
- 避免 step 中的逐环境 USD 或 prim 查询。

验收条件：

- contact、air-time 和 height scan tensor 的 shape、device 和更新周期稳定。
- flat 环境的 height scan 仍输出 187 维，并与 rough checkpoint 兼容。
- 相同 seed 生成相同 terrain mesh、origin 和环境分配。
- 2048 环境下传感器更新不引入明显 Python bottleneck。

### 阶段 4：轻量 MDP manager

预计工作量：5-7 人日。

只实现当前 locomotion 任务需要的 typed manager，不重建通用 Isaac Lab manager 框架。

任务：

- `ActionManager`：action clipping、scale、default offset 和 joint target。
- `CommandManager`：velocity command、standing ratio、heading 和定时重采样。
- `ObservationManager`：显式 term 顺序、噪声、corruption 开关和 schema 校验。
- `RewardManager`：term weight、逐 term value、episode sum 和 logging extras。
- `TerminationManager`：time-out、pelvis contact 和 bad orientation。
- `EventManager`：startup/reset/interval randomization 和 periodic push。
- `CurriculumManager`：根据 commanded walking performance 更新 terrain level。
- 将 reward、observation 和 termination term 实现为可独立测试的 tensor 函数。

需要迁移的随机化至少包括：

- friction
- pelvis mass 和 center of mass
- actuator stiffness/damping
- joint reset pose
- root pose/heading
- external force/torque
- periodic lateral/forward push

验收条件：

- 所有 term 都能使用合成 tensor 执行 CPU/CUDA 单元测试。
- observation schema 严格等于 259 维。
- 每个 reward term 与阶段 0 fixture 在定义容差内一致。
- terminated 和 truncated 语义符合 Gymnasium API。

### 阶段 5：任务注册与 RSL-RL 适配

预计工作量：4-6 人日。

任务：

- 实现本地 `TaskSpec` 和 task registry。
- 保留现有四个 Gym ID：
  - `PureRL-Velocity-Flat-TienKung-v0`
  - `PureRL-Velocity-Flat-TienKung-Play-v0`
  - `PureRL-Velocity-Rough-TienKung-v0`
  - `PureRL-Velocity-Rough-TienKung-Play-v0`
- 让环境实现标准 Gymnasium reset/step API 和 space 定义。
- 编写直接满足目标 `rsl-rl-lib` 版本 VecEnv contract 的 wrapper。
- 保持 `policy`/`critic` observation groups、auto-reset、episode extras 和 device 属性。
- 使用本地 dataclass 替换 `RslRl*Cfg`。
- 本地实现 checkpoint 查找、resume、pretrained initialization、JIT export 和 ONNX export。
- 删除 Hydra task config 和 Isaac Lab registry 依赖。
- 重写 train/play CLI，直接使用 Python 或 `uv run` 启动。

验收条件：

- 32 env、2 iteration 的 flat 和 rough 训练 smoke test 通过。
- checkpoint save、resume 和 pretrained initialization 通过。
- play 可加载 checkpoint，并可导出有效 JIT/ONNX 文件。
- flat checkpoint 可加载到 rough 环境且不存在输入 shape mismatch。

### 阶段 6：数值对齐、性能优化与切换

预计工作量：4-7 人日。

任务：

- 对照阶段 0 fixture 检查 action、observation、reward、termination 和 reset。
- 使用相同 seed 和 action sequence 比较短轨迹。
- 在 32、2048 和 4096 env 规模下测试吞吐、显存和稳定性。
- 优化 articulation state refresh、contact、raycast、terrain 和 buffer allocation。
- 确认训练循环中不存在 CPU/GPU 隐式同步热点。
- 删除 `source/purerl/config/extension.toml`。
- 删除所有 `isaaclab*` import、类型和配置。
- 删除 `isaaclab.sh`、外部 checkout 路径及相关安装说明。
- 更新 README、安装流程、架构说明和故障排查文档。

验收条件：

- 2048/4096 env 的吞吐不低于旧实现基准的 85%。
- 连续 10,000 step 无 NaN、内存持续增长或随机 simulator crash。
- 当前 checkpoint 推理输出在约定容差内，或者完成明确记录的 checkpoint migration。
- 干净 clone 在没有任何 IsaacLab 目录的机器上可以完成安装、训练、评估和导出。

## 7. 测试策略

### 7.1 CPU 单元测试

- dataclass config validation 和 serialization
- joint/body/schema ordering
- action scaling 和 PD target
- command resampling
- observation tensor 拼接和 noise
- 每个 reward/termination term
- terrain deterministic generation
- task registry
- checkpoint path selection
- 使用 fake environment 测试 RSL-RL wrapper contract

### 7.2 Isaac Sim 集成测试

- application start/close
- 1 env 和 32 env scene creation
- articulation state read/write
- selective reset
- contact sensor
- height scan
- flat/rough terrain collision
- Gymnasium reset/step contract
- video rendering

### 7.3 训练和部署测试

- flat/rough 两轮 PPO smoke training
- flat-to-rough checkpoint transfer
- checkpoint resume
- play inference
- JIT export 和加载
- ONNX export 和 runtime inference
- 2048/4096 env 性能基准

### 7.4 静态依赖门禁

CI 应检查 Python 源码和运行脚本中不存在：

```text
import isaaclab
from isaaclab
import isaaclab_tasks
from isaaclab_tasks
import isaaclab_rl
from isaaclab_rl
isaaclab.sh
```

迁移文档和许可证说明中可以出现这些名称，但不得存在可执行引用。

## 8. 主要风险与控制措施

### 8.1 Isaac Sim tensor API 稳定性

风险：直接 API 随 Isaac Sim 版本变化，尤其是 articulation、contact 和 raycast。

措施：将直接 Isaac Sim 调用集中在 `app/`、`sim/` 和 `sensors/`，其他模块只依赖 PureRL 自己的 tensor contract；固定 Isaac Sim 5.1 和扩展版本。

### 8.2 Contact 和 height scan 性能

风险：不恰当实现会导致逐环境查询、CPU/GPU 同步或显存开销过大。

措施：阶段 2 先验证批量 articulation，阶段 3 单独建立传感器 benchmark；禁止 step 热路径中的 USD traversal。

### 8.3 Checkpoint 行为漂移

风险：观测顺序、坐标系、噪声、reset 顺序或 terrain height 定义变化都会破坏已有策略。

措施：阶段 0 固定 schema 和数值 fixture；checkpoint 兼容性作为阶段 5 和阶段 6 的硬验收条件。

### 8.4 隐式默认配置遗漏

风险：当前配置继承了大量 Isaac Lab 默认值，源码中没有显式体现。

措施：删除旧依赖前导出解析后的完整配置；新配置禁止依赖多层继承，所有影响物理和 MDP 的参数必须显式声明。

### 8.5 许可证和代码来源

风险：训练脚本和可能迁移的算法实现来自 BSD-3-Clause 项目。

措施：保留已有版权和许可证声明；记录移植代码来源；优先独立实现小型任务专用模块。

## 9. 工作量与实施顺序

| 阶段 | 预计工作量 |
| --- | ---: |
| 阶段 0：冻结基准行为 | 2-3 人日 |
| 阶段 1：配置、依赖和启动层 | 3-4 人日 |
| 阶段 2：生命周期和基础场景 | 5-7 人日 |
| 阶段 3：传感器和地形 | 6-9 人日 |
| 阶段 4：MDP manager | 5-7 人日 |
| 阶段 5：注册和 RSL-RL | 4-6 人日 |
| 阶段 6：对齐、性能和切换 | 4-7 人日 |
| **合计** | **29-43 人日** |

单人连续实施预计约 6-9 周。建议按以下里程碑交付：

1. M1：完成行为 fixture 和独立 launcher。
2. M2：32 个 flat 环境可 reset/step。
3. M3：完整 observation、reward 和 termination 可运行。
4. M4：rough terrain、contact 和 height scan 可运行。
5. M5：RSL-RL 两轮 smoke training 和 checkpoint export 通过。
6. M6：4096 env 性能达标并删除全部 Isaac Lab 依赖。

## 10. 最终完成定义

只有同时满足以下条件，才能认为解耦完成：

- 根目录和用户环境不需要存在 IsaacLab checkout。
- 源码、脚本和 package metadata 不引用任何 `isaaclab*` 包。
- 不使用 `isaaclab.sh` 启动、安装或运行任务。
- 四个现有 Gym task ID 保持可用。
- action 维度为 20，policy observation 维度为 259。
- flat checkpoint 可用于 rough 环境。
- 训练、resume、play、JIT 和 ONNX export 全部通过。
- selective reset、domain randomization 和 terrain curriculum 有自动化测试。
- 2048/4096 env 性能满足约定基准。
- README 提供完全独立、可复现的安装和运行说明。

## 11. 当前实施进度（2026-08-05）

已完成并验证：

- 独立 dataclass 配置、任务 registry、Gymnasium 生命周期和 typed MDP manager。
- Isaac Sim 5.1 直接 backend、URDF 到 USD 缓存、批量 articulation/body/contact tensor view。
- flat 场景 32 环境连续 1,000 个随机 action step，无 NaN，并通过 selective reset 检查。
- 七类 generated terrain、deterministic tile assignment、terrain origin、curriculum 和 187 点批量高度扫描。
- contact history 按 200 Hz physics substep 更新；高度扫描按 50 Hz policy step 更新。
- EventManager 已接入 startup/reset/interval 生命周期，覆盖 pelvis mass/COM、PD gain、joint/root reset、external wrench 和 periodic push。
- 本地 RSL-RL 3.1.2 wrapper、checkpoint 查找、train/play CLI 和 JIT/ONNX export。
- flat 与 rough 均通过 32 环境、2 iteration PPO smoke training。
- flat checkpoint 已在 rough play 环境成功加载并执行，不存在 observation shape mismatch。
- 旧 extension manifest、旧任务配置、旧 asset 配置和训练脚本中的可执行 Isaac Lab 引用已移除。
- URDF 与 21 个 mesh 已迁入 Python package；wheel 构建及 package-data 内容检查通过。
- ObservationManager 已实现逐 term uniform corruption、独立 seeded CPU/CUDA generator、reset seed
  可复现和 play corruption 关闭；259 维 schema 和 height scan `[-1, 1]` clip 已由测试锁定。
- startup randomization 已实现 64 桶逐环境 static/dynamic friction，并通过 PhysX live material
  tensor 回读确认所有 robot collision shape 均已更新、restitution 为零。
- GPU PhysX contact、patch、pair、heap、buffer 和 collision stack 容量已显式配置，4096 rough
  环境不再依赖 Isaac Sim 的小规模默认值。
- rough 共址环境改用显式 USD/PhysX collision groups，terrain 作为 global collision object；
  4096 环境 50 步实测仅 3 次 pelvis contact，不再发生跨环境 robot collision。
- 新增 `scripts/tools/benchmark_env.py`，覆盖吞吐、逐项 termination、全状态 finite 检查、
  PyTorch/CUDA 整卡显存和显存增长硬门禁。
- 恢复并锁定历史 Isaac Lab/PureRL revision，fixture 覆盖 259 维 observation、20-joint
  articulation 顺序、逐 term reward、termination、selective reset、heading target、rough terrain
  mesh/origin/assignment 和 4-env 16-step 固定 action 轨迹。
- velocity command manager 已恢复 heading target、比例 mask、standing mask、wrap/clip、随机采样
  顺序及 reward 后更新时序；command 轨迹 RMSE 从未实现 heading 时的 `0.70035` 降至 `0.01992`。
- camera `rgb_array` 和 MP4 路径已完成，640x360 的 12 帧检查中 11 帧为非空画面。
- checkpoint 自然数字排序、save/resume 下一 iteration、flat-to-rough pretrained、play、JIT/ONNX
  export 均由 `check_rsl_workflow.py` 覆盖。
- W&B 支持 online/offline/disabled、project/entity/tags/run ID/resume；offline smoke 已确认数据写入
  当前训练 run 目录，writer 在退出时显式关闭。
- 源码和脚本禁止依赖扫描、61 个 CPU 测试、Ruff、flat/rough GPU smoke、wheel/package-data
  检查均通过。

RTX 4090、CPU governor 为 powersave 时的当前基准如下。rough 数据为显式碰撞过滤修复后的
零动作 50-step 测量；数值用于当前机器回归，不作为跨硬件绝对指标。

| Task | Envs | Transitions/s | CUDA device memory | Pelvis contacts |
| --- | ---: | ---: | ---: | ---: |
| Flat | 2048 | 20,712.32 | 8.26 -> 8.61 GiB | - |
| Flat | 4096 | 30,793.26 | 9.67 -> 9.92 GiB | - |
| Rough | 2048 | 13,768.42 | 9,918 MiB stable | 2 |
| Rough | 4096 | 18,276.59 | 10,932 -> 10,972 MiB | 3 |

长稳门禁已通过 flat 32 env、10,000 policy steps（40,000 physics substeps）：6,376 次
episode completion，无 NaN/Inf 或 simulator error，第 2,000 到 10,000 步整卡显存增长
`0.0 MiB`，PyTorch allocated 增长 `0.002 MiB`。

### 11.1 轨迹迁移结果

`scripts/tools/check_legacy_trajectory.py` 在固定初始 root/joint/command/heading target 后执行相同
16-step action。MDP 数学与 lifecycle 语义已对齐，但直接 Isaac Sim backend 与历史 Isaac Lab
封装的物理轨迹并非逐位一致。当前 RTX 4090 / Isaac Sim 5.1 实测如下：

| Field | RMSE | Max absolute error | Gate |
| --- | ---: | ---: | ---: |
| Root position | 0.04623 | 0.16251 | 0.18 |
| Joint position | 0.16576 | 0.61998 | 0.65 |
| Velocity command | 0.01992 | 0.11034 | 0.12 |
| Reward | 0.01382 | 0.05248 | 0.06 |

termination term、terminated 和 truncated agreement 均为 `1.0`。门禁同时要求上述误差有限且不
超过记录包络；这些阈值用于检测进一步回归，不表示两个物理 backend 数值等价。逐 term reward
数学使用 fixture state 的 CPU 测试保持 `rtol=2e-5, atol=2e-5`。

### 11.2 完成状态

阶段 0-6 的实现项和自动化工作流均已完成。仍需作为持续工程门禁维护的项目是不同 GPU/driver
上的吞吐基线、长时间训练稳定性和上述物理迁移包络；它们不再要求外部 Isaac Lab checkout。
