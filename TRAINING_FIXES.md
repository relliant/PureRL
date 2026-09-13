**2026-09-13：Flat 训练正确性修复**

本次修复针对 `2026-09-12_23-29-34_flat_baseline_seed5` 诊断发现的问题。新版本需要从头训练，输入观测由 259 维变为 261 维。已完成 109 项测试、3 个机器人真实物理检查，以及 32 个环境 / 5 轮 PPO 训练；这些验证覆盖实现正确性和运行流程，尚不能证明长期收敛。

**实现变更**

| 问题 | 修复后的行为 |
|---|---|
| auto-reset 改写 PPO 尚未存入 buffer 的动作 | Action manager 复制传入动作；失败和超时 reset 均不修改调用方动作或对应 old log-prob |
| Flat 仅依靠出生点间距避免碰撞 | Flat 和 Rough 都创建 clone 间碰撞过滤；Flat 使用 ground 对象的真实 prim 路径保留地面接触 |
| position drive 的力矩惩罚始终为零 | 使用 `clip(Kp * (q_target - q) - Kd * q_dot, -limit, limit)` 作为奖励的 PD 力矩估计；同步目标、随机增益和 reset，显式设置零速度目标 |
| 步态奖励使用策略不可见的时钟 | actor/critic 的公共 observation 追加无噪声 sin/cos 相位，和奖励使用同一个 clock |
| 落脚事件奖励重复乘 dt | 新增 `RewardTermCfg.is_event` / `RewardTermSpec.is_event`；feet_air_time 标为事件，其余项继续按 rate × dt 积分 |
| 难以区分终止原因 | 日志新增 `Termination/time_out`、`Termination/base_contact`、`Termination/bad_orientation` |

PD 力矩是用于奖励的**估计值**，不是 PhysX 隐式求解器的精确电机力矩测量。此设计参考 [Isaac Lab 2.2.1 ImplicitActuator 的奖励力矩估计](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/actuators/actuator_pd.py)。保留当前 position-drive 动力学；没有引入显式 torque 控制。

相位仍以 `episode_length_buf * step_dt / cycle_time` 为基础，观测顺序为旧 259 维前缀加 `[sin(2π phase), cos(2π phase)]`。地形扫描保持 `[72:259]`，相位为 `[259:261]`。export/play/train 和四套环境 preset 已同步更新。历史 Isaac Lab 数值 fixture 保持原样，只验证可比较的旧观测前缀；没有伪造新相位的仿真参考值。

事件奖励改变了实际训练尺度：Flat 中一次有效单脚落地最多 `0.12 × 0.75 = 0.09`，原来为 0.0018。终止惩罚仍为 `-200 × 0.02 = -4`。自定义完整 YAML 需更新 observation dimension，并给每个 reward 显式填写 `is_event`；当前 presets 已包含这些字段。

终止原因日志是在已结束 episode 上计算的比例，多个原因可能同时为真，不应假定它们之和总是 1。

**验证结果**

- `pytest`：109 passed，1 skipped；跳过的是安装了 Isaac Sim 时不适用的“缺少 Isaac Sim”异常测试。JIT/ONNX 测试覆盖 261 维输入。
- `ruff check source scripts tests` 和 `git diff --check` 通过。
- 配置检查：4 个任务均为 20 维 action / 261 维 observation。
- 实际 RSL-RL PPO + 环境 auto-reset 回归：terminated 和 truncated 都保留真实动作，策略不更新时重算 ratio 为 1。
- 真实物理：将两个机器人放在完全重叠的世界坐标，与远处第三个机器人作对照，运行 100 个物理步。最大位移差 0.000175 m，地面接触峰值约 1075.65 N，PD 力矩估计峰值 300 N·m，满足关节限幅。
- 真实训练：32 env × 60 steps × 5 iterations = 9600 transitions，正常打印 `TRAIN_OK` 并保存 `model_4.pt`。actor/critic 输入都为 261，所有模型张量有限。
- 5 轮 surrogate loss：`[0.01114, -0.00290, -0.00714, -0.00541, 0.01223]`；value loss 从 1.967 降至 0.834。力矩惩罚各轮为 -0.121 至 -0.286，腾空奖励已非零。这是小规模流程验证，不能与原来 4096 环境的曲线直接作收敛比较。

验证中宿主机仍报告 NVML 驱动/库版本不匹配及部分渲染组件错误；本次 CUDA/PhysX 物理和训练都成功完成。未修改系统驱动。

机器可读结果见 [fix_validation.json](reports/flat_baseline_seed5_20260913/fix_validation.json)。短程训练记录位于 `/tmp/purerl-fix-validation/tienkung_flat/2026-09-13_11-31-03_flat_fix_smoke_seed5`。

**复现与新训练**

在仓库根目录运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check source scripts tests
.venv/bin/python scripts/tools/check_task_config.py
.venv/bin/python scripts/tools/check_training_physics.py --steps 100
```

修复后的默认 Flat 正式训练命令：

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task PureRL-Velocity-Flat-TienKung-v0 \
  --seed 5 \
  --run-name flat_fixed_seed5
```

此命令从头初始化，不附加 `--resume` 或旧 checkpoint。旧 259 维模型在 train/play 加载时会收到明确错误，`--allow-unsafe-checkpoint` 不会绕过维度检查。修改代码不会使已启动的旧训练进程自动获得修复；本次没有停止它，也没有启动新的长程训练。

PPO 学习率、epochs、初始探索噪声和原有 domain randomization 保持当前配置，用于先验证修复的效果。后续根据新曲线分别比较探索尺度、GAE lambda、更新次数和更简单的命令/随机化课程。
