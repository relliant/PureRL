# 天工步态奖励调优记录（Gait Reward Tuning Notes）

> 目的：记录为改善 PureRL 天工（TienKung）locomotion 步态而新增的步态相关奖励，以及从天工 URDF 实测的物理参数，方便后续复现与调参。
> 参考开源实现：[humanoid-gym](https://github.com/roboterax/humanoid-gym)（Unitree G1/H1 的步态奖励设计）。
> 记录日期：2026-08-15。

---

## 1. 背景

原 PureRL 奖励只有「速度跟踪 + 能量惩罚」这类经典 legged 风格奖励，唯一与步态相关的 `feet_air_time` 实现也很弱（单支撑相取最小腾空时间），**没有可靠的步态事件约束**，导致训出来的策略步态异常（拖步、左右不对称、螃蟹步、原地晃等）。

参考 humanoid-gym，新增了一套**基于开环步态时钟（gait phase）**的步态奖励，强迫机器人走出交替清晰的步态。

---

## 2. 天工机器人物理参数（URDF 实测）

> 测量方法：正向运动学（默认关节姿态）+ 脚 STL 顶点扫描。URDF 位于
> `source/purerl/purerl/assets/robot_description/tienkung/tienkung2_lite.urdf`。

| 参数 | 值 | 来源 |
|------|----|------|
| 大腿长 | **0.400 m** | `hip_pitch→hip_yaw` z=-0.0987 + `hip_yaw→knee` z=-0.3013 |
| 小腿长 | **0.400 m** | `knee_pitch→ankle_pitch` z=-0.4 |
| 总腿长 | **0.800 m** | 大腿 + 小腿 |
| 默认站姿 pelvis→ankle 垂直距离 | **0.8338 m** | 正向运动学（默认关节角：hip_pitch=-0.5, knee_pitch=1.0, ankle_pitch=-0.5） |
| 脚 body 原点到脚底 | **0.0569 m** | `ankle_roll_l_link.STL` 顶点最低点 z=-0.0569 |
| 实际站高（pelvis→脚底） | **0.8907 m ≈ 0.89** | 0.8338 + 0.0569 |
| `default_root_height` | 0.89 m | `config/robot.py`，与实测一致 ✅ |

**关键结论**：`default_root_height=0.89` 是精确的；但脚底偏移是 **0.0569**（不是常见的 0.05），`base_height` 奖励里必须用这个值，否则站正了也只给约 0.54 的奖励（满分 1.0），会逼机器人踮脚/蹲腿。

### 关键名称

- root body：`pelvis`
- 脚 body：`ankle_roll_l_link` / `ankle_roll_r_link`
- 关节命名（`config/robot.py` 的 `JOINT_NAMES`）：
  - 腿：`hip_roll_*`、`hip_pitch_*`、`hip_yaw_*`、`knee_pitch_*`、`ankle_pitch_*`、`ankle_roll_*`
  - 臂：`shoulder_pitch_*`、`shoulder_roll_*`、`shoulder_yaw_*`、`elbow_pitch_*`

---

## 3. 步态奖励设计（核心：开环步态时钟）

```python
phase = episode_length_buf * step_dt / cycle_time     # cycle_time 见下表
sin_pos = sin(2π · phase)
左脚支撑相 = sin_pos >= 0，右脚支撑相 = sin_pos < 0   # 两脚反相（差 π）
双支撑相 = |sin_pos| < 0.1
```

所有步态奖励都挂在这个 `stance_mask` 上。相位是**开环时钟**（不依赖实际接触），简单稳定，缺点是固定频率。

---

## 4. 步态参数（`GaitCfg`，YAML 的 `gait:` 块）

| 参数 | 值 | 说明 |
|------|----|------|
| `cycle_time` | **0.5 s** | 步态周期（步频 2 Hz）。天工腿长 0.8m 属中小型人形，比 G1 的 0.64 快。**微调范围 0.45–0.6** |
| `contact_threshold` | 1.0 N | 脚接触力判定阈值 |
| `foot_min_dist` | 0.2 m | 步宽下限 |
| `foot_max_dist` | 0.5 m | 步宽上限 |
| `target_feet_height` | 0.06 m | 摆动脚目标离地高度 |
| `foot_height_offset` | **0.0569 m** | 脚 body 原点到脚底距离（天工实测，勿改成 0.05） |

---

## 5. 新增的 4 个步态奖励项

| 奖励项 | 权重 | 作用 |
|--------|------|------|
| `feet_contact_number` | 1.2 | 移动时接触与步态相位对齐（+1 匹配 / -0.3 不匹配），**核心** |
| `feet_distance` | 0.2 | 步宽约束（范围 0.5~1.0，惩罚交叉步/螃蟹步） |
| `base_height` | 0.2 | 保持躯干在脚上方 0.89m（惩罚蹲姿/踮脚） |
| `feet_clearance` | 1.0 | 平滑地奖励摆动脚抬到目标离地高度 |
| `feet_air_time` | 0.75 | 只在脚落地事件发生时奖励有效腾空时间 |

### 改动文件

- `source/purerl/purerl/config/env.py` — 新增 `GaitCfg`、4 个 reward 项、`EnvCfg.gait` 字段
- `source/purerl/purerl/mdp/rewards.py` — 新增 4 个纯函数
- `source/purerl/purerl/envs/tienkung_locomotion.py` — 新增 `_get_gait_phase()`、`_foot_contact_mask()` + 装配
- `source/purerl/purerl/config/presets/{flat,rough}{_play,}_env.yaml` — 4 个文件各加 4 项 + `gait:` 块

---

## 6. 调参指南（按顺序）

1. **`cycle_time`**：先观察步态是否稳定交替。拖沓/慢 → 调小（0.45）；急促/不稳 → 调大（0.55–0.6）。步态时钟只用于移动命令，站立命令不会被强制交替。
2. **`feet_clearance`**：初期可先关掉（权重设为 0 或移除），只保留 `feet_contact_number + feet_distance + base_height`，训几百 iter 看步态是否变清晰；稳定后再加回，`target_feet_height` 从 0.06 微调。
3. **`base_height`**：已自动对齐真机站高（`default_root_height=0.89` + `foot_height_offset=0.0569`），无需手填。
4. **warm-start**：可用旧 checkpoint 继续训（reward 项变了会有短暂适应期）。

---

## 7. 实现细节

- `ContactHistory.contact_events` 会跨越四个物理子步锁存落脚事件，并在策略奖励计算后清除。
- `feet_air_time` 使用 `last_air_time` 与落脚事件的乘积，只在落脚瞬间支付一次。
- `feet_clearance` 和 `base_height` 使用高斯形平滑奖励，避免原先布尔阈值没有梯度的问题。
- `feet_contact_number` 对低于 `gait.command_threshold` 的速度命令返回零，避免站立时被开环相位惩罚。

## 8. 验证结果

- 4 个新奖励函数用 torch 实跑，边界值全部正确
- `test_config + test_mdp + test_tienkung_env` 通过（含 `env.step` 真实触发 reward 装配）
- `base_height` 脚底偏移修正前后：站姿奖励 0.54 → 0.93

## 9. 已知事项

- `tests/test_legacy_fixture.py::test_legacy_schema_and_weights_match_current_contract` 会失败：这是「Isaac Lab 原始实现契约快照」测试，因新增 4 个 reward term 导致契约（有意）变更。新 reward 无 Isaac Lab reference 数值，需后续在 Isaac Lab 环境重新生成 reference 后更新 fixture。
