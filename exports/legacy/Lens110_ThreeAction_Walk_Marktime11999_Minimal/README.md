# Lens110 45D 走路策略最小提交包（Mark-time · 11999）

本包只含部署所需的策略文件与验证材料。严格保持 45D 观测、12 个 pitch/roll
动作合同；upper/lower 踝关节解算仍按既有部署方案在部署侧处理。

## 包内容

```
Lens110_ThreeAction_Walk_Marktime11999_Minimal/
├── policy_11999/           # 主策略
│   ├── policy.pt           # TorchScript（验证/回放用）
│   └── policy.onnx         # 部署用（obs [1,45] → actions [1,12]）
├── motions/                # 训练参考：mark-time 起步/停止 + walk
│   ├── lens110_mark_time_start.pkl
│   ├── lens110_mark_time_stop.pkl
│   └── lens110_walk_100hz.npz
├── replay/                 # MuJoCo 回放器 + mjcf + 网格（可选验证用）
├── config/robot_humanoid_lens110_config.yaml   # 实机部署参考 YAML（未改动）
└── obs_function/lens110_leg_pitch_roll_env_cfg.py  # 45D 观测布局定义（仅参考）
```

## ONNX IO 合同

| 项 | 值 |
|---|---|
| 输入 | `obs`，形状 `[1, 45]`，float32 |
| 输出 | `actions`，形状 `[1, 12]`，float32 |
| 观测布局 | base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + leg joint_pos rel(12) + leg joint_vel rel(12) + last_action(12) |
| 动作布局 | 12 个腿部关节（pitch/roll 系），顺序与 45D 框架一致 |
| 动作缩放 | residual × 0.25，叠加到默认关节位置 |
| 踝关节 PD | kp=kd=10（训练与回放一致） |

## MuJoCo 实测指标（sim 直驱，平底脚模型，stage1 11999）

行走 30 秒：

| 指标 | 实测 |
|---|---:|
| 前进距离 | 13.95~14.12 m |
| 横向偏移 | -0.28 ~ +0.40 m |
| 左脚步频 | 1.60~1.63 步/s |
| 躯干 roll std | 2.08° |
| 腿距 | 0.168 m |
| 航向漂移 | ±3.3° |

停止 5 秒后：

| 指标 | 实测 |
|---|---:|
| 躯干 roll | ±0.3° |
| 躯干 pitch | -1.26° |
| 左踝 roll | -2.15 ~ -0.12° |
| 右踝 roll | 0.07 ~ 2.09° |
| 脚底接触 | 平面接触，无侧棱着地 |

## 回放命令

### 原始 5 次按键、带航向闭环回放

```bash
cd replay

/home/srl/isaac-sim/python.sh sim2sim_lens110_12dof_keyboard.py \
  --policy ../policy_11999/policy.pt \
  --mjcf mjcf/lens110_21dof.xml \
  --cmd_vel 0.5 0 0 --forever
```

按键：`8/2` 起步/停止、`4/6` 横移、`7/9` 转向、`0` 复位、`Q/ESC` 退出。
回放默认开航向闭环；测原始漂移加 `--no_heading_feedback`。

### 一键 8/2 + 起步/停止斜坡回放（推荐看过渡效果）

```bash
cd /home/srl/lense110/walk/colleague_sim2sim_work/humanoid_sim2sim

/home/srl/Lens110_V28_test/Lens110_MuJoCo_Viewer_V28_OneClick/lens110_sim2sim_standalone_v1/.venv/bin/python play_sim_12dof_keyboard.py \
  --xml resource/lens_110/mjcf/lens110_21dof_sim_flatfoot.xml \
  --policy ../deliverables/Lens110_ThreeAction_Walk_Marktime11999_Minimal/policy_11999/policy.onnx
```

该回放器内置约 1 秒起步斜坡和约 1 秒停止斜坡，可通过 `--ramp-time` 调整。

## 说明

- 训练使用 `lens110_mark_time_start/stop.pkl` 作为过渡参考，观测保持 45D。
- 起步/停止的平滑过渡由回放器命令斜坡实现，不改变策略观测/动作合同。
- 已知剩余优化点：躯干 roll std 略高于 2.0°；停止后踝 pitch 尚未完全回到
  标准站姿。这些可作为下一轮训练优化目标。
