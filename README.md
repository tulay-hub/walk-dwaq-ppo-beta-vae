# 03 · 行走（DWAQ Walk）

## 项目定位

这是 Lens110 的盲行走项目，使用 DWAQ：`ActorCriticDWAQ + DWAQPPO + beta-VAE context`。

- 训练任务：`LeggedLab-Isaac--DWAQ-Lens110-v0`
- PLAY 任务：`LeggedLab-Isaac--DWAQ-Lens110-PLAY-v0`
- 物理频率：`500 Hz`（`sim.dt=0.002`）
- 策略频率：`100 Hz`（`decimation=5`）
- 回合上限：`60 s`
- actor：本体感知、命令、上一动作和历史，不直接接收 base linear velocity
- critic/VAE：privileged velocity 和历史序列用于 value/latent 学习

## 目录

```text
03_walk/
├── framework/isaaclab_shared -> framework/isaaclab_shared
├── data/terrain/source -> 共享地形实现
├── exports/packages/                 # DWAQ/框架发布包
├── exports/legacy/                   # 三动作旧行走包和旧 sim2sim 包
├── exports/training_exports -> DWAQ runs
├── experiments/dwaq_runs -> DWAQ logs/checkpoints
└── docs/
```

DWAQ 是 command-based walk，不依赖 DeepMimic reference motion；`data/terrain` 保存地形入口和项目说明。

## 训练

```bash
./scripts/train.sh \
  --headless --num_envs 4096
```

当前发布包 `exports/packages/lens110_dwaq_training_model36200_to_36700_2026-09-16.*` 只是当前训练窗口的
checkpoint 范围，不等于已经收敛。开始新训练前先做 foreground smoke test，再决定是否 resume。

## MuJoCo 回放

```bash
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd framework/isaaclab_shared/lens110/legged_lab_lbot
/path/to/mjlab/python scripts/play_dwaq_mujoco.py \
  --checkpoint /path/to/model_xxxx.pt \
  --xml "$REPOSITORY_ROOT/tools/retargeting/robot_retargeter/asset/robot/lens110/lens110_21dof.xml" \
  --terrain flat --demo --demo-duration 60
```

奖励结构见 [`docs/REWARD_FRAMEWORKS.md`](docs/REWARD_FRAMEWORKS.md) 的 DWAQ 章节；算法层的 VAE
velocity/reconstruction/KL loss 在共享框架的 `rsl_rl/rsl_rl/algorithms/dwaq_ppo.py` 中实现。

## English

This repository contains the command-conditioned walking project. It uses DWAQ: PPO optimization with a beta-VAE history encoder. The verified ONNX interface accepts `obs [1, 76]` and `obs_history [1, 380]`, and returns 21 actions. Physics runs at 500 Hz and policy control at 100 Hz.

Run `./scripts/train.sh --headless --num_envs 4096` from the unified workspace. For a standalone checkout, replace the workspace-relative shared-framework path with the `isaaclab-shared` dependency described by the release manifest. Record terrain seed, command range, checkpoint, joint order, and replay duration for every run.

Read `docs/REWARD_FRAMEWORKS.md` and `docs/INTERFACE_CONTRACTS.md` before loading a policy. Stair curriculum belongs to the separate stairs project; simulation success is not hardware validation.
