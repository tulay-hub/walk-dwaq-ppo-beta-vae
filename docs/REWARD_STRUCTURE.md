# 行走 DWAQ 奖励结构

环境奖励分为速度跟踪、身体姿态、能耗/加速度/动作平滑、关节限位、接触、脚步、关节姿态、生存、防偷懒和
步态相位先验。核心权重是 `track_lin_vel_xy_exp=+2.5`、`track_ang_vel_z_exp=+3.0`、
`termination_penalty=-200.0`，并以 `idle_penalty=-2.0` 防止有命令时原地不动。

此外，`DWAQPPO` 对 `ActorCriticDWAQ` 单独优化 velocity/reconstruction/KL 的 beta-VAE loss；它不是
`RewardTermCfg` 中的一项。逐项权重和源码位置见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)。

## 训练权重如何理解 / Interpreting training weights

本文按本仓库当前代码说明训练机制；已有策略的复现参数以对应 run 的 `params/env.yaml`、`params/agent.yaml` 和部署配置为准。奖励混合系数、逐项环境奖励权重、优化器 loss 系数、专家样本比例以及课程采样范围是不同概念。

混合系数可以写成 85%/15% 这样的配置比例，但不能代表训练过程中实际累计奖励贡献；单项 reward 的数值范围、门控、控制步长和出现频率都不同。需要实际贡献占比时，应统计同一 run 中每项加权回报，而不是把配置权重归一化成百分比。

Configuration mixing coefficients are not measured reward contributions. Environment weights, optimizer coefficients, expert sampling and curriculum schedules describe different parts of training. Reproduce a saved policy with its own run snapshots.

## PPO、β-VAE 与奖励权重的分工

当前策略由 `DWAQRunner → ActorCriticDWAQ → DWAQPPO` 构建；PPO 优化 actor/critic，β-VAE 通过独立 Adam 更新编码器/解码器。没有配置 AMP 判别器或专家动作模仿损失；腿部周期参考奖励是参数化步态先验，不等同于专家动作数据集。

```text
actor input = [context(3 velocity + 16 latent), policy_obs(76)]
L_RL = L_PPO_clip + 1.0*L_value - 0.008*entropy
L_VAE = 1.0*L_velocity_MSE + 1.0*L_next_observation_MSE + 1.0*KL_latent
LR_RL = LR_VAE = 1e-4 (fixed)
```

这里的 VAE 三项系数是 `1:1:1`，不是各贡献 33.3%：误差尺度不同，终止样本使用 live mask，重建还屏蔽 rollout 最后一步。重建目标实际为下一时刻观测 `o_(t+1)`。当前配置 gamma=0.99、lambda=0.95、clip=0.2，24 rollout steps、5 epochs、4 mini-batches。

环境 reward 仍按下表逐项求加权和，VAE loss 不加入环境回报。盲走上楼梯继承同一网络和权重，课程只改变 terrain row；成功判据是 timeout、无 failure、接近完整 60 s，以及距离出生点的 XY 净位移大于 10 m（20 m patch 长度的一半），并不检测机器人是否沿指定楼梯路径走完一整圈。连续成功两次升阶，失败降阶。

源码相对共享框架 `lens110/legged_lab_lbot/`：`rsl_rl/rsl_rl/algorithms/dwaq_ppo.py`、`rsl_rl/rsl_rl/modules/{actor_critic_dwaq,context_vae}.py`、`source/legged_lab/legged_lab/tasks/locomotion/dwaq/config/lens110/agents/rsl_rl_ppo_cfg.py` 和 `tasks/locomotion/dwaq/mdp/curriculums.py`（tasks 相对同一 legged_lab 包）。独立项目通过 `framework/isaaclab_shared/` 使用共享基座。

English: PPO and VAE have separate optimizers. RL loss is clipped PPO + value - 0.008 entropy; VAE coefficients for velocity MSE, next-observation MSE and latent KL are 1:1:1, not percentage contributions. Both fixed learning rates are 1e-4. Stairs reuse these losses and rewards; promotion tests timeout, no failure, nearly 60 s and more than 10 m XY displacement, not verified completion of the whole stair route.
