# 行走 DWAQ 奖励结构

环境奖励分为速度跟踪、身体姿态、能耗/加速度/动作平滑、关节限位、接触、脚步、关节姿态、生存、防偷懒和
步态相位先验。核心权重是 `track_lin_vel_xy_exp=+2.5`、`track_ang_vel_z_exp=+3.0`、
`termination_penalty=-200.0`，并以 `idle_penalty=-2.0` 防止有命令时原地不动。

此外，`DWAQPPO` 对 `ActorCriticDWAQ` 单独优化 velocity/reconstruction/KL 的 beta-VAE loss；它不是
`RewardTermCfg` 中的一项。逐项权重和源码位置见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)。
