# 行走 DWAQ 框架入口

`isaaclab_shared` 指向 `frameworks/shared/lens110_isaaclab`。行走使用共享 DWAQ 配置中的
`Lens110DwaqEnvCfg`、`SegmentedVelocityCommand`、DWAQ actor/VAE 和行走奖励；项目专属包在
`exports/packages`，日志通过 `experiments/dwaq_runs` 访问。
