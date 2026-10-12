# PRIMEU：ACT / Diffusion

新的数据适配、训练、部署、回放、机器人客户端和配置说明已集中到 [examples/primeu/README.md](../examples/primeu/README.md)。新训练入口会把数据契约和手部 basis 一并保存到 checkpoint，支持 60／33／27／15／12D action。

原入口继续保留：

```bash
robot-train --config scripts/config/primeu/act_smoke.yaml
robot-train --config scripts/config/primeu/diffusion_smoke.yaml
python -m scripts.tools.prepare_primeu_lerobot --source SOURCE --output NEW_OUTPUT
python -m scripts.tools.check_primeu_checkpoint --config CONFIG --checkpoint CHECKPOINT --report REPORT
```

原配置是单卡 batch 2、20 步的 60D 连通性测试，已有输出目录不能覆盖。旧 Diffusion 使用两帧观测，输出 7 帧；新实例改为单帧观测、输出 8 帧。旧 checkpoint 和运行记录保留在 `/user/users/szl/runs/primeu_lerobot_smoke_20261012`，不能冒充新单帧服务的验收结果。使用新的推理服务请按新 README 训练并导出包含 `primeu/` 资产的 checkpoint。
