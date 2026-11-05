# robot-stop

在本机退出遥操作/数采后使用：

```bash
robot-stop --dry-run     # 查看当前选择与目标，不连接硬件
robot-stop               # 默认停止双臂
robot-stop --arm left    # 仅左臂
robot-stop --arm right   # 仅右臂
robot-stop --config /absolute/path/to/record_cfg.yaml
```

默认读取 `scripts/config/record_cfg.yaml` 的 `record.robot_type`。
与 `robot-record` 使用同一硬件文件解析规则：默认按类型读取
`scripts/config/robots/*`，支持 `record.das_config_path` / `record.robot_config_path`，
并用 `record.robot` 覆盖硬件字段。命令执行前会打印实际目标。

| 机器人 | 操作与确认 |
| --- | --- |
| `nero_dual_arm` | 调用 NUC 的 `robot_stop`，发送 SDK `electronic_emergency_stop()`，等待 `arm_status=1`。这是阻尼急停，机械臂可能在重力下缓慢下沉，不代表物理电源已断开。 |
| `flexiv_dual_arm` | 直接 RDK `Stop()`，确认 `stopped()` 且模式为 `IDLE`。不使能、不清故障、不初始化夹爪、不回位；不保证失能或机械抱闸。 |
| Franka / 其他 | 报不支持并以非零退出，不发送硬件命令。 |

Flexiv 当前 RDK 没有 `Disable()`；`Brake(true)` 官方限制为医疗型号使用，
因此 Rizon4s 不调用该接口，也不会把停止成功报告为抱闸成功。
参考：[Flexiv 官方 Robot API](https://www.flexiv.com/software/rdk/api/classflexiv_1_1rdk_1_1_robot.html)。

命令不会自动回 reset 位置或操作夹爪。Nero 执行前应确认下沉空间无碰撞并妥善放置负载；
其他控制程序仍可能恢复使能或发送动作，所以应先退出遥操作/数采。
这属于软件停止，不能替代硬件急停。NUC server 保持运行；Nero 后续可用 `robot-reset` 恢复并回位。

双臂中一臂失败仍会尝试停止另一臂。两臂分别输出结果，失败返回码为 1，配置错误为 2。
`--timeout` 控制 RPC 超时和每臂状态等待时间，不限制 Flexiv RDK 自身的阻塞调用。

当前工作站的入口是 `/home/ziang/szl_ws/flexiv-local/bin/robot-stop`，
使用 `telelop` 环境。项目 `setup.py` 也已注册同名 console script，供重新安装时使用。

上线验证仅执行 dry-run 与模拟接口测试，没有实际触发停臂/下电。
