# Franka / Nero / Flexiv 统一数采

三端维护同一仓库的 `X-embodiment` 分支。`robot-record` 共用采集循环、
Quest `cross_embodiment_v1` 输入处理和 X-embodiment 数据格式。

```bash
robot-record --robot-type franka_dual_arm --repo-id '{robot_type}/my_task'
robot-record --robot-type nero_dual_arm --repo-id '{robot_type}/my_task'
robot-record --robot-type flexiv_dual_arm --repo-id '{robot_type}/my_task'
```

每次运行只控制所选本体。先追加 `--dry-run` 核对机器人地址、相机、坐标系、
Quest 映射和数据集字段；它仅加载配置，不连接设备或发送动作。
`robot-reset`、`robot-replay`、`robot-stop` 使用相同的 `--robot-type` 和工作站配置。
`robot-stop` 当前仅支持 Nero 和 Flexiv；Franka 仍使用现有 RPC 服务器停止流程。

## 各机器的配置

共享默认值在 `scripts/config/record_cfg.yaml` 和 `scripts/config/robots/`。
`scripts/config/station.local.yaml` 保存本机的默认本体、任务、模型路径、IP 和相机，
不进入 Git。两台机器可以保持同一提交和干净工作区，同时使用各自参数。

从 `scripts/config/station.example.yaml` 创建本机配置。格式为：

```yaml
default_robot_type: franka_dual_arm
record:
  repo_id: "{robot_type}/my_task"
  task:
    description: "任务描述"
    num_episodes: 50
robots:
  franka_dual_arm:
    robot:
      robot_ip: "172.16.0.1"
    cameras:
      head_cam_serial: "实际序列号"
```

配置优先级：共享默认值 → 本机工作站设置 → 命令行本体/数据集覆盖。
只应用 `robots` 中与所选本体对应的配置。
安装位置、基座变换、相机序列号和左右手映射属于各工作站；不要跨工位盲目复制标定。
全局任务/模型设置放在 `record`，硬件设置放在 `robots.<robot_type>`。
切换带有旧本体内联硬件的配置时会报错，避免把旧地址带给另一本体。

`--config PATH` 使用自定义会话配置；默认读取其同目录的 `station.local.yaml`。
用 `--station-config PATH` 显式指定工作站，或 `--no-station-config` 核对共享默认值。

## 统一数据契约

- 30 Hz；左右腕相机和头部 RGB 图像。
- observation：双臂绝对末端位姿（米、rotvec 弧度）和夹爪实测开口（米），共 14 维。
- action：双臂有效目标位姿增量和夹爪目标开口，共 14 维。
- Franka 使用服务器世界坐标；Nero/Flexiv 根据配置的安装变换转换。
- metadata 记录坐标、安装变换、有效指令语义和 Quest 处理参数；续采/合并拒绝不兼容语义。

统一格式便于跨本体训练，但不代表不同工位已经共用实物标定。
旧数据须确认语义再转换或合并；新数据的夹爪动作是目标开口，不能用观测开口代替。
Quest A 保存当前段并回位开始下一段，B 丢弃当前未保存段并回位重录；回位轨迹不进入数据。
参见 [按键说明](episode_controls.md)、[有效指令契约](effective_command_recording.txt)
和 [Quest 对齐说明](quest_processing_alignment.txt)。

## 安装、验证与同步

在已有 LeRobot 数采环境中运行：

```bash
git switch X-embodiment
git pull --ff-only origin X-embodiment
git submodule update --init
python -m pip install -e . --no-deps
python -m pytest -q tests robots/dual_flexiv_rizon4s/tests robots/dual_franka/test_franka_dual_arm.py
```

以上测试使用模拟设备和临时数据集，不连接实际机械臂。
保留本机 `station.local.yaml`，更新时先提交或备份源码改动，再做快进同步。
不要以整个目录覆盖另一台机器的工作区或数据。
