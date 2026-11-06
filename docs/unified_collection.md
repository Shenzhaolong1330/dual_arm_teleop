# Franka / Nero / Flexiv 统一数采

三端维护同一仓库的 `X-embodiment` 分支。`robot-record` 共用采集循环、
Quest `cross_embodiment_v1` 输入处理和 X-embodiment 数据格式。
物理语义修正、当前 TCP 核实结果和标定步骤见 [物理对齐说明](physical_alignment.md)。

```bash
robot-record --robot-type franka_dual_arm --repo-id '{robot_type}/my_task'
robot-record --robot-type nero_dual_arm --repo-id '{robot_type}/my_task'
robot-record --robot-type flexiv_dual_arm --repo-id '{robot_type}/my_task'
```

每次运行只控制所选本体。先追加 `--dry-run` 核对机器人地址、相机、坐标系、
Quest 映射和数据集字段；它仅加载配置，不连接设备或发送动作。
三种本体均启用 `mirror_teleop: true`，用于操作者和机器人面对面遥操：
操作者左手驱动机器人的右臂和右夹爪，右手驱动左臂和左夹爪。
数据中的 left/right 始终指机器人的左右臂，不是操作者的左右手。
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
- observation：双臂绝对末端位姿（目标契约为米、rotvec 弧度）和夹爪开口（米），共 14 维。
- action：双臂有效目标位姿增量和夹爪目标开口，共 14 维。
- Franka 使用服务器世界坐标；Nero/Flexiv 根据配置的安装变换转换。
- metadata 记录坐标、安装变换、有效指令语义和 Quest 处理参数；续采/合并拒绝不兼容语义。

统一格式便于跨本体训练，但不代表不同工位已经共用实物标定。
旧数据须确认语义再转换或合并；新数据的夹爪动作是目标开口，不能用观测开口代替。
Quest A 保存当前段并回位开始下一段，B 丢弃当前未保存段并回位重录；回位轨迹不进入数据。
参见 [按键说明](episode_controls.md)、[有效指令契约](effective_command_recording.txt)
和 [Quest 对齐说明](quest_processing_alignment.txt)。

## 2026-11-06 数据物理语义审查

字段顺序和维数在客户端适配层一致：

| 字段 | 目标物理含义 |
| --- | --- |
| `observation.state` 的前 12 维 | 左臂 xyz/rx/ry/rz，右臂 xyz/rx/ry/rz；位置为米，姿态应为绝对旋转向量（弧度） |
| `observation.state` 的后 2 维 | 左、右夹爪开口（米）；通常来自反馈，部分适配器在反馈失败时回退到缓存值，不能保证每帧都是新测量 |
| `action` 的前 12 维 | 同样的左右臂顺序；每个控制周期的目标位姿增量，平移为米、旋转应为旋转向量（弧度）；不是速度或绝对位姿 |
| `action` 的后 2 维 | 左、右夹爪目标开口（米）；不是二值开关，也不是观测到的开口 |
| `observation.images.left_wrist_image/right_wrist_image/head_image` | 机器人左腕、右腕、头部 RGB；当前共享配置为 424×240、30 Hz |

镜像在控制器输入到机器人动作之间完成。数据记录机器人侧字段和适配器返回的
控制目标，不记录“操作者左手坐标”，也不等同于机械臂实际运动或内部插补轨迹。
内部控制器还可能限幅、冻结或降低目标，客户端的布尔 ACK 不能证明最终执行量。

当前不能把三种本体标记为“物理语义已经实机完全验收”：

1. **Nero 旋转协议已核实并修正。** 已通过 NUC 新地址 `192.168.110.203`
   核实 `/home/dc/szl_ws/nero-runtime`：SDK 使用固定 XYZ RPY。
   新客户端取消分量倒序并正确转换 rotvec，要求 `nero.cartesian-target.v2`。
   新回执包含服务端限幅/缩放/冻结后的实际软件目标增量。
   安装代码不会启动服务或使能关节；工具点和 SDK/IK 末端参考点仍需核实。
2. **world 是每个工位的坐标约定。** Franka 由服务器提供；Nero 和 Flexiv 使用
   各自基座变换。仍需核实轴方向、原点、TCP 定义和相机角色，不能因为都叫 world
   就认为它们完成了跨工位标定。当前 Nero/Flexiv 基座间距配置分别为 0.54/0.32 m。
3. **夹爪范围和动力学不同。** Franka 开口范围为 0–0.085 m，Flexiv 为
   0–0.1 m；确认一代 Pika 后，Nero 名义范围修正为 0–0.095 m。
   同样的 0.04 是相同的目标距离，但相同扳机比例会对应不同距离。
   控制器限幅、死区、跟踪速度和延迟也不同。

可以开展跨本体联合训练实验，但应先关闭上述协议缺口，做每个本体的坐标/TCP
验证，并核查实际样本和 metadata。训练时保留本体来源，处理夹爪范围与动作分布，
统一任务语义并控制相机视角和采样比例，分别评估各本体的成功率。
Open X-Embodiment 和 Octo 展示了跨本体训练及迁移的可行性；这不能代替本系统的
协议校验或保证一个训练好的策略能直接在三台设备部署。

当前 `robot-train` 不接受 `dataset.repo_id` 列表，需要先生成一个合并后的标准
LeRobot 数据集。`tools-merge-datasets` 默认 `source.strict_robot_type: true`，
会拒绝不同本体；完成语义核验后可显式设置为 false，并保留 `strict_schema: true`。
新版跨本体合并要求合格 TCP/标定证据，输出 `robot_type=mixed`，逐帧保留
`embodiment_id`、`calibration_id` 及全部来源标定。训练配置需要显式使用本体条件。
改变 mirror 设置后，严格续采检查可能拒绝旧数据集；应新建采集版本，避免改写旧 metadata。

参考：[Open X-Embodiment](https://robotic-transformer-x.github.io/)、
[Octo](https://octo-models.github.io/)。

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
