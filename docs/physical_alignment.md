# 三本体物理语义与 TCP 对齐

新采集元数据使用 `physical_contract.version = x_embodiment_physical_v2`。
这表示客户端旋转、目标增量和反馈失败处理的契约版本；它本身不是实物标定证书。
`physical_calibration` 单独记录工具、安装、控制参数和标定来源。
改变这些内容后必须新建数据版本，不能修改旧数据的 metadata 后续采。

## 数据字段

`observation.state` 保持 14 维：左 xyz/rotvec、右 xyz/rotvec、左右夹爪开口。
`action` 保持 14 维：左目标平移/旋转增量、右目标平移/旋转增量、左右夹爪目标开口。
单位分别为 m、rad、m。左右始终指机器人左右，镜像只作用于 Quest 输入映射。

平移增量是当前参考目标到接受后的目标之差；旋转增量满足
`R_after = Exp(delta_rotvec) @ R_before`，表达在公共 world 坐标中。
它不是 RPY 分量、姿态向量的直接相减、速度或机械臂实际运动量。
Franka/Nero/Flexiv 都累计控制目标；观测反馈不能在每帧偷偷重置目标。
连接、回位和 Nero 超过空闲阈值后的明确重新同步除外。

夹爪 observation 来自反馈，action 为目标开口。Franka 当前反馈是从 Robotiq
寄存器位置线性估计的开口，不是直接测得的米制开口；元数据对此有单独来源标记。
Nero 和 Flexiv 使用 SDK 的 width(m)。新适配器遇到缺失/无效反馈或通信失败会
终止该帧，不能用上一次完整观测或目标宽度冒充当前观测。
仍需通过尺量验证各夹爪零点、最大行程及中间开口的误差。

采样默认 30 Hz；增量按每个控制步解释。各控制器插补频率、限制和动力学仍不同，
不会因为统一字段而变成相同机械臂。数据集标准 timestamp 是名义采样时间，
目前没有提供跨主机、相机和关节反馈的硬件时间同步保证。

## 2026-11-06 核实的安装与工具

| 本体 | 基座到 world | 原生工具点来源 |
| --- | --- | --- |
| Franka | 当前服务端 torso_extrinsics.yaml 的左右 Y=±0.25 m；左 Rx(-45°)，右 Rx(+45°)，以实际矩阵为准 | 两臂实时 FrankaRobotState.F_T_EE 均为 `[0,0,0.160,1,0,0,0]` |
| Nero | 操作者提供：平行直立，左右 Y=±0.27 m | NUC 当前 `tcp_offset_enabled=false`；一代带相机 Pika 直接安装在法兰上，操作者给出法兰至 TCP 为 170 mm，仍需确定距离起点及工具轴 |
| Flexiv | 操作者提供：平行直立，左右 Y=±0.16 m | 两臂当前工具 Flexiv-GN01-rs，Tool.params().tcp_location 均为 `[0,0,0.160,1,0,0,0]` |

安装变换为 `[x,y,z,qw,qx,qy,qz]`，m。Franka 的 world 变换已经在服务端执行，
客户端不能再乘一次。Nero/Flexiv 在客户端使用各自外参。Flexiv world 模式的动作
现在使用完整外参旋转，与观测完全相同；旧的 mount_yaw 只保留给 base 观测兼容模式。

两个控制器同样设置 0.160 m 只证明配置一致；仍需核实工具点确实是相同定义的
指间中心，及工具轴与夹合/接近方向的对应。完整跨本体 TCP 契约选择
`finger_midpoint_x_closing_z_approach_v1`：原点在约定接触高度的指间中心，
X 为夹合轴，Z 为接近方向，Y 组成右手系。非同向原生工具轴需要相应旋转标定。

## Nero 新协议与安装

当前 SDK 的六维反馈是固定 XYZ 欧拉角 RPY，客户端正确转换为 rotvec，取消旧分量倒序。
新 RPC `servo_p_OL_v2(arm, delta_rotvec, tcp_in_native)` 返回原生基座系中的
参考 TCP 位姿、接受后 TCP 位姿、有效增量和服务端单调时钟。旧 RPC 继续保留。
新版客户端必须完成协议握手，并核对服务端原生 TCP 偏移；不自动回退为 bool ACK。

服务端继续使用原有 IK、限幅、故障缩放和旋转冻结，不放宽安全参数。
回执捕获发生在空闲重新同步之后、目标限幅之后。IK 明确拒绝可记录零动作；
结果未知的异常必须终止该帧。该回执说明软件控制目标，不证明物理执行成功。

`left_tcp_in_native` / `right_tcp_in_native` 指定原生 SDK/控制点到统一 TCP 的
固定变换。它同时作用于观测和控制；通过变换参考位姿和目标位姿后求增量，
保留旋转引起的工具点杠杆臂平移。只旋转 delta 向量或只给观测加偏移均不正确。
原生 SDK 反馈点与 IK URDF 的末端点也必须核实一致，或分别补偿。

安装脚本只备份/编辑源码、检查语法，不导入服务器、不打开 CAN、不启动服务：

```bash
python robots/dual_agilex_nero/install_physical_protocol.py \
  --server /absolute/path/nero_interface_server.py --apply
```

原文件备份为 `nero_interface_server.py.before-physical-v2`，旧备份不会被覆盖。
NUC 服务启动会使能关节，安装代码不等于已启动或完成实机验收。

## 公开夹爪模型的边界

核对了 AgileX 官方
[Pika Gripper URDF](https://github.com/agilexrobotics/pika_ros/blob/1ec77880011af608f7f6af8d3c0e7c163b300984/src/pika_gripper_description/urdf/pika_gripper_description.urdf)、
[Pika 2 URDF](https://github.com/agilexrobotics/agx_arm_sim/blob/master/agx_arm_description/urdf/pika2_gripper.urdf)
和 [Nero/AGX Gripper 模型](https://github.com/agilexrobotics/agx_arm_urdf/blob/main/nero/urdf/nero_with_gripper_description.xacro)。
它们的根坐标、手指结构和安装变换不同。所查 Pika Gripper 模型没有给出当前
Nero 法兰到该夹爪根坐标的安装变换，也没有明确的目标接触中心 TCP。
不能用惯性质心、手指关节原点、另一个型号的长度或旧程序中的 0.19 m 常数替代标定。
操作者已确认实际安装为带相机的一代 Pika Gripper，直接安装在法兰上，
没有额外转接件；并提供法兰到 TCP 距离 170 mm。不再把转接件图纸或多点
实物 TCP 标定作为前提。有完整安装几何和接触点定义时，可采用 `model_based`。
已找到 [官方一代 Pika 手册](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/Pika_Product_User_Manual_beta_EN_ROS1.pdf?v=1792488910)：
第 4.1 节介绍适配其他机械臂的法兰，不能据此认定本机存在额外转接件。
第 16 页参数表给出的最大开口为 95 mm，新 Nero 配置据此使用 0.095 m，
不再沿用 AGX/其他夹爪的 0.100 m。实际零点和行程仍需按每只夹爪核对。

[Nero 官方法兰模型](https://github.com/agilexrobotics/agx_arm_urdf/blob/680fa925cd00db7e2628563d2edc5540963045de/nero/urdf/nero_with_flange_description.xacro)
另有 `link7 -> flange_link` 固定变换：平移 `[0.031,0,-0.0235]` m，
固定 XYZ RPY `[-1.5708,0,-1.5708]` rad。官方 SDK/ROS 文档的原生末端参考为
`link7`，不能把实际安装面中心与这个坐标原点默认为同一个点。
若 170 mm 是沿实际安装面外法向的长度，点在 link7 中约为 `[0.201,0,-0.0235]` m；
若 170 mm 已从 SDK 原生原点沿 X 量取，则为 `[0.170,0,0]` m。两个定义相差
约 39 mm，必须由测量起点确定。170 mm 是操作者数据，不冒充官方标称 TCP。
一代 Pika URDF 的夹合轴是夹爪模型自身的 Y 轴；跨本体公共工具 X 轴为夹合方向，
因此还要应用安装旋转和工具轴重排，不能只设置平移后保留错误轴向。
当前配置保留这些证据，未确认的完整变换不自动生效。

## 不驱动硬件的标定工具

`scripts/tools/physical_calibration.py` 只处理已经采集的 JSON，不连接机器人。

```bash
python scripts/tools/physical_calibration.py pivot \
  --input native_poses.json --output tcp_pivot.json
python scripts/tools/physical_calibration.py world \
  --input paired_points.json --output base_to_world.json
```

pivot 输入 `{"poses": [[x,y,z,rx,ry,rz], ...]}`，至少六个不同姿态保持同一接触点；
需要绕多个独立轴改变姿态。它解算 TCP 平移及残差，不会把 TCP 方向假装也标定出来。
world 输入成对的 `source_points`、`world_points`，至少三个不共线点。
默认 RMS 上限 2 mm；秩不足、病态或残差过大时拒绝结果。输出不覆盖已有文件。

将有依据的变换写入 `station.local.yaml` 对应机器人字段；记录标定来源和版本。
例如已核实的公共 TCP 可声明：

```yaml
physical_calibration:
  status: measured # 或有完整几何依据的 model_based
  tcp_definition: finger_midpoint_x_closing_z_approach_v1
  world_convention: base_midpoint_x_forward_y_left_z_up_v1
  evidence: "实际标定报告/制造商模型版本及安装变换"
```

这个声明必须对应实际生效的 TCP 变换和工具轴；不能只改标签来绕过核验。

## 合并与训练

新旧 physical_contract 或坐标语义不匹配时拒绝合并。跨本体合并还要求所有来源
具有合格 TCP 定义、`model_based`/`measured` 状态及标定证据。
不合格数据仍可分本体保存和审查，但不能被工具默认为已对齐。

跨本体合并输出 `robot_type=mixed`，逐帧附加 `embodiment_id`、`calibration_id`；
映射和所有来源标定保存在 metadata。不能再用第一个来源的 robot_type/外参代表全部样本。
这两个字段是独立来源字段，不改变 14 维 state/action。训练配置仍需显式读取
本体条件；仅有这个字段不代表现有 ACT 已自动消费它。分别报告各本体测试成功率。
旧 Nero action 缺少服务端有效目标回执，无法从 bool ACK 可靠重建限幅/冻结后的标签。

## 离线验证

`tests/test_physical_alignment.py` 覆盖旋转转换、TCP 杠杆臂、有效回执、缺失反馈、
标定可观测性/残差以及混合数据逐帧来源。
`scripts/tools/check_nero_server_contract.py` 从真实服务源码只提取 ServoP 和限幅函数，
使用假机械臂/IK 检查真实方法，完全不运行服务器构造函数或 SDK 硬件驱动。
