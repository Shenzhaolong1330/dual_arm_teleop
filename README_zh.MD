# Dual Arm Teleoperation, Data collection and policy training

这是一个基于 LeRobot 的双臂机器人遥操作、数据采集、回放、可视化、数据预处理和策略训练工具链。项目核心目标是把真实机器人操作流程串起来：用 Oculus Quest 做末端位姿遥操作，用 LeRobot 格式记录数据，再用 ACT / Diffusion Policy 训练并部署策略。

本仓库采集的数据可用于 [openpi-franka](https://github.com/Shenzhaolong1330/openpi-franka.git) 和 StarVLA 的策略训练，也可以直接用于本仓库提供的 ACT / Diffusion Policy 训练流程。

<p align="center">
  <img src="assets/record.png" alt="录制界面" width="820">
  <br>
  <sub>将遥操作过程直接录制成 LeRobot 数据集。</sub>
</p>

## 项目亮点

- **多机器人统一接口**：Franka、Dobot、ARX R5、AgileX Nero 等系统共用同一套记录、回放和训练流程。
- **Oculus Quest 遥操作**：通过 6-DoF 手柄位姿生成双臂末端增量位姿动作。
- **LeRobot 数据集格式**：同步记录相机图像、机器人状态、末端 delta action 和夹爪命令。
- **OpenPI 与 StarVLA 训练数据**：采集的数据可用于 [openpi-franka](https://github.com/Shenzhaolong1330/openpi-franka.git) 和 StarVLA 训练流程。
- **回放与可视化**：可以快速检查 episode，也可以把已录制 action 回放到机器人。
- **模仿学习训练与部署**：支持 ACT 和 Diffusion Policy，并通过 YAML 独立管理算法超参数。
- **ACT 数据清洗工具**：支持裁剪长静止段、平滑 action 毛刺、保护夹爪开合关键帧。

## 支持的机器人

机器人注册入口在 [robots/__init__.py](robots/__init__.py)，通过 [scripts/config/record_cfg.yaml](scripts/config/record_cfg.yaml) 里的 `record.robot_type` 选择。

| `robot_type` | 系统 | 典型动作空间 | 说明 |
| --- | --- | --- | --- |
| `franka_dual_arm` | 双 Franka + Robotiq | 左右臂 delta EE pose + 二值夹爪 | 当前主要部署路径。 |
| `dobot_dual_arm` | 双 Dobot Nova | 左右臂 delta EE pose / 关节动作 + 夹爪 | 包含 Dobot RPC client/server。 |
| `arx_dual_arm` | ARX R5 双臂 | 左右臂 delta EE pose / 关节动作 + 集成夹爪 | 包含 ARX 动作滤波、deadband 和 URDF 资源。 |
| `nero_dual_arm` | AgileX Nero 双臂 | 左右臂 delta EE pose + 夹爪 | 通过 `dual_agilx_nero` 配置。 |
| `franka` | 单 Franka | 右臂 delta EE pose / 关节动作 + 夹爪 | 适合单臂测试和开发。 |
| `flexiv_dual_arm` | 非夕 Flexiv 双臂 | 即将支持 | Flexiv support incoming。 |

<p align="center">
  <img src="assets/dual_franka.jpeg" alt="Dual Franka" width="24%">
  <img src="assets/dobot.jpeg" alt="Dual Dobot" width="32%">
  <img src="assets/arx.jpeg" alt="ARX R5" width="24%">
  <img src="assets/agilex.jpeg" alt="AgileX Nero" width="24%">
  <br>
  <sub>已接入/适配的机器人系统示例，非夕 Flexiv 即将支持。</sub>
</p>

其他硬件与遥操作设备：

<p align="center">
  <img src="assets/arx_lift.jpeg" alt="ARX lift setup" width="24%">
  <img src="assets/flexiv.jpeg" alt="Flexiv setup asset" width="24%">
  <img src="assets/quest3s.jpg" alt="Oculus Quest 3S" width="30%">
  <br>
  <sub>实验平台资产和 Oculus Quest 3S 遥操作设备。</sub>
</p>

## 支持的工具操作

| 命令 | 功能 |
| --- | --- |
| `robot-record` | 数据录制、策略部署或混合控制，具体由 `record.run_mode` 决定。 |
| `robot-replay` | 从 LeRobot 数据集中读取 episode，并把 action 回放到机器人。 |
| `robot-visualize` | 可视化数据集 episode、相机图像和状态/action。 |
| `robot-reset` | 复位或回 home，具体行为由机器人配置决定。 |
| `robot-train` | 使用 [scripts/config/train_cfg.yaml](scripts/config/train_cfg.yaml) 训练 ACT / Diffusion Policy。 |
| `robot-help` | 查看当前可用命令。 |
| `tools-check-rs` | 查看已连接 RealSense 相机序列号。 |
| `tools-check-dataset` | 检查/刷新本地数据集索引信息。 |
| `tools-preprocess-dataset` | 生成清洗后的训练数据集。 |
| `bash scripts/tools/check_robotiq_ports.sh` | 检查 Robotiq 夹爪串口。 |
| `bash scripts/tools/map_gripper.sh` | 夹爪设备映射辅助脚本。 |

常用运行模式：

```yaml
record:
  run_mode: "run_record"  # run_record, run_policy, run_mix
  robot_type: "franka_dual_arm"
  control_mode: "oculus"
```

## 支持的算法

策略配置放在 [scripts/config/policies](scripts/config/policies)。

| 算法 | 配置文件 | 适用场景 |
| --- | --- | --- |
| ACT | [act_cfg.yaml](scripts/config/policies/act_cfg.yaml) | Action Chunking Transformer，适合视觉/状态条件下的模仿学习。 |
| Diffusion Policy | [diffusion_cfg.yaml](scripts/config/policies/diffusion_cfg.yaml) | 基于扩散模型的动作序列生成。 |
| OpenPI | [openpi-franka](https://github.com/Shenzhaolong1330/openpi-franka.git) | 外部 OpenPI 训练流程，可使用本仓库采集的数据训练 Franka 策略。 |
| StarVLA | [starVLA/starVLA](https://github.com/starVLA/starVLA) | 外部 VLA 训练框架，可使用本仓库采集的机器人数据。 |

训练主配置在 [scripts/config/train_cfg.yaml](scripts/config/train_cfg.yaml)。当前训练脚本还支持按动作幅度加权采样，缓解数据中大量静止帧导致策略塌到零动作的问题：

```yaml
action_sampling:
  enabled: True
  min_translation_norm: 0.010
  min_rotation_norm: 0.030
  active_weight: 20.0
  inactive_weight: 0.2
```

## ACT 数据预处理

预处理配置在 [scripts/config/preprocess_dataset_cfg.yaml](scripts/config/preprocess_dataset_cfg.yaml)。工具会写出一个新数据集，不会修改原始数据。

| 功能 | 行为 |
| --- | --- |
| 裁剪长静止段 | 连续静止超过 `min_static_frames` 时，只保留开头 `keep_start_frames` 和结尾 `keep_end_frames`。 |
| action 平滑/去毛刺 | 对 Cartesian `delta_ee_pose` 标签做 median、EMA 或 median+EMA 平滑。 |
| 保留夹爪事件 | 夹爪开/合切换附近保留一段上下文，避免抓取关键帧被裁掉。 |

先 dry-run 看裁剪比例：

```bash
tools-preprocess-dataset --dry-run
```

生成或覆盖 cleaned 数据集：

```bash
tools-preprocess-dataset --overwrite
```

也可以直接运行 Python 脚本：

```bash
python scripts/tools/preprocess_dataset.py --overwrite
```

## 快速开始

### 1. 创建环境

```bash
conda create -n dual_arm_data python=3.10
conda activate dual_arm_data
```

### 2. 安装 LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

### 3. 安装本项目

```bash
mkdir -p dual_arm_ws
cd dual_arm_ws
git clone https://github.com/Shenzhaolong1330/dual_arm_teleop.git
cd dual_arm_teleop
pip install -e .
```

查看可用命令：

```bash
robot-help
```

## Oculus Quest 遥操作

本项目使用 Oculus Quest 3/3S 手柄位姿生成双臂末端 delta action。

<p align="center">
  <img src="assets/quest3s.jpg" alt="Oculus Quest 3S" width="520">
</p>

### 手柄按键

| 控制 | 功能 |
| --- | --- |
| 左手 Grip | 按住后允许左臂运动。 |
| 右手 Grip | 按住后允许右臂运动。 |
| 左/右 Trigger | 按下关闭对应夹爪，松开打开。 |
| A 按钮 | 请求机器人 reset / go-home，具体行为由机器人配置决定。 |
| 手柄位姿 | 生成末端执行器增量位姿。 |

### Oculus 配置

安装 ADB：

```bash
sudo apt install android-tools-adb
adb version
```

在 Meta Quest 手机 app 中开启开发者模式，连接头显后检查：

```bash
adb devices
```

无线使用时获取头显 IP，并写入 [scripts/config/record_cfg.yaml](scripts/config/record_cfg.yaml)：

```bash
adb shell ip route
```

```yaml
teleop:
  oculus_config:
    ip: "192.168.8.114"
```

如需安装 Oculus Reader APK：

```bash
cd teleoperators/oculus_teleoperator/oculus/oculus_reader/oculus_reader/APK
adb install -r teleop-debug.apk
```

## 数据集流程

### 录制

先检查 [scripts/config/record_cfg.yaml](scripts/config/record_cfg.yaml)，然后运行：

```bash
robot-record
```

<p align="center">
  <img src="assets/record.png" alt="数据录制" width="820">
</p>

录制按键：

| 按键 | 行为 |
| --- | --- |
| 右方向键 | 保存当前 episode，进入 reset 阶段，然后开始下一个 episode。 |
| 左方向键 | 重新录制当前 episode。 |
| Esc | 退出录制并保存已完成的数据。 |
| Ctrl+C / 异常 | 进入清理流程，询问是否删除未完成数据。 |

### 回放

```bash
robot-replay
```

### 可视化

```bash
robot-visualize
```

<p align="center">
  <img src="assets/visualize.png" alt="数据可视化" width="820">
</p>

## 数据集组织

数据默认保存在本地 LeRobot cache 下：

```bash
~/.cache/huggingface/lerobot
```

<p align="center">
  <img src="assets/dataset.png" alt="数据集目录" width="780">
  <br>
  <img src="assets/dataset_info.png" alt="数据集信息" width="780">
</p>

推荐命名格式：

```text
<user_or_namespace>/<task_description>
```

本地辅助工具会记录 task、date、version、user notes 等信息。可以用下面的命令检查或刷新：

```bash
tools-check-dataset
```

## 训练与部署

### 训练

修改 [scripts/config/train_cfg.yaml](scripts/config/train_cfg.yaml)，然后运行：

```bash
robot-train
```

关键字段示例：

```yaml
train:
  dataset:
    repo_id: "dual_franka_1/pick_up_and_seal_20260513_v01_cleaned"
    root: "/home/deepcybo/.cache/huggingface/lerobot/dual_franka_1/pick_up_and_seal_20260513_v01_cleaned"
  policy_cfg: "scripts/config/policies/act_cfg.yaml"
  policy:
    type: "act"
```

### 部署策略

将 `record.run_mode` 设置为 `run_policy`，并把 `record.policy.pretrained_path` 指向 checkpoint 的 pretrained model 目录：

```yaml
record:
  run_mode: "run_policy"
  policy_cfg: "scripts/config/policies/act_cfg.yaml"
  policy:
    type: "act"
    pretrained_path: "outputs/train/<run_name>/checkpoints/last/pretrained_model"
```

然后运行：

```bash
robot-record
```

## 配置文件索引

| 文件 | 作用 |
| --- | --- |
| [scripts/config/record_cfg.yaml](scripts/config/record_cfg.yaml) | 机器人、遥操作、相机、录制、回放、可视化和策略部署配置。 |
| [scripts/config/train_cfg.yaml](scripts/config/train_cfg.yaml) | 数据集、策略、优化器、日志、checkpoint 和训练配置。 |
| [scripts/config/preprocess_dataset_cfg.yaml](scripts/config/preprocess_dataset_cfg.yaml) | 数据清洗配置，包括静止段裁剪、动作平滑和夹爪事件保护。 |
| [scripts/config/policies/act_cfg.yaml](scripts/config/policies/act_cfg.yaml) | ACT 策略结构和优化器配置。 |
| [scripts/config/policies/diffusion_cfg.yaml](scripts/config/policies/diffusion_cfg.yaml) | Diffusion Policy 策略结构和优化器配置。 |

## 扩展新机器人

可以参考 [robots/example_robot](robots/example_robot)。一个新机器人接入通常需要：

1. 配置 dataclass。
2. 实现机器人类，包括 `connect`、`disconnect`、`get_observation`、`send_action`、`action_features`、`observation_features`。
3. 如有需要，实现 RPC client/server。
4. 在 [robots/__init__.py](robots/__init__.py) 中注册新的 `robot_type`。

尽量保持 action feature name 与现有遥操作和训练链路兼容：

```text
left_delta_ee_pose.{x,y,z,rx,ry,rz}
right_delta_ee_pose.{x,y,z,rx,ry,rz}
left_gripper_cmd_bin
right_gripper_cmd_bin
```
