# Le-nero 双臂遥操作与数据采集

本目录是 Le-nero 中的双臂遥操作与数据采集层，基于 LeRobot 提供机器人适配、Oculus 遥操作、数据集录制/回放/可视化、策略训练和轮次式 DAgger 闭环。

<p align="center">
  <img src="docs/pic/nero.png" alt="Dobot dual-arm system" width="600">
</p>

<p align="center">
  <img src="docs/pic/dobot.jpeg" alt="Dobot dual-arm system" width="600">
</p>

<p align="center">
  <img src="docs/pic/arx.jpeg" alt="ARX dual-arm system" width="600">
</p>

<p align="center">
  <img src="docs/pic/quest3s.jpg" alt="Oculus Quest" width="600">
</p>

## 仓库获取与环境配置

首次拉取仓库时建议直接带上子模块：

```bash
git clone --recurse-submodules https://github.com/Key-Zzs/Le-nero
cd Le-nero
```

如果已经 clone 过仓库，但子模块目录为空或缺文件，在仓库根目录执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

日常更新主仓库和子模块：

```bash
cd Le-nero
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --merge --recursive
```

切换主仓库分支：

```bash
git fetch origin
git switch <branch_name>
git submodule update --init --recursive
```

切换或更新本双臂遥操作包：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
git fetch origin
git switch main
git pull --ff-only
```

创建 Python 环境并安装根仓库与本包：

```bash
conda create -n dual_arm_teleop python=3.10 -y
conda activate dual_arm_teleop
python -m pip install --upgrade pip

cd Le-nero
pip install -e .

cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
pip install -e .
```

Oculus Reader 不是通过当前 `.gitmodules` 管理的子模块，需要单独 clone 到指定目录：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
cd oculus_reader
pip install -e .
```

如果该目录已经存在，只需要更新并重新安装：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus/oculus_reader
git pull --ff-only
pip install -e .
```

Oculus 连接还需要 ADB：

```bash
sudo apt install android-tools-adb
adb devices
```

首次 USB 连接时需要在头显中允许 USB 调试；无线连接时可以先通过 `adb shell ip route` 查看头显 IP，再执行 `adb connect <Oculus_IP>:5555`。

## 核心模块与调用机理

运行链路可以简化理解为：

```text
scripts/config/*.yaml
        |
        v
scripts/core/*.py 命令入口
        |
        +--> robots 创建真实机器人接口
        +--> teleoperators 创建 Oculus 遥操作输入
        +--> ../../src/lerobot/policies 创建策略模型
        |
        v
LeRobot dataset / train / replay / visualize
```

### 策略层

策略代码位于 Le-nero 根仓库：

```text
src/lerobot/policies
```

这里保留 LeRobot 的策略抽象和具体实现，例如 `act`、`diffusion`、`smolvla`、`pi0` 等。双臂遥操作脚本主要通过 `lerobot.policies.factory.make_policy` 和 `make_pre_post_processors` 创建策略对象及前后处理器。

本包最常用的策略配置文件是：

- `scripts/policy_config/act_train_config.yaml`：ACT 训练配置。
- `scripts/policy_config/act_reason_config.yaml`：ACT 推理/部署配置。
- `scripts/policy_config/diffusion_train_config.yaml`：Diffusion Policy 训练配置。
- `scripts/policy_config/diffusion_reason_config.yaml`：Diffusion Policy 推理/部署配置。

`scripts/core/policy_config_utils.py` 负责解析 `record_cfg.yaml`、`train_cfg.yaml` 或 `dagger_rounds_cfg.yaml` 中的策略配置路径。相对路径会优先按本包根目录解析，也支持直接写绝对路径。

### 机器人通讯接口定义层

机器人接口位于：

```text
robots
```

`robots/__init__.py` 是机器人注册表，当前注册的类型包括：

- `franka`
- `dobot_dual_arm`
- `nero_dual_arm`
- `franka_dual_arm`

脚本不会直接实例化某个具体机器人类，而是根据配置中的 `robot_type` 调用：

```python
create_robot_config(robot_type, **robot_cfg)
create_robot(robot_type, robot_config)
```

具体机器人类负责实现 LeRobot 期望的机器人接口，例如 `connect()`、`reset()`、`send_action()`、相机初始化、观测字段和动作字段定义。以 `nero_dual_arm` 为例，`robots/dual_agilex_nero/nero_dual_arm.py` 通过 `NeroDualArmClient` 连接双臂 zerorpc 服务，并把双臂末端位姿、关节状态、夹爪命令和 RealSense 相机组织成 LeRobot 可记录的数据结构。

硬件相关参数不建议直接写在运行脚本里，而是放在：

```text
scripts/DAS_config
```

例如 `nero_cofig.yaml` 定义 Nero 的机器人 IP、端口、夹爪参数、Oculus 映射和相机序列号。`run_record.py`、`run_replay.py`、`reset_robot.py` 会根据 `record.robot_type` 自动加载对应 DAS 配置；也可以在 `record_cfg.yaml` 中通过 `das_config_path` 显式指定。

### 工具脚本、数采系统与策略配置

主要脚本位于：

```text
scripts
```

常用目录含义：

- `scripts/core`：命令入口实现，包括采集、回放、可视化、重置、训练、DAgger。
- `scripts/config`：主流程配置，包含 `record_cfg.yaml`、`train_cfg.yaml`、`dagger_rounds_cfg.yaml`。
- `scripts/policy_config`：策略超参数配置，区分 train 和 reason 两类。
- `scripts/DAS_config`：硬件和遥操作细节配置。
- `scripts/tools`：数据集检查、RealSense 设备检查、数据集修补和重命名等工具。

核心配置文件：

- `record_cfg.yaml`：数据采集、策略推理、混合控制、回放和可视化共用的主配置。
- `train_cfg.yaml`：策略训练配置，包括数据集路径、输出目录、GPU、batch size、训练步数和 wandb。
- `dagger_rounds_cfg.yaml`：轮次式 DAgger 控制器配置，负责把采集、导出、训练串成闭环。
- `*_train_config.yaml`：策略训练时使用的模型结构和训练相关超参数。
- `*_reason_config.yaml`：策略推理或部署时使用的模型结构、设备和 checkpoint 参数。

`robot-record` 的三种运行模式由 `record.run_mode` 控制：

- `run_record`：纯遥操作采集。
- `run_policy`：加载策略 checkpoint，由策略控制机器人。
- `run_mix`：策略执行为主，操作者可接管，用于 DAgger 数据采集。

## 核心命令

安装本包后，`setup.py` 会注册以下命令：

| 命令 | 作用 | 默认配置 |
| --- | --- | --- |
| `robot-record` | 遥操作采集、策略执行或 run_mix 混合采集 | `scripts/config/record_cfg.yaml` |
| `robot-replay` | 回放已采集 episode | `scripts/config/record_cfg.yaml` 的 `replay` 段 |
| `robot-visualize` | 用 Rerun 可视化数据集 episode | `scripts/config/record_cfg.yaml` 的 `visualize` 段 |
| `robot-reset` | 根据配置连接机器人并回 home | `scripts/config/record_cfg.yaml` |
| `robot-train` | 训练 ACT 或 Diffusion Policy | `scripts/config/train_cfg.yaml` |
| `robot-dagger` | 运行轮次式 DAgger：采集、导出、训练下一轮策略 | `scripts/config/dagger_rounds_cfg.yaml` |
| `robot-dagger-export` | 从 raw run_mix 日志单独导出 DAgger 训练数据 | `scripts/config/dagger_rounds_cfg.yaml` 的 `dagger_export` 段 |
| `tools-check-dataset` | 检查本地 LeRobot 数据集信息 | 命令参数 |
| `tools-check-dagger-dataset` | 检查导出的 DAgger 数据集 | 命令参数 |
| `tools-check-rs` | 查看 RealSense 设备序列号 | 无 |
| `robot-help` | 打印命令摘要 | 无 |

所有核心命令都支持显式传入配置文件，推荐调试时总是写明路径：

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
robot-reset --config scripts/config/record_cfg.yaml
robot-train --config scripts/config/train_cfg.yaml
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
robot-dagger-export --config scripts/config/dagger_rounds_cfg.yaml
```

## 配置检查清单

采集前通常需要修改 `scripts/config/record_cfg.yaml`：

- `record.repo_id`：数据集名称，建议使用 `<robot_task<num>_step<num>/<description>`，例如 `nero_task3_step1/2mL_empty_right`。
- `record.robot_type`：选择 `nero_dual_arm`、`franka_dual_arm` 等机器人类型。
- `record.run_mode`：选择 `run_record`、`run_policy` 或 `run_mix`。
- `record.policy.type`、`config_path`、`pretrained_path`：仅在 `run_policy` 或 `run_mix` 时需要确认。
- `record.task`：任务描述、episode 数量、是否 resume、是否记录 success。
- `record.time`：episode 最大时长、reset 时长和 metadata 保存周期。
- `replay`、`visualize`：回放和可视化默认使用的数据集和 episode。

硬件参数通常在 `scripts/DAS_config/*.yaml` 中修改：

- `teleop.oculus_config.ip`：Oculus Quest IP。
- `teleop.oculus_config.*_pose_scaler` 和 `*_channel_signs`：左右手柄到机器人动作的映射。
- `robot.robot_ip`、`robot.robot_port`：机器人服务地址。
- `robot.use_gripper` 和夹爪参数：夹爪启用、开合阈值、最大开口和力。
- `cameras.*_serial`、`width`、`height`：RealSense 序列号和分辨率。

训练前通常需要修改 `scripts/config/train_cfg.yaml`：

- `train.dataset.repo_id` 和 `train.dataset.root`：训练数据集。
- `train.policy.type` 和 `train.policy.config_path`：策略类型和训练配置。
- `train.output_dir`、`job_name`：模型和日志输出位置。
- `train.training`：GPU 可见卡、显存限制、TF32 等训练设备设置。
- `train.steps`、`batch_size`、`num_workers`、`save_freq`：训练规模。
- `train.wandb`：wandb 项目和模式。

DAgger 前通常需要修改 `scripts/config/dagger_rounds_cfg.yaml`：

- `dagger_rounds.seed_repo_id` 或 `seed_dataset_path`：round 0 使用的种子数据集。
- `dagger_rounds.initial_pretrained_path`：可选，已有初始 checkpoint 时填写。
- `dagger_rounds.policy`：轮次中使用的策略类型，以及 train/reason 配置路径。
- `dagger_rounds.episodes_per_round`、`num_rounds`、`round_schedule`：每轮采集数量、轮数和训练步数策略。
- `dagger_rounds.output_root`：DAgger 轮次输出目录。
- `dagger_rounds.record_cfg_path`、`train_cfg_path`：被控制器动态改写并调用的基础配置。
- `dagger_rounds.policy_backend.export`：run_mix 日志导出为训练数据的规则。

## Oculus 设置与控制

### 安装并测试 Oculus Reader

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
cd teleoperators/oculus_teleoperator/oculus/oculus_reader
pip install -e .
python oculus_reader/reader.py
```

### 安装 Oculus Reader APK

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
cd teleoperators/oculus_teleoperator/oculus/oculus_reader/oculus_reader/APK
adb install -r teleop-debug.apk
```

安装完成后，应用将出现在 Oculus Quest 库中的 **未知来源** 下。

### 配置 Oculus 遥操作

在所选 DAS 配置中设置 Oculus IP 和映射参数，例如 `scripts/DAS_config/nero_cofig.yaml`：

```yaml
teleop:
  control_mode: "oculus"
  dual_arm: true
  oculus_config:
    ip: "192.168.110.62"
    use_gripper: true
    left_pose_scaler: [1.2, 1.2]
    right_pose_scaler: [1.2, 1.2]
    left_channel_signs: [-1, -1, 1, 1, 1, 1]
    right_channel_signs: [-1, -1, 1, 1, 1, 1]
```

### 控制器按键

| 控制键 | 功能 |
| --- | --- |
| 左握持键 `LG` | 按住以启动左臂末端运动。在 `run_mix` 中会开始或持续左臂专家接管。 |
| 右握持键 `RG` | 按住以启动右臂末端运动。在 `run_mix` 中会开始或持续右臂专家接管。 |
| 左扳机 `LTr` | 控制左夹爪；按下关闭，松开打开。 |
| 右扳机 `RTr` | 控制右夹爪；按下关闭，松开打开。 |
| `Y` 按钮 | 在 `run_mix` 中将左夹爪通道交还给策略控制。 |
| `B` 按钮 | 在 `run_mix` 中将右夹爪通道交还给策略控制。 |
| `A` 按钮 | 在当前 teleoperator/robot 实现支持时请求机器人复位。 |
| 控制器位姿 | 在对应握持键按住时，控制对应机械臂的末端增量位姿。 |

如果启用了 `mirror_teleop`，左右控制器的对应关系会交换，并在发送给机器人前对位姿增量做镜像。

### DAgger/run_mix 控制定义

- 默认由策略控制机器人；人工输入只覆盖正在主动控制的通道。
- 按住 `LG` 或 `RG` 会让对应手臂进入专家接管。接管的第一帧标记为 `takeover_start`，持续接管帧标记为 `recovery`。
- `LTr` 和 `RTr` 独立控制夹爪，不要求同时接管手臂。夹爪接管使用 soft takeover：扳机命令需要先接近当前保持的夹爪值，手动夹爪控制才会生效，避免夹爪突然跳变。
- 按 `Y` 可将左夹爪交还给策略，按 `B` 可将右夹爪交还给策略；交还后需要先松开对应扳机，才能再次手动接管该夹爪。
- 使用左箭头丢弃失败、不完整、质量差或不适合作为训练示范的 episode。`full_episode.success_policy` 为 `recorded_is_success` 时，这一点尤其重要。

### 坐标系映射

实际映射由 `*_pose_scaler` 和 `*_channel_signs` 配置。常见 Oculus 到机器人坐标映射如下：

| Oculus 轴 | 机器人轴 | 描述 |
| --- | --- | --- |
| X 向右 | -Y 向左 | 横向移动 |
| Y 向上 | Z 向上 | 垂直移动 |
| Z 向后 | X 向前 | 前后移动 |

### 故障排除

```bash
# 重启 ADB 服务器
adb kill-server
adb start-server

# 检查已连接设备
adb devices

# 停止 Oculus 应用
adb shell am force-stop com.rail.oculus.teleop

# 重新安装 APK
adb uninstall com.rail.oculus.teleop
adb install -r teleoperators/oculus_teleoperator/oculus/oculus_reader/oculus_reader/APK/teleop-debug.apk
```

## 常用流程

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop

# 1. 查看相机序列号，填入 scripts/DAS_config/*.yaml
tools-check-rs

# 2. 检查策略配置能否正常解析，run_policy/run_mix 前推荐执行
robot-record --config scripts/config/record_cfg.yaml --dry-run-policy-config

# 3. 连接机器人并回 home
robot-reset --config scripts/config/record_cfg.yaml

# 4. 遥操作采集数据
robot-record --config scripts/config/record_cfg.yaml

# 5. 可视化或回放数据
robot-visualize --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml

# 6. 训练策略
robot-train --config scripts/config/train_cfg.yaml

# 7. 运行 DAgger 轮次闭环
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
```

## 数据集操作

### 采集、回放与可视化

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
```

<p align="center">
  <img src="docs/pic/record.png" alt="Record" width="600">
  <br>
  <b>Figure 1: Record</b>
</p>

<p align="center">
  <img src="docs/pic/visualize.png" alt="Visualization" width="600">
  <br>
  <b>Figure 2: Visualization</b>
</p>

### 追加录制

如果目标 `repo_id` 下已经存在数据集，可以在 `record_cfg.yaml` 中设置 `record.task.resume: true` 并配置 `record.task.resume_dataset`，然后运行：

```bash
robot-record --config scripts/config/record_cfg.yaml
```

### 上传到 Hugging Face

在 `record_cfg.yaml` 中设置 `record.storage.push_to_hub: true`，然后登录：

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN}
huggingface-cli whoami
```

### 合并数据集

```bash
lerobot-edit-dataset \
    --repo_id <merged_repo_id> \
    --operation.type merge \
    --operation.repo_ids "['<repo_id_1>', '<repo_id_2>']"
```

更多数据集处理命令请参考 LeRobot 数据集工具文档。

## 数据集命名与本地元数据

<p align="center">
  <img src="docs/pic/dataset.png" alt="Dataset" width="600">
  <br>
  <b>Figure 3: Dataset</b>
</p>

<p align="center">
  <img src="docs/pic/dataset_info.png" alt="Dataset info" width="600">
  <br>
  <b>Figure 4: Dataset Info</b>
</p>

默认情况下，LeRobot 数据集保存在 `~/.cache/huggingface/lerobot` 下，除非配置了 `dataset_path` 或 `dataset_root`。本地数据集元数据可能包含：

- `dataset_info.txt`：本地数据集记录，例如 `record_id`、`name`、`task`、`date`、`version`、`user_info` 和 `type`。
- `dataset_info_backup`：`tools-check-dataset` 更新 `dataset_info.txt` 时生成的备份。
- 数据集文件夹：实际 LeRobot 数据集内容。

如果手动删除或移动过数据集，可通过以下命令刷新本地元数据：

```bash
tools-check-dataset
```

## 训练、评估与 DAgger

### 训练

修改 `scripts/config/train_cfg.yaml` 后运行：

```bash
robot-train --config scripts/config/train_cfg.yaml
```

### 策略评估

在 `scripts/config/record_cfg.yaml` 中设置 `record.run_mode: run_policy`，并配置 `record.policy.type`、`record.policy.config_path` 和 `record.policy.pretrained_path`，然后运行：

```bash
robot-record --config scripts/config/record_cfg.yaml
```

### DAgger full-episode success 策略

`full_success_episode` 和包含 full episode 的 `hybrid` 导出会使用 `dagger_rounds_cfg.yaml` 中的 `full_episode.success_policy`：

- `explicit`：最严格。每条保存的 run_mix episode 结束后人工确认 success，适合需要显式成功/失败标签的数据。
- `recorded_is_success`：保存即成功。适合“失败、不完整、轨迹质量差的数据用左箭头删除”的当前工作流；保存下来的 episode 会写入 `success=true`。
- `allow_missing_for_smoke`：仅用于调试。缺少 success 字段也允许导出，但会输出 warning，不建议正式训练。

使用 `recorded_is_success` 时，操作者必须严格删除失败、不完整、质量差或不适合作为完整示范的数据。保留下来的 success 标签后续可用于 VLA、Diffusion Policy、failure detector 的数据筛选与评估。

## 录制控制按键

采集时常用按键约定：

- 右箭头：停止当前 episode 并保存。
- 左箭头：丢弃当前 episode。
- Esc：停止整个录制任务。
- Enter：继续下一段遥操作或下一条 episode。
- Ctrl+C：中断并清理未完成数据集。
