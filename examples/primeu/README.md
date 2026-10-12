# PRIMEU：ACT / Diffusion 训练与部署

本实例放在 `dual_arm_teleop/examples/primeu/`。训练使用本仓库依赖的 Le-nero / LeRobot；推理通过现有 PRIMEU ROS runtime 的 `gr00t` 通信后端发送动作。后端名称只表示协议，模型仍是 ACT 或 Diffusion。其他机器人的配置和启动入口不变。

## 1. 环境与配置位置

开发机：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /user/users/szl/envs/teleop_primeu
unset PYTHONPATH LD_PRELOAD
cd /user/users/szl/code/primeu_lerobot/dual_arm_teleop
```

环境为 Python 3.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、torchcodec 0.4.0、FFmpeg 7。Conda 激活会设置环境库路径、独立缓存及 CPU 线程数。任务平台若直接调用解释器，需先设置 `LD_LIBRARY_PATH=/user/users/szl/envs/teleop_primeu/lib`，避免视频解码加载宿主机的旧版 libstdc++。

PRIMEU 代码使用本仓库的 `codex/primeu-lerobot` 分支；LeRobot 依赖使用 [Key-Zzs/Le-nero 的同名分支](https://github.com/Key-Zzs/Le-nero/tree/codex/primeu-lerobot)，已验证提交为 `f4742aaec7f742ba7610e1873723612fd8c5d628`。该版本包含单帧时序图像和 resize padding mask 的兼容性修复，不能仅安装未修复的默认分支后假定配置等价。新机器准备好上述依赖环境后，可在自己的代码父目录安装该源码版本：

```bash
git clone --branch codex/primeu-lerobot https://github.com/Key-Zzs/Le-nero.git Le-nero-primeu
git -C Le-nero-primeu checkout f4742aaec7f742ba7610e1873723612fd8c5d628
python -m pip install --no-deps -e ./Le-nero-primeu
```

本例 YAML 保留开发机实际验证过的路径；换机器时需修改数据、输出、checkpoint 和 ROS workspace 路径。两份开发机仓库均在同名分支上记录改动，更新代码使用 `git pull --ff-only`，不要在训练中途切换实现版本。

| 配置 | 用途 |
|---|---|
| `configs/data_{60,33,27,15,12}d.yaml` | 原始数据位置、转换输出位置 |
| `configs/{act,diffusion}_train.yaml` | 数据、batch、步数、输出、恢复、W&B |
| `configs/policies/{act,diffusion}.yaml` | 模型、动作窗口、优化器和调度参数 |
| `configs/{act,diffusion}_server.yaml` | checkpoint、设备、监听地址 |
| `configs/{act,diffusion}_replay.yaml` | 离线回放数据、样本、服务地址、报告 |
| `configs/robot.yaml` | 机器人端模型契约、外部 ROS workspace 和启动参数 |

训练配置中的相对数据／模型路径相对于该 YAML 解析。数据、模型权重、日志和凭据不要放进 Git。

## 2. 数据准备和动作维度

先修改对应的数据 YAML，再执行；输出必须是源目录之外的新目录，工具不覆盖已有目录：

```bash
python -m examples.primeu.prepare_data --config examples/primeu/configs/data_60d.yaml
# 换成 data_33d.yaml、data_27d.yaml、data_15d.yaml 或 data_12d.yaml 处理对应版本。
```

转换由 LeRobot v2.1 到 v3。原始数值标签不变；episode 编号重新连续排列。视频按字节复制并核对 SHA256，不重新编码。部分降维目录的视频软链接仍指向旧挂载路径，因此对应 YAML 显式设置 `video_source_root`；转换会逐帧比较源状态、时间戳和索引，确认属于同一轨迹后才使用该目录的视频，不修改原软链接。原记录时间与固定帧率视频存在漂移，因此副本 `timestamp` 按 `frame_index / fps` 对齐视频，原值逐值保留在 `source_timestamp`。转换报告位于输出的 `meta/primeu_conversion_report.json`；失败的 `.partial` 目录保留供排查，不当作完成数据使用。

所有版本的 **state 都是 60D**，顺序为左臂 7、左手 20、右臂 7、右手 20、头 3、腰 3。模型动作维度由经过校验的 metadata 确定：

| action 维度 | 模型预测内容 | 推理时保持不动的关节 |
|---:|---|---|
| 60 | 左臂 7、左手 20、右臂 7、右手 20、头 3、腰 3 | 无 |
| 33 | 左臂 7、左手压缩 8、右臂 7、右手压缩 5、头 3、腰 3 | 无 |
| 27 | 左臂 7、左手压缩 8、右臂 7、右手压缩 5 | 头、腰 |
| 15 | 左臂 7、左手压缩 8 | 右臂、右手、头、腰 |
| 12 | 右臂 7、右手压缩 5 | 左臂、左手、头、腰 |

手部压缩值通过该数据集的 Chebyshev basis 恢复为 20 个关节目标；不重新拟合、不换用其他数据集的 basis。契约、basis 及哈希随 checkpoint 保存。

**ACT / Diffusion 直接学习 Parquet 中存储的绝对目标**，不做 `action - state`，推理也不再次加 state。原数据布局中的 GR00T relative 标注描述其 processor 行为，不表示这里的原始标签已经变成相对量。

## 3. 训练、归一化和恢复

先校验配置，再训练：

```bash
python -m examples.primeu.train --config examples/primeu/configs/act_train.yaml --check
python -m examples.primeu.train --config examples/primeu/configs/act_train.yaml

# Diffusion 使用另一份配置：
python -m examples.primeu.train --config examples/primeu/configs/diffusion_train.yaml
```

默认都是 **单卡、batch 2、20 个 optimizer 更新步**，用于连通性验证，不是完成任务的策略。输出分别位于：

```text
/user/users/szl/runs/primeu_native_20261012/act_60d_smoke
/user/users/szl/runs/primeu_native_20261012/diffusion_60d_smoke
```

再次从头运行需换新 `train.output_dir`。正式训练在 YAML 调整 `steps`、`batch_size`、`num_workers`、`save_freq`、`log_freq` 和 GPU 配置；Diffusion 的 `scheduler_warmup_steps` 当前为短测用的 2，也需按正式训练设置。`batch_size` 是每进程 batch；当前示例只验证单卡，不提供未经验证的多卡吞吐承诺。

切换降维数据时修改 `train.dataset.root`、`repo_id`、`job_name` 和 `output_dir`；动作输出维度从数据自动构建，不在模型代码硬编码。一次运行对应一个数据契约，不能把不同手部 basis 的数据混在一起。当前 Step1 60D 的任务文本写“右手”，33D/27D 写“左手”；转换报告保留并注明这一原始标注差异，不自动纠正。ACT/Diffusion 本实例不编码任务文本，正式多任务扩展前仍需核查标注。

默认短测配置的训练预处理与部署共用保存的原生 processor（第 7 节的全尺寸配置跳过 resize）：

```text
三路 RGB + 60D 物理 state + 原始 action 标签
  → 图像双线性 resize 到 256×256（antialias，非补边）
  → 原生 normalizer
  → ACT / Diffusion
  → 原生 action unnormalizer
  → 原始数据单位的 action
```

- ACT：当前单帧三路图像和 state，学习 60 帧动作；state/action 使用 mean/std。服务每次返回新预测的 60 帧，机器人实际消费几帧由 ROS runtime 决定。
- Diffusion：**当前单帧 `n_obs_steps=1`**，horizon 8，服务返回 8 帧；state/action 使用 min/max。另有 160×160 确定性中心裁剪，可在 policy YAML 修改。
- 图像使用 ImageNet mean/std。state/action 统计按原始有效帧计算，保存于转换副本并随 checkpoint processor 保存。
- 当前 60D 短测使用全部 100 个 episode、49,682 帧，没有单独验证集。loss 和回放只能验证链路，不能据此评价泛化；正式划分验证集后应仅用训练部分重新计算统计。

完整 checkpoint 下的 `pretrained_model/` 用于部署，旁边的 `training_state/` 用于恢复。恢复时保持原 `output_dir`，设置 `train.resume: true`，把 `train.steps` 改为累计总步数，再执行同一训练命令。恢复会检查数据契约、basis 和 policy YAML，不允许悄悄换模型配置；它继续原生 optimizer/scheduler，不会自动重启一套学习率。原生恢复不保存精确的数据采样游标，因此不能视为与不中断运行逐位一致。

W&B 默认关闭。需要上传时修改训练 YAML 的 `train.wandb`：

```yaml
wandb:
  enable: true
  mode: online
  project: primeu-lerobot
  entity: shenzhaolong1330-beihang-university
  disable_artifact: true
```

在该环境先 `wandb login`，或由任务平台注入 `WANDB_API_KEY`；不要把 key 写进 YAML 或仓库。

## 4. 启动策略、回放和切换 checkpoint

先将服务 YAML 的 `checkpoint` 指向完整 `pretrained_model` 目录。例如 ACT 短测为：

```text
/user/users/szl/runs/primeu_native_20261012/act_60d_smoke/checkpoints/000020/pretrained_model
```

终端一启动服务；服务不会加载机器人 SDK、reset 机器人或发送动作：

```bash
python -m examples.primeu.server --config examples/primeu/configs/act_server.yaml
```

终端二在同一环境进行回放：

```bash
python -m examples.primeu.replay --config examples/primeu/configs/act_replay.yaml
```

回放默认 `mode: compare`，固定随机种子比较本进程直接加载与 ZMQ 服务输出。可在 YAML 改为 `direct`（不需要服务）或 `remote`（不在客户端加载模型）。回放按 episode 切换重置保持姿态，记录输出形状、数值差异和耗时；它不连接机器人。

换 Diffusion 时将上述 YAML 文件名换为 `diffusion_server.yaml`、`diffusion_replay.yaml`。切换任意 checkpoint 后重启服务，并同步更新回放 checkpoint；机器人端 `robot.yaml` 的 `expected_model_family`、`expected_model_action_dim`、`expected_horizon`、`expected_basis_sha256` 必须与新 checkpoint 一致。ACT 默认 horizon 60，当前 Diffusion 为 8。60D 的 basis 为 `{}`；降维版本将 `pretrained_model/primeu/contract.json` 的 `basis_sha256` 映射原样填入，保留 `left`／`right` 键及 SHA256，不能留空。

复制部署模型时必须复制整个 `pretrained_model/`，包括模型权重、processor 配置及统计、`primeu/` 契约与 basis，不只复制 safetensors。旧的两帧 Diffusion checkpoint 保留，但不支持这个单帧服务；旧 checkpoint 缺少 PRIMEU 导出资产时会明确拒绝加载。

## 5. 机器人端如何开始联动

现有外部运行环境位于机器人电脑 `/home/owen/prime_ros2`，需已有 `primeu_vla` ROS 包。这里只复用其 runtime，不迁移数采和底层驱动。机器人端可以只复制本目录的 `client.py` 与 `configs/robot.yaml`，Python 依赖为 `pyyaml`、`pyzmq`、`msgpack`，不需要安装 LeRobot 或 GPU 模型环境。

远端服务默认监听 `127.0.0.1:5559`。在机器人电脑建立 SSH 转发，并保持终端打开：

```bash
ssh -N -L 5559:127.0.0.1:5559 -p 30295 dev@10.30.13.100
```

修改机器人端 `robot.yaml`：服务地址指向 `127.0.0.1:5559`，模型契约与服务一致；`runtime.workspace`、`setup_file`、`node` 和任务文本对应现有机器人环境。本机 GPU 服务则直接用回环地址，不需要 SSH 转发。

以下命令在机器人电脑运行，`launch` 占一个终端；`start/stop` 在另一终端操作：

```bash
# 只打印将执行的命令，不访问网络或 ROS：
python3 client.py launch --config configs/robot.yaml --dry-run

# 检查模型类型、关节／相机顺序、动作窗口、basis 和绝对目标语义：
python3 client.py check --config configs/robot.yaml

# 启动已有 ROS runtime；不替代显式 start：
python3 client.py launch --config configs/robot.yaml

# 以下 start 会使现有 runtime 开始向机器人发送策略动作：
python3 client.py start --config configs/robot.yaml
python3 client.py status --config configs/robot.yaml
python3 client.py stop --config configs/robot.yaml
```

服务端完成图像处理、归一化、预测、反归一化和手部解码，将完整 60D 绝对关节目标按六组返回。原始 RGB 和物理 state 由客户端/runtime 发送，不在机器人端再做一套归一化或 resize。机器人端的限幅、频率、消费 chunk 长度及 home 行为继续由既有 runtime 管理；服务返回 60/8 帧不代表一次必须执行全部帧。

**未预测关节保持本次策略启动后的第一份有效观测姿态。** 后续反馈变化不会改变这组保持目标，不使用数据集参考姿态。每次 `start` 先校验服务并成功 reset 服务会话，才调用 ROS start；reset 清除模型缓存和姿态记录，不执行 home。`stop` 先停止 ROS loop，再尽力清除服务会话，模型掉线不阻止停止。服务或 ROS 调用超时不自动重试动作命令。一个服务只供一个机器人会话使用，不要在运行中同时回放或 reset。

## 6. 改动与验收记录

本实例增加 PRIMEU 数据契约与手部解码、原生训练薄封装、保存时的契约／basis 导出，以及现有协议的服务／客户端。共享训练入口只增加可选图像 resize 和 checkpoint 资产回调；Le-nero 图像 resize 支持时序 tensor 并跳过 padding mask；视频 loader 对显式单帧时间查询保留 T=1，修复单帧 Diffusion 的输入形状。其他机器人的调用不启用这些 PRIMEU 配置。

旧 `scripts/config/primeu/` 配置和 `scripts.tools.prepare_primeu_lerobot`、`scripts.tools.check_primeu_checkpoint` 入口继续保留；旧 Diffusion 两帧试验不会被本次单帧配置覆盖。

本轮记录目录为 `/user/users/szl/runs/primeu_native_20261012`。实际完成的转换、短训、保存／加载与回放结果以该目录报告和交付说明为准；不把旧试验的测试结果当成本轮通过。当前没有在真实 PRIMEU ROS runtime 上发送动作，实机联动与任务成功率仍需另行验证。

### 2026-10-12 实测结果

- 60 项 PRIMEU 专项测试、11 项已有 DAgger 回归测试通过；共享 policy 配置自检通过，独立环境 `pip check` 通过。
- 60D ACT、单帧 Diffusion 各完成 20 步、batch 2，已重新加载真实 checkpoint。ACT 另从第 20 步恢复到第 21 步，153 个 optimizer step 计数由 20 连续变为 21，权重发生更新。
- 33D、27D、15D、12D 分别完成 ACT 和 Diffusion 的一次真实优化更新、保存和重新加载，共 8 组；这只是接口验证，不是有效的任务训练。
- 十组 checkpoint 的归一化往返最大误差不超过 `8.2e-8`；训练／服务预处理一致，固定 seed 重置推理一致，所有 episode 首尾动作窗口不跨段，输出均为有限的完整 60D。
- 60D 和 15D 的 ACT／Diffusion 均通过直接调用与 ZMQ 的真实回放比较，最大动作差为 0；15D 回放覆盖 episode 切换及保持姿态重置。机器人客户端的模型／basis 握手通过。
- 当前机器为单张 RTX PRO 6000 96GB。回放预热后，ACT 服务推理约 10–13ms；Diffusion 使用 100 次 DDPM 去噪，约 407–410ms。它们是离线服务延迟，不代表实机闭环频率。

| 数据动作维度 | episode | 帧数 | 逐字节校验的视频数 |
|---:|---:|---:|---:|
| 60 | 100 | 49,682 | 300（已有转换再次用于训练／回放） |
| 33 | 100 | 49,682 | 300 |
| 27 | 100 | 49,682 | 300 |
| 15 | 94 | 39,394 | 282 |
| 12 | 50 | 13,681 | 150 |

完整结果见运行目录的 `acceptance.json`、`*_verify.json`、`*_replay.json`、`resume_verify.json` 和日志。复查指定 checkpoint：

```bash
python -m examples.primeu.verify \
  --config examples/primeu/configs/act_train.yaml \
  --checkpoint /user/users/szl/runs/primeu_native_20261012/act_60d_smoke/checkpoints/000020/pretrained_model \
  --report /user/users/szl/runs/primeu_native_20261012/act_recheck.json
```

原仓库的 Franka 实机 CLI 测试包含本机绝对路径，未纳入这次开发机离线回归，也没有通过它连接机器人。手部 basis 为有损拟合；解码器与原算法一致，并不表示能无损恢复最初 20D 手部命令。

## 7. Step5：15D ACT 原始分辨率、batch 32、5 万步

专用配置 `configs/act_step5_15d_fullres_50k.yaml` 使用原数据 `/user/datasets/primeu_data/demo_step5_release_liquid_1010_left8_15d` 的独立转换副本，共 75 个 episode、29,592 帧。`image_resize: null` 保留头部 **720×1280**、左右腕部各 **480×640**（均为高×宽），不缩放、不裁剪，仍使用 ImageNet 图像归一化。ACT 分别提取各相机特征后拼接，不要求三路分辨率一致。

使用现有原生 ACT：ImageNet ResNet18、60 帧 chunk、VAE/KL 权重 10、AdamW 学习率 `1e-5`、weight decay `1e-4`。输入 state 为 60D；动作是左臂 7D＋左手压缩 8D，使用这批数据自己的 `left` basis（SHA256：`db4033fe056dae0c8543bd0331435fe73be654957f66359be02cd66b8cca7577`）。推理解码左手，并保持首次有效观测中的右臂、右手、头和腰姿态。

配置为单张 RTX PRO 6000、FP32、batch **32**、**50,000** 个 optimizer 更新步，每 **10,000** 步保存，每 100 步记录日志，8 个数据 worker（torchcodec CPU 解码，每个 decoder 1 线程）。W&B 在该配置中关闭，日志保存在挂载盘。20 步全尺寸预检已通过：实测最大 allocated 约 43.65 GiB、reserved 约 47.31 GiB，纯模型更新约 0.49 秒/步；这不包含完整数据等待和保存开销，不代表 5 万步已经完成。

在第 1 节的环境和仓库目录中启动一次新训练：

```bash
python -m examples.primeu.train --config examples/primeu/configs/act_step5_15d_fullres_50k.yaml --check
mkdir -p /user/users/szl/runs/primeu_act
set -o pipefail
python -u -m examples.primeu.train \
  --config examples/primeu/configs/act_step5_15d_fullres_50k.yaml \
  2>&1 | tee /user/users/szl/runs/primeu_act/step5_left8_15d_fullres_b32_50k.log
```

不要与已运行的训练重复启动；恢复沿用第 3 节的 `resume: true` 流程。训练输出为 `/user/users/szl/runs/primeu_act/step5_left8_15d_fullres_b32_50k`，checkpoint 目录依次为 `010000` 至 `050000`。**以下配置预先指向最终第 50,000 步，须等该 checkpoint 实际保存后使用**：

```bash
# 一个终端启动服务，另一个终端回放；两者都不执行机器人动作。
python -m examples.primeu.server --config examples/primeu/configs/act_step5_15d_fullres_server.yaml
python -m examples.primeu.replay --config examples/primeu/configs/act_step5_15d_fullres_replay.yaml
```

机器人端使用 `configs/act_step5_15d_fullres_robot.yaml`，按第 5 节 `check/launch/start/stop` 操作，将命令中的 `configs/robot.yaml` 替换为这份文件。客户端通过 `expected_image_sizes` 校验三路尺寸；不要同时设置旧的 `expected_image_size`。runtime 必须发送训练对应的原始 RGB 分辨率，服务会拒绝不匹配的输入。切换中间 checkpoint 时修改 server/replay YAML 的 `checkpoint` 并重启服务；保持这份机器人配置中的维度、basis、60 帧动作窗口和图像尺寸与新模型一致。

本轮全尺寸验收记录：`/user/users/szl/runs/primeu_step5_fullres_validation_20261012/acceptance.json`。74 项 PRIMEU 测试通过；完整 batch32 的 20 步及 8-worker 40 步短训、保存重载、归一化往返与直接/ZMQ回放通过，回放动作差为 0。正式运行已于 2026-10-12 18:11（北京时间）后台启动，进程信息保存于输出同级 `.launch.json`；第 100 步日志平均模型更新 0.489 秒、数据等待 0.737 秒，初估 5 万步约 17–20 小时。该估计会随存储负载变化，短训和离线回放不代表任务成功率。正式训练使用全部 75 个 episode，未另划验证集。

查看正式训练进度：

```bash
tail -f /user/users/szl/runs/primeu_act/step5_left8_15d_fullres_b32_50k.log
```
