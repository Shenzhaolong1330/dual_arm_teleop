# Dual Arm Teleoperation

This project provides dual-arm teleoperation, data acquisition, replay/visualization, policy training, and a round-based DAgger loop. It uses the `lerobot` package from `Key-Zzs/Le-nero` for datasets, policies, and DAgger-related behavior, plus this repository's own robot and teleoperator adapters.

<p align="center">
  <img src="docs/images/supported_robots.jpg" alt="Supported robot systems" width="900">
</p>

The current registry supports these robot adapters: Dobot dual arm, AgileX Nero dual arm, ARX dual arm, Franka single/dual arm, and Flexiv Rizon4s dual arm. The exact `robot_type` values are `dobot_dual_arm`, `nero_dual_arm`, `arx_dual_arm`, `franka`, `franka_dual_arm`, and `flexiv_dual_arm`.

## Repository Setup and Environment

For a first-time clone:

```bash
git clone https://github.com/Shenzhaolong1330/dual_arm_teleop.git
cd dual_arm_teleop
```

To update this repository during daily development:

```bash
cd dual_arm_teleop
git pull --ff-only
```

To switch branches:

```bash
git fetch origin
git switch <branch_name>
```

Create the Python environment, install Le-nero first, then install this package:

```bash
conda create -n dual_arm_teleop python=3.10 -y
conda activate dual_arm_teleop
python -m pip install --upgrade pip

# Install the Le-nero repository first. This provides the lerobot package used by
# the policy and DAgger code in this project.
git clone https://github.com/Key-Zzs/Le-nero.git
cd Le-nero
pip install -e .

cd /path/to/dual_arm_teleop
pip install -e .
```

If `Le-nero` is already cloned locally, use that checkout instead of cloning it again. Install `Key-Zzs/Le-nero` rather than upstream vanilla LeRobot for the policy and DAgger stack expected by this project.

Oculus Reader is not vendored by this repository, so it must be cloned separately into the required location:

```bash
cd /path/to/dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
cd oculus_reader
pip install -e .
```

If the directory already exists, update it and reinstall:

```bash
cd /path/to/dual_arm_teleop/teleoperators/oculus_teleoperator/oculus/oculus_reader
git pull --ff-only
pip install -e .
```

Oculus connectivity also requires ADB:

```bash
sudo apt install android-tools-adb
adb devices
```

On the first USB connection, allow USB debugging in the headset. For wireless connection, first use `adb shell ip route` to find the headset IP, then run`adb tcpip 5555` and `adb connect <Oculus_IP>:5555`.

## Core Modules and Runtime Flow

The runtime flow can be understood as:

```text
scripts/config/*.yaml
        |
        v
scripts/core/*.py command entry points
        |
        +--> robots create the real robot interface
        +--> teleoperators create Oculus teleoperation input
        +--> Le-nero's lerobot.policies create policy models
        |
        v
Le-nero-provided lerobot dataset / train / replay / visualize
```

### Policy Layer

Policy and DAgger implementations come from the installed `Key-Zzs/Le-nero` repository, which provides the `lerobot` Python package used here:

```text
lerobot.policies
```

That package keeps the policy abstractions and implementations such as `act`, `diffusion`, `smolvla`, and `pi0`, plus project-specific behavior used by the training and DAgger flow. The dual-arm teleoperation scripts mainly use `lerobot.policies.factory.make_policy` and `make_pre_post_processors` to create policy objects and their pre/post-processors.

The most commonly used policy config files in this package are:

- `scripts/config/policies/act_train_config.yaml`: ACT training config.
- `scripts/config/policies/act_reason_config.yaml`: ACT inference/deployment config.
- `scripts/config/policies/diffusion_train_config.yaml`: Diffusion Policy training config.
- `scripts/config/policies/diffusion_reason_config.yaml`: Diffusion Policy inference/deployment config.

`scripts/core/policy_config_utils.py` resolves policy config paths from `record_cfg.yaml`, `train_cfg.yaml`, or `dagger_rounds_cfg.yaml`. Relative paths are resolved first against this package root, and absolute paths are also supported.

### Robot Communication Interface Layer

Robot interfaces are located in:

```text
robots
```

`robots/__init__.py` is the robot registry. The currently registered robot types include:

- `franka`
- `dobot_dual_arm`
- `nero_dual_arm`
- `arx_dual_arm`
- `franka_dual_arm`
- `flexiv_dual_arm`

Scripts do not instantiate a concrete robot class directly. Instead, they use the configured `robot_type` to call:

```python
create_robot_config(robot_type, **robot_cfg)
create_robot(robot_type, robot_config)
```

This package uses its own `robots` and `teleoperators` registries in the `robot-*` commands. It no longer ships top-level LeRobot plugin shim packages such as `lerobot_robot_*` or `lerobot_teleoperator_*`, so upstream LeRobot CLI auto-discovery aliases from `register_third_party_devices()` are not provided here.

Each concrete robot class implements the robot interface expected by LeRobot, such as `connect()`, `reset()`, `send_action()`, camera initialization, observation fields, and action fields. For example, `robots/dual_agilex_nero/nero_dual_arm.py` connects to the dual-arm zerorpc service through `NeroDualArmClient`, then organizes dual-arm end-effector poses, joint states, gripper commands, and RealSense cameras into a LeRobot-compatible data structure. `robots/dual_flexiv_rizon4s/flexiv_dual_arm.py` supports a dual Flexiv Rizon4s setup through Flexiv RDK, using the left/right robot serial numbers, Flexiv gripper/tool names, Cartesian limits, home joints, and RealSense cameras from `scripts/config/robots/flexiv_config.yaml`.

Hardware-specific parameters should usually live in config files instead of runtime scripts:

```text
scripts/config/robots
```

The default robot config mapping is:

- `dobot_dual_arm`: `scripts/config/robots/dobot_config.yaml`
- `nero_dual_arm`: `scripts/config/robots/nero_cofig.yaml`
- `arx_dual_arm`: `scripts/config/robots/arx_config.yaml`
- `franka`, `franka_dual_arm`: `scripts/config/robots/franka_config.yaml`
- `flexiv_dual_arm`: `scripts/config/robots/flexiv_config.yaml`

Each file defines the hardware connection details, gripper parameters, Oculus mapping, and camera serial numbers for that hardware profile. `run_record.py`, `run_replay.py`, and `reset_robot.py` automatically load the corresponding robot config based on `record.robot_type`. You can also explicitly set `das_config_path` in `record_cfg.yaml`.

### Tool Scripts, Data Collection, and Policy Configs

Main scripts are located in:

```text
scripts
```

Common directories:

- `scripts/core`: command entry implementations for record, replay, visualize, reset, train, and DAgger.
- `scripts/config`: main workflow configs, including `record_cfg.yaml`, `train_cfg.yaml`, `dagger_rounds_cfg.yaml`, and dataset tool configs.
- `scripts/config/policies`: policy hyperparameter configs, split into train and reason configs.
- `scripts/config/robots`: hardware and teleoperation detail configs.
- `scripts/tools`: dataset checks, RealSense device checks, dataset patching, renaming, and related utilities.

Core config files:

- `record_cfg.yaml`: main config shared by data collection, policy inference, mixed control, replay, and visualization.
- `train_cfg.yaml`: policy training config, including dataset paths, output directory, GPU settings, batch size, training steps, and wandb.
- `dagger_rounds_cfg.yaml`: round-based DAgger controller config that connects collection, export, and next-round training.
- `preprocess_dataset_cfg.yaml`: dataset cleanup and action smoothing config.
- `split_label_dataset_cfg.yaml`: sub-episode splitting, semantic labeling, and VLA manifest config.
- `merge_dataset_cfg.yaml`: merge multiple local LeRobot datasets into one output dataset.
- `*_train_config.yaml`: model structure and training-related hyperparameters used during policy training.
- `*_reason_config.yaml`: model structure, device, and checkpoint parameters used during inference or deployment.

The three `robot-record` modes are controlled by `record.run_mode`:

- `run_record`: pure teleoperation data collection.
- `run_policy`: load a policy checkpoint and let the policy control the robot.
- `run_mix`: policy execution with operator takeover, used for DAgger data collection.

## Core Commands

After installing this package, `setup.py` registers these console commands:

| Command | Purpose | Default config |
| --- | --- | --- |
| `robot-record` | Teleoperation collection, policy execution, or run_mix mixed collection | `scripts/config/record_cfg.yaml` |
| `robot-replay` | Replay a collected episode | `scripts/config/record_cfg.yaml` `replay` section |
| `robot-visualize` | Visualize a dataset episode with Rerun | `scripts/config/record_cfg.yaml` `visualize` section |
| `robot-reset` | Connect to the configured robot and return it home | `scripts/config/record_cfg.yaml` |
| `robot-train` | Train ACT or Diffusion Policy | `scripts/config/train_cfg.yaml` |
| `robot-dagger` | Run round-based DAgger: collect, export, then train the next-round policy | `scripts/config/dagger_rounds_cfg.yaml` |
| `robot-dagger-export` | Export DAgger training data from raw run_mix logs | `scripts/config/dagger_rounds_cfg.yaml` `dagger_export` section |
| `tools-check-dataset` | Clean stale entries from local `dataset_info.txt` | `scripts/config/record_cfg.yaml` |
| `tools-check-dagger-dataset` | Audit an exported DAgger dataset before training | `scripts/config/dagger_rounds_cfg.yaml`, `scripts/config/train_cfg.yaml` |
| `tools-check-rs` | Show RealSense device serial numbers | None |
| `tools-preprocess-dataset` | Rewrite a LeRobot dataset after trimming static spans and optionally smoothing actions | `scripts/config/preprocess_dataset_cfg.yaml` |
| `tools-split-label-dataset` | Split long episodes into labeled sub-episodes and optionally write a new dataset | `scripts/config/split_label_dataset_cfg.yaml` |
| `tools-merge-datasets` | Merge several local LeRobot datasets with the same schema into one materialized dataset | `scripts/config/merge_dataset_cfg.yaml` |
| `robot-help` | Print the command summary | None |

All core commands and config-driven tools support an explicit config file. It is recommended to pass the path during debugging:

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
robot-reset --config scripts/config/record_cfg.yaml
robot-train --config scripts/config/train_cfg.yaml
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
robot-dagger-export --config scripts/config/dagger_rounds_cfg.yaml
tools-preprocess-dataset --config scripts/config/preprocess_dataset_cfg.yaml --dry-run
tools-split-label-dataset --config scripts/config/split_label_dataset_cfg.yaml --dry-run
tools-merge-datasets --config scripts/config/merge_dataset_cfg.yaml --dry-run
```

## Tool Commands and Maintenance Scripts

The registered tool commands are:

| Command | Typical usage | Notes |
| --- | --- | --- |
| `tools-check-rs` | `tools-check-rs` | Requires `pyrealsense2`; prints connected RealSense names and serial numbers. |
| `tools-check-dataset` | `tools-check-dataset --config scripts/config/record_cfg.yaml` | Removes dataset records whose folders no longer exist. Use `--lerobot-home` to override the dataset cache root. |
| `tools-check-dagger-dataset` | `tools-check-dagger-dataset --config scripts/config/dagger_rounds_cfg.yaml --train-config scripts/config/train_cfg.yaml` | Audits exported DAgger data, schema, action values, source labels, and train compatibility. |
| `tools-preprocess-dataset` | `tools-preprocess-dataset --config scripts/config/preprocess_dataset_cfg.yaml --dry-run` | Supports `--overwrite` and `--max-episodes`; writes a cleaned dataset through LeRobot APIs. |
| `tools-split-label-dataset` | `tools-split-label-dataset --config scripts/config/split_label_dataset_cfg.yaml --dry-run` | Supports `--label-only`, `--write-dataset`, `--overwrite`, `--max-episodes`, and `--resume-cache`. |
| `tools-merge-datasets` | `tools-merge-datasets --config scripts/config/merge_dataset_cfg.yaml --dry-run` | Merges datasets listed in `source.datasets` under `source.parent_dir`; supports `--overwrite` and `--max-episodes`. |

Additional maintenance scripts are not installed as console commands, but can be run from this package root:

| Script | Typical usage | Purpose |
| --- | --- | --- |
| `scripts/tools/rename_lerobot_task.py` | `python scripts/tools/rename_lerobot_task.py --dataset-root <dataset> --new-task "<task>" --dry-run` | Rename a task prompt in `meta/tasks.parquet`, episode metadata, and optional `dataset_info.txt`. |
| `scripts/tools/merge_lerobot_tasks.py` | `python scripts/tools/merge_lerobot_tasks.py --dataset-root <dataset> --source-task "<old>" --target-task "<keep>" --dry-run` | Merge one task label into another inside a single dataset. |
| `scripts/tools/patch_lerobot_dataset_metadata.py` | `python scripts/tools/patch_lerobot_dataset_metadata.py --dataset-root <dataset> --check-only` | Inspect or patch explicit dataset metadata fields such as description and repo id. |
| `scripts/tools/run_dagger_export_train_experiment.py` | `python scripts/tools/run_dagger_export_train_experiment.py --raw-dataset <raw> --base-dataset <seed_or_agg> --initial-checkpoint <ckpt> --output-root <out> --dry-run` | Run export/train experiments for different DAgger export modes. |
| `scripts/core/check_dagger_sampling.py` | `python scripts/core/check_dagger_sampling.py --dataset <aggregated_dataset>` | Inspect source-aware DAgger sampler weights. |
| `scripts/tools/check_robotiq_ports.sh` | `bash scripts/tools/check_robotiq_ports.sh --verbose` | Locate Robotiq-like serial devices under `/dev/serial/by-id`; also supports `--json`. |
| `scripts/tools/map_gripper.sh` | `sudo bash scripts/tools/map_gripper.sh dobot_left_gripper` | Create a udev symlink for one connected USB gripper. |
| `rm_tmp.sh` | `bash rm_tmp.sh` | Remove Python caches and build artifacts from this package. |

Run dataset tools inside the same Le-nero/lerobot environment used for recording and training. Some tools import `lerobot`, `numpy`, `torch`, `pyarrow`, `pandas`, `pyrealsense2`, or hardware SDKs at startup.

## Configuration Checklist

Before data collection, usually edit `scripts/config/record_cfg.yaml`:

- `record.repo_id`: dataset name. Recommended format: `<robot_task<num>_step<num>/<description>`, for example `nero_task3_step1/2mL_empty_right`.
- `record.robot_type`: choose a robot type such as `nero_dual_arm`, `franka_dual_arm`, or `flexiv_dual_arm`.
- `record.run_mode`: choose `run_record`, `run_policy`, or `run_mix`.
- `record.save_depth_sidecar`, `record.save_ir_sidecar`, `record.save_rgbd_timestamps`: select native RealSense depth/IR and the scalar fields required to join them to the LeRobot timeline.
- `record.rgbd_sidecar_storage`: `zarr` is the default for new Flexiv recordings; `parquet` keeps the legacy array-in-Parquet path. Zarr mode requires `save_rgbd_timestamps: true` and never double-writes the large arrays.
- `record.rgbd_sidecar_zarr`: configures the relative store path, frame chunk size, bounded queue capacity, and compressor. The measured default is 8 frames with Blosc/LZ4 level 1 and bitshuffle.
- `record.rgb_camera_name_mode`: `rgb` records RGB video as `observation.images.head_rgb`, `observation.images.left_wrist_rgb`, and `observation.images.right_wrist_rgb`; use `legacy_image` only when an old checkpoint expects the previous `*_image` keys.
- `record.policy.type`, `config_path`, `pretrained_path`: required only for `run_policy` or `run_mix`.
- `record.task`: task description, number of episodes, resume behavior, success labels, and `episode_control_mode` (`keyboard` or keyboard-free `oculus`).
- `record.time`: max episode duration and metadata save period. `reset_time_sec` remains a legacy compatibility key; automatic inter-episode `robot.reset()` is not a timed manual-reset loop.
- `replay`, `visualize`: default dataset and episode used by replay and visualization.

Hardware parameters are usually edited in `scripts/config/robots/*.yaml`:

- `teleop.oculus_config.ip`: Oculus Quest IP.
- `teleop.oculus_config.*_pose_scaler` and `*_channel_signs`: mapping from left/right controllers to robot actions.
- `robot.robot_ip`, `robot.robot_port`: robot service address for RPC-backed robots.
- For `flexiv_dual_arm`, set `robot.left_robot_sn`, `robot.right_robot_sn`, Flexiv gripper/tool names, home joints, and Cartesian safety limits in `scripts/config/robots/flexiv_config.yaml`. Flexiv hardware control also requires `flexivrdk` and `spdlog` in the active Python environment.
- `robot.use_gripper` and gripper parameters: enable grippers, close/open thresholds, max opening width, and force.
- `cameras.*_serial`, `width`, `height`: RealSense serial numbers and resolution.

### Raw RGB-D Zarr v2 sidecar

New Flexiv recordings keep RGB in the normal LeRobot MP4 video fields and stream native depth/IR into a separate Zarr v2 acquisition store:

```text
<dataset_root>/
├── data/...                         # state, action, indices, and scalar sync only
├── videos/...                       # RGB MP4, unchanged
├── meta/info.json
├── meta/realsense_calibration.json
├── meta/rgbd_sidecar.json           # authoritative commit ledger
└── sidecars/realsense.zarr
```

The Zarr arrays are `/data/{head,left_wrist,right_wrist}/{depth,left_ir,right_ir,rgbd_timestamp,rgbd_reused}`, plus `/meta/{index,episode_index,frame_index,global_frame_index,robot_timestamp,episode_ends}`. Depth remains native `uint16` RealSense units; left/right IR remains lossless `uint8`. The nine 2-D depth/IR arrays are absent from `meta/info.json`, the LeRobot episode buffer, episode statistics, and main Parquet. Scalar timestamps, reused flags, and all existing state/action/RGB fields remain in Parquet.

`meta/rgbd_sidecar.json` is authoritative. During an active episode, physical arrays may contain an uncommitted tail, but readers expose only its committed prefix. Saving an episode drains the bounded writer queue, durably seals LeRobot Parquet/meta files, verifies scalar joins, appends `episode_ends`, and atomically advances the manifest. Rerecord clears both the LeRobot buffer and the uncommitted Zarr tail. Resume refuses short arrays, calibration/hash/schema conflicts, or any mismatch between manifest counts, `info.json`, and Parquet; it only truncates tails beyond an already recorded manifest prefix. Ctrl+C first leaves the dataset `incomplete` and then asks whether to delete the entire dataset folder; deletion occurs only after an explicit `y`. Writer/schema/queue failures remain preserved as `incomplete` or `corrupt` data for diagnosis.

The default configuration is:

```yaml
save_depth_sidecar: true
save_ir_sidecar: true
save_rgbd_timestamps: true
rgbd_sidecar_storage: zarr       # zarr | parquet
rgbd_sidecar_zarr:
  relative_path: sidecars/realsense.zarr
  chunk_frames: 8
  queue_capacity_frames: 64
  compressor: {codec: blosc, cname: lz4, clevel: 1, shuffle: bitshuffle}
```

This store is a raw acquisition sidecar, not a DP3 replay-buffer Zarr. DP3 conversion remains a later derived export that builds policy fields such as point clouds. Raw depth/IR remains available for preview and downstream stereo work.

The legacy `rgbd_sidecar_storage: parquet` path still writes `sidecar.head_depth`, `sidecar.head_left_ir`, and related arrays into Parquet. Tools auto-detect the format: a present manifest must validate as Zarr and is never silently bypassed; only datasets without a manifest use the legacy path.

Flexiv tracks the last RealSense SDK `frame_index` consumed from each camera independently. A repeated index sets that camera's `*_rgbd_reused=true` even when the reader thread did not raise; a read failure that reuses the last valid frameset is also marked true. The tracking state is reset on camera connect, robot reset, stop, and release. No additional frame-index dataset field is introduced, so the existing feature schema remains compatible.

After every RGB-D recording, run the full sidecar checker against the exact dataset root. It validates the manifest, calibration SHA-256 and stream shapes, every Zarr array, committed counts and episode boundaries, chunked Parquet/Zarr join keys, timestamp order, reused flags, SHA-256 uniqueness, exact adjacent equality, unmarked duplicates, and longest frozen runs. Legacy Parquet recordings retain the same content checks. The default maximum identical run is four frames:

```bash
python scripts/check_rgbd_sidecar_dataset.py --root /absolute/path/to/lerobot/dataset

# One RGB/depth/IR preview frame
python scripts/tools/export_rgbd_sidecar_preview.py \
  --root /absolute/path/to/lerobot/dataset --episode 0 --frame-index 0 --camera head

# Lossless raw IR pair; this exporter does not rectify images
python scripts/tools/export_ffs_stereo_pair.py \
  --root /absolute/path/to/lerobot/dataset --episode 0 --frame-index 0 --camera head

# Hardware-free storage benchmark
python scripts/tools/benchmark_rgbd_zarr_sidecar.py --frames 150
```

Record a new smoke dataset first and require the checker to exit 0 before any DP3 conversion or formal collection. Re-encoding cannot repair an older dataset whose borrowed-buffer depth/IR frames are already frozen; keep such data read-only for diagnosis and reacquire it.

Before training, usually edit `scripts/config/train_cfg.yaml`:

- `train.dataset.repo_id` and `train.dataset.root`: training dataset.
- `train.policy.type` and `train.policy.config_path`: policy type and training config.
- `train.output_dir`, `job_name`: model and log output location.
- `train.training`: visible GPUs, memory cap, TF32, and other training device settings.
- `train.steps`, `batch_size`, `num_workers`, `save_freq`: training scale.
- `train.wandb`: wandb project and mode.

Before DAgger, usually edit `scripts/config/dagger_rounds_cfg.yaml`:

- `dagger_rounds.seed_repo_id` or `seed_dataset_path`: seed dataset used by round 0.
- `dagger_rounds.initial_pretrained_path`: optional initial checkpoint, if one already exists.
- `dagger_rounds.policy`: policy type used in rounds, plus train/reason config paths.
- `dagger_rounds.episodes_per_round`, `num_rounds`, `round_schedule`: collection count per round, number of rounds, and training step schedule.
- `dagger_rounds.output_root`: DAgger round output directory.
- `dagger_rounds.record_cfg_path`, `train_cfg_path`: base configs dynamically modified and called by the controller.
- `dagger_rounds.policy_backend.export`: rules for exporting run_mix logs into training data.

## Oculus Setup and Controls

### Install and Test Oculus Reader

```bash
cd /path/to/dual_arm_teleop
cd teleoperators/oculus_teleoperator/oculus/oculus_reader
pip install -e .
python oculus_reader/reader.py
```

### Install the Oculus Reader APK

```bash
cd /path/to/dual_arm_teleop
cd teleoperators/oculus_teleoperator/oculus/oculus_reader/oculus_reader/APK
adb install -r teleop-debug.apk
```

After installation, the app appears in the Oculus Quest library under **Unknown Sources**.

### Configure Oculus Teleoperation

Set the Oculus IP and mapping in the selected robot config, for example `scripts/config/robots/nero_cofig.yaml` or `scripts/config/robots/flexiv_config.yaml`:

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

### Controller Controls

| Control | Function |
| --- | --- |
| Left grip `LG` | Hold to enable left-arm end-effector motion. In `run_mix`, this starts or continues expert override for the left arm. |
| Right grip `RG` | Hold to enable right-arm end-effector motion. In `run_mix`, this starts or continues expert override for the right arm. |
| Left trigger `LTr` | Control the left gripper. Pressing closes the gripper; releasing opens it. |
| Right trigger `RTr` | Control the right gripper. Pressing closes the gripper; releasing opens it. |
| `Y` button | In `run_mix`, release the left gripper channel back to policy control. |
| `B` button | In `run_mix`, release the right gripper channel back to policy control. |
| `A` button | Request robot reset, if supported by the active teleoperator/robot implementation. |
| Controller pose | Controls the corresponding end-effector delta pose while the corresponding grip is held. |

If `mirror_teleop` is enabled, the left/right controller assignment is swapped and pose deltas are mirrored before being sent to the robot.

### DAgger/run_mix Controller Definitions

- Policy is the default controller. Human input overrides only the channels being actively controlled.
- Holding `LG` or `RG` makes the corresponding arm an expert override. The first override frame is marked as `takeover_start`; continued override frames are marked as `recovery`.
- `LTr` and `RTr` control grippers independently from arm motion. Gripper takeover uses soft takeover: the trigger command must match the current held gripper value before manual gripper control becomes active, which avoids sudden jumps.
- Press `Y` for the left gripper or `B` for the right gripper to hand that gripper back to the policy. The trigger must be released before that gripper can be manually reacquired.
- Use the left arrow to discard failed, incomplete, low-quality, or not-trainable episodes before saving. This is required when `full_episode.success_policy` is `recorded_is_success`.

### Coordinate Mapping

The exact mapping is configured by `*_pose_scaler` and `*_channel_signs`. A common Oculus-to-robot mapping is:

| Oculus axis | Robot axis | Description |
| --- | --- | --- |
| X right | -Y left | Lateral movement |
| Y up | Z up | Vertical movement |
| Z backward | X forward | Forward/backward movement |

### Troubleshooting

```bash
# Restart ADB server
adb kill-server
adb start-server

# Check connected devices
adb devices

# Stop the Oculus app
adb shell am force-stop com.rail.oculus.teleop

# Reinstall APK
adb uninstall com.rail.oculus.teleop
adb install -r teleoperators/oculus_teleoperator/oculus/oculus_reader/oculus_reader/APK/teleop-debug.apk
```

## Common Workflow

```bash
cd /path/to/dual_arm_teleop

# 1. Show camera serial numbers and fill them into scripts/config/robots/*.yaml
tools-check-rs

# 2. Check that policy configs resolve correctly. Recommended before run_policy/run_mix
robot-record --config scripts/config/record_cfg.yaml --dry-run-policy-config

# 3. Connect to the robot and return it home
robot-reset --config scripts/config/record_cfg.yaml

# 4. Collect teleoperation data
robot-record --config scripts/config/record_cfg.yaml

# 5. Visualize or replay data
robot-visualize --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml

# Check every RGB-D/IR sidecar frame before any DP3 Zarr conversion
python scripts/check_rgbd_sidecar_dataset.py --root /absolute/path/to/lerobot/dataset

# 6. Train a policy
robot-train --config scripts/config/train_cfg.yaml

# 7. Run the round-based DAgger loop
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
```

## Dataset Operations

### Record, Replay, and Visualize

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
```

![Record](docs/images/record.png)

**Figure 1: Record**

![Visualization](docs/images/visualize.png)

**Figure 2: Visualization**

### Resume Recording

If a dataset already exists under the target `repo_id`, set `record.task.resume: true` and configure `record.task.resume_dataset` in `record_cfg.yaml`, then run:

```bash
robot-record --config scripts/config/record_cfg.yaml
```

### Upload to Hugging Face

Set `record.storage.push_to_hub: true` in `record_cfg.yaml`, then log in:

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN}
huggingface-cli whoami
```

### Merge Datasets

For local LeRobot datasets that share the same feature schema, use the config-driven tool:

```bash
tools-merge-datasets --config scripts/config/merge_dataset_cfg.yaml --dry-run
tools-merge-datasets --config scripts/config/merge_dataset_cfg.yaml
```

The important fields in `scripts/config/merge_dataset_cfg.yaml` are:

```yaml
merge_datasets:
  source:
    parent_dir: "/path/to/common/source/folder"
    repo_id_prefix: "franka_dual_arm"
    datasets:
      - "dataset_a"
      - "dataset_b"

  output:
    parent_dir: "/path/to/output/folder"
    dataset_name: "merged_dataset"
```

The tool uses `LeRobotDataset.create`, `add_frame`, `save_episode`, and `finalize` instead of directly editing parquet files. It preserves episode boundaries and task strings, validates fps/features/robot type by default, and writes `meta/merge_summary.json` in the output dataset.

To merge or rename task prompts inside a single dataset, use `scripts/tools/merge_lerobot_tasks.py` or `scripts/tools/rename_lerobot_task.py`.

## Dataset Naming and Local Metadata

![Dataset](docs/images/dataset.png)

**Figure 3: Dataset**

![Dataset info](docs/images/dataset_info.png)

**Figure 4: Dataset Info**

LeRobot datasets are stored under `~/.cache/huggingface/lerobot` by default unless `dataset_path` or `dataset_root` is configured. Local dataset metadata may include:

- `dataset_info.txt`: local dataset records such as `record_id`, `name`, `task`, `date`, `version`, `user_info`, and `type`.
- `dataset_info_backup`: backups created when `tools-check-dataset` updates `dataset_info.txt`.
- Dataset folders: the actual LeRobot dataset contents.

If datasets were manually deleted or moved, refresh local metadata with:

```bash
tools-check-dataset
```

## Training, Evaluation, and DAgger

### Training

Edit `scripts/config/train_cfg.yaml`, then run:

```bash
robot-train --config scripts/config/train_cfg.yaml
```

### Policy Evaluation

Set `record.run_mode: run_policy` in `scripts/config/record_cfg.yaml`, configure `record.policy.type`, `record.policy.config_path`, and `record.policy.pretrained_path`, then run:

```bash
robot-record --config scripts/config/record_cfg.yaml
```

### DAgger Full-Episode Success Policy

`full_success_episode` and `hybrid` exports with full episodes use `full_episode.success_policy` in `dagger_rounds_cfg.yaml`:

- `explicit`: strict mode. Each saved run_mix episode asks for a manual success label.
- `recorded_is_success`: saved means successful. This fits the workflow where failed, incomplete, low-quality, or not-trainable episodes are discarded with the left arrow before saving.
- `allow_missing_for_smoke`: debugging only. Missing success labels are allowed with a warning and should not be used for real training.

When using `recorded_is_success`, the operator must be strict about deleting any episode that should not become a full demonstration. The success labels can later support VLA, Diffusion Policy, failure-detector filtering, and evaluation.

## Recording Episode Control

Set `record.task.episode_control_mode: keyboard` for keyboard control:

- Right arrow while waiting: start the next episode.
- Right arrow while recording: stop and save the current episode.
- Left arrow while recording: discard the current episode and its uncommitted Zarr tail.

Set `record.task.episode_control_mode: oculus` for keyboard-free control:

- Quest X while waiting: start the next episode.
- Quest X while recording: stop and save the current episode.
- Quest Y while recording: discard the current episode. In this mode Y is reserved for episode discard and does not request left-gripper release.

After either save or discard, `robot-record` calls `robot.reset()` automatically and waits for the next Right-arrow/X start request. Enter and Esc are not part of this workflow. Ctrl+C stops the session, marks an active Zarr recording incomplete, and asks whether to delete the entire dataset folder.

Quest teleoperation remains live after reset-home while `robot-record` is waiting for the next episode start. You can reposition both arms and grippers during this interval; those waiting-state observations and actions are sent to the robot but are not written to LeRobot or the RGB-D Zarr sidecar. Because live control still reads robot observations, recorded `global_frame_index` values can contain strictly increasing gaps across episode boundaries by design.
