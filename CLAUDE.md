# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dual-arm teleoperation system using Oculus Quest 3 VR controllers to control robot arms (ARX R5, Dobot Nova5, Franka). Built on top of LeRobot (v0.3.4) for dataset collection, policy training (ACT/Diffusion), and replay. The primary active configuration is **ARX R5 dual-arm + Oculus Quest 3**.

## Setup & Installation

```bash
conda create -n dual_arm_data python=3.10
conda activate dual_arm_data

# LeRobot v0.3.4 (pinned commit)
git clone https://github.com/huggingface/lerobot.git && cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .

# This package
cd dual_arm_teleop && pip install -e .

# Oculus reader (optional, for VR teleoperation)
cd teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
pip install -e oculus_reader
sudo apt install android-tools-adb
```

## Key Commands

```bash
robot-record          # Start teleoperation recording (reads scripts/config/record_cfg.yaml)
robot-replay          # Replay a recorded dataset episode
robot-visualize       # Visualize dataset in Rerun viewer
robot-train           # Train ACT/Diffusion policy
robot-reset           # Reset robot to home position
tools-check-rs        # List connected RealSense camera serial numbers
tools-check-dataset   # Update dataset metadata
```

**Recording keyboard controls:** Right-arrow = end episode, Left-arrow = rerecord, Esc = exit.

## Architecture: Teleoperation Data Flow

```
Quest 3 (ADB logcat) → OculusReader → OculusDualArmRobot (delta EE pose)
    → OculusTeleop.get_action() → LeRobot record_loop
    → ArxDualArm.send_action() → Placo IK solver → ZMQ+msgpack RPC → ARX hardware
```

### Three-layer abstraction

1. **Teleoperator layer** (`teleoperators/`): Reads VR input, outputs `{left,right}_delta_ee_pose.{x,y,z,rx,ry,rz}` + gripper commands. Inherits from `lerobot.teleoperators.Teleoperator`.

2. **Robot layer** (`robots/`): Receives delta EE or direct joint commands, executes on hardware. Inherits from `lerobot.robots.Robot`. Factory pattern in `robots/__init__.py` via `ROBOT_CONFIG_REGISTRY`.

3. **Recording/Training layer** (`scripts/`): Orchestrates teleop + robot + cameras + dataset. Uses LeRobot's `record_loop`, `LeRobotDataset`, and policy training pipeline.

### ARX R5 specifics

- **IK**: `ArxR5IKSolver` (Placo-based) converts delta EE pose to 6-DOF joint angles. URDF: `assets/arx_r5/dual_R5a.urdf`. IK frames: `left_link6`, `right_link6`. Falls back to last good solution on solver failure.
- **Communication**: `ArxDualArmClient` uses ZMQ REQ/REP + msgpack. Protocol: `{"m": method, "a": args}` → `{"r": result}` or `{"e": error}`. Server runs on robot-side Linux (default `192.168.110.57:4242`).
- **Safety**: Joint delta clamping at `max_joint_delta=0.3` rad/step. Grip buttons (LG/RG) must be pressed to enable arm movement.
- **Gripper**: Integrated as joint 7, controlled separately via `set_{left,right}_gripper(0..1)`. 0=open, 1=closed.

### Coordinate transform (Oculus → Robot)

```
robot_x = -oculus_z   (backward → forward)
robot_y = -oculus_x   (right → left)  
robot_z =  oculus_y   (up → up)
```

Rotation mapping: `robot_roll = oculus_rz, robot_pitch = oculus_rx, robot_yaw = oculus_ry`.

## Configuration

All recording/replay config lives in `scripts/config/record_cfg.yaml`. Key fields:

- `robot_type`: `"arx_dual_arm"` | `"dobot_dual_arm"` | `"franka"`
- `debug`: When true, robot does NOT execute arm commands (gripper still works). **Currently set to `true` in `config_arx.py`** — must set to `false`/`0` for real teleoperation.
- `teleop.oculus_config.ip`: Quest 3 IP address
- `teleop.oculus_config.{left,right}_channel_signs`: Per-axis sign flips `[x,y,z,rx,ry,rz]` — critical for correct movement direction mapping
- `teleop.oculus_config.{left,right}_pose_scaler`: `[position_scale, orientation_scale]`
- `robot.robot_ip`: ARX RPC server IP

Robot-specific config dataclasses: `robots/dual_arx_r5/config_arx.py`, `robots/dual_dobot/config_dobot.py`.

## Testing

- `robots/dual_arx_r5/test_dual_arm.ipynb`: RPC client test (connect, read state, control gripper/joints, latency benchmark)
- IK solver standalone test: `python robots/dual_arx_r5/arx_ik_solver.py`
- Oculus teleop standalone test: `python teleoperators/oculus_teleoperator/oculus_teleop.py`
- Oculus dual-arm controller test: `python teleoperators/oculus_teleoperator/oculus/oculus_dual_arm_robot.py`

## Dependencies

- **placo**: Linux-only IK library, lazy-imported in `arx_ik_solver.py`
- **zmq + msgpack**: ARX RPC communication
- **zerorpc**: Dobot RPC communication (not used for ARX)
- **scipy**: Rotation transforms
- **pyrealsense2**: Camera interface (3 RealSense cameras: left_wrist, right_wrist, head)
- **lerobot** (v0.3.4): Dataset, recording loop, policy training, camera abstractions

## Current Status

Communication test with ARX R5 has passed. Next step is debugging the full teleoperation pipeline (Quest 3 → IK → ARX hardware).
