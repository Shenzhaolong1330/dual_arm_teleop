#!/usr/bin/env python
"""
Test script for IK computation and servo control.
Tests the full pipeline: Oculus -> IK -> ServoJ
"""

import logging
import numpy as np
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def test_ik_computation():
    """Test IK computation using Dobot's built-in IK."""
    log.info("\n" + "="*60)
    log.info("Testing IK Computation")
    log.info("="*60)
    
    # Import client
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "lerobot_robot/lerobot_robot"))
    from lerobot_robot.dobot_interface_client import DobotDualArmClient
    
    # Connect to robot
    client = DobotDualArmClient(ip='127.0.0.1', port=4242)
    
    # Get current state
    log.info("\n--- Getting Current State ---")
    left_ee = client.left_robot_get_ee_pose()
    right_ee = client.right_robot_get_ee_pose()
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    
    log.info(f"Left EE pose (m, rad): {left_ee}")
    log.info(f"Right EE pose (m, rad): {right_ee}")
    log.info(f"Left joints (rad): {left_joints}")
    log.info(f"Right joints (rad): {right_joints}")
    
    # Test IK with small delta
    log.info("\n--- Testing IK with Small Delta ---")
    
    # Add small delta to current pose (with proper rotation handling)
    from scipy.spatial.transform import Rotation
    
    delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1cm in X
    
    # Position: can be added directly
    left_target_pos = left_ee[:3] + delta[:3]
    right_target_pos = right_ee[:3] + delta[:3]
    
    # Orientation: need to use rotation matrices
    R_current_left = Rotation.from_euler("xyz", left_ee[3:])
    R_current_right = Rotation.from_euler("xyz", right_ee[3:])
    R_delta = Rotation.from_euler("xyz", delta[3:])
    
    # Target orientation: R_target = R_delta * R_current
    R_target_left = R_delta * R_current_left
    R_target_right = R_delta * R_current_right
    
    # Convert back to Euler angles
    left_target_rot = R_target_left.as_euler("xyz")
    right_target_rot = R_target_right.as_euler("xyz")
    
    # Combine position and orientation (in meters and radians)
    left_target = np.concatenate([left_target_pos, left_target_rot])
    right_target = np.concatenate([right_target_pos, right_target_rot])
    
    # Current joints in radians
    left_joints_rad = left_joints
    right_joints_rad = right_joints
    
    log.info(f"Left target (m, rad): {left_target}")
    log.info(f"Right target (m, rad): {right_target}")
    
    # Solve IK (client expects meters and radians, returns radians)
    log.info("\n--- Solving IK ---")
    left_ik_result = client.inverse_kinematics('left', left_target, left_joints_rad)
    right_ik_result = client.inverse_kinematics('right', right_target, right_joints_rad)
    
    if left_ik_result is not None:
        log.info(f"✓ Left IK success: {left_ik_result}")
        left_ik_rad = np.array(left_ik_result)
        log.info(f"  In radians: {left_ik_rad}")
        log.info(f"  Delta from current: {left_ik_rad - left_joints}")
    else:
        log.error("✗ Left IK failed")
    
    if right_ik_result is not None:
        log.info(f"✓ Right IK success: {right_ik_result}")
        right_ik_rad = np.array(right_ik_result)
        log.info(f"  In radians: {right_ik_rad}")
        log.info(f"  Delta from current: {right_ik_rad - right_joints}")
    else:
        log.error("✗ Right IK failed")
    
    return left_ik_result is not None and right_ik_result is not None


def test_teleop_ik():
    """Test IK in teleop pipeline."""
    log.info("\n" + "="*60)
    log.info("Testing Teleop IK Pipeline")
    log.info("="*60)
    
    from lerobot_teleoperator.oculus_dual_arm_teleop import OculusDualArmTeleop
    from lerobot_teleoperator.config_teleop import OculusDualArmTeleopConfig
    
    # Create config
    config = OculusDualArmTeleopConfig(
        ip="192.168.110.62",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=True,
        left_pose_scaler=[0.5, 0.5],
        right_pose_scaler=[0.5, 0.5],
    )
    
    # Create teleop
    teleop = OculusDualArmTeleop(config)
    
    log.info("\n--- Connecting to Robot ---")
    try:
        teleop.connect()
        log.info("✓ Connected to robot and Oculus")
    except Exception as e:
        log.error(f"✗ Connection failed: {e}")
        return False
    
    # Test getting action
    log.info("\n--- Testing Action Generation ---")
    try:
        action = teleop.get_action()
        
        log.info("\nAction keys:")
        for key in sorted(action.keys()):
            if 'joint' in key or 'delta' in key:
                log.info(f"  {key}: {action[key]:.6f}")
        
        # Check if joints are computed
        has_left_joints = all(f"left_joint_{i+1}.pos" in action for i in range(6))
        has_right_joints = all(f"right_joint_{i+1}.pos" in action for i in range(6))
        
        if has_left_joints and has_right_joints:
            log.info("\n✓ Joint positions computed successfully")
            
            # Print joint values
            left_joints = [action[f"left_joint_{i+1}.pos"] for i in range(6)]
            right_joints = [action[f"right_joint_{i+1}.pos"] for i in range(6)]
            log.info(f"  Left joints (rad): {left_joints}")
            log.info(f"  Right joints (rad): {right_joints}")
            
            return True
        else:
            log.error("✗ Joint positions not found in action")
            return False
            
    except Exception as e:
        log.error(f"✗ Action generation failed: {e}")
        return False
    finally:
        teleop.disconnect()


def test_servo_j():
    """Test servo_j_delta control."""
    log.info("\n" + "="*60)
    log.info("Testing ServoJ Control")
    log.info("="*60)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "lerobot_robot/lerobot_robot"))
    from lerobot_robot.dobot_interface_client import DobotDualArmClient
    
    # Connect to robot
    client = DobotDualArmClient(ip='127.0.0.1', port=4242)
    
    # Get current joints
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    
    log.info(f"Current left joints (rad): {left_joints}")
    log.info(f"Current right joints (rad): {right_joints}")
    
    # Test small delta
    log.info("\n--- Testing Small Joint Delta ---")
    delta_rad = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.01 radian on joint 1
    
    log.info(f"Sending delta (rad): {delta_rad}")
    
    try:
        # Send servo command (client expects radians)
        result = client.servo_j_delta('left', delta_rad, t=0.1, lookahead_time=0.05, gain=300)
        log.info(f"✓ ServoJ command sent: {result}")
        
        # Wait and check result
        time.sleep(0.2)
        new_left_joints = client.left_robot_get_joint_positions()
        log.info(f"New left joints (rad): {new_left_joints}")
        log.info(f"Actual delta (rad): {new_left_joints - left_joints}")
        log.info(f"Actual delta (rad): {new_left_joints - left_joints}")
        
        return True
    except Exception as e:
        log.error(f"✗ ServoJ failed: {e}")
        return False


def test_full_pipeline():
    """Test full pipeline: Oculus -> IK -> ServoJ."""
    log.info("\n" + "="*60)
    log.info("Testing Full Pipeline")
    log.info("="*60)
    
    from lerobot_robot.dobot_dual_arm import DobotDualArm
    from lerobot_robot.config_dobot import DobotDualArmConfig
    from lerobot_teleoperator.oculus_dual_arm_teleop import OculusDualArmTeleop
    from lerobot_teleoperator.config_teleop import OculusDualArmTeleopConfig
    
    # Create configs
    robot_config = DobotDualArmConfig(
        name="dobot_dual_arm",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        debug=False,
    )
    
    teleop_config = OculusDualArmTeleopConfig(
        ip="192.168.110.62",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        left_pose_scaler=[0.5, 0.5],
        right_pose_scaler=[0.5, 0.5],
    )
    
    # Create instances
    robot = DobotDualArm(robot_config)
    teleop = OculusDualArmTeleop(teleop_config)
    
    try:
        # Connect
        log.info("\n--- Connecting ---")
        robot.connect()
        teleop.connect()
        log.info("✓ Connected")
        
        # Test loop
        log.info("\n--- Running Control Loop (5 iterations) ---")
        for i in range(5):
            # Get action from teleop (includes IK)
            action = teleop.get_action()
            
            # Send to robot (uses servo_j_delta)
            robot.send_action(action)
            
            # Log
            left_joints = [action[f"left_joint_{i+1}.pos"] for i in range(6)]
            log.info(f"Iteration {i+1}: Left joints = {left_joints}")
            
            time.sleep(0.1)
        
        log.info("\n✓ Full pipeline test passed")
        return True
        
    except Exception as e:
        log.error(f"✗ Full pipeline failed: {e}")
        return False
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    log.info("\n" + "="*60)
    log.info("IK and Servo Control Test Suite")
    log.info("="*60)
    
    # Run tests
    tests = [
        ("IK Computation", test_ik_computation),
        # ("Teleop IK", test_teleop_ik),
        # ("ServoJ Control", test_servo_j),
        # ("Full Pipeline", test_full_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            log.error(f"Test '{name}' failed with error: {e}")
            results[name] = False
    
    # Summary
    log.info("\n" + "="*60)
    log.info("Test Summary")
    log.info("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info(f"{name}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    log.info("\n" + "="*60)
    if all_passed:
        log.info("✓ All tests passed!")
    else:
        log.info("✗ Some tests failed")
    log.info("="*60)