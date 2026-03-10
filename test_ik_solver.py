#!/usr/bin/env python
"""
Test script for ik_solver.py
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def test_ik_solver():
    """Test IK solver with known poses."""
    log.info("\n" + "="*60)
    log.info("Testing IK Solver")
    log.info("="*60)
    
    # Import IK solver
    from lerobot_teleoperator.ik_solver import DualArmIKSolver
    
    # Find URDF
    urdf_path = Path(__file__).parent / "assets/dobot_description/urdf/dual_nova5_robot.urdf"
    
    if not urdf_path.exists():
        log.error(f"URDF not found: {urdf_path}")
        return False
    
    log.info(f"URDF path: {urdf_path}")
    
    # Create IK solver
    try:
        ik_solver = DualArmIKSolver(str(urdf_path))
        log.info("✓ IK solver created successfully")
    except Exception as e:
        log.error(f"✗ Failed to create IK solver: {e}")
        return False
    
    # Test 1: Simple pose (home position)
    log.info("\n--- Test 1: Home Position ---")
    home_pose = np.array([0.3, 0.3, 1.0, 0.0, 0.0, 0.0])  # Approximate home pose
    
    left_joints, left_success = ik_solver.solve_left_ik(home_pose)
    log.info(f"Left IK success: {left_success}")
    log.info(f"Left joints: {left_joints}")
    
    right_joints, right_success = ik_solver.solve_right_ik(home_pose)
    log.info(f"Right IK success: {right_success}")
    log.info(f"Right joints: {right_joints}")
    
    # Test 2: Different pose
    log.info("\n--- Test 2: Different Pose ---")
    test_pose = np.array([0.4, 0.2, 0.9, 0.1, 0.0, 0.0])
    
    left_joints, left_success = ik_solver.solve_left_ik(test_pose)
    log.info(f"Left IK success: {left_success}")
    log.info(f"Left joints: {left_joints}")
    
    # Test 3: Dual arm IK
    log.info("\n--- Test 3: Dual Arm IK ---")
    left_joints, right_joints, left_ok, right_ok = ik_solver.solve_dual_ik(home_pose, home_pose)
    log.info(f"Success: left={left_ok}, right={right_ok}")
    log.info(f"Left joints: {left_joints}")
    log.info(f"Right joints: {right_joints}")
    
    # Test 4: Update joint positions
    log.info("\n--- Test 4: Update Joint Positions ---")
    ik_solver.update_joint_positions(left_joints=np.zeros(6), right_joints=np.zeros(6))
    positions = ik_solver.get_joint_positions()
    log.info(f"Updated positions: {positions}")
    
    return True


def test_ik_with_robot_state():
    """Test IK with real robot state."""
    log.info("\n" + "="*60)
    log.info("Testing IK with Robot State")
    log.info("="*60)
    
    from lerobot_teleoperator.ik_solver import DualArmIKSolver
    from lerobot_robot.dobot_interface_client import DobotDualArmClient
    
    # Find URDF
    urdf_path = Path(__file__).parent / "assets/dobot_description/urdf/dual_nova5_robot.urdf"
    
    if not urdf_path.exists():
        log.error(f"URDF not found: {urdf_path}")
        return False
    
    # Create IK solver
    try:
        ik_solver = DualArmIKSolver(str(urdf_path))
        log.info("✓ IK solver created")
    except Exception as e:
        log.error(f"✗ Failed to create IK solver: {e}")
        return False
    
    # Connect to robot
    try:
        client = DobotDualArmClient(ip='127.0.0.1', port=4242)
        log.info("✓ Connected to robot")
    except Exception as e:
        log.error(f"✗ Failed to connect to robot: {e}")
        return False
    
    # Get current state
    left_ee = client.left_robot_get_ee_pose()
    right_ee = client.right_robot_get_ee_pose()
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    
    log.info(f"\nCurrent left EE pose (m, rad): {left_ee}")
    log.info(f"Current right EE pose (m, rad): {right_ee}")
    log.info(f"Current left joints (rad): {left_joints}")
    log.info(f"Current right joints (rad): {right_joints}")
    
    # Update IK solver with current joints
    ik_solver.update_joint_positions(left_joints=left_joints, right_joints=right_joints)
    
    # Test IK with current pose (should return current joints)
    log.info("\n--- Testing IK with Current Pose ---")
    left_ik_joints, left_success = ik_solver.solve_left_ik(left_ee, left_joints)
    right_ik_joints, right_success = ik_solver.solve_right_ik(right_ee, right_joints)
    
    log.info(f"Left IK success: {left_success}")
    log.info(f"Left IK joints: {left_ik_joints}")
    log.info(f"Difference from current: {left_ik_joints - left_joints}")
    
    log.info(f"Right IK success: {right_success}")
    log.info(f"Right IK joints: {right_ik_joints}")
    log.info(f"Difference from current: {right_ik_joints - right_joints}")
    
    # Test IK with small delta
    log.info("\n--- Testing IK with Small Delta ---")
    from scipy.spatial.transform import Rotation
    
    delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1cm in X
    
    # Compute target pose with proper rotation
    left_target_pos = left_ee[:3] + delta[:3]
    R_current = Rotation.from_euler("xyz", left_ee[3:])
    R_delta = Rotation.from_euler("xyz", delta[3:])
    R_target = R_delta * R_current
    left_target_rot = R_target.as_euler("xyz")
    left_target = np.concatenate([left_target_pos, left_target_rot])
    
    log.info(f"Target pose: {left_target}")
    
    left_ik_joints, left_success = ik_solver.solve_left_ik(left_target, left_joints)
    log.info(f"Left IK success: {left_success}")
    log.info(f"Left IK joints: {left_ik_joints}")
    log.info(f"Delta from current: {left_ik_joints - left_joints}")
    
    return True


if __name__ == "__main__":
    log.info("\n" + "="*60)
    log.info("IK Solver Test Suite")
    log.info("="*60)
    
    # Run tests
    tests = [
        ("IK Solver Basic", test_ik_solver),
        ("IK with Robot State", test_ik_with_robot_state),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            log.error(f"Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
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
