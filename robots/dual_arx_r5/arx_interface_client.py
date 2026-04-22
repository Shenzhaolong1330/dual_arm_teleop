"""
ARX R5 dual-arm RPC client (ZMQ + msgpack).

Communicates with the ARX ROS2 RPC server running on the robot-side Linux.
Protocol:
  Request:  {"m": method_name, "a": [args...]}  → msgpack
  Response: {"r": result}  or  {"e": error_msg}  → msgpack

Write commands (set_*) use fire-and-forget: server replies immediately
with {"r": null}, then executes the ROS2 publish asynchronously.
"""

import logging
import numpy as np
import zmq
import msgpack

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 7  # 6 arm + 1 gripper
NUM_POSE_DIMS = 6   # x, y, z, roll, pitch, yaw

_DEFAULT_TIMEOUT_MS = 5000
_CONNECT_TIMEOUT_MS = 30000


class ArxDualArmClient:
    """ZMQ+msgpack RPC client for ARX R5 dual-arm robot."""

    def __init__(self, ip: str = "localhost", port: int = 4242):
        self._addr = f"tcp://{ip}:{port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, _DEFAULT_TIMEOUT_MS)
        self._socket.setsockopt(zmq.SNDTIMEO, _DEFAULT_TIMEOUT_MS)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._addr)
        logger.info(f"ZMQ REQ client connected to {self._addr}")

    def _call(self, method: str, *args, timeout_ms: int | None = None):
        if self._socket is None:
            raise RuntimeError("RPC client already closed")

        if timeout_ms is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            payload = msgpack.packb({"m": method, "a": list(args)}, use_bin_type=True)
            self._socket.send(payload)
            raw = self._socket.recv()
            resp = msgpack.unpackb(raw, raw=False)
            if "e" in resp:
                raise RuntimeError(f"Server error in {method}: {resp['e']}")
            return resp.get("r")
        finally:
            if timeout_ms is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, _DEFAULT_TIMEOUT_MS)

    # ========== System ==========

    def system_connect(self, timeout: float = 10.0) -> bool:
        try:
            return self._call("system_connect", timeout, timeout_ms=_CONNECT_TIMEOUT_MS)
        except Exception as e:
            logger.error(f"Error connecting to ARX system: {e}")
            return False

    def disconnect(self):
        try:
            self._call("disconnect")
            logger.info("Disconnected from ARX system")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    def is_connected(self) -> bool:
        try:
            return self._call("is_connected")
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            return False

    def emergency_stop(self):
        try:
            self._call("emergency_stop")
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    # ========== Full state ==========

    def get_full_state(self):
        """Get full system state. Returns dict with left_arm/right_arm sub-dicts,
        each containing numpy arrays for joint_positions[7], joint_velocities[7],
        joint_currents[7], end_pose[6], and float gripper (0-1)."""
        try:
            state = self._call("get_full_state")
            if state is None:
                return None

            deserialized = {}
            for side in ("left_arm", "right_arm"):
                if side in state:
                    arm = state[side]
                    deserialized[side] = {
                        "joint_positions": np.array(arm["joint_positions"]),
                        "joint_velocities": np.array(arm["joint_velocities"]),
                        "joint_currents": np.array(arm["joint_currents"]),
                        "end_pose": np.array(arm["end_pose"]),
                        "gripper": arm["gripper"],
                    }
            return deserialized
        except Exception as e:
            logger.error(f"Error getting full state: {e}")
            return None

    # ========== Left arm ==========

    def get_left_joint_positions(self) -> np.ndarray:
        try:
            return np.array(self._call("get_left_joint_positions"))
        except Exception as e:
            logger.error(f"Error getting left joint positions: {e}")
            return np.zeros(NUM_ARM_JOINTS)

    def get_left_end_pose(self) -> np.ndarray:
        try:
            return np.array(self._call("get_left_end_pose"))
        except Exception as e:
            logger.error(f"Error getting left end pose: {e}")
            return np.zeros(NUM_POSE_DIMS)

    def get_left_gripper_position(self) -> float:
        try:
            return float(self._call("get_left_gripper_position"))
        except Exception as e:
            logger.error(f"Error getting left gripper position: {e}")
            return 0.0

    def set_left_gripper(self, position: float):
        try:
            self._call("set_left_gripper", position)
        except Exception as e:
            logger.error(f"Error setting left gripper: {e}")

    # ========== Right arm ==========

    def get_right_joint_positions(self) -> np.ndarray:
        try:
            return np.array(self._call("get_right_joint_positions"))
        except Exception as e:
            logger.error(f"Error getting right joint positions: {e}")
            return np.zeros(NUM_ARM_JOINTS)

    def get_right_end_pose(self) -> np.ndarray:
        try:
            return np.array(self._call("get_right_end_pose"))
        except Exception as e:
            logger.error(f"Error getting right end pose: {e}")
            return np.zeros(NUM_POSE_DIMS)

    def get_right_gripper_position(self) -> float:
        try:
            return float(self._call("get_right_gripper_position"))
        except Exception as e:
            logger.error(f"Error getting right gripper position: {e}")
            return 0.0

    def set_right_gripper(self, position: float):
        try:
            self._call("set_right_gripper", position)
        except Exception as e:
            logger.error(f"Error setting right gripper: {e}")

    # ========== Joint 控制 (归零等场景) ==========

    def set_dual_joint_positions(self, left_positions, right_positions):
        """发送双臂关节位置. 每臂 7 个值 (6 arm + 1 gripper)."""
        try:
            left = left_positions.tolist() if hasattr(left_positions, 'tolist') else list(left_positions)
            right = right_positions.tolist() if hasattr(right_positions, 'tolist') else list(right_positions)
            self._call("set_dual_joint_positions", left, right)
        except Exception as e:
            logger.error(f"Error setting dual joint positions: {e}")

    # ========== EE Pose 控制 (服务端内置 IK) ==========

    def set_left_ee_pose(self, pose, gripper: float = 0.0):
        """设置左臂末端位姿. pose: [x,y,z,roll,pitch,yaw] 米+弧度, gripper: 0-1."""
        try:
            p = pose.tolist() if hasattr(pose, 'tolist') else list(pose)
            self._call("set_left_ee_pose", p, gripper)
        except Exception as e:
            logger.error(f"Error setting left ee pose: {e}")

    def set_right_ee_pose(self, pose, gripper: float = 0.0):
        """设置右臂末端位姿. pose: [x,y,z,roll,pitch,yaw] 米+弧度, gripper: 0-1."""
        try:
            p = pose.tolist() if hasattr(pose, 'tolist') else list(pose)
            self._call("set_right_ee_pose", p, gripper)
        except Exception as e:
            logger.error(f"Error setting right ee pose: {e}")

    def set_dual_ee_poses(self, left_pose, right_pose,
                          left_gripper: float = 0.0, right_gripper: float = 0.0):
        """同时设置双臂末端位姿 + 夹爪. pose: [x,y,z,roll,pitch,yaw], gripper: 0-1."""
        try:
            lp = left_pose.tolist() if hasattr(left_pose, 'tolist') else list(left_pose)
            rp = right_pose.tolist() if hasattr(right_pose, 'tolist') else list(right_pose)
            self._call("set_dual_ee_poses", lp, rp, left_gripper, right_gripper)
        except Exception as e:
            logger.error(f"Error setting dual ee poses: {e}")

    # ========== Lifecycle ==========

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception as e:
                logger.error(f"Error terminating context: {e}")
            self._context = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = ArxDualArmClient("localhost", 4242)
    if client.system_connect(timeout=10.0):
        print("Connected!")
        state = client.get_full_state()
        if state:
            for side in ("left_arm", "right_arm"):
                if side in state:
                    print(f"\n{side}:")
                    print(f"  joints: {state[side]['joint_positions']}")
                    print(f"  ee_pose: {state[side]['end_pose']}")
                    print(f"  gripper: {state[side]['gripper']}")
        client.disconnect()
    client.close()
