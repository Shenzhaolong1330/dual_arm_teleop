"""
ARX R5 dual-arm RPC client (ZeroRPC).

Communicates with the ARX ROS2 RPC server (`arx_ros2_rpc_server.py`) running on
the robot-side Linux. The server exposes a ZeroRPC service whose methods mirror
the names called below.

Notes:
- Methods are always dispatched via ``Client.__call__`` (i.e. ``self._client(method, *args)``)
  to avoid name collisions with ZeroRPC's own socket APIs (e.g. ``disconnect``,
  ``close``).
- The class name and method signatures are kept identical to the previous
  ZMQ+msgpack implementation so callers (``arx_dual_arm.ArxDualArm``,
  ``test_*``) require no changes.
"""

import logging
from typing import Optional

import numpy as np
import zerorpc

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 7  # 6 arm + 1 gripper
NUM_POSE_DIMS = 6   # x, y, z, roll, pitch, yaw

# ZeroRPC 默认参数: timeout 仅约束“无响应”超时, 不影响正常请求延迟
_RPC_TIMEOUT_S = 30
_RPC_HEARTBEAT_S = 20


class ArxDualArmClient:
    """ZeroRPC client for ARX R5 dual-arm robot."""

    def __init__(self, ip: str = "localhost", port: int = 4242):
        self._addr = f"tcp://{ip}:{port}"
        self._client: Optional[zerorpc.Client] = None
        # 兼容旧调用习惯 (例如 client.server.<method>) 暴露同一个底层连接
        self.server: Optional[zerorpc.Client] = None

        client = zerorpc.Client(timeout=_RPC_TIMEOUT_S, heartbeat=_RPC_HEARTBEAT_S)
        client.connect(self._addr)
        self._client = client
        self.server = client
        logger.info(f"ZeroRPC client connected to {self._addr}")

    def _call(self, method: str, *args):
        """统一 RPC 调用入口.

        总是走 ``Client.__call__`` 派发, 避免与 ZeroRPC 内置方法 (如 disconnect/close)
        名字冲突.
        """
        if self._client is None:
            raise RuntimeError("RPC client already closed")
        try:
            return self._client(method, *args)
        except Exception as e:
            raise RuntimeError(f"RPC call failed in {method}: {e}") from e

    @staticmethod
    def _to_list(data):
        if hasattr(data, "tolist"):
            return data.tolist()
        return list(data)

    # ========== System ==========

    def system_connect(self, timeout: float = 10.0) -> bool:
        try:
            return bool(self._call("system_connect", float(timeout)))
        except Exception as e:
            logger.error(f"Error connecting to ARX system: {e}")
            return False

    def disconnect(self):
        """请求服务端断开硬件连接 (本地 socket 不在此处关闭)."""
        try:
            self._call("disconnect")
            logger.info("Disconnected from ARX system")
        except NameError:
            logger.debug("Server does not expose disconnect method - skipping")
        except Exception as e:
            logger.warning(f"Error disconnecting (continuing): {e}")

    def is_connected(self) -> bool:
        try:
            return bool(self._call("is_connected"))
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
        """Get full system state.

        Returns a dict with ``left_arm`` / ``right_arm`` sub-dicts, each containing
        numpy arrays for ``joint_positions[7]``, ``joint_velocities[7]``,
        ``joint_currents[7]``, ``end_pose[6]``, and a float ``gripper`` (0-1).
        若服务端额外返回 ``chassis``, 也会原样透传.
        """
        try:
            state = self._call("get_full_state")
            if state is None:
                return None

            deserialized = {}
            if "chassis" in state:
                deserialized["chassis"] = state["chassis"]

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
            self._call("set_left_gripper", float(position))
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
            self._call("set_right_gripper", float(position))
        except Exception as e:
            logger.error(f"Error setting right gripper: {e}")

    # ========== Joint 控制 (归零等场景) ==========

    def set_dual_joint_positions(self, left_positions, right_positions):
        """发送双臂关节位置. 每臂 7 个值 (6 arm + 1 gripper)."""
        try:
            self._call(
                "set_dual_joint_positions",
                self._to_list(left_positions),
                self._to_list(right_positions),
            )
        except Exception as e:
            logger.error(f"Error setting dual joint positions: {e}")

    # ========== EE Pose 控制 (服务端内置 IK) ==========

    def set_left_ee_pose(self, pose, gripper: float = 0.0):
        """设置左臂末端位姿. pose: [x,y,z,roll,pitch,yaw] 米+弧度, gripper: 0-1."""
        try:
            self._call("set_left_ee_pose", self._to_list(pose), float(gripper))
        except Exception as e:
            logger.error(f"Error setting left ee pose: {e}")

    def set_right_ee_pose(self, pose, gripper: float = 0.0):
        """设置右臂末端位姿. pose: [x,y,z,roll,pitch,yaw] 米+弧度, gripper: 0-1."""
        try:
            self._call("set_right_ee_pose", self._to_list(pose), float(gripper))
        except Exception as e:
            logger.error(f"Error setting right ee pose: {e}")

    def set_dual_ee_poses(self, left_pose, right_pose,
                          left_gripper: float = 0.0, right_gripper: float = 0.0):
        """同时设置双臂末端位姿 + 夹爪. pose: [x,y,z,roll,pitch,yaw], gripper: 0-1."""
        try:
            self._call(
                "set_dual_ee_poses",
                self._to_list(left_pose),
                self._to_list(right_pose),
                float(left_gripper),
                float(right_gripper),
            )
        except Exception as e:
            logger.error(f"Error setting dual ee poses: {e}")

    # ========== Lifecycle ==========

    def close(self):
        """关闭 ZeroRPC 连接 (幂等)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.error(f"Error closing RPC client: {e}")
            self._client = None
            self.server = None


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
