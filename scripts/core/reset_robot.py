import yaml
from pathlib import Path
from typing import Dict, Any
from lerobot_robot import DobotDualArmConfig, DobotDualArm
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 创建机器人配置
    robot_config = DobotDualArmConfig(
        left_arm_port=cfg["record"]["robot"].get("left_arm_port", 4242),
        right_arm_port=cfg["record"]["robot"].get("right_arm_port", 4243),
        use_gripper=cfg["record"]["robot"]["use_gripper"],
        close_threshold=cfg["record"]["robot"].get("close_threshold", 0.5),
        gripper_max_open=cfg["record"]["robot"].get("gripper_max_open", 0.085),
        debug=False
    )
    
    # 创建机器人实例并连接
    robot = DobotDualArm(robot_config)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # 断开连接
    robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()