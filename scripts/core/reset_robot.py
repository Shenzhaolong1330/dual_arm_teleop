import yaml
from pathlib import Path
from typing import Dict, Any
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    robot_type = cfg["record"].get("robot_type", "dobot_dual_arm")
    
    # 创建机器人配置
    robot_config = create_robot_config(
        robot_type=robot_type,
        robot_ip=cfg["record"]["robot"].get("robot_ip", "localhost"),
        robot_port=cfg["record"]["robot"].get("robot_port", 4242),
        use_gripper=cfg["record"]["robot"]["use_gripper"],
        close_threshold=cfg["record"]["robot"].get("close_threshold", 0.5),
        gripper_max_open=cfg["record"]["robot"].get("gripper_max_open", 0.085),
        debug=False
    )
    
    # 创建机器人实例并连接
    robot = create_robot(robot_type, robot_config)
    print("----------",robot.name)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # 断开连接
    # robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()