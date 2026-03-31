import yaml
from pathlib import Path
from typing import Dict, Any
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
from teleoperators import (
    OculusTeleopConfig,
    OculusTeleop,
)
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
from send2trash import send2trash
import termios, sys
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from dataclasses import field

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


class RecordConfig:
    """Configuration class for recording sessions."""
    
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        policy = cfg["policy"]
        teleop = cfg["teleop"]
        
        # Global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)
        self.run_mode: str = cfg.get("run_mode", "run_record")
        self.rename_map: dict[str, str] = field(default_factory=dict)
        
        # Robot type selection
        self.robot_type: str = cfg.get("robot_type", "dobot_dual_arm")
        if self.robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type: {self.robot_type}. "
                f"Supported types: {SUPPORTED_ROBOTS}"
            )
        
        # Teleop config - parse based on control mode
        self.control_mode = teleop.get("control_mode", "oculus")
        self.dual_arm = teleop.get("dual_arm", True)
        self._parse_teleop_config(teleop)
        
        # Policy config
        self._parse_policy_config(policy)
        
        # Robot config
        self.robot_ip: str = robot.get("robot_ip", "localhost")
        self.robot_port: int = robot.get("robot_port", 4242)
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot.get("close_threshold", 0.5)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.085)
        self.gripper_force: float = robot.get("gripper_force", 10.0)
        self.gripper_speed: float = robot.get("gripper_speed", 0.1)
        
        # Task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", "")
        
        # Time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        # save metadata period (number of episodes between metadata writes)
        # YAML uses `save_meta_period` — use the same name here.
        self.save_meta_period: int = time.get("save_meta_period", 1)
        
        # Cameras config (3 RealSense cameras: left wrist, right wrist, head)
        self.left_wrist_cam_serial: str = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial: str = cam["right_wrist_cam_serial"]
        self.head_cam_serial: str = cam["head_cam_serial"]
        self.cam_width: int = cam["width"]
        self.cam_height: int = cam["height"]
        
        # Storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)
    
    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """Parse teleoperation configuration based on control mode."""
        if self.control_mode == "oculus":
            oculus_cfg = teleop.get("oculus_config", {})
            self.use_gripper = oculus_cfg.get("use_gripper", True)
            self.oculus_ip = oculus_cfg.get("ip", "192.168.110.62")
            self.pose_scaler = oculus_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = oculus_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            self.visualize_placo = oculus_cfg.get("visualize_placo", False)
            if self.dual_arm:
                self.left_pose_scaler = oculus_cfg.get("left_pose_scaler", self.pose_scaler)
                self.right_pose_scaler = oculus_cfg.get("right_pose_scaler", self.pose_scaler)
                self.left_channel_signs = oculus_cfg.get("left_channel_signs", self.channel_signs)
                self.right_channel_signs = oculus_cfg.get("right_channel_signs", self.channel_signs)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")
    
    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """Parse policy configuration."""
        policy_type = policy["type"]
        if policy_type == "act":
            from lerobot.policies import ACTConfig
            self.policy = ACTConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        else:
            raise ValueError(f"No config for policy type: {policy_type}")
        
        if policy.get("pretrained_path"):
            self.policy.pretrained_path = policy["pretrained_path"]
    
    def create_teleop_config(self):
        """Create teleoperation configuration object."""
        if self.control_mode == "oculus":
            if self.dual_arm:
                return OculusTeleopConfig(
                    use_gripper=self.use_gripper,
                    ip=self.oculus_ip,
                    left_pose_scaler=self.left_pose_scaler,
                    right_pose_scaler=self.right_pose_scaler,
                    left_channel_signs=self.left_channel_signs,
                    right_channel_signs=self.right_channel_signs,
                    visualize_placo=self.visualize_placo,
                )
            return OculusTeleopConfig(
                use_gripper=self.use_gripper,
                ip=self.oculus_ip,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")


def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        print(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            print(f"====== [DELETE] Removing folder: {dataset_path} ======")
            # Send to trash
            send2trash(dataset_path)
            print("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            print("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")

def run_record(record_cfg: RecordConfig):
    print("====== [START] Starting recording ======")
    try:
        dataset_name, data_version = generate_dataset_name(record_cfg)

        # Check joint offsets
        # if not record_cfg.debug:
        #     check_joint_offsets(record_cfg)        
        
        # Create RealSenseCamera configurations (3 cameras: left wrist, right wrist, head)
        left_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.left_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        right_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.right_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        head_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.head_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        # Create the robot and teleoperator configurations
        camera_config = {
            "left_wrist_image": left_wrist_image_cfg,
            "right_wrist_image": right_wrist_image_cfg,
            "head_image": head_image_cfg,
        }
        
        # Create teleop config using the new method
        teleop_config = record_cfg.create_teleop_config()
        
        # Create robot configuration dynamically based on robot_type
        robot_config = create_robot_config(
            record_cfg.robot_type,
            robot_ip=record_cfg.robot_ip,
            robot_port=record_cfg.robot_port,
            cameras=camera_config,
            debug=record_cfg.debug,
            use_gripper=record_cfg.use_gripper,
            gripper_max_open=record_cfg.gripper_max_open,
            gripper_force=record_cfg.gripper_force,
            gripper_speed=record_cfg.gripper_speed,
            close_threshold=record_cfg.close_threshold,
            gripper_reverse=record_cfg.gripper_reverse,
            control_mode=record_cfg.control_mode,
        )
        
        # Initialize the robot dynamically based on robot_type
        robot = create_robot(record_cfg.robot_type, robot_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_meta_period

        # Initialize the keyboard listener and rerun visualization
        _, events = init_keyboard_listener()
        init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        preprocessor = None
        postprocessor = None

        # configure the teleop and policy
        if record_cfg.run_mode == "run_record":
            logging.info("====== [INFO] Running in teleoperation mode ======")
            teleop = OculusTeleop(teleop_config)
            policy = None
        elif record_cfg.run_mode == "run_policy":
            logging.info("====== [INFO] Running in policy mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = None
        elif record_cfg.run_mode == "run_mix":
            logging.info("====== [INFO] Running in mixed mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = OculusTeleop(teleop_config)
        
        if policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=record_cfg.policy,
                pretrained_path=record_cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, {}),  # 使用空字典作为rename_map
                preprocessor_overrides={
                    "device_processor": {"device": record_cfg.policy.device},
                    "rename_observations_processor": {"rename_map": {}},  # 使用空字典作为rename_map
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        episode_idx = 0

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            record_loop(
                robot=robot,
                events=events,
                fps=record_cfg.fps,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=record_cfg.episode_time_sec,
                single_task=record_cfg.task_description,
                display_data=record_cfg.display,
            )

            if events["rerecord_episode"]:
                logging.info("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or events["rerecord_episode"]):
                while True:
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    user_input = input("====== [WAIT] Press Enter to reset the environment ======")
                    if user_input == "":
                        break  
                    else:
                        logging.info("====== [WARNING] Please press only Enter to continue ======")

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            episode_idx += 1

        # Clean up
        logging.info("Stop recording")
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(record_cfg)

if __name__ == "__main__":
    main()