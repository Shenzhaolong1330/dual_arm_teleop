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
        # Finish behavior: by default reset to home and keep connection to avoid server stop on close.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        
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
        
        # Policy config - load from policy_cfg file if specified
        policy_cfg_path = cfg.get("policy_cfg")
        self._parse_policy_config(policy, policy_cfg_path)
        
        # Robot config
        self.robot_ip: str = robot.get("robot_ip", "localhost")
        self.robot_port: int = robot.get("robot_port", 4242)
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot.get("close_threshold", 0.5)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.085)
        self.gripper_force: float = robot.get("gripper_force", 10.0)
        self.gripper_speed: float = robot.get("gripper_speed", 0.1)
        self.reset_go_home: bool = robot.get("reset_go_home", True)
        self.go_home_duration_sec: float | None = robot.get("go_home_duration_sec", None)
        self.go_home_rate_hz: float | None = robot.get("go_home_rate_hz", None)
        self.max_cartesian_delta: float | None = robot.get("max_cartesian_delta", None)
        self.max_rotation_delta: float | None = robot.get("max_rotation_delta", None)
        
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
        # Debugging: verbose action logging (prints raw, postprocessed and robot action values)
        self.verbose_action_debug: bool = cfg.get("verbose_action_debug", False)
    
    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """Parse teleoperation configuration based on control mode."""
        if self.control_mode == "oculus":
            oculus_cfg = teleop.get("oculus_config", {})
            self.use_gripper = oculus_cfg.get("use_gripper", True)
            self.oculus_ip = oculus_cfg.get("ip", "192.168.110.62")
            self.pose_scaler = oculus_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = oculus_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            self.visualize_placo = oculus_cfg.get("visualize_placo", False)
            self.action_smoothing_alpha = oculus_cfg.get("action_smoothing_alpha", 0.35)
            self.mirror_teleop = oculus_cfg.get("mirror_teleop", False)
            if self.dual_arm:
                self.left_pose_scaler = oculus_cfg.get("left_pose_scaler", self.pose_scaler)
                self.right_pose_scaler = oculus_cfg.get("right_pose_scaler", self.pose_scaler)
                self.left_channel_signs = oculus_cfg.get("left_channel_signs", self.channel_signs)
                self.right_channel_signs = oculus_cfg.get("right_channel_signs", self.channel_signs)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")
    
    def _parse_policy_config(self, policy: Dict[str, Any], policy_cfg_path: str = None) -> None:
        """Parse policy configuration."""
        pretrained_path = policy.get("pretrained_path")
        if pretrained_path:
            project_root = Path(__file__).resolve().parent.parent.parent
            pretrained_path_text = str(pretrained_path)
            pretrained_local_path = Path(pretrained_path_text).expanduser()
            looks_like_local_path = (
                pretrained_local_path.is_absolute()
                or pretrained_path_text.startswith((".", "~"))
                or len(pretrained_local_path.parts) > 2
            )
            if looks_like_local_path:
                if not pretrained_local_path.is_absolute():
                    pretrained_local_path = project_root / pretrained_local_path
                pretrained_local_path = pretrained_local_path.resolve()
                if not pretrained_local_path.exists():
                    raise FileNotFoundError(
                        "[POLICY] pretrained_path does not exist:\n"
                        f"  {pretrained_local_path}\n"
                        "Train that checkpoint first, or point `record.policy.pretrained_path` "
                        "to an existing model directory."
                    )
                pretrained_path = str(pretrained_local_path)

            try:
                self.policy = PreTrainedConfig.from_pretrained(pretrained_path)
                self.policy.pretrained_path = pretrained_path
                if policy.get("device"):
                    self.policy.device = policy["device"]
                if "push_to_hub" in policy:
                    self.policy.push_to_hub = policy["push_to_hub"]
                logging.info(f"[POLICY] Loaded pretrained policy config from: {pretrained_path}")
                logging.info(
                    "[POLICY] Effective config: type=%s chunk_size=%s n_action_steps=%s "
                    "temporal_ensemble_coeff=%s optimizer_lr=%s kl_weight=%s",
                    getattr(self.policy, "type", None),
                    getattr(self.policy, "chunk_size", None),
                    getattr(self.policy, "n_action_steps", None),
                    getattr(self.policy, "temporal_ensemble_coeff", None),
                    getattr(self.policy, "optimizer_lr", None),
                    getattr(self.policy, "kl_weight", None),
                )
                return
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "[POLICY] Failed to load pretrained policy config from "
                    f"{pretrained_path}. Check that the directory contains config.json."
                ) from exc

        # 加载策略配置文件（如果指定）
        policy_defaults = {}
        if policy_cfg_path:
            # 支持相对路径：相对于项目根目录
            project_root = Path(__file__).resolve().parent.parent.parent
            cfg_path = Path(policy_cfg_path)
            if not cfg_path.is_absolute():
                cfg_path = project_root / cfg_path
            
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    policy_defaults = yaml.safe_load(f).get("policy", {})
                logging.info(f"[POLICY] Loaded policy config from: {cfg_path}")
            else:
                logging.warning(f"[POLICY] Policy config file not found: {cfg_path}")
        
        # 合并配置：policy 中的值优先于 policy_cfg 文件
        def get_policy_param(key, default=None):
            return policy.get(key, policy_defaults.get(key, default))
        
        def normalize_temporal_ensemble_coeff(value: Any) -> float | None:
            """Treat non-positive and None-like values as disabled temporal ensembling."""
            if value is None:
                return None

            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"", "none", "null", "~"}:
                    return None
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        "`policy.temporal_ensemble_coeff` must be a number, null, or None-like string. "
                        f"Got: {value!r}"
                    ) from exc

            if isinstance(value, (int, float)):
                return float(value) if value > 0 else None

            raise ValueError(
                "`policy.temporal_ensemble_coeff` must be numeric or null-like. "
                f"Got type: {type(value).__name__}"
            )

        policy_type = get_policy_param("type")
        if policy_type == "act":
            from lerobot.policies import ACTConfig

            temporal_ensemble_coeff = normalize_temporal_ensemble_coeff(
                get_policy_param("temporal_ensemble_coeff")
            )
            self.policy = ACTConfig(
                device=get_policy_param("device", "cuda"),
                push_to_hub=get_policy_param("push_to_hub", False),
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                # 输入/输出结构
                n_obs_steps=get_policy_param("n_obs_steps", 1),
                chunk_size=get_policy_param("chunk_size", 100),
                n_action_steps=get_policy_param("n_action_steps", 100),
                # Transformer 架构
                dim_model=get_policy_param("dim_model", 512),
                n_heads=get_policy_param("n_heads", 8),
                n_encoder_layers=get_policy_param("n_encoder_layers", 4),
                n_decoder_layers=get_policy_param("n_decoder_layers", 1),
                dim_feedforward=get_policy_param("dim_feedforward", 3200),
                feedforward_activation=get_policy_param("feedforward_activation", "relu"),
                pre_norm=get_policy_param("pre_norm", False),
                dropout=get_policy_param("dropout", 0.1),
                # VAE 相关
                use_vae=get_policy_param("use_vae", True),
                latent_dim=get_policy_param("latent_dim", 32),
                n_vae_encoder_layers=get_policy_param("n_vae_encoder_layers", 4),
                kl_weight=get_policy_param("kl_weight", 10.0),
                # 视觉骨干网络
                vision_backbone=get_policy_param("vision_backbone", "resnet18"),
                pretrained_backbone_weights=get_policy_param("pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"),
                replace_final_stride_with_dilation=get_policy_param("replace_final_stride_with_dilation", False),
                # 优化器
                optimizer_lr=get_policy_param("optimizer_lr", 1e-5),
                optimizer_weight_decay=get_policy_param("optimizer_weight_decay", 1e-4),
                optimizer_lr_backbone=get_policy_param("optimizer_lr_backbone", 1e-5),
            )
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device=get_policy_param("device", "cuda"),
                push_to_hub=get_policy_param("push_to_hub", False),
                # 输入/输出结构
                n_obs_steps=get_policy_param("n_obs_steps", 2),
                horizon=get_policy_param("horizon", 16),
                n_action_steps=get_policy_param("n_action_steps", 8),
                # 视觉骨干网络
                vision_backbone=get_policy_param("vision_backbone", "resnet18"),
                crop_shape=tuple(get_policy_param("crop_shape", [84, 84])) if get_policy_param("crop_shape") else None,
                crop_is_random=get_policy_param("crop_is_random", True),
                pretrained_backbone_weights=get_policy_param("pretrained_backbone_weights", None),
                use_group_norm=get_policy_param("use_group_norm", True),
                spatial_softmax_num_keypoints=get_policy_param("spatial_softmax_num_keypoints", 32),
                use_separate_rgb_encoder_per_camera=get_policy_param("use_separate_rgb_encoder_per_camera", False),
                # U-Net 架构
                down_dims=tuple(get_policy_param("down_dims", [512, 1024, 2048])),
                kernel_size=get_policy_param("kernel_size", 5),
                n_groups=get_policy_param("n_groups", 8),
                diffusion_step_embed_dim=get_policy_param("diffusion_step_embed_dim", 128),
                use_film_scale_modulation=get_policy_param("use_film_scale_modulation", True),
                # 噪声调度器
                noise_scheduler_type=get_policy_param("noise_scheduler_type", "DDPM"),
                num_train_timesteps=get_policy_param("num_train_timesteps", 100),
                beta_schedule=get_policy_param("beta_schedule", "squaredcos_cap_v2"),
                beta_start=get_policy_param("beta_start", 0.0001),
                beta_end=get_policy_param("beta_end", 0.02),
                prediction_type=get_policy_param("prediction_type", "epsilon"),
                clip_sample=get_policy_param("clip_sample", True),
                clip_sample_range=get_policy_param("clip_sample_range", 1.0),
                num_inference_steps=get_policy_param("num_inference_steps", None),
                # 损失计算
                do_mask_loss_for_padding=get_policy_param("do_mask_loss_for_padding", False),
                # 优化器
                optimizer_lr=get_policy_param("optimizer_lr", 1e-4),
                optimizer_betas=tuple(get_policy_param("optimizer_betas", [0.95, 0.999])),
                optimizer_eps=get_policy_param("optimizer_eps", 1e-8),
                optimizer_weight_decay=get_policy_param("optimizer_weight_decay", 1e-6),
                # 学习率调度器
                scheduler_name=get_policy_param("scheduler_name", "cosine"),
                scheduler_warmup_steps=get_policy_param("scheduler_warmup_steps", 500),
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
                    action_smoothing_alpha=self.action_smoothing_alpha,
                    mirror_teleop=self.mirror_teleop,
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
        robot_kwargs = dict(
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
        if record_cfg.robot_type == "franka_dual_arm":
            robot_kwargs.update(
                reset_go_home=record_cfg.reset_go_home,
                go_home_duration_sec=record_cfg.go_home_duration_sec,
                go_home_rate_hz=record_cfg.go_home_rate_hz,
            )
            if record_cfg.max_cartesian_delta is not None:
                robot_kwargs["max_cartesian_delta"] = record_cfg.max_cartesian_delta
            if record_cfg.max_rotation_delta is not None:
                robot_kwargs["max_rotation_delta"] = record_cfg.max_rotation_delta
        robot_config = create_robot_config(record_cfg.robot_type, **robot_kwargs)
        
        # Initialize the robot dynamically based on robot_type
        robot = create_robot(record_cfg.robot_type, robot_config)
        if record_cfg.verbose_action_debug and hasattr(robot, "set_action_debug"):
            robot.set_action_debug(True)

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

        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        _, events = init_keyboard_listener()
        if record_cfg.display:
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

        # Reset robot to home position at the end (same intent as pressing A in teleop).
        if record_cfg.reset_on_finish:
            try:
                robot.reset()
            except Exception as reset_err:
                logging.warning(f"[WARNING] reset_on_finish failed: {reset_err}")

        # Optional disconnect. For Nero, disconnect triggers client.close() -> robot_stop on server.
        if record_cfg.disconnect_on_finish:
            robot.disconnect()
        else:
            logging.info("[INFO] Skip robot.disconnect() to avoid stop/e-stop at session end.")

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
