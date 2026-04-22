import yaml
import argparse
import time
from datetime import datetime
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


class _RuntimeRecordLatencyProfiler:
    """Optional runtime profiler for `robot-record` main loop."""

    def __init__(self, fps: int, csv_path: str | None = None):
        from scripts.tools.profile_dual_arx_r5_latency import StageLatencyCollector

        self.collector = StageLatencyCollector()
        self.fps = int(fps)
        self.target_ms = 1000.0 / max(self.fps, 1)
        self.csv_path = Path(csv_path) if csv_path else None

        self._session_start = time.perf_counter()
        self._frame_idx = 0
        self._current_frame_start: float | None = None
        self._installed = False
        self._records: list[tuple[object, str, object]] = []
        self._busy_wait_restore: tuple[object, object, object, object] | None = None
        self._robot_hook_restore: tuple[object, object] | None = None

    def wrap_callable(self, fn, stage_name: str):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.collector.record(stage_name, (time.perf_counter() - t0) * 1e3)

        return wrapped

    def install(self, robot, teleop, dataset):
        if self._installed:
            return

        self._wrap_method(robot, "get_observation", self._wrap_get_observation)
        self._wrap_method(robot, "send_action", self._wrap_send_action)

        if teleop is not None:
            self._wrap_method(teleop, "get_action", self._wrap_teleop_get_action)
        if dataset is not None:
            self._wrap_method(dataset, "add_frame", self._wrap_dataset_add_frame)

        try:
            import lerobot.utils.robot_utils as ru_mod
            import lerobot.scripts.lerobot_record as rec_mod

            orig_ru = ru_mod.busy_wait
            orig_rec = rec_mod.busy_wait

            def wrapped_busy_wait(seconds):
                self.collector.record("busy_wait_budget_ms", float(seconds) * 1e3)
                t0 = time.perf_counter()
                orig_ru(seconds)
                self.collector.record("busy_wait_actual_ms", (time.perf_counter() - t0) * 1e3)
                if self._current_frame_start is not None:
                    loop_total_ms = (time.perf_counter() - self._current_frame_start) * 1e3
                    self.collector.record("loop_total", loop_total_ms)
                    self.collector.record("loop_overrun_ms", max(0.0, loop_total_ms - self.target_ms))
                    self.collector.end_frame()
                    self._current_frame_start = None

            ru_mod.busy_wait = wrapped_busy_wait
            rec_mod.busy_wait = wrapped_busy_wait
            self._busy_wait_restore = (ru_mod, rec_mod, orig_ru, orig_rec)
        except Exception as e:
            logging.warning(f"[PROFILE] busy_wait patch failed: {e}")

        if hasattr(robot, "set_latency_hook"):
            try:
                prev_hook = getattr(robot, "_latency_hook", None)
                robot.set_latency_hook(self.collector.record)
                self._robot_hook_restore = (robot, prev_hook)
            except Exception as e:
                logging.warning(f"[PROFILE] robot latency hook install failed: {e}")

        self._installed = True
        logging.info("[PROFILE] runtime latency profiling enabled for robot-record")

    def _wrap_method(self, obj, method_name: str, wrapper_builder):
        if not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)
        wrapped = wrapper_builder(original)
        setattr(obj, method_name, wrapped)
        self._records.append((obj, method_name, original))

    def _wrap_get_observation(self, original):
        def wrapped(*args, **kwargs):
            if self._current_frame_start is not None:
                self.collector.end_frame()
            self.collector.begin_frame(self._frame_idx)
            self._frame_idx += 1
            self._current_frame_start = time.perf_counter()

            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.collector.record("robot_get_observation_total", (time.perf_counter() - t0) * 1e3)

        return wrapped

    def _wrap_teleop_get_action(self, original):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.collector.record("teleop_get_action_total", (time.perf_counter() - t0) * 1e3)

        return wrapped

    def _wrap_send_action(self, original):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.collector.record("robot_send_action_total", (time.perf_counter() - t0) * 1e3)

        return wrapped

    def _wrap_dataset_add_frame(self, original):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.collector.record("dataset_add_frame", (time.perf_counter() - t0) * 1e3)

        return wrapped

    def _report_paths(self) -> tuple[Path, Path]:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt = Path("logs") / f"profile_{ts}.txt"
        csv = self.csv_path if self.csv_path else txt.with_suffix(".csv")
        return txt, csv

    def close(self):
        if self._current_frame_start is not None:
            self.collector.end_frame()
            self._current_frame_start = None

        if self._busy_wait_restore is not None:
            ru_mod, rec_mod, orig_ru, orig_rec = self._busy_wait_restore
            ru_mod.busy_wait = orig_ru
            rec_mod.busy_wait = orig_rec
            self._busy_wait_restore = None

        while self._records:
            obj, method_name, original = self._records.pop()
            setattr(obj, method_name, original)

        if self._robot_hook_restore is not None:
            robot, prev_hook = self._robot_hook_restore
            try:
                robot.set_latency_hook(prev_hook)
            except Exception:
                pass
            self._robot_hook_restore = None

        self._installed = False

        if not self.collector.has_data():
            logging.warning("[PROFILE] no latency data collected")
            return

        duration_s = time.perf_counter() - self._session_start
        report = self.collector.format_report(fps=self.fps, duration_s=duration_s)
        txt_path, csv_path = self._report_paths()
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(report, encoding="utf-8")
        self.collector.export_csv(csv_path)

        logging.info(f"[PROFILE] report txt: {txt_path}")
        logging.info(f"[PROFILE] report csv: {csv_path}")


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
        self.enable_ee_action_filter: bool = robot.get("enable_ee_action_filter", True)
        self.ee_action_filter_alpha_pos: float = robot.get("ee_action_filter_alpha_pos", 0.35)
        self.ee_action_filter_alpha_rot: float = robot.get("ee_action_filter_alpha_rot", 0.25)
        self.enable_ee_action_deadband: bool = robot.get("enable_ee_action_deadband", True)
        self.ee_action_deadband_pos_norm: float = robot.get("ee_action_deadband_pos_norm", 0.00035)
        self.ee_action_deadband_rot_norm: float = robot.get("ee_action_deadband_rot_norm", 0.0025)
        
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
            self.action_smoothing_alpha = oculus_cfg.get("action_smoothing_alpha", 0.35)
            if self.dual_arm:
                self.left_pose_scaler = oculus_cfg.get("left_pose_scaler", self.pose_scaler)
                self.right_pose_scaler = oculus_cfg.get("right_pose_scaler", self.pose_scaler)
                self.left_channel_signs = oculus_cfg.get("left_channel_signs", self.channel_signs)
                self.right_channel_signs = oculus_cfg.get("right_channel_signs", self.channel_signs)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")
    
    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """Parse policy configuration."""
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

        policy_type = policy["type"]
        if policy_type == "act":
            from lerobot.policies import ACTConfig

            temporal_ensemble_coeff = normalize_temporal_ensemble_coeff(
                policy.get("temporal_ensemble_coeff")
            )
            self.policy = ACTConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                chunk_size=policy.get("chunk_size", 100),
                n_action_steps=policy.get("n_action_steps", 100),
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
                    action_smoothing_alpha=self.action_smoothing_alpha,
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


def run_record(record_cfg: RecordConfig, enable_profile_latency: bool = False, profile_csv: str | None = None):
    print("====== [START] Starting recording ======")
    profiler = _RuntimeRecordLatencyProfiler(int(record_cfg.fps), profile_csv) if enable_profile_latency else None

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
            enable_ee_action_filter=record_cfg.enable_ee_action_filter,
            ee_action_filter_alpha_pos=record_cfg.ee_action_filter_alpha_pos,
            ee_action_filter_alpha_rot=record_cfg.ee_action_filter_alpha_rot,
            enable_ee_action_deadband=record_cfg.enable_ee_action_deadband,
            ee_action_deadband_pos_norm=record_cfg.ee_action_deadband_pos_norm,
            ee_action_deadband_rot_norm=record_cfg.ee_action_deadband_rot_norm,
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

        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        _, events = init_keyboard_listener()
        if record_cfg.display:
            init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        if profiler is not None:
            teleop_action_processor = profiler.wrap_callable(teleop_action_processor, "teleop_action_processor")
            robot_action_processor = profiler.wrap_callable(robot_action_processor, "robot_action_processor")
            robot_observation_processor = profiler.wrap_callable(
                robot_observation_processor, "robot_observation_processor"
            )
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
        if profiler is not None:
            profiler.install(robot=robot, teleop=teleop, dataset=dataset)

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
        if profiler is not None:
            profiler.close()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        if profiler is not None:
            profiler.close()
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        if profiler is not None:
            profiler.close()
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run dual-arm teleoperation recording.")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to record config YAML (default: scripts/config/record_cfg.yaml)",
    )
    parser.add_argument(
        "--profile-latency",
        action="store_true",
        help="Enable runtime latency profiling during robot-record (writes TXT+CSV under logs/).",
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
        help="Optional CSV output path when --profile-latency is enabled.",
    )
    args = parser.parse_args()

    parent_path = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(
        record_cfg,
        enable_profile_latency=args.profile_latency,
        profile_csv=args.profile_csv,
    )

if __name__ == "__main__":
    main()
