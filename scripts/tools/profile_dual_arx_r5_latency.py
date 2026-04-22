"""
Dual ARX R5 teleoperation latency profiler.

This tool runs a standalone recording session for ARX dual-arm + Oculus teleop,
collects per-stage latency statistics, and writes TXT + CSV reports.

Usage:
  python -m scripts.tools.profile_dual_arx_r5_latency \
      --config scripts/config/record_cfg.yaml \
      --duration 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class SessionArtifacts:
    report_txt: Path
    report_csv: Path


class StageLatencyCollector:
    """Collects stage latency samples and per-frame rows for reporting."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._frame_rows: list[dict[str, float]] = []
        self._current_frame: dict[str, float] | None = None

    def begin_frame(self, frame_idx: int) -> None:
        self._current_frame = {"frame_idx": float(frame_idx)}

    def end_frame(self) -> None:
        if self._current_frame is None:
            return
        self._frame_rows.append(self._current_frame)
        self._current_frame = None

    def record(self, stage_name: str, elapsed_ms: float) -> None:
        elapsed_ms = float(elapsed_ms)
        self._samples[stage_name].append(elapsed_ms)
        if self._current_frame is not None:
            self._current_frame[stage_name] = elapsed_ms

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage_name, (time.perf_counter() - start) * 1e3)

    def has_data(self) -> bool:
        return bool(self._samples)

    def stage_names(self) -> list[str]:
        return sorted(self._samples.keys())

    def stats(self, stage_name: str) -> dict[str, float] | None:
        values = self._samples.get(stage_name)
        if not values:
            return None
        arr = np.array(values, dtype=np.float64)
        return {
            "count": float(arr.size),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    def export_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(self._frame_rows)
        if not rows:
            logger.warning("[PROFILE] no frame data available for CSV export")
            return

        all_columns = {"frame_idx"}
        for row in rows:
            all_columns.update(row.keys())

        columns = ["frame_idx"] + sorted(col for col in all_columns if col != "frame_idx")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def format_report(self, fps: int, duration_s: float) -> str:
        target_ms = 1000.0 / fps
        lines: list[str] = []
        w = lines.append

        w("=" * 100)
        w("Dual ARX R5 Teleop Data Collection Latency Report")
        w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"Target FPS: {fps} (target period={target_ms:.2f} ms), session duration={duration_s:.1f}s")
        w("=" * 100)
        w(
            f"{'stage':<36} {'count':>7} {'min':>8} {'mean':>8} {'p50':>8} "
            f"{'p95':>8} {'p99':>8} {'max':>8}"
        )
        w("-" * 100)

        for stage_name in self.stage_names():
            s = self.stats(stage_name)
            if s is None:
                continue
            w(
                f"{stage_name:<36} {int(s['count']):>7d} {s['min']:>8.3f} {s['mean']:>8.3f} "
                f"{s['p50']:>8.3f} {s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f}"
            )

        w("-" * 100)
        loop_stats = self.stats("loop_total")
        if loop_stats is not None:
            loop_values = self._samples.get("loop_total", [])
            over_count = sum(1 for x in loop_values if x > target_ms)
            total = len(loop_values)
            over_ratio = (100.0 * over_count / total) if total else 0.0
            w(f"Loop overruns: {over_count}/{total} ({over_ratio:.1f}%)")

        w("")
        w("Top bottlenecks by p95:")
        candidates: list[tuple[str, float, float]] = []
        for stage_name in self.stage_names():
            if stage_name in {"loop_overrun_ms", "busy_wait_budget_ms"}:
                continue
            s = self.stats(stage_name)
            if s is None:
                continue
            candidates.append((stage_name, s["p95"], s["mean"]))
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            w("- No stage data collected.")
        else:
            for stage_name, p95, mean in candidates[:8]:
                pct = 100.0 * mean / target_ms if target_ms > 0 else 0.0
                w(f"- {stage_name}: p95={p95:.3f} ms, mean={mean:.3f} ms ({pct:.1f}% of period)")

        w("=" * 100)
        return "\n".join(lines) + "\n"


class RobotLatencyHook:
    """Bridge object passed to ArxDualArm.set_latency_hook."""

    def __init__(self, collector: StageLatencyCollector):
        self._collector = collector

    def __call__(self, stage_name: str, elapsed_ms: float) -> None:
        self._collector.record(stage_name, elapsed_ms)


def _generate_artifact_paths(base_dir: Path | str = "logs") -> SessionArtifacts:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(base_dir)
    return SessionArtifacts(
        report_txt=base / f"profile_{ts}.txt",
        report_csv=base / f"profile_{ts}.csv",
    )


def _build_camera_config(record_cfg, fps: int):
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

    left_wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.left_wrist_cam_serial,
        fps=fps,
        width=record_cfg.cam_width,
        height=record_cfg.cam_height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    right_wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.right_wrist_cam_serial,
        fps=fps,
        width=record_cfg.cam_width,
        height=record_cfg.cam_height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    head_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.head_cam_serial,
        fps=fps,
        width=record_cfg.cam_width,
        height=record_cfg.cam_height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )

    return {
        "left_wrist_image": left_wrist_image_cfg,
        "right_wrist_image": right_wrist_image_cfg,
        "head_image": head_image_cfg,
    }


def profiled_record_loop(
    *,
    robot,
    teleop,
    dataset,
    fps: int,
    duration_s: float,
    collector: StageLatencyCollector,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
) -> int:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import busy_wait

    frame_idx = 0
    start_t = time.perf_counter()
    target_ms = 1000.0 / fps

    while (time.perf_counter() - start_t) < duration_s:
        collector.begin_frame(frame_idx)
        loop_start = time.perf_counter()

        with collector.measure("robot_get_observation_total"):
            obs = robot.get_observation()

        with collector.measure("robot_observation_processor"):
            obs_processed = robot_observation_processor(obs)

        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        with collector.measure("teleop_get_action_total"):
            action = teleop.get_action()

        with collector.measure("teleop_action_processor"):
            action_processed = teleop_action_processor((action, obs))

        with collector.measure("robot_action_processor"):
            robot_action_to_send = robot_action_processor((action_processed, obs))

        with collector.measure("robot_send_action_total"):
            sent_action = robot.send_action(robot_action_to_send)

        action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
        frame = {**observation_frame, **action_frame, "task": "profiling_session"}

        with collector.measure("dataset_add_frame"):
            dataset.add_frame(frame)

        elapsed_before_wait_s = time.perf_counter() - loop_start
        budget_s = (1.0 / fps) - elapsed_before_wait_s
        collector.record("busy_wait_budget_ms", budget_s * 1e3)

        wait_start = time.perf_counter()
        busy_wait(budget_s)
        collector.record("busy_wait_actual_ms", (time.perf_counter() - wait_start) * 1e3)

        loop_total_ms = (time.perf_counter() - loop_start) * 1e3
        collector.record("loop_total", loop_total_ms)
        collector.record("loop_overrun_ms", max(0.0, loop_total_ms - target_ms))

        collector.end_frame()
        frame_idx += 1

    return frame_idx


def run_profiled_session(config_path: Path, duration_s: float, csv_output: Path | None) -> SessionArtifacts:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from robots import create_robot, create_robot_config
    from scripts.core.run_record import RecordConfig
    from teleoperators import OculusTeleop

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    fps = int(record_cfg.fps)

    if record_cfg.robot_type != "arx_dual_arm":
        raise ValueError(
            f"This profiler only supports robot_type='arx_dual_arm', got '{record_cfg.robot_type}'."
        )

    if record_cfg.run_mode != "run_record":
        logger.info("[PROFILE] overriding run_mode to 'run_record' for latency profiling")

    collector = StageLatencyCollector()
    hook = RobotLatencyHook(collector)

    artifacts = _generate_artifact_paths()
    if csv_output is not None:
        artifacts = SessionArtifacts(report_txt=artifacts.report_txt, report_csv=csv_output)

    temp_repo_id = f"profiling/arx_latency_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dataset_path = Path(HF_LEROBOT_HOME) / temp_repo_id

    robot = None
    teleop = None
    dataset = None
    frame_count = 0

    try:
        camera_config = _build_camera_config(record_cfg, fps)
        teleop_config = record_cfg.create_teleop_config()

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
        robot = create_robot(record_cfg.robot_type, robot_config)
        teleop = OculusTeleop(teleop_config)

        if hasattr(robot, "set_latency_hook"):
            robot.set_latency_hook(hook)

        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}

        dataset = LeRobotDataset.create(
            repo_id=temp_repo_id,
            fps=fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        dataset.meta.metadata_buffer_size = 1

        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        logger.info("\n" + "=" * 72)
        logger.info("Starting ARX latency profiling session")
        logger.info(f"Config: {config_path}")
        logger.info(f"Duration: {duration_s:.1f}s | FPS: {fps}")
        logger.info("=" * 72 + "\n")

        robot.connect()
        teleop.connect()

        frame_count = profiled_record_loop(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=fps,
            duration_s=duration_s,
            collector=collector,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if frame_count > 0:
            dataset.save_episode()

    except KeyboardInterrupt:
        logger.info("\n[PROFILE] interrupted by Ctrl+C, finalizing report...")
    finally:
        if robot is not None and hasattr(robot, "set_latency_hook"):
            robot.set_latency_hook(None)

        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as e:  # pragma: no cover - best effort cleanup
                logger.warning(f"[PROFILE] teleop disconnect failed: {e}")

        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:  # pragma: no cover - best effort cleanup
                logger.warning(f"[PROFILE] robot disconnect failed: {e}")

        if dataset is not None:
            try:
                dataset.finalize()
            except Exception as e:  # pragma: no cover - best effort cleanup
                logger.warning(f"[PROFILE] dataset finalize failed: {e}")

        if temp_dataset_path.exists():
            shutil.rmtree(temp_dataset_path, ignore_errors=True)

    if not collector.has_data():
        raise RuntimeError("No profiling data collected. Session ended before first frame.")

    report = collector.format_report(fps=fps, duration_s=duration_s)
    print(report)

    artifacts.report_txt.parent.mkdir(parents=True, exist_ok=True)
    artifacts.report_txt.write_text(report, encoding="utf-8")
    collector.export_csv(artifacts.report_csv)

    logger.info(f"[PROFILE] frames collected: {frame_count}")
    logger.info(f"[PROFILE] report txt: {artifacts.report_txt}")
    logger.info(f"[PROFILE] report csv: {artifacts.report_csv}")

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual ARX R5 teleop latency profiler")
    parser.add_argument(
        "--config",
        "-c",
        default=str(Path(__file__).resolve().parent.parent / "config" / "record_cfg.yaml"),
        help="Path to record_cfg.yaml",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Profiling duration in seconds",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path (default: logs/profile_<timestamp>.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    run_profiled_session(Path(args.config), args.duration, csv_path)


if __name__ == "__main__":
    main()
