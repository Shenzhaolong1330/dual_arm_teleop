import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_flexiv_dual_arm_class():
    package_dir = Path(__file__).parents[1] / "robots" / "dual_flexiv_rizon4s"
    package_name = "_test_dual_flexiv_rizon4s"
    package = ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package

    for module_name in ("config_flexiv", "flexiv_dual_arm"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, package_dir / f"{module_name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.flexiv_dual_arm"].FlexivDualArm


FlexivDualArm = _load_flexiv_dual_arm_class()


def _frame(frame_index: int) -> dict:
    return {
        "rgb": np.full((2, 3, 3), frame_index, dtype=np.uint8),
        "depth": np.full((2, 3), frame_index, dtype=np.uint16),
        "left_ir": np.full((2, 3), frame_index, dtype=np.uint8),
        "right_ir": np.full((2, 3), frame_index, dtype=np.uint8),
        "timestamp": float(frame_index),
        "frame_index": frame_index,
        "reused": False,
    }


def _robot(camera_names=("head_rgb",)) -> FlexivDualArm:
    robot = FlexivDualArm.__new__(FlexivDualArm)
    robot.config = SimpleNamespace(
        save_depth_sidecar=True,
        save_ir_sidecar=True,
        save_rgbd_timestamps=True,
        camera_read_timeout_ms=200,
    )
    robot.cameras = {name: object() for name in camera_names}
    robot._frame_lock = threading.Lock()
    robot._latest_frames = {}
    robot._last_published_camera_frame_indices = {}
    return robot


def _publish(robot: FlexivDualArm) -> dict:
    observation = {}
    robot._add_camera_observations(observation)
    return observation


def test_rgbd_reused_tracks_consumed_frame_index_per_camera():
    robot = _robot(("head_rgb", "left_wrist_rgb"))

    robot._latest_frames = {"head_rgb": _frame(10), "left_wrist_rgb": _frame(10)}
    first = _publish(robot)
    assert first["head_rgbd_reused"] is False
    assert first["left_wrist_rgbd_reused"] is False

    robot._latest_frames = {"head_rgb": _frame(11), "left_wrist_rgb": _frame(10)}
    second = _publish(robot)
    assert second["head_rgbd_reused"] is False
    assert second["left_wrist_rgbd_reused"] is True

    repeated = _publish(robot)
    assert repeated["head_rgbd_reused"] is True
    assert repeated["left_wrist_rgbd_reused"] is True

    robot._latest_frames = {"head_rgb": _frame(12), "left_wrist_rgb": _frame(11)}
    next_new = _publish(robot)
    assert next_new["head_rgbd_reused"] is False
    assert next_new["left_wrist_rgbd_reused"] is False

    robot._mark_latest_camera_frame_reused("head_rgb")
    failed_read_reuse = _publish(robot)
    assert failed_read_reuse["head_rgbd_reused"] is True
    assert failed_read_reuse["left_wrist_rgbd_reused"] is True


def test_rgbd_reused_state_reset_treats_current_frame_as_new():
    robot = _robot()
    robot._latest_frames = {"head_rgb": _frame(5)}

    assert _publish(robot)["head_rgbd_reused"] is False
    assert _publish(robot)["head_rgbd_reused"] is True

    robot._reset_published_camera_frame_indices()

    assert _publish(robot)["head_rgbd_reused"] is False
