#!/usr/bin/env python
"""Utilities for exporting RGB-D/IR sidecar frames from LeRobot datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = ("head", "left_wrist", "right_wrist")


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def resolve_dataset(repo_id: str | None, root: Path | None) -> LeRobotDataset:
    if root is not None:
        root = root.expanduser()

    if root is not None and (root / "meta" / "info.json").is_file():
        return LeRobotDataset(repo_id or root.name, root=root)

    if repo_id and root is not None:
        candidate = root / repo_id
        if (candidate / "meta" / "info.json").is_file():
            return LeRobotDataset(repo_id, root=candidate)

    if repo_id is None:
        raise ValueError("Pass --repo-id, or pass --root pointing directly to a LeRobot dataset root.")

    return LeRobotDataset(repo_id, root=root)


def camera_rgb_key(camera: str, rgb_camera_name_mode: str = "rgb") -> str:
    if rgb_camera_name_mode == "legacy_image":
        return f"observation.images.{camera}_image"
    if rgb_camera_name_mode != "rgb":
        raise ValueError(f"Unsupported rgb_camera_name_mode: {rgb_camera_name_mode!r}")
    return f"observation.images.{camera}_rgb"


def sidecar_keys(camera: str) -> dict[str, str]:
    return {
        "depth": f"sidecar.{camera}_depth",
        "left_ir": f"sidecar.{camera}_left_ir",
        "right_ir": f"sidecar.{camera}_right_ir",
    }


def require_keys(features: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in features]
    if missing:
        raise KeyError(f"Dataset is missing required feature(s): {missing}")


def global_index_for_episode_frame(dataset: LeRobotDataset, episode_index: int, frame_index: int) -> int:
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    if episode_index < 0 or episode_index >= dataset.num_episodes:
        raise ValueError(
            f"episode_index {episode_index} is out of range. Dataset has {dataset.num_episodes} episode(s)."
        )

    start = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    end = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    episode_length = end - start
    if frame_index >= episode_length:
        raise ValueError(
            f"frame_index {frame_index} is out of range for episode {episode_index}; "
            f"episode length is {episode_length}."
        )
    return start + frame_index


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def scalar_or_none(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    array = as_numpy(value).reshape(-1)
    if array.size == 0:
        return None
    item = array[0].item() if hasattr(array[0], "item") else array[0]
    if isinstance(item, np.generic):
        item = item.item()
    return item


def rgb_to_uint8_hwc(value: Any) -> np.ndarray:
    image = as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got shape={image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image * 255.0 if float(np.nanmax(image)) <= 1.5 else image
    return np.clip(image, 0, 255).astype(np.uint8)


def grayscale_to_uint8(value: Any) -> np.ndarray:
    image = as_numpy(value)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected single-channel image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def depth_to_colormap(value: Any) -> np.ndarray:
    depth = np.squeeze(as_numpy(value)).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape HxW, got {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        low = float(np.nanpercentile(depth[valid], 1))
        high = float(np.nanpercentile(depth[valid], 99))
    else:
        low = float(np.nanmin(depth)) if depth.size else 0.0
        high = float(np.nanmax(depth)) if depth.size else 1.0
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    gray = (normalized * 255.0).astype(np.uint8)

    try:
        import cv2  # noqa: PLC0415

        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.repeat(gray[..., None], 3, axis=-1)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_output_dir(dataset: LeRobotDataset, tool_name: str) -> Path:
    return dataset.root / "outputs" / tool_name
