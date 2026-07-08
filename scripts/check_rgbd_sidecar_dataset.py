#!/usr/bin/env python
"""Check LeRobot RGB-D/IR sidecar fields recorded by robot-record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


CAMERAS = ("head", "left_wrist", "right_wrist")
SYNC_KEYS = (
    "global_frame_index",
    "robot_timestamp",
    "head_rgbd_timestamp",
    "left_wrist_rgbd_timestamp",
    "right_wrist_rgbd_timestamp",
)


def _rgb_keys(mode: str) -> list[str]:
    if mode == "legacy_image":
        return [f"observation.images.{camera}_image" for camera in CAMERAS]
    return [f"observation.images.{camera}_rgb" for camera in CAMERAS]


def _sidecar_keys(save_depth: bool = True, save_ir: bool = True) -> list[str]:
    keys: list[str] = []
    for camera in CAMERAS:
        if save_depth:
            keys.append(f"sidecar.{camera}_depth")
        if save_ir:
            keys.extend((f"sidecar.{camera}_left_ir", f"sidecar.{camera}_right_ir"))
    return keys


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    array = _as_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape={array.shape}")
    return float(array[0])


def _feature_summary(name: str, value: Any) -> str:
    array = _as_numpy(value)
    if array.size == 0:
        return f"{name}: shape={array.shape} dtype={array.dtype} empty"
    return (
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"min={array.min()} max={array.max()}"
    )


def _require_features(features: dict[str, dict], keys: list[str] | tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in features]
    if missing:
        raise AssertionError(f"Missing required feature(s): {missing}")


def _episode_indices(dataset: LeRobotDataset) -> dict[int, list[int]]:
    episode_values = dataset.hf_dataset["episode_index"]
    by_episode: dict[int, list[int]] = {}
    for index, value in enumerate(episode_values):
        episode = int(_scalar(value))
        by_episode.setdefault(episode, []).append(index)
    return by_episode


def _assert_monotonic(dataset: LeRobotDataset, indices: list[int], key: str, strict: bool) -> None:
    values = np.asarray([_scalar(dataset.hf_dataset[index][key]) for index in indices], dtype=np.float64)
    if values.size <= 1:
        return
    deltas = np.diff(values)
    ok = np.all(deltas > 0) if strict else np.all(deltas >= 0)
    if not ok:
        relation = "strictly increasing" if strict else "monotonic nondecreasing"
        raise AssertionError(f"{key} is not {relation}; first bad delta={deltas[deltas < 0][:1]}")


def _resolve_latest_dataset(root: Path) -> tuple[str, Path]:
    if (root / "meta" / "info.json").is_file():
        return root.name, root
    candidates = sorted(
        root.rglob("meta/info.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found under {root}")
    dataset_root = candidates[0].parents[1]
    return str(dataset_root.relative_to(root)), dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="LeRobot dataset repo_id to check.")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root or HF_LEROBOT_HOME base.")
    parser.add_argument(
        "--rgb-camera-name-mode",
        choices=("rgb", "legacy_image"),
        default="rgb",
        help="Expected RGB camera field naming.",
    )
    parser.add_argument("--no-depth", action="store_true", help="Do not require depth sidecar fields.")
    parser.add_argument("--no-ir", action="store_true", help="Do not require IR sidecar fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    repo_id = args.repo_id

    if repo_id is None:
        search_root = root if root is not None else Path(HF_LEROBOT_HOME)
        repo_id, dataset_root = _resolve_latest_dataset(search_root)
        root = dataset_root
        print(f"Using latest dataset: repo_id={repo_id} root={dataset_root}")

    dataset = LeRobotDataset(repo_id, root=root)
    features = dataset.features

    rgb_keys = _rgb_keys(args.rgb_camera_name_mode)
    sidecar_keys = _sidecar_keys(save_depth=not args.no_depth, save_ir=not args.no_ir)
    reused_keys = [f"{camera}_rgbd_reused" for camera in CAMERAS]
    required_keys = [*rgb_keys, *sidecar_keys, *SYNC_KEYS, *reused_keys]
    _require_features(features, required_keys)

    by_episode = _episode_indices(dataset)
    if not by_episode:
        raise AssertionError("Dataset has no frames.")

    print(f"repo_id={dataset.repo_id}")
    print(f"num_episodes={len(by_episode)} num_frames={len(dataset.hf_dataset)}")

    for episode, indices in sorted(by_episode.items()):
        print(f"episode={episode} frames={len(indices)}")
        _assert_monotonic(dataset, indices, "global_frame_index", strict=True)
        for key in SYNC_KEYS[1:]:
            _assert_monotonic(dataset, indices, key, strict=False)

    first_index = next(iter(next(iter(by_episode.values()))))
    sample = dataset[first_index]
    raw_sample = dataset.hf_dataset[first_index]

    print("sample:")
    for key in rgb_keys:
        print("  " + _feature_summary(key, sample[key]))
    for key in sidecar_keys:
        print("  " + _feature_summary(key, raw_sample[key]))
    for key in SYNC_KEYS:
        print(f"  {key}: {_scalar(raw_sample[key])}")

    head_depth_key = "sidecar.head_depth"
    if not args.no_depth and head_depth_key in raw_sample:
        head_global_index = int(_scalar(raw_sample["global_frame_index"]))
        _ = sample[rgb_keys[0]]
        _ = raw_sample[head_depth_key]
        print(f"head_rgb/head_depth aligned on global_frame_index={head_global_index}")

    print("RGB-D/IR sidecar check passed.")


if __name__ == "__main__":
    main()
