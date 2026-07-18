#!/usr/bin/env python

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from robots.dual_flexiv_rizon4s.flexiv_state_schema import (
    propagate_flexiv_dataset_schema,
    validate_flexiv_dataset_schema,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)["preprocess_dataset"]


def _as_path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _resolve_for_safety(left)
    right = _resolve_for_safety(right)
    return left == right or left in right.parents or right in left.parents


def _assert_output_is_separate_from_source(source: LeRobotDataset, cfg: dict[str, Any]) -> None:
    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is None:
        if str(output_cfg.get("repo_id", "")).strip("/") == str(source.repo_id).strip("/"):
            raise ValueError(
                "Refusing to preprocess in place: output.repo_id is the same as source.repo_id "
                "and output.root is not set. Choose a separate output.root/repo_id."
            )
        return

    source_root = _resolve_for_safety(Path(source.root))
    if _paths_overlap(output_root, source_root):
        raise ValueError(
            "Refusing to preprocess in place: output.root overlaps source.root. "
            f"source.root={source_root} output.root={_resolve_for_safety(output_root)}"
        )


def _select_episodes(dataset: LeRobotDataset, cfg: dict[str, Any]) -> list[int]:
    episodes = cfg["source"].get("episodes")
    if episodes is None:
        episodes = list(range(dataset.meta.total_episodes))
    else:
        episodes = [int(ep) for ep in episodes]

    max_episodes = cfg["source"].get("max_episodes")
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    return episodes


def _indices_matching(action_names: list[str], suffixes: set[str]) -> list[int]:
    return [
        idx
        for idx, name in enumerate(action_names)
        if "delta_ee_pose" in name and name.rsplit(".", 1)[-1] in suffixes
    ]


def _gripper_indices(action_names: list[str]) -> list[int]:
    return [
        idx
        for idx, name in enumerate(action_names)
        if "gripper" in name and not _force_or_wrench_name(name)
    ]


def _force_or_wrench_name(name: str) -> bool:
    lower = name.lower()
    return "force" in lower or "wrench" in lower


def _expanded_mask(event_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not event_mask.any():
        return event_mask.copy()

    keep = event_mask.copy()
    event_indices = np.flatnonzero(event_mask)
    for idx in event_indices:
        start = max(0, idx - radius)
        end = min(len(keep), idx + radius + 1)
        keep[start:end] = True
    return keep


def _gripper_event_mask(
    actions: np.ndarray,
    action_names: list[str],
    states: np.ndarray | None,
    state_names: list[str],
    cfg: dict[str, Any],
) -> np.ndarray:
    gripper_cfg = cfg.get("gripper_events", {}) or {}
    if not gripper_cfg.get("enabled", False):
        return np.zeros(actions.shape[0], dtype=bool)

    signal = None
    indices: list[int] = []
    if states is not None and state_names:
        state_array = np.asarray(states, dtype=np.float32)
        if state_array.ndim == 1:
            state_array = state_array[:, None]
        indices = [
            idx
            for idx, name in enumerate(state_names)
            if "gripper" in name.lower() and not _force_or_wrench_name(name)
            and idx < state_array.shape[1]
        ]
        if indices:
            signal = state_array

    # Keep compatibility for datasets without a usable observation.state, but
    # never let an action command override a real gripper width/state signal.
    if signal is None:
        indices = _gripper_indices(action_names)
        if indices and actions.shape[0] > 0:
            signal = actions

    if signal is None or not indices or signal.shape[0] == 0:
        return np.zeros(actions.shape[0], dtype=bool)

    threshold = float(gripper_cfg.get("change_threshold", 0.5))
    diffs = np.abs(np.diff(signal[:, indices], axis=0))
    event_mask = np.zeros(actions.shape[0], dtype=bool)
    event_mask[1:] = (diffs >= threshold).any(axis=1)
    event_mask[:-1] |= (diffs >= threshold).any(axis=1)
    return _expanded_mask(event_mask, int(gripper_cfg.get("keep_radius_frames", 15)))


def _feature_indices(
    names: list[str],
    *,
    include: tuple[str, ...] = (),
    suffixes: set[str] | None = None,
    exclude: tuple[str, ...] = (),
) -> list[int]:
    suffixes = suffixes or set()
    indices: list[int] = []
    for idx, name in enumerate(names):
        lower = name.lower()
        if exclude and any(token in lower for token in exclude):
            continue
        if include and not all(token in lower for token in include):
            continue
        if suffixes and lower.rsplit(".", 1)[-1] not in suffixes:
            continue
        indices.append(idx)
    return indices


def _state_rate_motion_mask(
    states: np.ndarray | None,
    state_names: list[str],
    fps: float,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    trim_cfg = cfg.get("static_trim", {}) or {}
    state_cfg = trim_cfg.get("state_rate", {}) or {}
    if states is None or states.size == 0:
        return None

    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[:, None]
    if states.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    diffs = np.zeros_like(states, dtype=np.float32)
    if states.shape[0] > 1:
        diffs[1:] = np.abs(np.diff(states, axis=0)) * max(float(fps), 1e-6)

    ignore_gripper = bool(state_cfg.get("ignore_gripper", False))
    include_force = bool(state_cfg.get("include_force", False))
    exclude = ("gripper",) if ignore_gripper else ()
    if not include_force:
        exclude = (*exclude, "force", "wrench")
    gripper_indices = (
        []
        if ignore_gripper
        else _feature_indices(
            state_names,
            include=("gripper",),
            exclude=exclude,
        )
    )
    translation_indices = _feature_indices(
        state_names,
        include=("ee_pose",),
        suffixes={"x", "y", "z"},
        exclude=exclude,
    )
    rotation_indices = _feature_indices(
        state_names,
        include=("ee_pose",),
        suffixes={"rx", "ry", "rz", "roll", "pitch", "yaw"},
        exclude=exclude,
    )
    joint_indices = _feature_indices(
        state_names,
        include=("joint",),
        exclude=exclude,
    )

    motion = np.zeros(states.shape[0], dtype=bool)

    def mark(indices: list[int], threshold_key: str, default: float) -> None:
        nonlocal motion
        if not indices:
            return
        norms = np.linalg.norm(diffs[:, indices], axis=1)
        active = norms > float(state_cfg.get(threshold_key, default))
        motion |= active
        motion[:-1] |= active[1:]

    mark(joint_indices, "joint_norm_threshold", 1e-6)
    mark(translation_indices, "translation_norm_threshold", 1e-6)
    mark(rotation_indices, "rotation_norm_threshold", 1e-6)
    mark(gripper_indices, "gripper_norm_threshold", 1e-6)

    if not (joint_indices or translation_indices or rotation_indices or gripper_indices):
        all_indices = [
            idx
            for idx, name in enumerate(state_names)
            if not (ignore_gripper and "gripper" in name.lower())
            and (include_force or not _force_or_wrench_name(name))
        ]
        if not state_names:
            all_indices = list(range(states.shape[1]))
        if all_indices:
            norms = np.linalg.norm(diffs[:, all_indices], axis=1)
            active = norms > float(state_cfg.get("norm_threshold", 1e-6))
            motion |= active
            motion[:-1] |= active[1:]

    return motion


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("action_smoothing.median_window must be odd.")

    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(values)
    for idx in range(values.shape[0]):
        out[idx] = np.median(padded[idx : idx + window], axis=0)
    return out


def _ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.shape[0] == 0:
        return values.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _smooth_actions(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    smoothing_cfg = cfg.get("action_smoothing", {}) or {}
    if not smoothing_cfg.get("enabled", False):
        return actions.copy()

    smoothed = actions.copy()
    indices: list[int] = []
    if smoothing_cfg.get("smooth_cartesian", True):
        indices.extend(_indices_matching(action_names, {"x", "y", "z", "rx", "ry", "rz"}))
    if smoothing_cfg.get("smooth_gripper", False):
        indices.extend(_gripper_indices(action_names))
    indices = sorted(set(indices))
    if not indices:
        return smoothed

    values = actions[:, indices]
    method = str(smoothing_cfg.get("method", "median")).lower()
    if method == "median":
        values = _median_filter(values, int(smoothing_cfg.get("median_window", 3)))
    elif method == "ema":
        values = _ema_filter(values, float(smoothing_cfg.get("ema_alpha", 0.35)))
    elif method == "median_ema":
        values = _median_filter(values, int(smoothing_cfg.get("median_window", 3)))
        values = _ema_filter(values, float(smoothing_cfg.get("ema_alpha", 0.35)))
    else:
        raise ValueError("action_smoothing.method must be one of: median, ema, median_ema")
    smoothed[:, indices] = values

    max_translation = smoothing_cfg.get("max_translation_delta")
    if max_translation is not None:
        max_translation = float(max_translation)
        for idx in _indices_matching(action_names, {"x", "y", "z"}):
            smoothed[:, idx] = np.clip(smoothed[:, idx], -max_translation, max_translation)

    max_rotation = smoothing_cfg.get("max_rotation_delta")
    if max_rotation is not None:
        max_rotation = float(max_rotation)
        for idx in _indices_matching(action_names, {"rx", "ry", "rz"}):
            smoothed[:, idx] = np.clip(smoothed[:, idx], -max_rotation, max_rotation)

    return smoothed


def _action_motion_mask(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    trim_cfg = cfg.get("static_trim", {}) or {}
    translation_indices = _indices_matching(action_names, {"x", "y", "z"})
    rotation_indices = _indices_matching(action_names, {"rx", "ry", "rz"})

    translation_norm = (
        np.linalg.norm(actions[:, translation_indices], axis=1)
        if translation_indices
        else np.zeros(actions.shape[0], dtype=np.float32)
    )
    rotation_norm = (
        np.linalg.norm(actions[:, rotation_indices], axis=1)
        if rotation_indices
        else np.zeros(actions.shape[0], dtype=np.float32)
    )
    return (translation_norm >= float(trim_cfg.get("translation_norm_threshold", 0.001))) | (
        rotation_norm >= float(trim_cfg.get("rotation_norm_threshold", 0.005))
    )


def _motion_mask(
    actions: np.ndarray,
    action_names: list[str],
    states: np.ndarray | None,
    state_names: list[str],
    fps: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, str]:
    trim_cfg = cfg.get("static_trim", {}) or {}
    motion_source = str(trim_cfg.get("motion_source", "state_rate")).lower()
    if motion_source in {"state", "state_rate", "observation.state"}:
        state_motion = _state_rate_motion_mask(states, state_names, fps, cfg)
        if state_motion is not None:
            return state_motion, "state_rate"
        logger.warning("[preprocess] observation.state unavailable; falling back to action delta motion.")

    return _action_motion_mask(actions, action_names, cfg), "action_delta"


def _trim_static_runs(moving_or_protected: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    trim_cfg = cfg.get("static_trim", {}) or {}
    if not trim_cfg.get("enabled", False):
        return np.ones_like(moving_or_protected, dtype=bool)

    min_static = int(trim_cfg.get("min_static_frames", 10))
    keep_start = int(trim_cfg.get("keep_start_frames", 5))
    keep_end = int(trim_cfg.get("keep_end_frames", 5))
    keep = moving_or_protected.copy()
    static = ~moving_or_protected

    idx = 0
    while idx < len(static):
        if not static[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(static) and static[idx]:
            idx += 1
        end = idx
        run_len = end - start
        if run_len < min_static:
            keep[start:end] = True
            continue
        keep[start : min(end, start + keep_start)] = True
        keep[max(start, end - keep_end) : end] = True
    return keep


def _episode_columns(dataset: LeRobotDataset, start: int, end: int, columns: list[str]) -> dict[str, np.ndarray]:
    raw_dataset = dataset.hf_dataset.with_format(None)
    batch = raw_dataset[start:end]
    out: dict[str, np.ndarray] = {}
    for key in columns:
        if key in batch:
            out[key] = np.asarray(batch[key], dtype=np.float32)
    return out


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def _coerce_numeric_feature(value: Any, feature: dict[str, Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value, dtype=np.dtype(feature["dtype"]))
    expected_shape = tuple(feature.get("shape", ()))
    if array.shape != expected_shape and array.size == int(np.prod(expected_shape, dtype=np.int64)):
        array = array.reshape(expected_shape)
    return array


def _restore_image_layout(value: Any, feature: dict[str, Any]) -> Any:
    array = _to_numpy(value)
    if not isinstance(array, np.ndarray):
        return array

    expected_shape = tuple(feature.get("shape", ()))
    if (
        array.ndim == 3
        and len(expected_shape) == 3
        and expected_shape[-1] in (1, 3, 4)
        and array.shape[0] == expected_shape[-1]
        and array.shape[1:] == expected_shape[:2]
    ):
        return np.transpose(array, (1, 2, 0))
    return array


def _frame_from_source_item(
    source: LeRobotDataset,
    item: dict[str, Any],
    action: np.ndarray,
) -> dict[str, Any]:
    frame = {}
    for key in source.features:
        feature = source.features[key]
        if key in DEFAULT_FEATURES:
            continue
        if key == "action":
            frame[key] = _coerce_numeric_feature(action, feature)
        elif feature["dtype"] in ["image", "video"]:
            frame[key] = _restore_image_layout(item[key], feature)
        elif feature["dtype"] == "string":
            frame[key] = item[key]
        else:
            frame[key] = _coerce_numeric_feature(item[key], feature)
    frame["task"] = item["task"]
    return frame


def _create_output_dataset(source: LeRobotDataset, cfg: dict[str, Any]) -> LeRobotDataset:
    _assert_output_is_separate_from_source(source, cfg)
    if source.meta.info.get("robot_type") == "flexiv_dual_arm":
        validate_flexiv_dataset_schema(
            source.meta.info,
            source.features,
            source="preprocess_dataset source",
        )
    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is not None and output_root.exists():
        if output_cfg.get("overwrite", False):
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                "Set output.overwrite=true to replace it."
            )

    output = LeRobotDataset.create(
        repo_id=output_cfg["repo_id"],
        root=output_root,
        fps=source.fps,
        features={
            key: value
            for key, value in source.features.items()
            if key not in DEFAULT_FEATURES
        },
        robot_type=source.meta.info.get("robot_type"),
        use_videos=len(source.meta.video_keys) > 0,
        image_writer_threads=int(output_cfg.get("image_writer_threads", 4)),
        batch_encoding_size=int(output_cfg.get("batch_encoding_size", 1)),
    )
    propagate_flexiv_dataset_schema(
        source.meta.info,
        output,
        source_features=source.features,
        output_features=output.features,
        source="preprocess_dataset",
    )
    return output


def preprocess_dataset(cfg: dict[str, Any]) -> None:
    source_cfg = cfg["source"]
    source = LeRobotDataset(
        source_cfg["repo_id"],
        root=_as_path_or_none(source_cfg.get("root")),
    )
    episodes = _select_episodes(source, cfg)
    action_names = source.features["action"]["names"]
    state_names = source.features.get("observation.state", {}).get("names") or []
    state_names = [str(name) for name in state_names]
    dry_run = bool(cfg.get("dry_run", False))

    output = None if dry_run else _create_output_dataset(source, cfg)
    total_in = 0
    total_out = 0

    try:
        for new_ep_idx, ep_idx in enumerate(episodes):
            ep = source.meta.episodes[int(ep_idx)]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            arrays = _episode_columns(source, start, end, ["action", "observation.state"])
            actions = arrays["action"]
            states = arrays.get("observation.state")

            smoothed_actions = _smooth_actions(actions, action_names, cfg)
            gripper_keep = _gripper_event_mask(
                actions,
                action_names,
                states,
                state_names,
                cfg,
            )
            motion, motion_source = _motion_mask(
                smoothed_actions,
                action_names,
                states,
                state_names,
                float(source.fps),
                cfg,
            )
            keep_mask = _trim_static_runs(motion | gripper_keep, cfg)
            keep_indices = np.flatnonzero(keep_mask) + start

            total_in += end - start
            total_out += len(keep_indices)
            logger.info(
                "[EP %s -> %s] frames %d -> %d (%.1f%% kept), motion=%s %.1f%% gripper_keep=%.1f%%",
                ep_idx,
                new_ep_idx,
                end - start,
                len(keep_indices),
                100.0 * len(keep_indices) / max(end - start, 1),
                motion_source,
                100.0 * float(motion.mean()) if len(motion) else 0.0,
                100.0 * float(gripper_keep.mean()) if len(gripper_keep) else 0.0,
            )

            if dry_run:
                continue

            for source_idx in keep_indices:
                local_idx = int(source_idx - start)
                item = source[int(source_idx)]
                frame = _frame_from_source_item(source, item, smoothed_actions[local_idx])
                output.add_frame(frame)
            output.save_episode()
    finally:
        if output is not None:
            output.finalize()

    logger.info(
        "[DONE] frames %d -> %d (%.1f%% kept)%s",
        total_in,
        total_out,
        100.0 * total_out / max(total_in, 1),
        " [dry-run]" if dry_run else f" output={cfg['output']['repo_id']}",
    )


def main() -> None:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "preprocess_dataset_cfg.yaml"
    parser = argparse.ArgumentParser(description="Preprocess a LeRobot dataset for ACT training.")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--dry-run", action="store_true", help="Only report frame counts; do not write output.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Override source.max_episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite=true.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.max_episodes is not None:
        cfg["source"]["max_episodes"] = args.max_episodes
    if args.overwrite:
        cfg["output"]["overwrite"] = True

    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
