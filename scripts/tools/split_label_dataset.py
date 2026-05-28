#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:  # Keep pure segmentation tests importable outside the LeRobot env.
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    torch = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    Image = None

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_FEATURES
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    LeRobotDataset = None
    DEFAULT_FEATURES = {
        "timestamp": {},
        "frame_index": {},
        "episode_index": {},
        "index": {},
        "task_index": {},
    }

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class ArmEvent:
    arm: str
    event: str
    frame: int
    value: float


@dataclass
class RawSegment:
    parent_episode: int
    segment_id: int
    stage_id: str
    active_arm: str
    start: int
    end: int
    core_start: int
    core_end: int
    close_frames: dict[str, int] = field(default_factory=dict)
    open_frames: dict[str, int] = field(default_factory=dict)
    source_segments: list[dict[str, Any]] = field(default_factory=list)
    semantic_label: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class LabelResult:
    canonical_instruction: str
    stage_label: str
    object: str
    target: str
    confidence: float
    variants: list[str]
    needs_review: bool
    source: str
    error: str | None = None
    arm: str = "unknown"
    approach_direction: str = "unknown"
    object_color: str = "unknown"
    object_size: str = "unknown"
    object_name: str = "object"
    object_description: str = "unknown object"
    action: str = "manipulate"
    target_direction: str = "unknown"
    target_color: str = "unknown"
    target_size: str = "unknown"
    target_name: str = "target"
    target_description: str = "unknown target"
    placement_relation: str = "at"


ARM_VALUES = ["left arm", "right arm", "both arms", "unknown"]
DIRECTION_VALUES = [
    "front",
    "back",
    "left",
    "right",
    "front-left",
    "front-right",
    "back-left",
    "back-right",
    "center",
    "up",
    "down",
    "unknown",
]
ACTION_VALUES = [
    "grasp",
    "pick up",
    "hold",
    "move",
    "place",
    "release",
    "push",
    "pull",
    "align",
    "insert",
    "remove",
    "open",
    "close",
    "seal",
    "unseal",
    "handover",
    "manipulate",
]
RELATION_VALUES = ["in", "inside", "on", "onto", "under", "over", "next to", "near", "against", "through", "at"]


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if "split_label_dataset" not in data:
        raise ValueError(f"Invalid config, missing split_label_dataset: {path}")
    return data["split_label_dataset"]


def _as_path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _resolve_for_safety(left)
    right = _resolve_for_safety(right)
    return left == right or left in right.parents or right in left.parents


def _path_inside(path: Path, root: Path) -> bool:
    path = _resolve_for_safety(path)
    root = _resolve_for_safety(root)
    return path == root or root in path.parents


def _assert_output_root_is_separate_from_source(source: Any, cfg: dict[str, Any]) -> None:
    output_cfg = cfg.get("output", {}) or {}
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is None:
        if str(output_cfg.get("repo_id", "")).strip("/") == str(source.repo_id).strip("/"):
            raise ValueError(
                "Refusing to split in place: output.repo_id is the same as source.repo_id "
                "and output.root is not set. Choose a separate output.root/repo_id."
            )
        return

    source_root = _resolve_for_safety(Path(source.root))
    if _paths_overlap(output_root, source_root):
        raise ValueError(
            "Refusing to split in place: output.root overlaps source.root. "
            f"source.root={source_root} output.root={_resolve_for_safety(output_root)}"
        )


def _assert_write_path_outside_source(source: Any, path: Path | None, label: str) -> None:
    if path is None:
        return
    source_root = _resolve_for_safety(Path(source.root))
    if _path_inside(path, source_root):
        raise ValueError(
            f"Refusing to write {label} inside the source dataset. "
            f"source.root={source_root} {label}={_resolve_for_safety(path)}"
        )


def _assert_no_source_writes(
    source: Any,
    cfg: dict[str, Any],
    *,
    write_dataset: bool,
    dry_run: bool,
    semantic_on_dry_run: bool,
) -> None:
    if write_dataset:
        _assert_output_root_is_separate_from_source(source, cfg)

    if not dry_run:
        _assert_write_path_outside_source(source, _manifest_dir(cfg), "manifests.dir")
        modality_path = _as_path_or_none((cfg.get("vla_export", {}) or {}).get("starvla_modality_path"))
        _assert_write_path_outside_source(source, modality_path, "vla_export.starvla_modality_path")

    if not dry_run or semantic_on_dry_run:
        cache_path = _as_path_or_none((cfg.get("openrouter", {}) or {}).get("cache_path"))
        _assert_write_path_outside_source(source, cache_path, "openrouter.cache_path")


def _require_lerobot() -> None:
    if LeRobotDataset is None:
        raise ModuleNotFoundError(
            "lerobot is not importable in this Python environment. "
            "Activate the project environment before running this tool."
        )


def _select_episodes(dataset: Any, cfg: dict[str, Any]) -> list[int]:
    episodes = cfg["source"].get("episodes")
    if episodes is None:
        episodes = list(range(dataset.meta.total_episodes))
    else:
        episodes = [int(ep) for ep in episodes]

    max_episodes = cfg["source"].get("max_episodes")
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    return episodes


def _feature_names(source: Any, key: str) -> list[str]:
    feature = source.features.get(key, {})
    names = feature.get("names") or []
    return [str(name) for name in names]


def _side_from_name(name: str, fallback_index: int) -> str:
    lower = name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return f"gripper_{fallback_index}"


def _gripper_indices(names: list[str], *, prefer_state: bool = False) -> list[int]:
    indices = [idx for idx, name in enumerate(names) if "gripper" in name.lower()]
    if not prefer_state:
        return indices

    state_like = [
        idx
        for idx in indices
        if any(token in names[idx].lower() for token in ("state", "pos", "position", "width", "open"))
        and "cmd" not in names[idx].lower()
    ]
    return state_like or indices


def _episode_columns(dataset: Any, start: int, end: int, columns: list[str]) -> dict[str, np.ndarray]:
    raw_dataset = dataset.hf_dataset.with_format(None)
    batch = raw_dataset[start:end]
    out: dict[str, np.ndarray] = {}
    for key in columns:
        if key in batch:
            out[key] = np.asarray(batch[key], dtype=np.float32)
    return out


def _extract_gripper_signal(
    arrays: dict[str, np.ndarray],
    action_names: list[str],
    state_names: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray | None, list[str], str]:
    split_cfg = cfg.get("segmentation", {}) or {}
    if split_cfg.get("prefer_action_gripper", True) and "action" in arrays:
        indices = _gripper_indices(action_names)
        if indices:
            side_names = [_side_from_name(action_names[idx], i) for i, idx in enumerate(indices)]
            return arrays["action"][:, indices], side_names, "action"

    if "observation.state" in arrays:
        indices = _gripper_indices(state_names, prefer_state=True)
        if indices:
            side_names = [_side_from_name(state_names[idx], i) for i, idx in enumerate(indices)]
            return arrays["observation.state"][:, indices], side_names, "observation.state"

    return None, [], "none"


def _motion_mask(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    split_cfg = cfg.get("segmentation", {}) or {}
    suffix_groups = {
        "translation": {"x", "y", "z"},
        "rotation": {"rx", "ry", "rz", "roll", "pitch", "yaw"},
    }

    def matching(suffixes: set[str]) -> list[int]:
        return [
            idx
            for idx, name in enumerate(action_names)
            if "delta_ee_pose" in name and name.rsplit(".", 1)[-1] in suffixes
        ]

    translation_indices = matching(suffix_groups["translation"])
    rotation_indices = matching(suffix_groups["rotation"])
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
    return (translation_norm >= float(split_cfg.get("motion_translation_threshold", 0.001))) | (
        rotation_norm >= float(split_cfg.get("motion_rotation_threshold", 0.005))
    )


def _action_filter_indices(action_names: list[str], action_dim: int, cfg: dict[str, Any]) -> list[int]:
    filter_cfg = cfg.get("action_filter", {}) or {}
    configured = filter_cfg.get("indices")
    if configured is not None:
        indices = [int(idx) for idx in configured]
        return [idx for idx in indices if 0 <= idx < action_dim]

    indices = list(range(action_dim))
    if filter_cfg.get("ignore_gripper", True) and action_names:
        non_gripper = [idx for idx, name in enumerate(action_names[:action_dim]) if "gripper" not in name.lower()]
        if non_gripper:
            indices = non_gripper
    return indices


def _action_activity_mask(actions: np.ndarray | None, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray | None:
    filter_cfg = cfg.get("action_filter", {}) or {}
    if actions is None or actions.size == 0:
        return None

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    indices = _action_filter_indices(action_names, int(actions.shape[1]), cfg)
    if not indices:
        return None

    selected = actions[:, indices]
    norms = np.linalg.norm(selected, axis=1)
    return norms > float(filter_cfg.get("norm_threshold", 1e-6))


def _filter_segments_by_action(
    segments: list[RawSegment],
    actions: np.ndarray | None,
    action_names: list[str],
    cfg: dict[str, Any],
) -> tuple[list[RawSegment], dict[str, int]]:
    filter_cfg = cfg.get("action_filter", {}) or {}
    stats = {
        "before": len(segments),
        "after": len(segments),
        "dropped_zero_action": 0,
        "trimmed": 0,
    }
    if not filter_cfg.get("enabled", True) or not segments:
        return segments, stats

    activity = _action_activity_mask(actions, action_names, cfg)
    if activity is None:
        return segments, stats

    fps = float(cfg.get("_fps", 30))
    keep_context = max(0, int(round(float(filter_cfg.get("keep_context_sec", 0.15)) * fps)))
    min_active_frames = max(0, int(filter_cfg.get("min_active_frames", 1)))
    min_active_ratio = float(filter_cfg.get("min_active_ratio", 0.0))
    min_len = max(1, int(round(float(filter_cfg.get("min_segment_sec_after_trim", 0.2)) * fps)))
    trim_edges = bool(filter_cfg.get("trim_leading_trailing", True))

    filtered: list[RawSegment] = []
    for segment in segments:
        start = max(0, min(len(activity), int(segment.start)))
        end = max(start, min(len(activity), int(segment.end)))
        if end <= start:
            stats["dropped_zero_action"] += 1
            continue

        local_active = activity[start:end]
        active_indices = np.flatnonzero(local_active)
        active_count = int(active_indices.size)
        active_ratio = active_count / max(1, end - start)
        if active_count < min_active_frames or active_ratio < min_active_ratio:
            stats["dropped_zero_action"] += 1
            continue

        if trim_edges:
            new_start = max(start, start + int(active_indices[0]) - keep_context)
            new_end = min(end, start + int(active_indices[-1]) + 1 + keep_context)
        else:
            new_start, new_end = start, end

        if new_end - new_start < min_len:
            stats["dropped_zero_action"] += 1
            continue

        if new_start != segment.start or new_end != segment.end:
            stats["trimmed"] += 1
            segment.source_segments.append(
                {
                    "source": "action_filter",
                    "original_start": int(segment.start),
                    "original_end": int(segment.end),
                    "trimmed_start": int(new_start),
                    "trimmed_end": int(new_end),
                    "active_frames": active_count,
                    "active_ratio": active_ratio,
                }
            )
            segment.start = int(new_start)
            segment.end = int(new_end)
            segment.core_start = max(segment.start, min(segment.end - 1, int(segment.core_start)))
            segment.core_end = max(segment.core_start, min(segment.end - 1, int(segment.core_end)))
        filtered.append(segment)

    for idx, segment in enumerate(filtered):
        segment.segment_id = idx
        segment.stage_id = f"episode_{segment.parent_episode:06d}_stage_{idx:03d}"

    stats["after"] = len(filtered)
    return filtered, stats


def _stable_state_events(
    values: np.ndarray,
    arm: str,
    *,
    open_threshold: float,
    closed_threshold: float,
    debounce_frames: int,
    close_is_low: bool = True,
) -> list[ArmEvent]:
    if values.size == 0:
        return []

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not close_is_low:
        values = 1.0 - values

    raw = np.full(values.shape[0], fill_value=-1, dtype=np.int8)
    raw[values >= open_threshold] = 1
    raw[values <= closed_threshold] = 0

    first_known = next((int(v) for v in raw if v >= 0), -1)
    current = first_known
    candidate = -1
    candidate_count = 0
    events: list[ArmEvent] = []

    for idx, state in enumerate(raw):
        state = int(state)
        if state < 0:
            candidate = -1
            candidate_count = 0
            continue
        if current < 0:
            current = state
            continue
        if state == current:
            candidate = -1
            candidate_count = 0
            continue
        if state != candidate:
            candidate = state
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= max(1, debounce_frames):
            event_frame = idx - candidate_count + 1
            event_name = "open" if candidate == 1 else "close"
            events.append(
                ArmEvent(
                    arm=arm,
                    event=event_name,
                    frame=int(event_frame),
                    value=float(values[event_frame]),
                )
            )
            current = candidate
            candidate = -1
            candidate_count = 0

    return events


def _arm_segments_from_events(
    parent_episode: int,
    arm: str,
    events: list[ArmEvent],
    episode_len: int,
    cfg: dict[str, Any],
) -> list[RawSegment]:
    split_cfg = cfg.get("segmentation", {}) or {}
    fps = float(cfg.get("_fps", split_cfg.get("fps", 30)))
    context_before = int(round(float(split_cfg.get("context_before_sec", 1.5)) * fps))
    context_after = int(round(float(split_cfg.get("context_after_sec", 1.5)) * fps))
    min_len = int(round(float(split_cfg.get("min_segment_sec", 1.0)) * fps))

    segments: list[RawSegment] = []
    pending_close: ArmEvent | None = None
    for event in events:
        if event.event == "close":
            pending_close = event
            continue
        if event.event != "open" or pending_close is None:
            continue

        start = max(0, pending_close.frame - context_before)
        end = min(episode_len, event.frame + context_after + 1)
        if end - start >= min_len:
            segments.append(
                RawSegment(
                    parent_episode=parent_episode,
                    segment_id=-1,
                    stage_id="",
                    active_arm=arm,
                    start=start,
                    end=end,
                    core_start=pending_close.frame,
                    core_end=event.frame,
                    close_frames={arm: pending_close.frame},
                    open_frames={arm: event.frame},
                    source_segments=[
                        {
                            "active_arm": arm,
                            "core_start": pending_close.frame,
                            "core_end": event.frame,
                            "close_frame": pending_close.frame,
                            "open_frame": event.frame,
                        }
                    ],
                )
            )
        pending_close = None

    return segments


def _motion_segments(
    parent_episode: int,
    moving: np.ndarray,
    episode_len: int,
    cfg: dict[str, Any],
) -> list[RawSegment]:
    split_cfg = cfg.get("segmentation", {}) or {}
    if not split_cfg.get("motion_fallback_enabled", True) or moving.size == 0:
        return []

    fps = float(cfg.get("_fps", split_cfg.get("fps", 30)))
    context = int(round(float(split_cfg.get("motion_context_sec", 1.0)) * fps))
    min_len = int(round(float(split_cfg.get("min_segment_sec", 1.0)) * fps))
    min_motion = int(round(float(split_cfg.get("min_motion_sec", 0.4)) * fps))

    segments: list[RawSegment] = []
    idx = 0
    while idx < len(moving):
        if not moving[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < len(moving) and moving[idx]:
            idx += 1
        run_end = idx
        if run_end - run_start < min_motion:
            continue
        start = max(0, run_start - context)
        end = min(episode_len, run_end + context)
        if end - start < min_len:
            continue
        segments.append(
            RawSegment(
                parent_episode=parent_episode,
                segment_id=-1,
                stage_id="",
                active_arm="motion",
                start=start,
                end=end,
                core_start=run_start,
                core_end=run_end - 1,
                source_segments=[
                    {
                        "active_arm": "motion",
                        "core_start": run_start,
                        "core_end": run_end - 1,
                    }
                ],
            )
        )
    return segments


def _merge_segments(
    segments: list[RawSegment],
    parent_episode: int,
    cfg: dict[str, Any],
) -> list[RawSegment]:
    if not segments:
        return []

    split_cfg = cfg.get("segmentation", {}) or {}
    fps = float(cfg.get("_fps", split_cfg.get("fps", 30)))
    merge_gap = int(round(float(split_cfg.get("merge_gap_sec", 0.5)) * fps))

    ordered = sorted(segments, key=lambda seg: (seg.start, seg.end))
    merged: list[RawSegment] = []
    for seg in ordered:
        if not merged or seg.start - merged[-1].end > merge_gap:
            merged.append(seg)
            continue

        current = merged[-1]
        current.end = max(current.end, seg.end)
        current.core_start = min(current.core_start, seg.core_start)
        current.core_end = max(current.core_end, seg.core_end)
        current.source_segments.extend(seg.source_segments)
        current.close_frames.update(seg.close_frames)
        current.open_frames.update(seg.open_frames)
        arms = sorted(
            {
                src.get("active_arm", current.active_arm)
                for src in current.source_segments
                if src.get("active_arm") != "motion"
            }
        )
        if len(arms) > 1:
            current.active_arm = "both"
        elif len(arms) == 1:
            current.active_arm = arms[0]

    for idx, seg in enumerate(merged):
        seg.segment_id = idx
        seg.stage_id = f"episode_{parent_episode:06d}_stage_{idx:03d}"
    return merged


def split_episode(
    parent_episode: int,
    gripper_values: np.ndarray | None,
    gripper_sides: list[str],
    actions: np.ndarray | None,
    action_names: list[str],
    cfg: dict[str, Any],
) -> list[RawSegment]:
    episode_len = 0
    if gripper_values is not None:
        episode_len = int(gripper_values.shape[0])
    elif actions is not None:
        episode_len = int(actions.shape[0])
    if episode_len <= 0:
        return []

    split_cfg = cfg.get("segmentation", {}) or {}
    segments: list[RawSegment] = []
    if gripper_values is not None and gripper_values.size > 0:
        for col, arm in enumerate(gripper_sides):
            events = _stable_state_events(
                gripper_values[:, col],
                arm,
                open_threshold=float(split_cfg.get("open_threshold", 0.75)),
                closed_threshold=float(split_cfg.get("closed_threshold", 0.25)),
                debounce_frames=int(split_cfg.get("debounce_frames", 5)),
                close_is_low=bool(split_cfg.get("close_is_low", True)),
            )
            segments.extend(_arm_segments_from_events(parent_episode, arm, events, episode_len, cfg))

    if not segments and actions is not None:
        segments.extend(_motion_segments(parent_episode, _motion_mask(actions, action_names, cfg), episode_len, cfg))

    return _merge_segments(segments, parent_episode, cfg)


def _template_instruction(segment: RawSegment, parent_task: str, total_segments: int) -> LabelResult:
    arm_text = {
        "left": "left arm",
        "right": "right arm",
        "both": "both arms",
        "motion": "unknown",
    }.get(segment.active_arm, segment.active_arm)
    index_text = f"subtask {segment.segment_id + 1} of {max(total_segments, 1)}"
    parent_task = parent_task or "the long-horizon task"
    canonical = f"{_arm_prefix(arm_text)}: manipulate the object."
    variants = [
        f"Use the {arm_text} to perform {index_text} for: {parent_task}.",
        f"Use the {arm_text} to continue the manipulation step for: {parent_task}.",
        f"Use the {arm_text} to complete this subtask toward: {parent_task}.",
    ]
    return LabelResult(
        canonical_instruction=canonical,
        stage_label="grasp_release_primitive",
        object="unknown",
        target="unknown",
        confidence=0.0,
        variants=variants,
        needs_review=True,
        source="template",
        arm=arm_text if arm_text in ARM_VALUES else "unknown",
    )


def _field(raw: dict[str, Any], key: str, fallback: str = "unknown") -> str:
    value = raw.get(key, fallback)
    value = str(value).strip().lower()
    return value or fallback


def _enum_field(raw: dict[str, Any], key: str, allowed: list[str], fallback: str = "unknown") -> str:
    value = _field(raw, key, fallback)
    aliases = {
        "left": "left arm",
        "right": "right arm",
        "both": "both arms",
        "dual": "both arms",
        "two arms": "both arms",
        "centre": "center",
        "middle": "center",
        "pickup": "pick up",
        "pick": "pick up",
        "grab": "grasp",
        "put": "place",
    }
    value = aliases.get(value, value)
    return value if value in allowed else fallback


def _description_from_parts(color: str, size: str, name: str, fallback: str) -> str:
    parts = [part for part in [size, color, name] if part and part != "unknown"]
    return " ".join(parts) if parts else fallback


def _strip_articles(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text or "unknown"


def _noun_phrase(text: str, fallback: str) -> str:
    text = _strip_articles(text)
    if text == "unknown" or not text:
        return f"the {fallback}"
    return f"the {text}"


def _arm_prefix(arm: str) -> str:
    arm = str(arm).strip().lower()
    if arm == "left arm":
        return "Left arm"
    if arm == "right arm":
        return "Right arm"
    if arm == "both arms":
        return "Both arms"
    return "Robot"


def _cap_action(action: str) -> str:
    action = str(action).strip().lower() or "manipulate"
    return action[:1].upper() + action[1:]


def _direction_phrase(direction: str, prefix: str) -> str:
    direction = str(direction).strip().lower()
    if not direction or direction == "unknown":
        return ""
    return f" {prefix} the {direction}"


def _place_phrase(relation: str, target: str, target_direction: str) -> str:
    return f"{relation} {target}{_direction_phrase(target_direction, 'at')}"


def _canonical_from_fields(fields: dict[str, str]) -> str:
    prefix = _arm_prefix(fields["arm"])
    obj = _noun_phrase(fields["object_description"], "object")
    target = _noun_phrase(fields["target_description"], "target")
    action = fields["action"]
    relation = fields["placement_relation"]
    from_direction = _direction_phrase(fields["approach_direction"], "from")
    place = _place_phrase(relation, target, fields["target_direction"])

    if action in {"pick up", "grasp", "hold"}:
        return f"{prefix}: pick up {obj}{from_direction} and place it {place}."
    if action in {"place", "release"}:
        return f"{prefix}: place {obj} {place}."
    if action in {"seal", "close", "open", "unseal", "align", "insert", "remove", "push", "pull"}:
        if target != obj:
            return f"{prefix}: {action} {obj} {relation} {target}{_direction_phrase(fields['target_direction'], 'at')}."
        return f"{prefix}: {action} {obj}{from_direction}."
    if action == "handover":
        return f"{prefix}: hand over {obj}{from_direction}."
    if action == "move":
        return f"{prefix}: move {obj} {place}."
    return f"{prefix}: manipulate {obj}{from_direction}."


def _variants_from_fields(fields: dict[str, str]) -> list[str]:
    prefix = _arm_prefix(fields["arm"])
    obj = _noun_phrase(fields["object_description"], "object")
    target = _noun_phrase(fields["target_description"], "target")
    action = fields["action"]
    relation = fields["placement_relation"]
    place = _place_phrase(relation, target, fields["target_direction"])

    if action in {"pick up", "grasp", "hold"}:
        return [
            _canonical_from_fields(fields),
            f"{prefix}: move {obj} {place}.",
            f"{prefix}: put {obj} {place}.",
        ]
    if action in {"seal", "close", "open", "unseal"}:
        return [
            _canonical_from_fields(fields),
            f"{prefix}: {_cap_action(action).lower()} {target}.",
            f"{prefix}: finish {_cap_action(action).lower()}ing {target}.",
        ]
    return [
        _canonical_from_fields(fields),
        f"{prefix}: {_cap_action(action).lower()} {obj}.",
        f"{prefix}: move {obj} {place}.",
    ]


def _normalize_language_fields(raw: dict[str, Any], fallback: LabelResult) -> dict[str, str]:
    arm = _enum_field(raw, "arm", ARM_VALUES, fallback.arm)
    approach_direction = _enum_field(raw, "approach_direction", DIRECTION_VALUES, fallback.approach_direction)
    object_color = _field(raw, "object_color", fallback.object_color)
    object_size = _field(raw, "object_size", fallback.object_size)
    object_name = _field(raw, "object_name", fallback.object_name)
    object_description = str(raw.get("object_description") or "").strip().lower()
    if not object_description:
        object_description = _description_from_parts(
            object_color, object_size, object_name, fallback.object_description
        )
    action = _enum_field(raw, "action", ACTION_VALUES, fallback.action)
    target_direction = _enum_field(raw, "target_direction", DIRECTION_VALUES, fallback.target_direction)
    target_color = _field(raw, "target_color", fallback.target_color)
    target_size = _field(raw, "target_size", fallback.target_size)
    target_name = _field(raw, "target_name", fallback.target_name)
    target_description = str(raw.get("target_description") or "").strip().lower()
    if not target_description:
        target_description = _description_from_parts(
            target_color, target_size, target_name, fallback.target_description
        )
    placement_relation = _enum_field(raw, "placement_relation", RELATION_VALUES, fallback.placement_relation)
    return {
        "arm": arm,
        "approach_direction": approach_direction,
        "object_color": object_color,
        "object_size": object_size,
        "object_name": object_name,
        "object_description": object_description,
        "action": action,
        "target_direction": target_direction,
        "target_color": target_color,
        "target_size": target_size,
        "target_name": target_name,
        "target_description": target_description,
        "placement_relation": placement_relation,
    }


def _normalize_label(raw: dict[str, Any], fallback: LabelResult, min_confidence: float) -> LabelResult:
    variants = raw.get("variants")
    if not isinstance(variants, list):
        variants = []
    variants = [str(item).strip() for item in variants if str(item).strip()]
    if len(variants) < 3:
        variants.extend(fallback.variants)
    variants = variants[:3]

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    language_fields = _normalize_language_fields(raw, fallback)
    has_structured_fields = any(key in raw for key in language_fields)
    canonical_instruction = (
        _canonical_from_fields(language_fields)
        if has_structured_fields
        else str(raw.get("canonical_instruction") or fallback.canonical_instruction).strip()
    )
    if has_structured_fields:
        variants = _variants_from_fields(language_fields)

    label = LabelResult(
        canonical_instruction=canonical_instruction,
        stage_label=str(raw.get("stage_label") or fallback.stage_label).strip(),
        object=str(raw.get("object") or raw.get("object_description") or fallback.object).strip(),
        target=str(raw.get("target") or raw.get("target_description") or fallback.target).strip(),
        confidence=confidence,
        variants=variants,
        needs_review=bool(raw.get("needs_review", False)) or confidence < min_confidence,
        source="openrouter",
        **language_fields,
    )
    if not label.canonical_instruction:
        return fallback
    return label


def _to_numpy(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


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
    source: Any,
    item: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    frame = {}
    for key in source.features:
        if key in DEFAULT_FEATURES:
            continue
        if source.features[key]["dtype"] in ["image", "video"]:
            frame[key] = _restore_image_layout(item[key], source.features[key])
        else:
            frame[key] = _to_numpy(item[key])
    frame["task"] = task
    return frame


def _camera_keys_for_segment(source: Any, segment: RawSegment, cfg: dict[str, Any]) -> list[str]:
    key_cfg = cfg.get("keyframes", {}) or {}
    available = set(getattr(source.meta, "camera_keys", [])) or {
        key for key, feature in source.features.items() if feature.get("dtype") in ("image", "video")
    }
    head_key = key_cfg.get("head_camera_key", "observation.images.head_image")
    left_key = key_cfg.get("left_wrist_camera_key", "observation.images.left_wrist_image")
    right_key = key_cfg.get("right_wrist_camera_key", "observation.images.right_wrist_image")

    keys = [head_key]
    if segment.active_arm == "left":
        keys.append(left_key)
    elif segment.active_arm == "right":
        keys.append(right_key)
    else:
        keys.extend([left_key, right_key])
    return [key for key in dict.fromkeys(keys) if key in available]


def _keyframe_indices(segment: RawSegment, episode_start: int, episode_len: int, cfg: dict[str, Any]) -> dict[str, int]:
    key_cfg = cfg.get("keyframes", {}) or {}
    fps = float(cfg.get("_fps", 30))
    pre_offset = int(round(float(key_cfg.get("pre_close_sec", 0.5)) * fps))
    return {
        "pre_interaction": episode_start + max(0, min(episode_len - 1, segment.core_start - pre_offset)),
        "grasp": episode_start + max(0, min(episode_len - 1, segment.core_start)),
        "release": episode_start + max(0, min(episode_len - 1, segment.core_end)),
    }


def _image_to_jpeg_bytes(value: Any, feature: dict[str, Any], quality: int) -> bytes:
    if Image is None:
        raise ModuleNotFoundError("Pillow is required for OpenRouter image labeling.")

    value = _restore_image_layout(value, feature)
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            if array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        image = Image.fromarray(array).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def _collect_keyframe_images(
    source: Any,
    segment: RawSegment,
    episode_start: int,
    episode_len: int,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    key_cfg = cfg.get("keyframes", {}) or {}
    camera_keys = _camera_keys_for_segment(source, segment, cfg)
    indices = _keyframe_indices(segment, episode_start, episode_len, cfg)
    quality = int(key_cfg.get("jpeg_quality", 85))

    images: list[dict[str, str]] = []
    manifest: dict[str, Any] = {
        "indices": indices,
        "camera_keys": camera_keys,
        "hashes": [],
    }
    for role, source_idx in indices.items():
        item = source[int(source_idx)]
        for camera_key in camera_keys:
            if camera_key not in item:
                continue
            image_bytes = _image_to_jpeg_bytes(item[camera_key], source.features[camera_key], quality)
            digest = hashlib.sha256(image_bytes).hexdigest()
            manifest["hashes"].append(
                {
                    "role": role,
                    "camera_key": camera_key,
                    "source_index": int(source_idx),
                    "sha256": digest,
                }
            )
            images.append(
                {
                    "role": role,
                    "camera_key": camera_key,
                    "source_index": str(source_idx),
                    "sha256": digest,
                    "data_url": "data:image/jpeg;base64,"
                    + base64.b64encode(image_bytes).decode("ascii"),
                }
            )
    return images, manifest


def _semantic_camera_keys(source: Any, cfg: dict[str, Any]) -> list[str]:
    sem_cfg = cfg.get("semantic_segmentation", {}) or {}
    key_cfg = cfg.get("keyframes", {}) or {}
    available = set(getattr(source.meta, "camera_keys", [])) or {
        key for key, feature in source.features.items() if feature.get("dtype") in ("image", "video")
    }
    configured = sem_cfg.get("camera_keys")
    if configured == "all":
        keys = sorted(available)
    elif configured:
        keys = [str(key) for key in configured]
    else:
        keys = [key_cfg.get("head_camera_key", "observation.images.head_image")]
    return [key for key in dict.fromkeys(keys) if key in available]


def _semantic_sample_local_indices(episode_len: int, cfg: dict[str, Any]) -> list[int]:
    if episode_len <= 0:
        return []
    sem_cfg = cfg.get("semantic_segmentation", {}) or {}
    fps = float(cfg.get("_fps", 30))
    max_frames = max(1, int(sem_cfg.get("max_sample_frames", 18)))
    interval = max(1, int(round(float(sem_cfg.get("sample_interval_sec", 2.0)) * fps)))

    indices = list(range(0, episode_len, interval))
    if not indices or indices[-1] != episode_len - 1:
        indices.append(episode_len - 1)
    if len(indices) > max_frames:
        indices = sorted({int(round(v)) for v in np.linspace(0, episode_len - 1, max_frames)})
    return [max(0, min(episode_len - 1, int(idx))) for idx in indices]


def _collect_episode_overview_images(
    source: Any,
    episode_start: int,
    episode_len: int,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    sem_cfg = cfg.get("semantic_segmentation", {}) or {}
    key_cfg = cfg.get("keyframes", {}) or {}
    camera_keys = _semantic_camera_keys(source, cfg)
    local_indices = _semantic_sample_local_indices(episode_len, cfg)
    quality = int(sem_cfg.get("jpeg_quality", key_cfg.get("jpeg_quality", 85)))

    images: list[dict[str, str]] = []
    manifest: dict[str, Any] = {
        "local_indices": local_indices,
        "source_indices": [int(episode_start + idx) for idx in local_indices],
        "camera_keys": camera_keys,
        "hashes": [],
    }
    for local_idx in local_indices:
        source_idx = int(episode_start + local_idx)
        item = source[source_idx]
        for camera_key in camera_keys:
            if camera_key not in item:
                continue
            image_bytes = _image_to_jpeg_bytes(item[camera_key], source.features[camera_key], quality)
            digest = hashlib.sha256(image_bytes).hexdigest()
            role = f"overview_frame_{local_idx:06d}"
            manifest["hashes"].append(
                {
                    "role": role,
                    "camera_key": camera_key,
                    "local_frame": int(local_idx),
                    "source_index": int(source_idx),
                    "sha256": digest,
                }
            )
            images.append(
                {
                    "role": role,
                    "camera_key": camera_key,
                    "local_frame": str(local_idx),
                    "source_index": str(source_idx),
                    "sha256": digest,
                    "data_url": "data:image/jpeg;base64,"
                    + base64.b64encode(image_bytes).decode("ascii"),
                }
            )
    return images, manifest


def _auxiliary_segments_payload(segments: list[RawSegment]) -> list[dict[str, Any]]:
    return [
        {
            "active_arm": segment.active_arm,
            "start_frame": int(segment.start),
            "end_frame": int(segment.end),
            "core_start_frame": int(segment.core_start),
            "core_end_frame": int(segment.core_end),
            "close_frames": segment.close_frames,
            "open_frames": segment.open_frames,
        }
        for segment in segments
    ]


def _frame_from_semantic_item(
    item: dict[str, Any],
    *,
    frame_keys: tuple[str, ...],
    time_keys: tuple[str, ...],
    fps: float,
    default: int,
) -> int:
    for key in frame_keys:
        if key in item and item[key] is not None:
            return int(round(float(item[key])))
    for key in time_keys:
        if key in item and item[key] is not None:
            return int(round(float(item[key]) * fps))
    return int(default)


def _normalize_semantic_segments(
    raw: dict[str, Any] | list[Any],
    parent_episode: int,
    episode_len: int,
    cfg: dict[str, Any],
) -> list[RawSegment]:
    raw_segments = raw.get("segments") if isinstance(raw, dict) else raw
    if not isinstance(raw_segments, list):
        return []

    sem_cfg = cfg.get("semantic_segmentation", {}) or {}
    fps = float(cfg.get("_fps", 30))
    min_len = max(1, int(round(float(sem_cfg.get("min_segment_sec", 1.0)) * fps)))
    pad = max(0, int(round(float(sem_cfg.get("boundary_padding_sec", 0.0)) * fps)))
    valid_arms = {"left", "right", "both"}

    segments: list[RawSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        raw_start = _frame_from_semantic_item(
            item,
            frame_keys=("start_frame", "start"),
            time_keys=("start_time_sec", "start_sec"),
            fps=fps,
            default=0,
        )
        raw_end = _frame_from_semantic_item(
            item,
            frame_keys=("end_frame", "end"),
            time_keys=("end_time_sec", "end_sec"),
            fps=fps,
            default=episode_len,
        )
        raw_start = max(0, min(episode_len - 1, raw_start))
        raw_end = max(raw_start + 1, min(episode_len, raw_end))

        start = max(0, raw_start - pad)
        end = min(episode_len, raw_end + pad)
        if end - start < min_len:
            continue

        core_start = _frame_from_semantic_item(
            item,
            frame_keys=("core_start_frame",),
            time_keys=("core_start_time_sec",),
            fps=fps,
            default=raw_start,
        )
        core_end = _frame_from_semantic_item(
            item,
            frame_keys=("core_end_frame",),
            time_keys=("core_end_time_sec",),
            fps=fps,
            default=raw_end - 1,
        )
        core_start = max(start, min(end - 1, core_start))
        core_end = max(core_start, min(end - 1, core_end))

        active_arm = str(item.get("active_arm", "both")).strip().lower()
        if active_arm not in valid_arms:
            active_arm = "both"

        semantic_label = {
            "canonical_instruction": item.get("canonical_instruction", ""),
            "stage_label": item.get("stage_label", "semantic_subtask"),
            "object": item.get("object", "unknown"),
            "target": item.get("target", "unknown"),
            "confidence": item.get("confidence", 0.0),
            "variants": item.get("variants", []),
            "needs_review": item.get("needs_review", False),
            "arm": item.get("arm", item.get("active_arm", "unknown")),
            "approach_direction": item.get("approach_direction", "unknown"),
            "object_color": item.get("object_color", "unknown"),
            "object_size": item.get("object_size", "unknown"),
            "object_name": item.get("object_name", item.get("object", "object")),
            "object_description": item.get("object_description", item.get("object", "unknown object")),
            "action": item.get("action", "manipulate"),
            "target_direction": item.get("target_direction", "unknown"),
            "target_color": item.get("target_color", "unknown"),
            "target_size": item.get("target_size", "unknown"),
            "target_name": item.get("target_name", item.get("target", "target")),
            "target_description": item.get("target_description", item.get("target", "unknown target")),
            "placement_relation": item.get("placement_relation", "at"),
        }
        segments.append(
            RawSegment(
                parent_episode=parent_episode,
                segment_id=-1,
                stage_id="",
                active_arm=active_arm,
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                source_segments=[
                    {
                        "source": "vlm_semantic_segmentation",
                        "active_arm": active_arm,
                        "start_frame": raw_start,
                        "end_frame": raw_end,
                        "reasoning": str(item.get("reasoning", "")),
                    }
                ],
                semantic_label=semantic_label,
            )
        )

    segments = sorted(segments, key=lambda seg: (seg.start, seg.end))
    for idx, segment in enumerate(segments):
        segment.segment_id = idx
        segment.stage_id = f"episode_{parent_episode:06d}_stage_{idx:03d}"
    return segments


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_candidate(text: str) -> str:
    text = _strip_json_fences(text)
    if not text:
        return text

    object_start = text.find("{")
    object_end = text.rfind("}")
    if 0 <= object_start < object_end:
        return text[object_start : object_end + 1]

    array_start = text.find("[")
    array_end = text.rfind("]")
    if 0 <= array_start < array_end:
        return text[array_start : array_end + 1]

    return text


def _parse_json_content(content: Any, *, context: str) -> dict[str, Any]:
    if isinstance(content, dict):
        return content

    text = _message_content_to_text(content)
    candidate = _extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", candidate)
        if no_trailing_commas != candidate:
            try:
                return json.loads(no_trailing_commas)
            except json.JSONDecodeError:
                pass

        preview = candidate[:1200].replace("\n", "\\n")
        raise ValueError(
            f"{context} returned invalid JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno} char {exc.pos}. raw_preview={preview}"
        ) from exc


class JsonlCache:
    def __init__(self, path: Path | None, *, resume: bool):
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        if resume and path is not None and path.exists():
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    cache_key = record.get("cache_key")
                    if cache_key:
                        self.records[str(cache_key)] = record

    def get(self, cache_key: str) -> dict[str, Any] | None:
        return self.records.get(cache_key)

    def append(self, cache_key: str, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cache_key": cache_key, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.records[cache_key] = payload


class OpenRouterLabeler:
    def __init__(self, cfg: dict[str, Any], cache: JsonlCache | None):
        self.cfg = cfg
        self.openrouter_cfg = cfg.get("openrouter", {}) or {}
        self.enabled = bool(self.openrouter_cfg.get("enabled", False))
        self.cache = cache
        self.api_key = str(self.openrouter_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.model = str(self.openrouter_cfg.get("model", "google/gemini-3.1-flash-lite"))
        self.timeout = float(self.openrouter_cfg.get("timeout_sec", 60))
        self.min_confidence = float((cfg.get("labeling", {}) or {}).get("min_confidence", 0.55))

    def validate_model(self) -> None:
        if not self.enabled or not self.openrouter_cfg.get("validate_model", True):
            return
        if not self.api_key:
            logger.warning("[LABEL] openrouter.api_key / OPENROUTER_API_KEY is not set; template labels will be used.")
            self.enabled = False
            return

        request = urllib.request.Request(OPENROUTER_MODELS_URL, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LABEL] Could not validate OpenRouter model %s: %s", self.model, exc)
            return

        model_info = next((item for item in payload.get("data", []) if item.get("id") == self.model), None)
        if not model_info:
            logger.warning("[LABEL] Model %s was not found in OpenRouter model list.", self.model)
            return
        input_modalities = set(((model_info.get("architecture") or {}).get("input_modalities") or []))
        supported = set(model_info.get("supported_parameters") or [])
        if "image" not in input_modalities:
            logger.warning("[LABEL] Model %s does not advertise image input support.", self.model)
        if not ({"response_format", "structured_outputs"} & supported):
            logger.warning("[LABEL] Model %s may not support JSON schema output.", self.model)

    def segment_episode(
        self,
        *,
        parent_episode: int,
        parent_task: str,
        episode_len: int,
        auxiliary_segments: list[RawSegment],
        overview_images: list[dict[str, str]],
    ) -> list[RawSegment]:
        sem_cfg = self.cfg.get("semantic_segmentation", {}) or {}
        if not sem_cfg.get("enabled", False):
            return []
        if not self.enabled or not self.api_key:
            return []
        if not overview_images:
            logger.warning("[SEG] No overview images available for semantic segmentation.")
            return []

        cache_key = self._semantic_cache_key(parent_episode, parent_task, episode_len, auxiliary_segments, overview_images)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            cached_segments = cached.get("semantic_segments") if cached else None
            if cached_segments is not None:
                segments = _normalize_semantic_segments(cached_segments, parent_episode, episode_len, self.cfg)
                if segments:
                    return segments

        try:
            raw_segments = self._request_semantic_segments(
                parent_episode=parent_episode,
                parent_task=parent_task,
                episode_len=episode_len,
                auxiliary_segments=auxiliary_segments,
                overview_images=overview_images,
            )
            segments = _normalize_semantic_segments(raw_segments, parent_episode, episode_len, self.cfg)
            if self.cache is not None:
                self.cache.append(
                    cache_key,
                    {
                        "semantic_segments": raw_segments,
                        "parent_episode": parent_episode,
                        "created_at": time.time(),
                    },
                )
            return segments
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SEG] OpenRouter semantic segmentation failed for episode=%s: %s", parent_episode, exc)
            return []

    def label(
        self,
        segment: RawSegment,
        parent_task: str,
        total_segments: int,
        images: list[dict[str, str]],
    ) -> LabelResult:
        fallback = _template_instruction(segment, parent_task, total_segments)
        sem_cfg = self.cfg.get("semantic_segmentation", {}) or {}
        if (
            segment.semantic_label
            and sem_cfg.get("use_segment_labels", True)
            and not sem_cfg.get("relabel_segments", False)
        ):
            label = _normalize_label(segment.semantic_label, fallback, self.min_confidence)
            label.source = "openrouter_semantic_segmentation"
            return label
        if not self.enabled:
            return fallback
        if not self.api_key:
            fallback.error = "openrouter.api_key / OPENROUTER_API_KEY is not set"
            return fallback

        cache_key = self._cache_key(segment, parent_task, images)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached and isinstance(cached.get("label"), dict):
                label = _normalize_label(cached["label"], fallback, self.min_confidence)
                label.source = "cache"
                return label

        try:
            raw_label = self._request_label(segment, parent_task, images)
            label = _normalize_label(raw_label, fallback, self.min_confidence)
            if self.cache is not None:
                self.cache.append(
                    cache_key,
                    {
                        "label": asdict(label),
                        "parent_episode": segment.parent_episode,
                        "segment_id": segment.segment_id,
                        "created_at": time.time(),
                    },
                )
            return label
        except Exception as exc:  # noqa: BLE001
            fallback.error = str(exc)
            logger.warning(
                "[LABEL] OpenRouter failed for episode=%s segment=%s: %s",
                segment.parent_episode,
                segment.segment_id,
                exc,
            )
            return fallback

    def _semantic_cache_key(
        self,
        parent_episode: int,
        parent_task: str,
        episode_len: int,
        auxiliary_segments: list[RawSegment],
        overview_images: list[dict[str, str]],
    ) -> str:
        payload = {
            "type": "semantic_segmentation",
            "language_schema_version": "slot_v2",
            "model": self.model,
            "parent_episode": parent_episode,
            "parent_task": parent_task,
            "episode_len": episode_len,
            "auxiliary_segments": _auxiliary_segments_payload(auxiliary_segments),
            "image_hashes": [image["sha256"] for image in overview_images],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_key(self, segment: RawSegment, parent_task: str, images: list[dict[str, str]]) -> str:
        payload = {
            "language_schema_version": "slot_v2",
            "model": self.model,
            "parent_task": parent_task,
            "segment": {
                "episode": segment.parent_episode,
                "start": segment.start,
                "end": segment.end,
                "core_start": segment.core_start,
                "core_end": segment.core_end,
                "active_arm": segment.active_arm,
            },
            "image_hashes": [image["sha256"] for image in images],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _request_semantic_segments(
        self,
        *,
        parent_episode: int,
        parent_task: str,
        episode_len: int,
        auxiliary_segments: list[RawSegment],
        overview_images: list[dict[str, str]],
    ) -> dict[str, Any]:
        sem_cfg = self.cfg.get("semantic_segmentation", {}) or {}
        max_segments = max(1, int(sem_cfg.get("max_segments", 12)))
        sample_metadata = [
            {
                "role": image["role"],
                "local_frame": int(image["local_frame"]),
                "camera_key": image["camera_key"],
            }
            for image in overview_images
        ]
        auxiliary = _auxiliary_segments_payload(auxiliary_segments)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are segmenting a long dual-arm robot manipulation episode into semantic subtasks. "
                    "Use the chronological overview images as the primary evidence. Gripper close/open and "
                    "motion segments are only auxiliary hints; if the visual task semantics disagree with "
                    "the gripper heuristic, trust the visual semantics.\n"
                    f"Parent task: {parent_task or 'unknown'}\n"
                    f"Parent episode: {parent_episode}\n"
                    f"Episode length in local frames: {episode_len}\n"
                    "Frame numbers are local to the episode. `start_frame` is inclusive and `end_frame` is exclusive. "
                    "Create one segment per meaningful subtask, such as moving a specific object, placing it at "
                    "a target, sealing/closing, handing off between arms, or coordinated dual-arm manipulation. "
                    "Do not split solely because the gripper opened or closed.\n"
                    "For every segment, fill the structured language slots first: arm, approach_direction, "
                    "object_color, object_size, object_name, action, target_direction, target_color, "
                    "target_size, target_name, and placement_relation. "
                    "Use `unknown` when an attribute is not visually clear. "
                    "Keep canonical_instruction short and imperative, for example: "
                    "`Left arm: pick up the yellow cube and place it in the black basket.` "
                    "`Right arm: place the blue package in the black basket.` "
                    "`Both arms: seal the black basket.` "
                    "Use generic object or target names only when the images do not support a specific name.\n"
                    f"Sampled frames: {json.dumps(sample_metadata, ensure_ascii=False)}\n"
                    f"Auxiliary gripper/motion segments: {json.dumps(auxiliary, ensure_ascii=False)}"
                ),
            }
        ]
        for image in overview_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image["data_url"],
                    },
                }
            )

        segment_schema = {
            "type": "object",
            "properties": {
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
                "core_start_frame": {"type": "integer"},
                "core_end_frame": {"type": "integer"},
                "active_arm": {"type": "string", "enum": ["left", "right", "both"]},
                "arm": {"type": "string", "enum": ARM_VALUES},
                "approach_direction": {"type": "string", "enum": DIRECTION_VALUES},
                "object_color": {"type": "string"},
                "object_size": {"type": "string"},
                "object_name": {"type": "string"},
                "object_description": {"type": "string"},
                "action": {"type": "string", "enum": ACTION_VALUES},
                "target_direction": {"type": "string", "enum": DIRECTION_VALUES},
                "target_color": {"type": "string"},
                "target_size": {"type": "string"},
                "target_name": {"type": "string"},
                "target_description": {"type": "string"},
                "placement_relation": {"type": "string", "enum": RELATION_VALUES},
                "stage_label": {"type": "string"},
                "canonical_instruction": {"type": "string"},
                "object": {"type": "string"},
                "target": {"type": "string"},
                "confidence": {"type": "number"},
                "variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "needs_review": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": [
                "start_frame",
                "end_frame",
                "core_start_frame",
                "core_end_frame",
                "active_arm",
                "arm",
                "approach_direction",
                "object_color",
                "object_size",
                "object_name",
                "object_description",
                "action",
                "target_direction",
                "target_color",
                "target_size",
                "target_name",
                "target_description",
                "placement_relation",
                "stage_label",
                "canonical_instruction",
                "object",
                "target",
                "confidence",
                "variants",
                "needs_review",
                "reasoning",
            ],
            "additionalProperties": False,
        }
        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You identify semantic stages in robot manipulation data for imitation learning. "
                        "Return only valid JSON. Do not wrap JSON in markdown. Do not include comments."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "temperature": float(self.openrouter_cfg.get("temperature", 0.2)),
            "max_tokens": int(sem_cfg.get("max_tokens", self.openrouter_cfg.get("max_tokens", 800))),
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_semantic_subtasks",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array",
                                "items": segment_schema,
                                "minItems": 1,
                                "maxItems": max_segments,
                            }
                        },
                        "required": ["segments"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": str(self.openrouter_cfg.get("referer", "dual_arm_teleop")),
                "X-Title": str(self.openrouter_cfg.get("title", "dual_arm_teleop_split_label_dataset")),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        message = response_payload["choices"][0]["message"]
        content_text = message.get("content", "")
        try:
            return _parse_json_content(content_text, context="semantic segmentation")
        except ValueError as exc:
            if not sem_cfg.get("repair_invalid_json", True):
                raise
            logger.warning("[SEG] Invalid JSON from OpenRouter, trying one repair request: %s", exc)
            return self._repair_json_response(
                content_text,
                context="semantic segmentation",
                schema_hint=(
                    "Return a JSON object with exactly one key `segments`. `segments` is an array of objects. "
                    "Each object must include integer start_frame, end_frame, core_start_frame, core_end_frame; "
                    "active_arm as left/right/both; arm, approach_direction, object_color, object_size, "
                    "object_name, object_description, action, target_direction, target_color, target_size, "
                    "target_name, target_description, placement_relation; stage_label, canonical_instruction, "
                    "object, target, confidence, variants array of exactly 3 strings, needs_review boolean, and reasoning."
                ),
                max_tokens=int(sem_cfg.get("repair_max_tokens", sem_cfg.get("max_tokens", 1200))),
            )

    def _repair_json_response(
        self,
        content: Any,
        *,
        context: str,
        schema_hint: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        raw_text = _message_content_to_text(content)
        repair_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You repair malformed JSON. Return only valid JSON matching the requested shape. "
                        "Do not add markdown fences or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{schema_hint}\n\n"
                        "Repair this malformed JSON without changing the intended values:\n"
                        f"{raw_text}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }
        body = json.dumps(repair_payload).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": str(self.openrouter_cfg.get("referer", "dual_arm_teleop")),
                "X-Title": str(self.openrouter_cfg.get("title", "dual_arm_teleop_split_label_dataset")),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"repair HTTP {exc.code}: {detail}") from exc

        message = response_payload["choices"][0]["message"]
        return _parse_json_content(message.get("content", ""), context=f"{context} repair")

    def _request_label(self, segment: RawSegment, parent_task: str, images: list[dict[str, str]]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are labeling robot manipulation sub-episodes for imitation learning. "
                    "Use the images and metadata to fill a structured language template. "
                    "First decide the slots: arm, approach_direction, object_color, object_size, object_name, "
                    "action, target_direction, target_color, target_size, target_name, and placement_relation. "
                    "Use `unknown` for visually unclear attributes. "
                    "Keep the canonical instruction short and imperative, for example: "
                    "`Left arm: pick up the yellow cube and place it in the black basket.` "
                    "`Right arm: place the blue package in the black basket.` "
                    "`Both arms: seal the black basket.` "
                    "Do not invent object names when the object is unclear; use generic wording. "
                    f"Parent task: {parent_task}\n"
                    f"Active arm: {segment.active_arm}\n"
                    f"Segment index: {segment.segment_id}\n"
                    f"Segment frame range: {segment.start}:{segment.end}\n"
                    f"Auxiliary event frames: close={segment.close_frames}, open={segment.open_frames}\n"
                    "Return exactly the requested JSON schema."
                ),
            }
        ]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image["data_url"],
                    },
                }
            )

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You write grounded robot task labels for VLA training datasets.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "temperature": float(self.openrouter_cfg.get("temperature", 0.2)),
            "max_tokens": int(self.openrouter_cfg.get("max_tokens", 500)),
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_subepisode_label",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "canonical_instruction": {"type": "string"},
                            "arm": {"type": "string", "enum": ARM_VALUES},
                            "approach_direction": {"type": "string", "enum": DIRECTION_VALUES},
                            "object_color": {"type": "string"},
                            "object_size": {"type": "string"},
                            "object_name": {"type": "string"},
                            "object_description": {"type": "string"},
                            "action": {"type": "string", "enum": ACTION_VALUES},
                            "target_direction": {"type": "string", "enum": DIRECTION_VALUES},
                            "target_color": {"type": "string"},
                            "target_size": {"type": "string"},
                            "target_name": {"type": "string"},
                            "target_description": {"type": "string"},
                            "placement_relation": {"type": "string", "enum": RELATION_VALUES},
                            "stage_label": {"type": "string"},
                            "object": {"type": "string"},
                            "target": {"type": "string"},
                            "confidence": {"type": "number"},
                            "variants": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "needs_review": {"type": "boolean"},
                        },
                        "required": [
                            "canonical_instruction",
                            "arm",
                            "approach_direction",
                            "object_color",
                            "object_size",
                            "object_name",
                            "object_description",
                            "action",
                            "target_direction",
                            "target_color",
                            "target_size",
                            "target_name",
                            "target_description",
                            "placement_relation",
                            "stage_label",
                            "object",
                            "target",
                            "confidence",
                            "variants",
                            "needs_review",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": str(self.openrouter_cfg.get("referer", "dual_arm_teleop")),
                "X-Title": str(self.openrouter_cfg.get("title", "dual_arm_teleop_split_label_dataset")),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        message = response_payload["choices"][0]["message"]
        content_text = message.get("content", "")
        try:
            return _parse_json_content(content_text, context="segment label")
        except ValueError as exc:
            if not (self.cfg.get("semantic_segmentation", {}) or {}).get("repair_invalid_json", True):
                raise
            logger.warning("[LABEL] Invalid JSON from OpenRouter, trying one repair request: %s", exc)
            return self._repair_json_response(
                content_text,
                context="segment label",
                schema_hint=(
                    "Return a JSON object with canonical_instruction, stage_label, object, target, "
                    "arm, approach_direction, object_color, object_size, object_name, object_description, "
                    "action, target_direction, target_color, target_size, target_name, target_description, "
                    "placement_relation, confidence, variants array of exactly 3 strings, and needs_review boolean."
                ),
                max_tokens=int(self.openrouter_cfg.get("repair_max_tokens", self.openrouter_cfg.get("max_tokens", 800))),
            )


def _create_output_dataset(source: Any, cfg: dict[str, Any]) -> Any:
    _require_lerobot()
    _assert_output_root_is_separate_from_source(source, cfg)
    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is not None and output_root.exists():
        if output_cfg.get("overwrite", False):
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                "Set output.overwrite=true or pass --overwrite to replace it."
            )

    return LeRobotDataset.create(
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


def _manifest_dir(cfg: dict[str, Any]) -> Path:
    manifest_cfg = cfg.get("manifests", {}) or {}
    if manifest_cfg.get("dir"):
        return Path(manifest_cfg["dir"]).expanduser()
    output_root = _as_path_or_none((cfg.get("output", {}) or {}).get("root"))
    if output_root is not None:
        return output_root / "meta"
    return Path("outputs") / "split_label_dataset"


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _segment_manifest(
    source: Any,
    segment: RawSegment,
    label: LabelResult,
    parent_task: str,
    keyframes: dict[str, Any],
    output_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parent_repo_id": source.repo_id,
        "parent_episode": segment.parent_episode,
        "stage_id": segment.stage_id,
        "segment_id": segment.segment_id,
        "active_arm": segment.active_arm,
        "frame_range": [segment.start, segment.end],
        "source_index_range": [
            int(source.meta.episodes[int(segment.parent_episode)]["dataset_from_index"]) + segment.start,
            int(source.meta.episodes[int(segment.parent_episode)]["dataset_from_index"]) + segment.end,
        ],
        "event_frames": {
            "close": segment.close_frames,
            "open": segment.open_frames,
        },
        "source_segments": segment.source_segments,
        "parent_task": parent_task,
        "label": asdict(label),
        "keyframes": keyframes,
        "output_episodes": output_episodes,
    }


def _vla_manifest_record(split_record: dict[str, Any], output_repo_id: str) -> dict[str, Any]:
    return {
        "dataset_type": "lerobot_v3",
        "target": "openpi",
        "repo_id": output_repo_id,
        "parent_repo_id": split_record["parent_repo_id"],
        "parent_episode": split_record["parent_episode"],
        "segment_id": split_record["segment_id"],
        "stage_id": split_record["stage_id"],
        "active_arm": split_record["active_arm"],
        "instruction": split_record["label"]["canonical_instruction"],
        "instruction_variants": split_record["label"]["variants"],
        "needs_review": split_record["label"]["needs_review"],
        "output_episodes": split_record["output_episodes"],
    }


def _starvla_modality(source: Any) -> dict[str, Any]:
    video: dict[str, dict[str, str]] = {}
    for key in getattr(source.meta, "camera_keys", []):
        name = key.removeprefix("observation.images.")
        if name == "head_image":
            name = "base_view"
        video[name] = {"original_key": key}

    action_shape = tuple(source.features.get("action", {}).get("shape", [0]))
    state_shape = tuple(source.features.get("observation.state", {}).get("shape", [0]))
    return {
        "state": {
            "dual_arm_state": {
                "start": 0,
                "end": int(state_shape[0]) if state_shape else 0,
                "original_key": "observation.state",
            }
        },
        "action": {
            "dual_arm_action": {
                "start": 0,
                "end": int(action_shape[0]) if action_shape else 0,
                "original_key": "action",
                "absolute": False,
            }
        },
        "video": video,
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def _tasks_for_output(label: LabelResult, cfg: dict[str, Any]) -> list[str]:
    language_cfg = cfg.get("language_augmentation", {}) or {}
    tasks = [label.canonical_instruction]
    if language_cfg.get("write_variants_as_episodes", True):
        tasks.extend(label.variants[: int(language_cfg.get("num_variants", 3))])
    deduped: list[str] = []
    for task in tasks:
        task = str(task).strip()
        if task and task not in deduped:
            deduped.append(task)
    return deduped


def _report_enabled(cfg: dict[str, Any], key: str, default: bool = True) -> bool:
    return bool((cfg.get("report", {}) or {}).get(key, default))


def _report_line(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def _fmt_seconds(frame: int, fps: float) -> str:
    return f"{frame / max(fps, 1e-6):.2f}s"


def _segment_semantic_summary(segment: RawSegment) -> str:
    label = segment.semantic_label or {}
    instruction = str(label.get("canonical_instruction", "")).strip()
    if not instruction:
        return ""
    details = [
        f"instruction={instruction}",
        f"arm={label.get('arm', label.get('active_arm', 'unknown'))}",
        f"direction={label.get('approach_direction', 'unknown')}->{label.get('target_direction', 'unknown')}",
        f"action={label.get('action', 'unknown')}",
        f"object={label.get('object', 'unknown')}",
        f"target={label.get('target', 'unknown')}",
        f"confidence={label.get('confidence', 0.0)}",
    ]
    return " | ".join(details)


def _print_episode_segmentation_report(
    cfg: dict[str, Any],
    *,
    ep_idx: int,
    episode_len: int,
    parent_task: str,
    segmentation_source: str,
    gripper_source: str,
    gripper_sides: list[str],
    segments: list[RawSegment],
    action_filter_stats: dict[str, int] | None = None,
) -> None:
    if not _report_enabled(cfg, "print_segments", True):
        return

    fps = float(cfg.get("_fps", 30))
    _report_line(
        f"\n[SPLIT] episode={ep_idx} frames={episode_len} duration={episode_len / max(fps, 1e-6):.2f}s "
        f"segments={len(segments)} source={segmentation_source}"
    )
    _report_line(f"        task={parent_task}")
    _report_line(f"        auxiliary_gripper_source={gripper_source} sides={gripper_sides}")
    if action_filter_stats:
        _report_line(
            "        action_filter="
            f"before:{action_filter_stats.get('before', 0)} "
            f"after:{action_filter_stats.get('after', 0)} "
            f"trimmed:{action_filter_stats.get('trimmed', 0)} "
            f"dropped_zero_action:{action_filter_stats.get('dropped_zero_action', 0)}"
        )
    for segment in segments:
        duration = (segment.end - segment.start) / max(fps, 1e-6)
        _report_line(
            f"  - seg={segment.segment_id:03d} arm={segment.active_arm} "
            f"frames={segment.start}:{segment.end} "
            f"time={_fmt_seconds(segment.start, fps)}-{_fmt_seconds(segment.end, fps)} "
            f"len={duration:.2f}s core={segment.core_start}:{segment.core_end}"
        )
        if segment.close_frames or segment.open_frames:
            _report_line(f"    gripper_hint: close={segment.close_frames} open={segment.open_frames}")
        semantic_summary = _segment_semantic_summary(segment)
        if semantic_summary:
            _report_line(f"    semantic: {semantic_summary}")


def _print_label_report(
    cfg: dict[str, Any],
    *,
    segment: RawSegment,
    label: LabelResult,
    task_variants: list[str],
) -> None:
    if not _report_enabled(cfg, "print_labels", True):
        return

    _report_line(f"    [LABEL seg={segment.segment_id:03d}] {label.canonical_instruction}")
    _report_line(
        f"      stage={label.stage_label} arm={label.arm} action={label.action} "
        f"direction={label.approach_direction}->{label.target_direction} "
        f"object={label.object_description} target={label.target_description} "
        f"relation={label.placement_relation} confidence={label.confidence:.2f} "
        f"needs_review={label.needs_review} source={label.source}"
    )
    if _report_enabled(cfg, "print_variants", True):
        for idx, variant in enumerate(task_variants[1:], start=1):
            _report_line(f"      variant_{idx}: {variant}")


def split_label_dataset(cfg: dict[str, Any]) -> None:
    _require_lerobot()
    source_cfg = cfg["source"]
    source = LeRobotDataset(
        source_cfg["repo_id"],
        root=_as_path_or_none(source_cfg.get("root")),
    )
    cfg["_fps"] = source.fps
    episodes = _select_episodes(source, cfg)
    dry_run = bool(cfg.get("dry_run", False))
    write_dataset = bool((cfg.get("output", {}) or {}).get("write_dataset", True)) and not dry_run
    label_only = bool(cfg.get("label_only", False))
    if label_only:
        write_dataset = False

    semantic_cfg = cfg.get("semantic_segmentation", {}) or {}
    semantic_on_dry_run = bool(semantic_cfg.get("enabled", False) and semantic_cfg.get("run_on_dry_run", False))
    _assert_no_source_writes(
        source,
        cfg,
        write_dataset=write_dataset,
        dry_run=dry_run,
        semantic_on_dry_run=semantic_on_dry_run,
    )

    manifest_dir = _manifest_dir(cfg)
    split_manifest_path = manifest_dir / "split_manifest.jsonl"
    vla_manifest_path = manifest_dir / "vla_manifest.jsonl"
    cache_path = _as_path_or_none((cfg.get("openrouter", {}) or {}).get("cache_path"))
    cache = JsonlCache(cache_path, resume=bool(cfg.get("resume_cache", True)))
    labeler = OpenRouterLabeler(cfg, cache)
    if not dry_run or semantic_on_dry_run:
        labeler.validate_model()

    if not dry_run:
        if split_manifest_path.exists() and (cfg.get("output", {}) or {}).get("overwrite", False):
            split_manifest_path.unlink()
        if vla_manifest_path.exists() and (cfg.get("output", {}) or {}).get("overwrite", False):
            vla_manifest_path.unlink()

    output = None if not write_dataset else _create_output_dataset(source, cfg)
    output_episode_idx = 0
    total_segments = 0
    total_input_frames = 0
    total_output_frames = 0

    action_names = _feature_names(source, "action")
    state_names = _feature_names(source, "observation.state")

    try:
        for ep_idx in episodes:
            ep = source.meta.episodes[int(ep_idx)]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            episode_len = end - start
            total_input_frames += episode_len
            arrays = _episode_columns(source, start, end, ["action", "observation.state"])
            gripper_values, gripper_sides, gripper_source = _extract_gripper_signal(
                arrays, action_names, state_names, cfg
            )
            auxiliary_segments = split_episode(
                parent_episode=int(ep_idx),
                gripper_values=gripper_values,
                gripper_sides=gripper_sides,
                actions=arrays.get("action"),
                action_names=action_names,
                cfg=cfg,
            )
            parent_task = source[int(start)]["task"] if episode_len else ""
            segments = auxiliary_segments
            segmentation_source = "gripper_motion_fallback"
            if semantic_cfg.get("enabled", False) and (not dry_run or semantic_on_dry_run):
                overview_images, _overview_manifest = _collect_episode_overview_images(source, start, episode_len, cfg)
                semantic_segments = labeler.segment_episode(
                    parent_episode=int(ep_idx),
                    parent_task=str(parent_task),
                    episode_len=episode_len,
                    auxiliary_segments=auxiliary_segments,
                    overview_images=overview_images,
                )
                if semantic_segments:
                    segments = semantic_segments
                    segmentation_source = "vlm_semantic"
                elif not semantic_cfg.get("fallback_to_gripper", True):
                    segments = []
                    segmentation_source = "vlm_semantic_failed_no_fallback"

            segments, action_filter_stats = _filter_segments_by_action(
                segments,
                arrays.get("action"),
                action_names,
                cfg,
            )
            total_segments += len(segments)
            logger.info(
                "[EP %s] frames=%d segmentation=%s gripper_source=%s sides=%s segments=%d action_filter=%s",
                ep_idx,
                episode_len,
                segmentation_source,
                gripper_source,
                gripper_sides,
                len(segments),
                action_filter_stats,
            )
            for segment in segments:
                logger.info(
                    "  [SEG %d] arm=%s frames=%d:%d core=%d:%d close=%s open=%s",
                    segment.segment_id,
                    segment.active_arm,
                    segment.start,
                    segment.end,
                    segment.core_start,
                    segment.core_end,
                    segment.close_frames,
                    segment.open_frames,
                )
            _print_episode_segmentation_report(
                cfg,
                ep_idx=int(ep_idx),
                episode_len=episode_len,
                parent_task=str(parent_task),
                segmentation_source=segmentation_source,
                gripper_source=gripper_source,
                gripper_sides=gripper_sides,
                segments=segments,
                action_filter_stats=action_filter_stats,
            )

            if dry_run:
                continue

            split_records: list[dict[str, Any]] = []
            vla_records: list[dict[str, Any]] = []
            for segment in segments:
                images, keyframes = _collect_keyframe_images(source, segment, start, episode_len, cfg)
                label = labeler.label(segment, parent_task, len(segments), images)
                output_episodes: list[dict[str, Any]] = []
                task_variants = _tasks_for_output(label, cfg)
                _print_label_report(cfg, segment=segment, label=label, task_variants=task_variants)

                if output is not None:
                    for variant_idx, task in enumerate(task_variants):
                        for local_idx in range(segment.start, segment.end):
                            item = source[int(start + local_idx)]
                            output.add_frame(_frame_from_source_item(source, item, task))
                        output.save_episode()
                        output_episodes.append(
                            {
                                "episode_index": output_episode_idx,
                                "variant_index": variant_idx,
                                "task": task,
                            }
                        )
                        output_episode_idx += 1
                        total_output_frames += segment.length

                split_record = _segment_manifest(
                    source, segment, label, parent_task, keyframes, output_episodes
                )
                split_records.append(split_record)
                vla_records.append(_vla_manifest_record(split_record, cfg["output"]["repo_id"]))

            _append_jsonl(split_manifest_path, split_records)
            _append_jsonl(vla_manifest_path, vla_records)

        if not dry_run and (cfg.get("vla_export", {}) or {}).get("write_starvla_modality", True):
            modality_path = _as_path_or_none((cfg.get("vla_export", {}) or {}).get("starvla_modality_path"))
            if modality_path is None:
                modality_path = manifest_dir / "modality.json"
            _write_json(modality_path, _starvla_modality(source))
            logger.info("[VLA] StarVLA modality draft: %s", modality_path)
    finally:
        if output is not None:
            output.finalize()

    logger.info(
        "[DONE] episodes=%d input_frames=%d segments=%d output_frames=%d%s",
        len(episodes),
        total_input_frames,
        total_segments,
        total_output_frames,
        " [dry-run]" if dry_run else f" manifests={manifest_dir}",
    )


def main() -> None:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "split_label_dataset_cfg.yaml"
    parser = argparse.ArgumentParser(description="Split long LeRobot episodes and label sub-episodes.")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--dry-run", action="store_true", help="Only report segments; do not label or write.")
    parser.add_argument("--label-only", action="store_true", help="Label and write manifests without dataset output.")
    parser.add_argument("--write-dataset", action="store_true", help="Force writing the output LeRobot dataset.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Override source.max_episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite=true.")
    parser.add_argument("--resume-cache", action="store_true", help="Reuse OpenRouter label cache if present.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.label_only:
        cfg["label_only"] = True
        cfg.setdefault("output", {})["write_dataset"] = False
    if args.write_dataset:
        cfg.setdefault("output", {})["write_dataset"] = True
        cfg["label_only"] = False
    if args.max_episodes is not None:
        cfg["source"]["max_episodes"] = args.max_episodes
    if args.overwrite:
        cfg.setdefault("output", {})["overwrite"] = True
    if args.resume_cache:
        cfg["resume_cache"] = True

    split_label_dataset(cfg)


if __name__ == "__main__":
    main()
