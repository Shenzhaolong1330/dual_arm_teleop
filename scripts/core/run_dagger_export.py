#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME


DEFAULT_KEEP_FRAME_ROLES = ("takeover_start", "recovery")
EXPORT_MODE_INTERVENTION_SEGMENTS = "intervention_segments"
EXPORT_MODE_FULL_SUCCESS_EPISODE = "full_success_episode"
EXPORT_MODE_HYBRID = "hybrid"
VALID_EXPORT_MODES = {
    EXPORT_MODE_INTERVENTION_SEGMENTS,
    EXPORT_MODE_FULL_SUCCESS_EPISODE,
    EXPORT_MODE_HYBRID,
}
SUCCESS_POLICY_EXPLICIT = "explicit"
SUCCESS_POLICY_RECORDED_IS_SUCCESS = "recorded_is_success"
SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE = "allow_missing_for_smoke"
VALID_SUCCESS_POLICIES = {
    SUCCESS_POLICY_EXPLICIT,
    SUCCESS_POLICY_RECORDED_IS_SUCCESS,
    SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE,
}
SMOKE_MISSING_SUCCESS_WARNING = (
    "WARNING: success field missing; exporting without success validation."
)

DEFAULT_FULL_EPISODE_CFG = {
    "require_success": True,
    "success_field": "success",
    "fallback_success_from_episode_metadata": True,
    "action_label_source": "sent_action",
    "drop_reset_ignore_frames": True,
    "require_no_reset": True,
    "require_min_len": True,
}
DEFAULT_INTERVENTION_SEGMENTS_CFG = {
    "keep_frame_roles": list(DEFAULT_KEEP_FRAME_ROLES),
    "label_source": "expert_action",
    "require_complete_expert_action": True,
    "max_segments_per_episode": None,
}
DEFAULT_HYBRID_CFG = {
    "include_full_success_episode": True,
    "include_intervention_segments": True,
    "max_recovery_segments_per_episode": 1,
    "max_recovery_frames_ratio": 0.25,
    "prefer_full_episode_when_duplicate": True,
}


@dataclass
class DAggerExportConfig:
    raw_root: str | Path
    output_root: str | Path
    seed_root: str | Path | None = None
    raw_repo_id: str | None = None
    output_repo_id: str | None = None
    seed_repo_id: str | None = None
    keep_frame_roles: tuple[str, ...] = DEFAULT_KEEP_FRAME_ROLES
    min_segment_frames: int = 1
    min_episode_len_for_act: int | None = None
    export_mode: str = EXPORT_MODE_INTERVENTION_SEGMENTS
    profile: str | None = None
    pre_takeover_context: int = 0
    require_complete_expert_action: bool = True
    full_episode: dict[str, Any] | None = None
    intervention_segments: dict[str, Any] | None = None
    hybrid: dict[str, Any] | None = None
    gripper_action_indices: list[int] | tuple[int, ...] | None = None
    gripper_open_threshold: float = 0.5
    overwrite: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 4


def _resolve_root(repo_id: str | None, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    if repo_id is None:
        raise ValueError("Either repo_id or root must be provided.")
    return HF_LEROBOT_HOME / repo_id


def _repo_id_from_root(root: Path, fallback: str) -> str:
    if root.name:
        return root.name
    return fallback


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _to_scalar(value[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_bool(value: Any) -> bool:
    return bool(_to_scalar(value))


def _to_int(value: Any, default: int = -1) -> int:
    value = _to_scalar(value)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_str(value: Any) -> str:
    value = _to_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _coerce_numpy_feature(value: Any, feature: dict[str, Any]) -> Any:
    dtype = feature["dtype"]
    if dtype == "string":
        return value
    if dtype in {"image", "video"}:
        return _coerce_image_or_video_feature(value, feature)

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.dtype(dtype))
    expected_shape = tuple(feature["shape"])
    if array.shape == ():
        array = array.reshape(expected_shape)
    return array


def _coerce_image_or_video_feature(value: Any, feature: dict[str, Any]) -> Any:
    """Match decoded camera tensors to the seed dataset image layout.

    LeRobot video decoding may return channel-first arrays/tensors (C, H, W),
    while the seed feature schema in this project stores camera frames as
    channel-last (H, W, C). The exported DAgger dataset must follow the seed
    schema exactly so it can be aggregated before backend training.
    """

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)
    expected_shape = tuple(feature.get("shape", ()))
    if array.shape == expected_shape:
        return array

    if array.ndim == 3 and len(expected_shape) == 3:
        expected_h, expected_w, expected_c = expected_shape
        if expected_c in (1, 3, 4) and array.shape == (expected_c, expected_h, expected_w):
            return np.transpose(array, (1, 2, 0))

    return array


def _feature_shape(feature: dict[str, Any]) -> tuple:
    return tuple(feature.get("shape", ()))


def _feature_names(feature: dict[str, Any]) -> Any:
    names = feature.get("names")
    if isinstance(names, list):
        return list(names)
    return names


def assert_lerobot_schema_compatible(
    reference_meta: LeRobotDatasetMetadata,
    candidate_meta: LeRobotDatasetMetadata,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> None:
    """Fail loudly when two LeRobot datasets cannot be aggregated for backend training."""

    errors: list[str] = []
    if reference_meta.fps != candidate_meta.fps:
        errors.append(f"fps differs: {reference_name}={reference_meta.fps}, {candidate_name}={candidate_meta.fps}")
    if reference_meta.robot_type != candidate_meta.robot_type:
        errors.append(
            f"robot_type differs: {reference_name}={reference_meta.robot_type}, "
            f"{candidate_name}={candidate_meta.robot_type}"
        )

    ref_features = reference_meta.features
    cand_features = candidate_meta.features
    ref_keys = set(ref_features)
    cand_keys = set(cand_features)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    if missing:
        errors.append(f"{candidate_name} missing feature keys: {missing}")
    if extra:
        errors.append(f"{candidate_name} has extra feature keys: {extra}")

    for key in sorted(ref_keys & cand_keys):
        ref = ref_features[key]
        cand = cand_features[key]
        if ref.get("dtype") != cand.get("dtype"):
            errors.append(
                f"{key}.dtype differs: {reference_name}={ref.get('dtype')}, "
                f"{candidate_name}={cand.get('dtype')}"
            )
        if _feature_shape(ref) != _feature_shape(cand):
            errors.append(
                f"{key}.shape differs: {reference_name}={_feature_shape(ref)}, "
                f"{candidate_name}={_feature_shape(cand)}"
            )
        if _feature_names(ref) != _feature_names(cand):
            errors.append(
                f"{key}.names differs: {reference_name}={_feature_names(ref)}, "
                f"{candidate_name}={_feature_names(cand)}"
            )

    if ACTION not in ref_features or ACTION not in cand_features:
        errors.append("Both datasets must contain the standard `action` feature.")
    elif _feature_shape(ref_features[ACTION]) != _feature_shape(cand_features[ACTION]):
        errors.append(
            f"action dimension differs: {reference_name}={_feature_shape(ref_features[ACTION])}, "
            f"{candidate_name}={_feature_shape(cand_features[ACTION])}"
        )

    if errors:
        raise ValueError(
            "LeRobot dataset schema mismatch; cannot safely aggregate or train the selected backend:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def assert_dataset_roots_schema_compatible(
    reference_repo_id: str,
    reference_root: str | Path,
    candidate_repo_id: str,
    candidate_root: str | Path,
) -> None:
    reference_meta = LeRobotDatasetMetadata(reference_repo_id, root=reference_root)
    candidate_meta = LeRobotDatasetMetadata(candidate_repo_id, root=candidate_root)
    assert_lerobot_schema_compatible(
        reference_meta,
        candidate_meta,
        reference_name=str(reference_root),
        candidate_name=str(candidate_root),
    )


def _frame_role_at(raw_hf_dataset, idx: int) -> str:
    column_names = raw_hf_dataset.column_names
    if "frame_role" in column_names:
        return _to_str(raw_hf_dataset["frame_role"][idx])
    if "is_expert" in column_names and _to_bool(raw_hf_dataset["is_expert"][idx]):
        return "recovery"
    return "policy"


def _segment_id_at(raw_hf_dataset, idx: int, fallback: int) -> int:
    if "intervention_segment_id" not in raw_hf_dataset.column_names:
        return fallback
    return _to_int(raw_hf_dataset["intervention_segment_id"][idx], default=fallback)


def _expert_label_complete_at(raw_hf_dataset, idx: int, require_complete: bool) -> bool:
    if not require_complete:
        return True
    if "expert_label_complete" not in raw_hf_dataset.column_names:
        raise ValueError(
            "Raw run_mix dataset is missing `expert_label_complete`. "
            "Recollect raw logs with the updated run_mix recorder, or set "
            "require_complete_expert_action=False only if you have manually verified labels."
        )
    return _to_bool(raw_hf_dataset["expert_label_complete"][idx])


def _context_indices_before(
    raw_hf_dataset,
    start: int,
    first_idx: int,
    pre_takeover_context: int,
    require_complete_expert_action: bool,
) -> list[int]:
    if pre_takeover_context <= 0:
        return []

    context: list[int] = []
    idx = first_idx - 1
    while idx >= start and len(context) < pre_takeover_context:
        role = _frame_role_at(raw_hf_dataset, idx)
        if role in {"reset", "ignore"}:
            break
        if not _expert_label_complete_at(raw_hf_dataset, idx, require_complete_expert_action):
            break
        context.append(idx)
        idx -= 1

    context.reverse()
    return context


def _iter_export_segments(
    raw_dataset: LeRobotDataset,
    keep_frame_roles: set[str],
    pre_takeover_context: int,
    require_complete_expert_action: bool,
) -> list[dict[str, Any]]:
    """Return truly continuous intervention slices from the raw run_mix timeline.

    Export keeps only complete expert-labeled takeover/recovery roles by default,
    drops reset/ignore/policy frames, and flushes whenever the raw index or
    intervention_segment_id is no longer continuous. Optional pre-takeover
    context is included only when it is adjacent and has complete expert labels.
    This prevents sparse takeover frames from being interpreted as one continuous
    action chunk.
    """

    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    segments: list[dict[str, Any]] = []

    for raw_episode_index in range(raw_dataset.num_episodes):
        episode = raw_dataset.meta.episodes[raw_episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])

        current_indices: list[int] = []
        current_segment_id: int | None = None
        last_idx: int | None = None

        def flush() -> None:
            nonlocal current_indices, current_segment_id, last_idx
            if current_indices:
                segments.append(
                    {
                        "raw_episode_index": raw_episode_index,
                        "intervention_segment_id": current_segment_id,
                        "indices": current_indices,
                    }
                )
            current_indices = []
            current_segment_id = None
            last_idx = None

        for idx in range(start, end):
            role = _frame_role_at(raw_hf_dataset, idx)
            if role not in keep_frame_roles:
                flush()
                continue

            if not _expert_label_complete_at(raw_hf_dataset, idx, require_complete_expert_action):
                flush()
                continue

            segment_id = _segment_id_at(raw_hf_dataset, idx, fallback=idx)
            if (
                current_indices
                and (segment_id != current_segment_id or last_idx is None or idx != last_idx + 1)
            ):
                flush()

            if not current_indices:
                current_indices.extend(
                    _context_indices_before(
                        raw_hf_dataset=raw_hf_dataset,
                        start=start,
                        first_idx=idx,
                        pre_takeover_context=pre_takeover_context,
                        require_complete_expert_action=require_complete_expert_action,
                    )
                )

            current_indices.append(idx)
            current_segment_id = segment_id
            last_idx = idx

        flush()

    return segments


def _merge_cfg(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(defaults)
    if override:
        cfg.update({key: value for key, value in override.items() if value is not None})
    return cfg


def _validate_export_mode(export_mode: str | None) -> str:
    mode = str(export_mode or EXPORT_MODE_INTERVENTION_SEGMENTS).strip().lower()
    if mode not in VALID_EXPORT_MODES:
        raise ValueError(
            f"Unknown DAgger export_mode `{export_mode}`. "
            f"Expected one of: {sorted(VALID_EXPORT_MODES)}."
        )
    return mode


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _indices_are_contiguous(indices: list[int]) -> bool:
    return all(next_idx == idx + 1 for idx, next_idx in zip(indices, indices[1:], strict=False))


def _has_timestamp_continuity(raw_hf_dataset, indices: list[int], fps: int) -> bool:
    if len(indices) < 2 or "timestamp" not in raw_hf_dataset.column_names:
        return True

    timestamps = [float(_to_scalar(raw_hf_dataset["timestamp"][idx])) for idx in indices]
    expected_dt = 1.0 / max(1, fps)
    tolerance = max(1e-4, expected_dt * 0.5)
    for left, right in zip(timestamps, timestamps[1:], strict=False):
        dt = right - left
        if dt <= 0 or abs(dt - expected_dt) > tolerance:
            return False
    return True


def _label_field_for_raw_item(raw_item: dict[str, Any], label_source: str) -> str:
    label_source = str(label_source)
    if label_source == "hybrid_by_frame_role":
        role = _to_str(raw_item.get("frame_role", "policy"))
        if role in DEFAULT_KEEP_FRAME_ROLES and _to_bool(raw_item.get("expert_label_complete", False)):
            return "expert_action"
        return "sent_action"
    if label_source not in {"expert_action", "sent_action", "policy_action", ACTION}:
        raise ValueError(
            f"Unknown action label source `{label_source}`. "
            "Expected one of: sent_action, expert_action, policy_action, action, hybrid_by_frame_role."
        )
    return label_source


def _make_standard_frame(
    raw_item: dict[str, Any],
    output_features: dict[str, dict],
    label_source: str = "expert_action",
) -> dict[str, Any]:
    action_field = _label_field_for_raw_item(raw_item, label_source)
    if action_field not in raw_item:
        raise KeyError(
            f"Raw run_mix dataset is missing `{action_field}`; "
            f"cannot export action labels from `{label_source}` safely."
        )

    frame: dict[str, Any] = {"task": raw_item["task"]}
    for key, feature in output_features.items():
        if key in DEFAULT_FEATURES:
            continue
        if key == ACTION:
            frame[key] = _coerce_numpy_feature(raw_item[action_field], feature)
            continue
        if key not in raw_item:
            raise KeyError(f"Raw run_mix frame is missing required training feature `{key}`.")
        frame[key] = _coerce_numpy_feature(raw_item[key], feature)
    return frame


def _flatten_feature_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    if isinstance(names, dict):
        flattened: list[str] = []
        for value in names.values():
            flattened.extend(_flatten_feature_names(value))
        return flattened
    if isinstance(names, (list, tuple)):
        flattened = []
        for value in names:
            flattened.extend(_flatten_feature_names(value))
        return flattened
    return [str(names)]


def _action_dim(action_feature: dict[str, Any]) -> int:
    shape = _feature_shape(action_feature)
    if not shape:
        return 1
    dim = 1
    for value in shape:
        dim *= int(value)
    return dim


def _resolve_gripper_indices(
    output_features: dict[str, dict],
    gripper_action_indices: list[int] | tuple[int, ...] | None,
) -> tuple[list[int], str]:
    action_feature = output_features[ACTION]
    action_dim = _action_dim(action_feature)
    if gripper_action_indices is not None:
        indices = [int(index) for index in gripper_action_indices]
        if not indices:
            return [], "disabled_by_empty_config"
        invalid = [index for index in indices if index < 0 or index >= action_dim]
        if invalid:
            raise ValueError(
                f"gripper_action_indices contains invalid indices {invalid}; action_dim={action_dim}."
            )
        return indices, "config"

    action_names = _flatten_feature_names(action_feature.get("names"))
    if len(action_names) != action_dim:
        return [], "not_configured"
    indices = [
        index
        for index, name in enumerate(action_names)
        if "grip" in name.lower() or "gripper" in name.lower()
    ]
    return indices, "feature_names" if indices else "not_configured"


class GripperDiagnostics:
    def __init__(
        self,
        output_features: dict[str, dict],
        gripper_action_indices: list[int] | tuple[int, ...] | None,
        open_threshold: float,
    ) -> None:
        self.indices, self.source = _resolve_gripper_indices(output_features, gripper_action_indices)
        self.open_threshold = float(open_threshold)
        self.current_subtype: str | None = None
        self.current_open_steps: list[bool] = []
        self.by_subtype: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "frames": 0,
                "open_frames": 0,
                "close_frames": 0,
                "episodes": 0,
                "first_open_timesteps": [],
                "early_open_episodes": 0,
            }
        )

    @property
    def enabled(self) -> bool:
        return bool(self.indices)

    def start_episode(self, subtype: str) -> None:
        self.current_subtype = subtype
        self.current_open_steps = []

    def observe_action(self, action: Any) -> None:
        if not self.enabled or self.current_subtype is None:
            return
        action_array = np.asarray(action).reshape(-1)
        is_open = bool(np.any(action_array[self.indices] > self.open_threshold))
        self.current_open_steps.append(is_open)

    def finish_episode(self) -> None:
        if not self.enabled or self.current_subtype is None:
            self.current_subtype = None
            self.current_open_steps = []
            return

        subtype_stats = self.by_subtype[self.current_subtype]
        episode_len = len(self.current_open_steps)
        open_frames = sum(1 for is_open in self.current_open_steps if is_open)
        first_open = next(
            (idx for idx, is_open in enumerate(self.current_open_steps) if is_open),
            None,
        )
        subtype_stats["frames"] += episode_len
        subtype_stats["open_frames"] += open_frames
        subtype_stats["close_frames"] += episode_len - open_frames
        subtype_stats["episodes"] += 1
        subtype_stats["first_open_timesteps"].append(first_open)
        if first_open is not None and first_open < episode_len * 0.5:
            subtype_stats["early_open_episodes"] += 1

        self.current_subtype = None
        self.current_open_steps = []

    def summary(self) -> dict[str, Any]:
        warnings: list[str] = []
        if not self.enabled:
            return {
                "enabled": False,
                "indices": [],
                "index_source": self.source,
                "open_threshold": self.open_threshold,
                "by_subtype": {},
                "warnings": warnings,
            }

        by_subtype: dict[str, dict[str, Any]] = {}
        for subtype, stats in sorted(self.by_subtype.items()):
            frames = int(stats["frames"])
            episodes = int(stats["episodes"])
            open_ratio = stats["open_frames"] / max(1, frames)
            early_ratio = stats["early_open_episodes"] / max(1, episodes)
            by_subtype[subtype] = {
                "frames": frames,
                "episodes": episodes,
                "open_ratio": open_ratio,
                "close_ratio": stats["close_frames"] / max(1, frames),
                "first_open_timesteps": stats["first_open_timesteps"],
                "early_open_episode_ratio": early_ratio,
            }

            if subtype == "intervention_segment":
                if open_ratio > 0.6:
                    warnings.append(
                        "Recovery/intervention segments have high gripper-open ratio; "
                        "check whether release labels dominate recovery data."
                    )
                if early_ratio > 0.5:
                    warnings.append(
                        "Many recovery/intervention segments open the gripper in the first half."
                    )
            elif subtype == "full_success_episode" and early_ratio > 0.5:
                warnings.append(
                    "Many full successful episodes open the gripper in the first half; "
                    "verify that success annotations include the full insert/release timing."
                )

        return {
            "enabled": True,
            "indices": self.indices,
            "index_source": self.source,
            "open_threshold": self.open_threshold,
            "by_subtype": by_subtype,
            "warnings": warnings,
        }


SUCCESS_TRUE_STRINGS = {"1", "true", "yes", "y", "success", "succeeded", "complete", "completed", "done"}
SUCCESS_FALSE_STRINGS = {"0", "false", "no", "n", "fail", "failed", "failure", "incomplete", "aborted"}


def _success_value_to_bool(value: Any) -> bool | None:
    value = _to_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(float(value)):
            return None
        return float(value) > 0.5
    value_str = str(value).strip().lower()
    if value_str in SUCCESS_TRUE_STRINGS:
        return True
    if value_str in SUCCESS_FALSE_STRINGS:
        return False
    return None


def _resolve_success_policy(full_episode_cfg: dict[str, Any]) -> str:
    success_policy = full_episode_cfg.get("success_policy")
    if success_policy is None:
        return (
            SUCCESS_POLICY_EXPLICIT
            if bool(full_episode_cfg.get("require_success", True))
            else SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE
        )

    success_policy = str(success_policy).strip().lower()
    if success_policy not in VALID_SUCCESS_POLICIES:
        raise ValueError(
            "`full_episode.success_policy` must be one of "
            f"{sorted(VALID_SUCCESS_POLICIES)}. Got: {success_policy!r}"
        )
    return success_policy


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _episode_success_from_metadata(
    episode: dict[str, Any],
    success_field: str,
    fallback_aliases: bool,
) -> tuple[bool | None, str | None]:
    candidates = [success_field]
    if fallback_aliases:
        candidates.extend(
            [
                "success",
                "is_success",
                "episode_success",
                "task_success",
                "task_status",
                "status",
                "done",
                "completed",
                f"info/{success_field}",
                f"stats/{success_field}/max",
                f"stats/{success_field}/mean",
            ]
        )
    for key in _ordered_unique(candidates):
        if key not in episode:
            continue
        success = _success_value_to_bool(episode[key])
        if success is not None:
            return success, f"episode_metadata:{key}"
    return None, None


def _episode_success_from_frames(
    raw_hf_dataset,
    start: int,
    end: int,
    success_field: str,
    fallback_aliases: bool,
) -> tuple[bool | None, str | None]:
    candidates = [success_field]
    if fallback_aliases:
        candidates.extend(
            [
                "success",
                "is_success",
                "episode_success",
                "task_success",
                "task_status",
                "status",
                "done",
                "next.success",
            ]
        )

    columns = set(raw_hf_dataset.column_names)
    for key in _ordered_unique(candidates):
        if key not in columns:
            continue
        values = [_success_value_to_bool(raw_hf_dataset[key][idx]) for idx in range(start, end)]
        values = [value for value in values if value is not None]
        if not values:
            continue
        if key in {"done", "status", "task_status"}:
            return values[-1], f"frame_field:{key}"
        return any(values), f"frame_field:{key}"
    return None, None


def _episode_success(
    raw_hf_dataset,
    episode: dict[str, Any],
    start: int,
    end: int,
    full_episode_cfg: dict[str, Any],
) -> tuple[bool | None, str | None]:
    success_field = str(full_episode_cfg.get("success_field") or "success")
    fallback_aliases = bool(full_episode_cfg.get("fallback_success_from_episode_metadata", True))
    success, source = _episode_success_from_metadata(episode, success_field, fallback_aliases)
    if success is not None:
        return success, source
    return _episode_success_from_frames(raw_hf_dataset, start, end, success_field, fallback_aliases)


def _episode_has_recorded_success_inference(
    raw_hf_dataset,
    episode: dict[str, Any],
    start: int,
    end: int,
) -> bool:
    for key in (
        "success_inferred_from_recorded_episode",
        "info/success_inferred_from_recorded_episode",
    ):
        if key in episode:
            inferred = _success_value_to_bool(episode[key])
            if inferred is not None:
                return inferred

    columns = set(raw_hf_dataset.column_names)
    if "success_inferred_from_recorded_episode" in columns:
        values = [
            _success_value_to_bool(raw_hf_dataset["success_inferred_from_recorded_episode"][idx])
            for idx in range(start, end)
        ]
        if any(value is True for value in values):
            return True

    if "success_policy" in columns:
        policies = {
            str(_to_scalar(raw_hf_dataset["success_policy"][idx]) or "").strip().lower()
            for idx in range(start, end)
        }
        return SUCCESS_POLICY_RECORDED_IS_SUCCESS in policies

    return False


def _episode_success_for_policy(
    raw_hf_dataset,
    episode: dict[str, Any],
    start: int,
    end: int,
    full_episode_cfg: dict[str, Any],
    success_policy: str,
) -> dict[str, Any]:
    success, success_source = _episode_success(raw_hf_dataset, episode, start, end, full_episode_cfg)

    if success is False:
        return {
            "success": False,
            "success_source": success_source,
            "success_inferred_from_recorded_episode": False,
            "success_validation_skipped": False,
            "skip_reason": "explicit_failure",
        }

    if success is True:
        inferred_from_recorded = (
            success_policy == SUCCESS_POLICY_RECORDED_IS_SUCCESS
            and _episode_has_recorded_success_inference(raw_hf_dataset, episode, start, end)
        )
        return {
            "success": True,
            "success_source": success_source,
            "success_inferred_from_recorded_episode": inferred_from_recorded,
            "success_validation_skipped": False,
            "skip_reason": None,
        }

    if success_policy == SUCCESS_POLICY_RECORDED_IS_SUCCESS:
        return {
            "success": True,
            "success_source": "recorded_episode",
            "success_inferred_from_recorded_episode": True,
            "success_validation_skipped": False,
            "skip_reason": None,
        }

    if success_policy == SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE:
        return {
            "success": True,
            "success_source": "missing_allowed_for_smoke",
            "success_inferred_from_recorded_episode": False,
            "success_validation_skipped": True,
            "skip_reason": None,
        }

    return {
        "success": None,
        "success_source": None,
        "success_inferred_from_recorded_episode": False,
        "success_validation_skipped": False,
        "skip_reason": "missing_success",
    }


def _raw_required_columns(
    output_features: dict[str, dict],
    label_source: str,
) -> set[str]:
    columns = {key for key in output_features if key not in DEFAULT_FEATURES and key != ACTION}
    if label_source == "hybrid_by_frame_role":
        columns.update({"sent_action", "expert_action", "frame_role", "expert_label_complete"})
    else:
        columns.add(label_source)
    return columns


def _raw_available_keys(raw_dataset: LeRobotDataset, raw_hf_dataset) -> set[str]:
    # Video/image features may be decoded by LeRobotDataset.__getitem__ from
    # dataset metadata rather than appearing directly in hf_dataset.column_names.
    return set(raw_dataset.features) | set(raw_hf_dataset.column_names)


def _raw_extra_features(raw_dataset: LeRobotDataset, output_features: dict[str, dict]) -> list[str]:
    return sorted(set(raw_dataset.features) - set(output_features))


def _init_export_state() -> dict[str, Any]:
    return {
        "exported_frames": 0,
        "exported_episodes": 0,
        "exported_raw_episode_indices": set(),
        "exported_intervention_ids": set(),
        "episode_metadata": [],
        "subtype_frames": Counter(),
        "subtype_episodes": Counter(),
    }


def _save_exported_episode(
    *,
    raw_dataset: LeRobotDataset,
    output_dataset: LeRobotDataset,
    output_features: dict[str, dict],
    indices: list[int],
    label_source: str,
    export_mode: str,
    export_subtype: str,
    source_raw_episode_index: int,
    source_intervention_segment_id: int | None,
    state: dict[str, Any],
    gripper_diagnostics: GripperDiagnostics,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    gripper_diagnostics.start_episode(export_subtype)
    for raw_idx in indices:
        raw_item = raw_dataset[raw_idx]
        frame = _make_standard_frame(raw_item, output_features, label_source=label_source)
        gripper_diagnostics.observe_action(frame[ACTION])
        output_dataset.add_frame(frame)

    exported_episode_index = int(state["exported_episodes"])
    output_dataset.save_episode()
    gripper_diagnostics.finish_episode()

    state["exported_episodes"] += 1
    state["exported_frames"] += len(indices)
    state["exported_raw_episode_indices"].add(int(source_raw_episode_index))
    state["subtype_frames"][export_subtype] += len(indices)
    state["subtype_episodes"][export_subtype] += 1
    if source_intervention_segment_id is not None:
        state["exported_intervention_ids"].add(
            (int(source_raw_episode_index), int(source_intervention_segment_id))
        )
    record = {
        "exported_episode_index": exported_episode_index,
        "export_mode": export_mode,
        "export_subtype": export_subtype,
        "source_raw_episode_index": int(source_raw_episode_index),
        "source_intervention_segment_id": source_intervention_segment_id,
        "source_raw_start_index": int(indices[0]),
        "source_raw_end_index_inclusive": int(indices[-1]),
        "length": len(indices),
        "action_label_source": label_source,
    }
    if extra_metadata:
        record.update(extra_metadata)
    state["episode_metadata"].append(record)


def _filter_segments_by_episode_limit(
    segments: list[dict[str, Any]],
    max_segments_per_episode: int | None,
) -> list[dict[str, Any]]:
    if max_segments_per_episode is None:
        return segments

    counts: Counter[int] = Counter()
    kept: list[dict[str, Any]] = []
    for segment in segments:
        raw_episode_index = int(segment["raw_episode_index"])
        if counts[raw_episode_index] >= max_segments_per_episode:
            continue
        counts[raw_episode_index] += 1
        kept.append(segment)
    return kept


def _export_intervention_segments(
    *,
    raw_dataset: LeRobotDataset,
    output_dataset: LeRobotDataset,
    output_features: dict[str, dict],
    export_mode: str,
    intervention_cfg: dict[str, Any],
    effective_min_episode_len: int,
    pre_takeover_context: int,
    state: dict[str, Any],
    gripper_diagnostics: GripperDiagnostics,
    skip_raw_episode_indices: set[int] | None = None,
    max_total_frames: int | None = None,
) -> dict[str, Any]:
    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    label_source = str(intervention_cfg.get("label_source") or "expert_action")
    keep_frame_roles = tuple(intervention_cfg.get("keep_frame_roles") or DEFAULT_KEEP_FRAME_ROLES)
    require_complete = bool(intervention_cfg.get("require_complete_expert_action", True))
    max_segments_per_episode = _normalize_optional_int(intervention_cfg.get("max_segments_per_episode"))
    missing_columns = sorted(_raw_required_columns(output_features, label_source) - _raw_available_keys(raw_dataset, raw_hf_dataset))
    if missing_columns:
        raise KeyError(
            "Raw run_mix dataset is missing required intervention export columns: "
            f"{missing_columns}."
        )

    segments = _iter_export_segments(
        raw_dataset,
        set(keep_frame_roles),
        pre_takeover_context=pre_takeover_context,
        require_complete_expert_action=require_complete,
    )
    segments = _filter_segments_by_episode_limit(segments, max_segments_per_episode)

    stats = {
        "intervention_segment_candidates": len(segments),
        "intervention_segments_exported": 0,
        "intervention_segment_frames_exported": 0,
        "skipped_short_segments": 0,
        "intervention_segments_skipped_duplicate_full_episode": 0,
        "intervention_segments_skipped_ratio_cap": 0,
    }
    skip_raw_episode_indices = skip_raw_episode_indices or set()
    remaining_frames = max_total_frames

    for segment in segments:
        raw_episode_index = int(segment["raw_episode_index"])
        if raw_episode_index in skip_raw_episode_indices:
            stats["intervention_segments_skipped_duplicate_full_episode"] += 1
            continue

        indices = list(segment["indices"])
        if len(indices) < effective_min_episode_len:
            stats["skipped_short_segments"] += 1
            continue
        if remaining_frames is not None and len(indices) > remaining_frames:
            stats["intervention_segments_skipped_ratio_cap"] += 1
            continue

        _save_exported_episode(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            indices=indices,
            label_source=label_source,
            export_mode=export_mode,
            export_subtype="intervention_segment",
            source_raw_episode_index=raw_episode_index,
            source_intervention_segment_id=segment["intervention_segment_id"],
            state=state,
            gripper_diagnostics=gripper_diagnostics,
        )
        stats["intervention_segments_exported"] += 1
        stats["intervention_segment_frames_exported"] += len(indices)
        if remaining_frames is not None:
            remaining_frames -= len(indices)

    return stats


def _select_full_success_episodes(
    *,
    raw_dataset: LeRobotDataset,
    output_features: dict[str, dict],
    full_episode_cfg: dict[str, Any],
    effective_min_episode_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    raw_columns = _raw_available_keys(raw_dataset, raw_hf_dataset)
    label_source = str(full_episode_cfg.get("action_label_source") or "sent_action")
    success_policy = _resolve_success_policy(full_episode_cfg)
    missing_columns = sorted(_raw_required_columns(output_features, label_source) - raw_columns)

    stats = {
        "full_episode_candidates": raw_dataset.num_episodes,
        "full_episode_exported": 0,
        "full_episode_skipped_no_success": 0,
        "full_episode_skipped_reset": 0,
        "full_episode_skipped_too_short": 0,
        "full_episode_frames_exported": 0,
        "full_episode_skipped_missing_success_info": 0,
        "full_episode_skipped_missing_label_source": 0,
        "full_episode_skipped_discontinuous": 0,
        "full_episode_success_sources": Counter(),
        "full_episode_success_policy": success_policy,
        "full_episode_success_required": success_policy != SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE,
        "full_episode_missing_required_columns": missing_columns,
        "success_policy": success_policy,
        "success_inferred_episodes": 0,
        "success_explicit_true_episodes": 0,
        "success_explicit_false_episodes": 0,
        "success_missing_allowed_episodes": 0,
        "success_validation_skipped_episodes": 0,
        "skipped_no_success": 0,
        "skipped_explicit_failure": 0,
        "warnings": [],
    }
    selected: list[dict[str, Any]] = []
    if missing_columns:
        stats["full_episode_skipped_missing_label_source"] = raw_dataset.num_episodes
        return selected, stats

    require_no_reset = bool(full_episode_cfg.get("require_no_reset", True))
    require_min_len = bool(full_episode_cfg.get("require_min_len", True))
    drop_reset_ignore_frames = bool(full_episode_cfg.get("drop_reset_ignore_frames", True))

    for raw_episode_index in range(raw_dataset.num_episodes):
        episode = raw_dataset.meta.episodes[raw_episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])

        success_info = _episode_success_for_policy(
            raw_hf_dataset,
            episode,
            start,
            end,
            full_episode_cfg,
            success_policy,
        )
        success = success_info["success"]
        success_source = success_info["success_source"]
        if success_info["skip_reason"] == "missing_success":
            stats["full_episode_skipped_missing_success_info"] += 1
            stats["full_episode_skipped_no_success"] += 1
            stats["skipped_no_success"] += 1
            continue
        if success_info["skip_reason"] == "explicit_failure":
            stats["full_episode_skipped_no_success"] += 1
            stats["success_explicit_false_episodes"] += 1
            stats["skipped_explicit_failure"] += 1
            continue
        indices = list(range(start, end))
        roles = [_frame_role_at(raw_hf_dataset, idx) for idx in indices]
        if require_no_reset and any(role == "reset" for role in roles):
            stats["full_episode_skipped_reset"] += 1
            continue

        if drop_reset_ignore_frames:
            indices = [
                idx
                for idx, role in zip(indices, roles, strict=False)
                if role not in {"reset", "ignore"}
            ]

        if require_min_len and len(indices) < effective_min_episode_len:
            stats["full_episode_skipped_too_short"] += 1
            continue

        if not _indices_are_contiguous(indices) or not _has_timestamp_continuity(
            raw_hf_dataset, indices, raw_dataset.fps
        ):
            stats["full_episode_skipped_discontinuous"] += 1
            continue

        if success_info["success_inferred_from_recorded_episode"]:
            stats["success_inferred_episodes"] += 1
        elif success_info["success_validation_skipped"]:
            stats["success_missing_allowed_episodes"] += 1
            stats["success_validation_skipped_episodes"] += 1
        elif success is True:
            stats["success_explicit_true_episodes"] += 1

        if success_source is not None:
            stats["full_episode_success_sources"][success_source] += 1
        selected.append(
            {
                "raw_episode_index": raw_episode_index,
                "indices": indices,
                "success": success,
                "success_source": success_source,
                "success_policy": success_policy,
                "success_inferred_from_recorded_episode": success_info[
                    "success_inferred_from_recorded_episode"
                ],
                "success_validation_skipped": success_info["success_validation_skipped"],
                "action_label_source": label_source,
            }
        )

    if stats["success_validation_skipped_episodes"] > 0:
        stats["warnings"].append(SMOKE_MISSING_SUCCESS_WARNING)

    return selected, stats


def _export_full_success_episodes(
    *,
    raw_dataset: LeRobotDataset,
    output_dataset: LeRobotDataset,
    output_features: dict[str, dict],
    export_mode: str,
    full_episode_cfg: dict[str, Any],
    effective_min_episode_len: int,
    state: dict[str, Any],
    gripper_diagnostics: GripperDiagnostics,
) -> dict[str, Any]:
    selected, stats = _select_full_success_episodes(
        raw_dataset=raw_dataset,
        output_features=output_features,
        full_episode_cfg=full_episode_cfg,
        effective_min_episode_len=effective_min_episode_len,
    )

    for episode in selected:
        indices = episode["indices"]
        _save_exported_episode(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            indices=indices,
            label_source=episode["action_label_source"],
            export_mode=export_mode,
            export_subtype="full_success_episode",
            source_raw_episode_index=episode["raw_episode_index"],
            source_intervention_segment_id=None,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
            extra_metadata={
                "success": episode["success"],
                "success_source": episode["success_source"],
                "success_policy": episode["success_policy"],
                "success_inferred_from_recorded_episode": episode[
                    "success_inferred_from_recorded_episode"
                ],
                "success_validation_skipped": episode["success_validation_skipped"],
            },
        )
        stats["full_episode_exported"] += 1
        stats["full_episode_frames_exported"] += len(indices)

    stats["full_episode_success_sources"] = dict(stats["full_episode_success_sources"])
    return stats


def _export_hybrid(
    *,
    raw_dataset: LeRobotDataset,
    output_dataset: LeRobotDataset,
    output_features: dict[str, dict],
    full_episode_cfg: dict[str, Any],
    intervention_cfg: dict[str, Any],
    hybrid_cfg: dict[str, Any],
    effective_min_episode_len: int,
    pre_takeover_context: int,
    state: dict[str, Any],
    gripper_diagnostics: GripperDiagnostics,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    full_episode_stats: dict[str, Any] = {}
    intervention_stats: dict[str, Any] = {}

    if bool(hybrid_cfg.get("include_full_success_episode", True)):
        full_episode_stats = _export_full_success_episodes(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            export_mode=EXPORT_MODE_HYBRID,
            full_episode_cfg=full_episode_cfg,
            effective_min_episode_len=effective_min_episode_len,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
        )
        stats.update(full_episode_stats)

    max_recovery_segments_per_episode = hybrid_cfg.get("max_recovery_segments_per_episode")
    hybrid_intervention_cfg = dict(intervention_cfg)
    if max_recovery_segments_per_episode is not None:
        hybrid_intervention_cfg["max_segments_per_episode"] = int(max_recovery_segments_per_episode)

    full_frames = int(state["subtype_frames"].get("full_success_episode", 0))
    recovery_frame_cap: int | None = None
    max_ratio = hybrid_cfg.get("max_recovery_frames_ratio")
    if max_ratio is not None and full_frames > 0:
        recovery_frame_cap = max(0, int(full_frames * float(max_ratio)))

    duplicate_skip = set()
    if bool(hybrid_cfg.get("prefer_full_episode_when_duplicate", True)):
        duplicate_skip = set(state["exported_raw_episode_indices"])

    if bool(hybrid_cfg.get("include_intervention_segments", True)):
        intervention_stats = _export_intervention_segments(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            export_mode=EXPORT_MODE_HYBRID,
            intervention_cfg=hybrid_intervention_cfg,
            effective_min_episode_len=effective_min_episode_len,
            pre_takeover_context=pre_takeover_context,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
            skip_raw_episode_indices=duplicate_skip,
            max_total_frames=recovery_frame_cap,
        )
        stats.update(intervention_stats)

    hybrid_full_frames = int(state["subtype_frames"].get("full_success_episode", 0))
    hybrid_recovery_frames = int(state["subtype_frames"].get("intervention_segment", 0))
    stats.update(
        {
            "hybrid_full_frames": hybrid_full_frames,
            "hybrid_recovery_frames": hybrid_recovery_frames,
            "hybrid_full_episodes": int(state["subtype_episodes"].get("full_success_episode", 0)),
            "hybrid_recovery_segments": int(state["subtype_episodes"].get("intervention_segment", 0)),
            "hybrid_recovery_ratio": hybrid_recovery_frames / max(1, hybrid_full_frames),
        }
    )
    return stats


def export_dagger_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    seed_root: str | Path | None = None,
    raw_repo_id: str | None = None,
    output_repo_id: str | None = None,
    seed_repo_id: str | None = None,
    keep_frame_roles: tuple[str, ...] = DEFAULT_KEEP_FRAME_ROLES,
    min_segment_frames: int = 1,
    min_episode_len_for_act: int | None = None,
    export_mode: str = EXPORT_MODE_INTERVENTION_SEGMENTS,
    pre_takeover_context: int = 0,
    require_complete_expert_action: bool = True,
    full_episode: dict[str, Any] | None = None,
    intervention_segments: dict[str, Any] | None = None,
    hybrid: dict[str, Any] | None = None,
    gripper_action_indices: list[int] | tuple[int, ...] | None = None,
    gripper_open_threshold: float = 0.5,
    overwrite: bool = False,
    image_writer_processes: int = 0,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    """Export the current ACT chunk DAgger profile into a standard LeRobot dataset.

    Round controllers should let the selected backend/export profile prepare
    these arguments instead of hard-coding policy-specific export rules.
    """

    export_mode = _validate_export_mode(export_mode)
    full_episode_cfg = _merge_cfg(DEFAULT_FULL_EPISODE_CFG, full_episode)
    full_episode_cfg["success_policy"] = _resolve_success_policy(full_episode_cfg)
    intervention_cfg = _merge_cfg(DEFAULT_INTERVENTION_SEGMENTS_CFG, intervention_segments)
    hybrid_cfg = _merge_cfg(DEFAULT_HYBRID_CFG, hybrid)

    if intervention_segments is None or "keep_frame_roles" not in intervention_segments:
        intervention_cfg["keep_frame_roles"] = list(keep_frame_roles)
    if intervention_segments is None or "require_complete_expert_action" not in intervention_segments:
        intervention_cfg["require_complete_expert_action"] = bool(require_complete_expert_action)

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    seed_root = _resolve_root(seed_repo_id, seed_root)
    raw_repo_id = raw_repo_id or _repo_id_from_root(raw_root, "raw_run_mix")
    output_repo_id = output_repo_id or _repo_id_from_root(output_root, "dagger_export")
    seed_repo_id = seed_repo_id or _repo_id_from_root(seed_root, "seed")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)

    seed_meta = LeRobotDatasetMetadata(seed_repo_id, root=seed_root)
    raw_dataset = LeRobotDataset(raw_repo_id, root=raw_root)
    output_features = seed_meta.features
    use_videos = any(feature["dtype"] == "video" for feature in output_features.values())
    effective_min_episode_len = max(min_segment_frames, min_episode_len_for_act or 1)
    gripper_diagnostics = GripperDiagnostics(
        output_features=output_features,
        gripper_action_indices=gripper_action_indices,
        open_threshold=gripper_open_threshold,
    )

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=seed_meta.fps,
        features=output_features,
        robot_type=seed_meta.robot_type,
        root=output_root,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    output_dataset.meta.metadata_buffer_size = 1

    state = _init_export_state()
    mode_stats: dict[str, Any]
    if export_mode == EXPORT_MODE_INTERVENTION_SEGMENTS:
        mode_stats = _export_intervention_segments(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            export_mode=export_mode,
            intervention_cfg=intervention_cfg,
            effective_min_episode_len=effective_min_episode_len,
            pre_takeover_context=pre_takeover_context,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
        )
    elif export_mode == EXPORT_MODE_FULL_SUCCESS_EPISODE:
        mode_stats = _export_full_success_episodes(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            export_mode=export_mode,
            full_episode_cfg=full_episode_cfg,
            effective_min_episode_len=effective_min_episode_len,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
        )
    else:
        mode_stats = _export_hybrid(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            output_features=output_features,
            full_episode_cfg=full_episode_cfg,
            intervention_cfg=intervention_cfg,
            hybrid_cfg=hybrid_cfg,
            effective_min_episode_len=effective_min_episode_len,
            pre_takeover_context=pre_takeover_context,
            state=state,
            gripper_diagnostics=gripper_diagnostics,
        )

    exported_frames = int(state["exported_frames"])
    exported_episodes = int(state["exported_episodes"])
    output_dataset.finalize()
    if exported_frames > 0:
        exported_meta = LeRobotDatasetMetadata(output_repo_id, root=output_root)
        assert_lerobot_schema_compatible(
            seed_meta,
            exported_meta,
            reference_name="seed",
            candidate_name="exported_dagger",
        )

    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    total_raw_frames = len(raw_dataset)
    expert_frames = (
        sum(1 for value in raw_hf_dataset["is_expert"] if _to_bool(value))
        if "is_expert" in raw_hf_dataset.column_names
        else exported_frames
    )
    intervention_ids = set()
    if "intervention_segment_id" in raw_hf_dataset.column_names:
        for value in raw_hf_dataset["intervention_segment_id"]:
            segment_id = _to_int(value)
            if segment_id >= 0:
                intervention_ids.add(segment_id)
    incomplete_expert_label_frames = None
    if "expert_label_complete" in raw_hf_dataset.column_names and "is_expert" in raw_hf_dataset.column_names:
        incomplete_expert_label_frames = sum(
            1
            for is_expert, is_complete in zip(
                raw_hf_dataset["is_expert"],
                raw_hf_dataset["expert_label_complete"],
                strict=False,
            )
            if _to_bool(is_expert) and not _to_bool(is_complete)
        )
    raw_extra_features = _raw_extra_features(raw_dataset, output_features)
    gripper_summary = gripper_diagnostics.summary()
    export_warnings = list(mode_stats.get("warnings", []))
    if mode_stats.get("full_episode_skipped_missing_success_info"):
        logging.warning(
            "[dagger_export] skipped %d full episode candidate(s) because no success indicator was found; "
            "set full_episode.require_success=false only for smoke tests.",
            mode_stats["full_episode_skipped_missing_success_info"],
        )
    for warning in export_warnings:
        logging.warning("[dagger_export] %s", warning)
    for warning in gripper_summary.get("warnings", []):
        logging.warning("[dagger_export] %s", warning)

    exported_intervention_ids = state["exported_intervention_ids"]
    exported_raw_episode_indices = state["exported_raw_episode_indices"]
    label_source = (
        intervention_cfg.get("label_source")
        if export_mode == EXPORT_MODE_INTERVENTION_SEGMENTS
        else full_episode_cfg.get("action_label_source")
        if export_mode == EXPORT_MODE_FULL_SUCCESS_EPISODE
        else "by_export_subtype"
    )
    dropped_frame_roles = (
        ["policy", "reset", "ignore"]
        if export_mode == EXPORT_MODE_INTERVENTION_SEGMENTS
        else ["reset", "ignore"]
        if export_mode == EXPORT_MODE_FULL_SUCCESS_EPISODE and full_episode_cfg.get("drop_reset_ignore_frames", True)
        else "by_export_subtype"
        if export_mode == EXPORT_MODE_HYBRID
        else []
    )
    full_episode_frames = int(
        state["subtype_frames"].get(
            "full_success_episode",
            mode_stats.get("full_episode_frames_exported", 0),
        )
        or 0
    )
    full_episode_count = int(
        state["subtype_episodes"].get(
            "full_success_episode",
            mode_stats.get("full_episode_exported", 0),
        )
        or 0
    )
    recovery_frames = int(
        state["subtype_frames"].get(
            "intervention_segment",
            mode_stats.get("intervention_segment_frames_exported", 0),
        )
        or 0
    )
    recovery_segment_count = int(
        state["subtype_episodes"].get(
            "intervention_segment",
            mode_stats.get("intervention_segments_exported", 0),
        )
        or 0
    )
    hybrid_full_frames = int(mode_stats.get("hybrid_full_frames", full_episode_frames if export_mode == EXPORT_MODE_HYBRID else 0) or 0)
    hybrid_full_episodes = int(
        mode_stats.get("hybrid_full_episodes", full_episode_count if export_mode == EXPORT_MODE_HYBRID else 0)
        or 0
    )
    hybrid_recovery_frames = int(
        mode_stats.get("hybrid_recovery_frames", recovery_frames if export_mode == EXPORT_MODE_HYBRID else 0)
        or 0
    )
    hybrid_recovery_segments = int(
        mode_stats.get("hybrid_recovery_segments", recovery_segment_count if export_mode == EXPORT_MODE_HYBRID else 0)
        or 0
    )
    hybrid_recovery_ratio = float(mode_stats.get("hybrid_recovery_ratio", 0.0) or 0.0)

    summary = {
        "export_mode": export_mode,
        "exported_frames": exported_frames,
        "exported_episodes": exported_episodes,
        "full_episode_frames": full_episode_frames,
        "full_episode_count": full_episode_count,
        "recovery_frames": recovery_frames,
        "recovery_segment_count": recovery_segment_count,
        "hybrid_full_frames": hybrid_full_frames,
        "hybrid_full_episodes": hybrid_full_episodes,
        "hybrid_recovery_frames": hybrid_recovery_frames,
        "hybrid_recovery_segments": hybrid_recovery_segments,
        "hybrid_recovery_ratio": hybrid_recovery_ratio,
        "raw_dataset": {
            "repo_id": raw_repo_id,
            "root": str(raw_root),
            "total_frames": total_raw_frames,
            "total_episodes": raw_dataset.num_episodes,
        },
        "seed_dataset": {
            "repo_id": seed_repo_id,
            "root": str(seed_root),
        },
        "exported_dataset": {
            "repo_id": output_repo_id,
            "root": str(output_root),
            "total_frames": exported_frames,
            "total_episodes": exported_episodes,
        },
        "rules": {
            "export_mode": export_mode,
            "label_source": label_source,
            "standard_action_field": ACTION,
            "keep_frame_roles": list(intervention_cfg.get("keep_frame_roles") or DEFAULT_KEEP_FRAME_ROLES),
            "dropped_frame_roles": dropped_frame_roles,
            "min_segment_frames": min_segment_frames,
            "min_episode_len_for_act": min_episode_len_for_act,
            "effective_min_episode_len": effective_min_episode_len,
            "pre_takeover_context": pre_takeover_context,
            "require_complete_expert_action": bool(
                intervention_cfg.get("require_complete_expert_action", require_complete_expert_action)
            ),
            "full_episode": full_episode_cfg,
            "intervention_segments": intervention_cfg,
            "hybrid": hybrid_cfg,
            "raw_extra_features_dropped": raw_extra_features,
        },
        "stats": {
            "expert_frame_ratio": expert_frames / max(1, total_raw_frames),
            "expert_episode_count": len(exported_raw_episode_indices),
            "intervention_count": len(exported_intervention_ids)
            if exported_intervention_ids
            else (len(intervention_ids) if intervention_ids else mode_stats.get("intervention_segment_candidates", 0)),
            "skipped_short_segments": mode_stats.get("skipped_short_segments", 0),
            "incomplete_expert_label_frames": incomplete_expert_label_frames,
            "subtype_frames": dict(state["subtype_frames"]),
            "subtype_episodes": dict(state["subtype_episodes"]),
            **mode_stats,
            "full_episode_frames": full_episode_frames,
            "full_episode_count": full_episode_count,
            "recovery_frames": recovery_frames,
            "recovery_segment_count": recovery_segment_count,
        },
        "gripper_diagnostics": gripper_summary,
        "warnings": export_warnings + list(gripper_summary.get("warnings", [])),
    }

    episode_metadata_path = output_root / "dagger_export_episode_metadata.json"
    with open(episode_metadata_path, "w") as f:
        json.dump(state["episode_metadata"], f, indent=2)

    summary_path = output_root / "dagger_export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        "[dagger_export] mode=%s raw=%s exported=%s frames=%d episodes=%d",
        export_mode,
        raw_root,
        output_root,
        exported_frames,
        exported_episodes,
    )
    return summary


def export_from_config(config: DAggerExportConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        config = dict(config)
        if config.get("min_episode_len") is not None and config.get("min_episode_len_for_act") is None:
            config["min_episode_len_for_act"] = config["min_episode_len"]
        config.pop("min_episode_len", None)
        allowed = {field.name for field in fields(DAggerExportConfig)}
        config = {key: value for key, value in config.items() if key in allowed}
        config = DAggerExportConfig(**config)
    kwargs = config.__dict__.copy()
    kwargs.pop("profile", None)
    return export_dagger_dataset(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export standardized ACT DAgger data from raw run_mix logs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "dagger_rounds_cfg.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    export_cfg = cfg.get("dagger_export", cfg)
    export_from_config(export_cfg)


if __name__ == "__main__":
    main()
