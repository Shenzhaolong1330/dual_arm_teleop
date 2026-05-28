#!/usr/bin/env python
"""Offline diagnostics for the LeRobot Diffusion Policy deployment path.

This script is intentionally read-only. It checks:
- train/record policy selection and diffusion train-vs-reason config drift
- dataset action schema, gripper dimensions, gripper close timing
- simple action/observation gripper lag evidence
- optional checkpoint inference without connecting to a robot
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ModuleNotFoundError:  # Allow config-only diagnostics on a minimal Python.
    torch = None


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, REPO_ROOT / "src", REPO_ROOT):
    path = str(candidate)
    if path not in sys.path:
        sys.path.insert(0, path)


IMPORTANT_DIFFUSION_FIELDS = [
    "n_obs_steps",
    "horizon",
    "n_action_steps",
    "drop_n_last_frames",
    "normalization_mapping",
    "vision_backbone",
    "crop_shape",
    "crop_is_random",
    "pretrained_backbone_weights",
    "use_group_norm",
    "spatial_softmax_num_keypoints",
    "use_separate_rgb_encoder_per_camera",
    "down_dims",
    "kernel_size",
    "n_groups",
    "diffusion_step_embed_dim",
    "use_film_scale_modulation",
    "noise_scheduler_type",
    "num_train_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
    "num_inference_steps",
    "do_mask_loss_for_padding",
    "device",
    "use_amp",
    "pretrained_path",
]

MUST_MATCH_FIELDS = {
    "n_obs_steps",
    "horizon",
    "normalization_mapping",
    "vision_backbone",
    "crop_shape",
    "pretrained_backbone_weights",
    "use_group_norm",
    "spatial_softmax_num_keypoints",
    "use_separate_rgb_encoder_per_camera",
    "down_dims",
    "kernel_size",
    "n_groups",
    "diffusion_step_embed_dim",
    "use_film_scale_modulation",
    "noise_scheduler_type",
    "num_train_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
}

ALLOWED_DIFF_FIELDS = {
    "device",
    "use_amp",
    "pretrained_path",
    "n_action_steps",
    "num_inference_steps",
    "crop_is_random",
    "drop_n_last_frames",
}

DIFFUSION_DEFAULTS = {
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
    "drop_n_last_frames": 7,
    "vision_backbone": "resnet18",
    "crop_shape": [84, 84],
    "crop_is_random": True,
    "pretrained_backbone_weights": None,
    "use_group_norm": True,
    "spatial_softmax_num_keypoints": 32,
    "use_separate_rgb_encoder_per_camera": False,
    "down_dims": [512, 1024, 2048],
    "kernel_size": 5,
    "n_groups": 8,
    "diffusion_step_embed_dim": 128,
    "use_film_scale_modulation": True,
    "noise_scheduler_type": "DDPM",
    "num_train_timesteps": 100,
    "beta_schedule": "squaredcos_cap_v2",
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "prediction_type": "epsilon",
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "num_inference_steps": None,
    "do_mask_loss_for_padding": False,
    "device": "cuda",
    "use_amp": False,
    "pretrained_path": None,
}


class ConfigView(SimpleNamespace):
    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - int(self.n_obs_steps), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        start = 1 - int(self.n_obs_steps)
        return list(range(start, start + int(self.horizon)))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    return value if isinstance(value, dict) else data


def normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if torch is not None and isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(normalize_value(k)): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def short(value: Any, limit: int = 72) -> str:
    text = repr(normalize_value(value))
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def resolve_main_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    for base in (PROJECT_ROOT, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def build_policy_from_yaml(
    policy_type: str,
    config_path: Path,
    mode: str,
    legacy_policy: dict[str, Any] | None = None,
):
    try:
        from scripts.core.policy_config_utils import build_policy_config, load_policy_yaml

        policy_yaml = load_policy_yaml(config_path)
        return build_policy_config(
            policy_type,
            policy_yaml,
            legacy_policy_dict=legacy_policy,
            legacy_source_name=f"{mode}_cfg.yaml",
            config_path=config_path,
            mode=mode,
        )
    except ModuleNotFoundError:
        raw = load_yaml(config_path)
        if policy_type == "diffusion":
            merged = dict(DIFFUSION_DEFAULTS)
            merged.update(raw)
            if legacy_policy:
                for key, value in legacy_policy.items():
                    if key in merged:
                        merged[key] = value
            return ConfigView(**merged)
        merged = dict(raw)
        if legacy_policy:
            merged.update({k: v for k, v in legacy_policy.items() if k not in {"type", "config_path"}})
        return ConfigView(**merged)


def resolve_policy_path(policy: dict[str, Any], policy_type: str, mode: str) -> Path:
    try:
        from scripts.core.policy_config_utils import resolve_policy_config_path

        return resolve_policy_config_path(
            policy,
            scripts_dir=SCRIPTS_DIR,
            project_root=PROJECT_ROOT,
            mode=mode,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raw_path = policy.get("config_path")
        if raw_path is None:
            stem = "diffusion" if policy_type in {"diffusion", "dp", "diffusion_policy"} else "act"
            raw_path = f"scripts/policy_config/{stem}_{mode}_config.yaml"
        return resolve_main_path(raw_path)


def load_active_policy(main_cfg_path: Path, section_name: str, mode: str):
    data = load_yaml(main_cfg_path)
    cfg_section = section(data, section_name)
    policy = cfg_section.get("policy", {})
    policy_type = str(policy.get("type", "")).strip().lower()
    if not policy_type:
        return cfg_section, None, None, None
    policy_path = resolve_policy_path(policy, policy_type, mode)
    policy_cfg = build_policy_from_yaml(policy_type, policy_path, mode, legacy_policy=policy)
    return cfg_section, policy_type, policy_path, policy_cfg


def load_explicit_diffusion_configs(args: argparse.Namespace):
    train_path = resolve_main_path(args.diffusion_train_config)
    reason_path = resolve_main_path(args.diffusion_reason_config)
    train_cfg = build_policy_from_yaml("diffusion", train_path, "train", legacy_policy=None)
    reason_cfg = build_policy_from_yaml("diffusion", reason_path, "reason", legacy_policy=None)
    return train_path, train_cfg, reason_path, reason_cfg


def print_config_summary(
    train_main: tuple[dict[str, Any], str | None, Path | None, Any | None],
    record_main: tuple[dict[str, Any], str | None, Path | None, Any | None],
    train_diff_path: Path,
    train_diff_cfg: Any,
    reason_diff_path: Path,
    reason_diff_cfg: Any,
) -> None:
    _, train_policy_type, train_policy_path, _ = train_main
    _, record_policy_type, record_policy_path, _ = record_main

    print("\n== Active policy selection ==")
    print(f"train_cfg policy.type:  {train_policy_type} ({train_policy_path})")
    print(f"record_cfg policy.type: {record_policy_type} ({record_policy_path})")
    if train_policy_type != "diffusion":
        print("ERROR: train_cfg active policy is not diffusion.")
    if record_policy_type != "diffusion":
        print("ERROR: record_cfg active policy is not diffusion; run_record.py will not load DP.")

    print("\n== Explicit diffusion config paths ==")
    print(f"train:  {train_diff_path}")
    print(f"reason: {reason_diff_path}")

    print("\n== Diffusion train vs reason config diff ==")
    print("| field | train | reason | status |")
    print("| --- | --- | --- | --- |")
    for field in IMPORTANT_DIFFUSION_FIELDS:
        train_value = normalize_value(getattr(train_diff_cfg, field, None))
        reason_value = normalize_value(getattr(reason_diff_cfg, field, None))
        same = train_value == reason_value
        if same:
            status = "OK"
        elif field in MUST_MATCH_FIELDS:
            status = "ERROR"
        elif field in ALLOWED_DIFF_FIELDS:
            status = "WARNING"
        else:
            status = "WARNING"
        if field == "crop_is_random" and bool(reason_value):
            status = "ERROR"
        print(f"| {field} | {short(train_value)} | {short(reason_value)} | {status} |")

    pretrained_path = getattr(reason_diff_cfg, "pretrained_path", None)
    if pretrained_path:
        exists = Path(str(pretrained_path)).expanduser().exists()
        print(f"\nreason pretrained_path exists: {exists} ({pretrained_path})")

    max_action_steps = reason_diff_cfg.horizon - reason_diff_cfg.n_obs_steps + 1
    print("\n== Diffusion indexing ==")
    print(f"observation_delta_indices: {train_diff_cfg.observation_delta_indices}")
    print(f"action_delta_indices:      {train_diff_cfg.action_delta_indices}")
    print(
        "reason select_action executes generated action indices "
        f"[{reason_diff_cfg.n_obs_steps - 1}, "
        f"{reason_diff_cfg.n_obs_steps - 1 + reason_diff_cfg.n_action_steps - 1}]"
    )
    print(f"n_action_steps <= horizon - n_obs_steps + 1: {reason_diff_cfg.n_action_steps <= max_action_steps}")
    print(f"do_mask_loss_for_padding: train={train_diff_cfg.do_mask_loss_for_padding}")


def tensor_to_np(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def dataset_episode_records(dataset: Any) -> list[tuple[int, int, int]]:
    episodes = dataset.meta.episodes
    records: list[tuple[int, int, int]] = []
    if hasattr(episodes, "iterrows"):
        for row_idx, row in episodes.iterrows():
            ep_idx = int(row["episode_index"]) if "episode_index" in row else int(row_idx)
            records.append((ep_idx, int(row["dataset_from_index"]), int(row["dataset_to_index"])))
    else:
        for row_idx, row in enumerate(episodes):
            ep_idx = int(row.get("episode_index", row_idx))
            records.append((ep_idx, int(row["dataset_from_index"]), int(row["dataset_to_index"])))
    return records


def load_key_range(dataset: Any, key: str, start: int, end: int) -> np.ndarray:
    rows = [tensor_to_np(dataset.hf_dataset[i][key]) for i in range(start, end)]
    return np.stack(rows, axis=0)


def action_names_from_dataset(dataset: Any) -> list[str]:
    action_feature = dataset.features.get("action", {})
    names = action_feature.get("names")
    if names:
        return list(names)
    shape = action_feature.get("shape", [0])
    dim = int(shape[0]) if shape else 0
    return [f"action[{i}]" for i in range(dim)]


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def normalize_dims(dims: list[int] | None, action_dim: int) -> list[int] | None:
    if dims is None:
        return None
    normalized = []
    for dim in dims:
        if dim < 0:
            dim += action_dim
        if dim < 0 or dim >= action_dim:
            raise ValueError(f"Invalid gripper dim {dim} for action_dim={action_dim}.")
        normalized.append(dim)
    return normalized


def state_names_from_dataset(dataset: Any) -> list[str]:
    state_feature = dataset.features.get("observation.state", {})
    names = state_feature.get("names")
    if names:
        return list(names)
    shape = state_feature.get("shape", [0])
    dim = int(shape[0]) if shape else 0
    return [f"state[{i}]" for i in range(dim)]


def infer_action_semantics(action_names: list[str]) -> str:
    lowered = [name.lower() for name in action_names]
    if any("delta_ee_pose" in name for name in lowered):
        return "Cartesian delta EE pose + gripper command"
    if any("ee_pose" in name for name in lowered):
        return "Cartesian absolute/target EE pose + gripper command (check robot parser)"
    if any("joint" in name and ".pos" in name for name in lowered):
        return "joint position target"
    return "unknown from names"


def first_close_frame(values: np.ndarray, close_threshold: float) -> int | None:
    if len(values) == 0:
        return None
    if values[0] <= close_threshold:
        return 0
    transitions = np.flatnonzero((values[:-1] > close_threshold) & (values[1:] <= close_threshold))
    if len(transitions) == 0:
        return None
    return int(transitions[0] + 1)


def count_static_prefix_suffix(actions: np.ndarray, gripper_indices: list[int], eps: float) -> tuple[int, int]:
    arm_indices = [i for i in range(actions.shape[1]) if i not in gripper_indices]
    if not arm_indices:
        return 0, 0
    norms = np.linalg.norm(actions[:, arm_indices], axis=1)
    moving = norms > eps
    if not np.any(moving):
        return len(actions), len(actions)
    first = int(np.argmax(moving))
    last = int(len(moving) - 1 - np.argmax(moving[::-1]))
    return first, len(actions) - 1 - last


def print_close_context(
    dataset: Any,
    ep_start: int,
    ep_end: int,
    close_idx: int,
    action_names: list[str],
    state_names: list[str],
    gripper_indices: list[int],
    context: int,
) -> None:
    lo = max(ep_start, ep_start + close_idx - context)
    hi = min(ep_end, ep_start + close_idx + context + 1)
    actions = load_key_range(dataset, "action", lo, hi)
    arm_indices = [i for i in range(actions.shape[1]) if i not in gripper_indices]
    delta_norms = (
        np.linalg.norm(actions[:, arm_indices], axis=1) if arm_indices else np.zeros(actions.shape[0])
    )
    states = None
    if "observation.state" in dataset.features:
        states = load_key_range(dataset, "observation.state", lo, hi)

    selected_state_indices = [
        i
        for i, name in enumerate(state_names)
        if "ee_pose" in name.lower() or "gripper" in name.lower()
    ]
    selected_state_indices = selected_state_indices[:16]

    print("  close context rows:")
    for offset, global_idx in enumerate(range(lo, hi)):
        rel = global_idx - ep_start
        grip_bits = ", ".join(
            f"{action_names[i]}={actions[offset, i]:+.4f}" for i in gripper_indices
        )
        state_bits = ""
        if states is not None and selected_state_indices:
            state_bits = "; state " + ", ".join(
                f"{state_names[i]}={states[offset, i]:+.4f}" for i in selected_state_indices
            )
        print(f"    frame={rel:04d}: delta_norm={delta_norms[offset]:.6f}; {grip_bits}{state_bits}")


def print_dataset_diagnostics(
    dataset: Any,
    max_episodes: int,
    close_threshold: float,
    static_eps: float,
    context: int,
    gripper_dims: list[int] | None = None,
) -> None:
    print("\n== Dataset action/gripper diagnostics ==")
    action_names = action_names_from_dataset(dataset)
    state_names = state_names_from_dataset(dataset)
    gripper_dims = normalize_dims(gripper_dims, len(action_names))
    gripper_indices = gripper_dims or [
        i for i, name in enumerate(action_names) if "gripper" in name.lower()
    ]

    print(f"dataset root: {dataset.root}")
    print(f"repo_id: {dataset.repo_id}")
    print(f"fps: {dataset.fps}")
    print(f"frames: {dataset.num_frames}, episodes: {dataset.num_episodes}")
    print(f"action_dim: {len(action_names)}")
    print(f"action_names: {action_names}")
    print(f"inferred action semantics: {infer_action_semantics(action_names)}")
    print(f"gripper_indices: {[(i, action_names[i]) for i in gripper_indices]}")

    if not gripper_indices:
        print("WARNING: no action dimension containing 'gripper' was found.")
        return

    records = dataset_episode_records(dataset)[:max_episodes]
    all_actions = []
    close_ratios: list[float] = []
    all_gripper_values: list[np.ndarray] = []
    print("\nPer-episode gripper close timing:")
    print("| episode | length | static_head | static_tail | gripper | close_frame | close_ratio | min | max |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    first_context_printed = False
    lag_rows: list[tuple[str, float, float]] = []

    for ep_idx, start, end in records:
        actions = load_key_range(dataset, "action", start, end)
        all_actions.append(actions)
        static_head, static_tail = count_static_prefix_suffix(actions, gripper_indices, static_eps)
        for grip_idx in gripper_indices:
            values = actions[:, grip_idx].astype(float)
            all_gripper_values.append(values)
            close_idx = first_close_frame(values, close_threshold)
            close_display = "" if close_idx is None else str(close_idx)
            ratio = math.nan if close_idx is None else close_idx / max(1, len(values))
            if close_idx is not None:
                close_ratios.append(ratio)
            print(
                f"| {ep_idx} | {len(values)} | {static_head} | {static_tail} | "
                f"{action_names[grip_idx]} | {close_display} | {ratio:.3f} | "
                f"{float(np.min(values)):.3f} | {float(np.max(values)):.3f} |"
            )
            if close_idx is not None and not first_context_printed:
                print_close_context(
                    dataset,
                    start,
                    end,
                    close_idx,
                    action_names,
                    state_names,
                    gripper_indices,
                    context,
                )
                first_context_printed = True

        if "observation.state" in dataset.features:
            states = load_key_range(dataset, "observation.state", start, end)
            for grip_idx in gripper_indices:
                name = action_names[grip_idx]
                if name in state_names and len(actions) > 1:
                    state_idx = state_names.index(name)
                    same = float(np.mean(np.abs(states[1:, state_idx] - actions[1:, grip_idx])))
                    prev = float(np.mean(np.abs(states[1:, state_idx] - actions[:-1, grip_idx])))
                    lag_rows.append((name, same, prev))

    if all_actions:
        all_actions_np = np.concatenate(all_actions, axis=0)
        print("\nAction min/max over sampled episodes:")
        for idx, name in enumerate(action_names):
            print(f"  {idx:02d} {name}: min={np.min(all_actions_np[:, idx]):+.6f} max={np.max(all_actions_np[:, idx]):+.6f}")

    if close_ratios:
        close_ratios_np = np.asarray(close_ratios)
        print("\nFirst close ratio summary:")
        print(f"  mean={np.mean(close_ratios_np):.4f}")
        print(f"  std={np.std(close_ratios_np):.4f}")
        print(f"  min={np.min(close_ratios_np):.4f}")
        print(f"  max={np.max(close_ratios_np):.4f}")
        print(f"  ratio < 0.3 episodes/gripper-series: {int(np.sum(close_ratios_np < 0.3))}")
    else:
        print("\nFirst close ratio summary: no close transition found in sampled episodes.")

    if all_gripper_values:
        gripper_values = np.concatenate(all_gripper_values)
        quantiles = np.quantile(gripper_values, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
        print("\nGripper value distribution over sampled episodes:")
        print(
            "  "
            + ", ".join(
                f"q{int(q * 100):02d}={value:.4f}"
                for q, value in zip([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], quantiles)
            )
        )
        print(f"  mean={np.mean(gripper_values):.4f}, std={np.std(gripper_values):.4f}")

    if lag_rows:
        print("\nGripper observation/action lag check (lower error suggests alignment):")
        print("| gripper | mean_abs obs[t]-action[t] | mean_abs obs[t]-action[t-1] |")
        print("| --- | ---: | ---: |")
        for name, same, prev in lag_rows[: len(gripper_indices) * max_episodes]:
            print(f"| {name} | {same:.5f} | {prev:.5f} |")


def parse_float_xyz(raw: str | None) -> np.ndarray | None:
    if raw is None or raw.strip() == "":
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected --tube-pick-point as 'x,y,z', got {raw!r}.")
    return np.asarray([float(part) for part in parts], dtype=float)


def parse_name_list(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if len(names) != 3:
        raise ValueError(f"Expected three comma-separated names, got {raw!r}.")
    return names


def resolve_output_path(raw: str | None) -> Path | None:
    if raw is None or raw.strip() == "":
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_episode_records(
    dataset: Any,
    max_episodes: int,
    episode_indices: list[int] | None,
) -> list[tuple[int, int, int]]:
    records = dataset_episode_records(dataset)
    if episode_indices is not None:
        wanted = set(episode_indices)
        records = [record for record in records if record[0] in wanted]
    return records[:max_episodes]


def find_right_gripper_dim(action_names: list[str], user_dim: int | None) -> int | None:
    if user_dim is not None:
        dim = user_dim
        if dim < 0:
            dim += len(action_names)
        if dim < 0 or dim >= len(action_names):
            raise ValueError(f"Invalid --gripper-dim {user_dim} for action_dim={len(action_names)}.")
        return dim

    exact = ["right_gripper_cmd", "action.right_gripper_cmd"]
    for target in exact:
        for idx, name in enumerate(action_names):
            if name == target:
                return idx

    for idx, name in enumerate(action_names):
        lowered = name.lower()
        if "right" in lowered and "gripper" in lowered:
            return idx
    return None


def _axis_from_name(name: str) -> str | None:
    lowered = name.lower()
    for axis in ("x", "y", "z"):
        if lowered.endswith(f".{axis}") or lowered.endswith(f"_{axis}") or lowered.endswith(f"/{axis}"):
            return axis
    return None


def find_right_ee_xyz_indices(state_names: list[str], user_names: list[str] | None) -> dict[str, int] | None:
    if user_names is not None:
        indices = {}
        for axis, name in zip(("x", "y", "z"), user_names, strict=True):
            if name not in state_names:
                print(f"WARNING: --ee-position-names entry {name!r} was not found in state names.")
                return None
            indices[axis] = state_names.index(name)
        return indices

    candidate_stems = [
        "right_ee_pose",
        "observation.state.right_ee_pose",
        "right_current_ee_pose",
        "right_tcp_pose",
        "right_end_effector_pose",
    ]
    for stem in candidate_stems:
        names = [f"{stem}.{axis}" for axis in ("x", "y", "z")]
        if all(name in state_names for name in names):
            return {axis: state_names.index(name) for axis, name in zip(("x", "y", "z"), names, strict=True)}

    fuzzy: dict[str, int] = {}
    for idx, name in enumerate(state_names):
        lowered = name.lower()
        axis = _axis_from_name(name)
        if axis is None or axis in fuzzy:
            continue
        if "right" in lowered and (
            "ee" in lowered or "tcp" in lowered or "end_effector" in lowered or "end effector" in lowered
        ):
            fuzzy[axis] = idx
    if all(axis in fuzzy for axis in ("x", "y", "z")):
        return fuzzy
    return None


def sample_at_offset(values: np.ndarray, close_idx: int | None, offset: int) -> float | str:
    if close_idx is None:
        return ""
    idx = close_idx + offset
    if idx < 0 or idx >= len(values):
        return ""
    return float(values[idx])


def xyz_at_offset(
    states: np.ndarray | None,
    xyz_indices: dict[str, int] | None,
    close_idx: int | None,
    offset: int,
) -> tuple[float | str, float | str, float | str]:
    if states is None or xyz_indices is None or close_idx is None:
        return "", "", ""
    idx = close_idx + offset
    if idx < 0 or idx >= len(states):
        return "", "", ""
    return (
        float(states[idx, xyz_indices["x"]]),
        float(states[idx, xyz_indices["y"]]),
        float(states[idx, xyz_indices["z"]]),
    )


def dist_at_offset(
    states: np.ndarray | None,
    xyz_indices: dict[str, int] | None,
    pick_point: np.ndarray | None,
    close_idx: int | None,
    offset: int,
) -> float | str:
    x, y, z = xyz_at_offset(states, xyz_indices, close_idx, offset)
    if pick_point is None or x == "" or y == "" or z == "":
        return ""
    return float(np.linalg.norm(np.asarray([x, y, z], dtype=float) - pick_point))


def close_context_image_keys(dataset: Any) -> list[str]:
    return [
        key
        for key in dataset.features
        if key.startswith("observation.") and "image" in key.lower()
    ]


def save_image_array(array: Any, path: Path) -> bool:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        print("WARNING: Pillow is not installed; close context images cannot be saved.")
        return False

    img = tensor_to_np(array)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in {1, 3, 4}:
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        max_value = float(np.nanmax(img)) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return True


def save_close_context_images(
    dataset: Any,
    ep_idx: int,
    ep_start: int,
    ep_end: int,
    close_idx: int | None,
    offsets: list[int],
) -> None:
    if close_idx is None:
        return
    image_keys = close_context_image_keys(dataset)
    if not image_keys:
        print("WARNING: dataset has no image keys; close context images skipped.")
        return

    out_dir = PROJECT_ROOT / "logs" / "close_context_images" / f"episode_{ep_idx:03d}"
    saved_any = False
    for offset in offsets:
        rel_idx = close_idx + offset
        if rel_idx < 0 or ep_start + rel_idx >= ep_end:
            continue
        item = dataset[ep_start + rel_idx]
        for key in image_keys:
            if key not in item:
                continue
            safe_key = key.replace("/", "_").replace(".", "_")
            path = out_dir / f"frame_{rel_idx:05d}_offset_{offset:+04d}_{safe_key}.png"
            try:
                saved_any = save_image_array(item[key], path) or saved_any
            except Exception as exc:
                print(f"WARNING: failed to save image {key} at episode {ep_idx} frame {rel_idx}: {exc}")
    if saved_any:
        print(f"Saved close context images to: {out_dir}")


def run_dataset_close_spatial_stats(
    dataset: Any,
    max_episodes: int,
    episode_indices: list[int] | None,
    gripper_dim: int | None,
    gripper_threshold: float,
    ee_position_names: list[str] | None,
    tube_pick_point: np.ndarray | None,
    save_images: bool,
    output_csv: Path | None,
) -> None:
    print("\n== Dataset close spatial/context diagnostics ==")
    action_names = action_names_from_dataset(dataset)
    state_names = state_names_from_dataset(dataset)
    selected_gripper_dim = find_right_gripper_dim(action_names, gripper_dim)

    print(f"action_names: {action_names}")
    if selected_gripper_dim is None:
        print("WARNING: Could not find right_gripper_cmd in action names; spatial close stats skipped.")
        return
    print(f"selected gripper dim: {selected_gripper_dim} ({action_names[selected_gripper_dim]})")

    xyz_indices = find_right_ee_xyz_indices(state_names, ee_position_names)
    if xyz_indices is None:
        print("WARNING: No absolute right EE pose found; distance-to-pick-point cannot be computed.")
        print(f"Available state names: {state_names}")
    else:
        print(
            "right EE xyz state names: "
            + ", ".join(f"{axis}={state_names[idx]}" for axis, idx in xyz_indices.items())
        )

    if tube_pick_point is not None and xyz_indices is None:
        print("WARNING: --tube-pick-point was provided but right EE pose was not found; distances skipped.")

    records = select_episode_records(dataset, max_episodes, episode_indices)
    offsets = [-60, -30, -10, 0, 10, 30, 60]
    offset_labels = {
        -60: "60_before",
        -30: "30_before",
        -10: "10_before",
        0: "close",
        10: "10_after",
        30: "30_after",
        60: "60_after",
    }
    rows: list[dict[str, Any]] = []
    close_ratios: list[float] = []
    dist_at_close_values: list[float] = []
    delta_sum_30_values: list[float] = []
    delta_sum_60_values: list[float] = []

    for ep_idx, start, end in records:
        actions = load_key_range(dataset, "action", start, end)
        values = actions[:, selected_gripper_dim].astype(float)
        close_idx = int(np.argmax(values < gripper_threshold)) if np.any(values < gripper_threshold) else None
        gripper_like_indices = {
            i for i, name in enumerate(action_names) if "gripper" in name.lower()
        }
        gripper_like_indices.add(selected_gripper_dim)
        arm_indices = [i for i in range(actions.shape[1]) if i not in gripper_like_indices]
        delta_norms = np.linalg.norm(actions[:, arm_indices], axis=1) if arm_indices else np.zeros(len(values))

        states = None
        if "observation.state" in dataset.features and xyz_indices is not None:
            states = load_key_range(dataset, "observation.state", start, end)

        row: dict[str, Any] = {
            "episode_index": ep_idx,
            "episode_length": len(values),
            "first_close_frame": "" if close_idx is None else close_idx,
            "first_close_ratio": "" if close_idx is None else close_idx / max(1, len(values)),
            "gripper_threshold": gripper_threshold,
            "gripper_value_at_close": "" if close_idx is None else float(values[close_idx]),
            "action_delta_norm_sum_30_after_close": "",
            "action_delta_norm_sum_60_after_close": "",
        }

        if close_idx is not None:
            ratio = close_idx / max(1, len(values))
            close_ratios.append(ratio)
            sum_30 = float(np.sum(delta_norms[close_idx + 1 : min(len(delta_norms), close_idx + 31)]))
            sum_60 = float(np.sum(delta_norms[close_idx + 1 : min(len(delta_norms), close_idx + 61)]))
            row["action_delta_norm_sum_30_after_close"] = sum_30
            row["action_delta_norm_sum_60_after_close"] = sum_60
            delta_sum_30_values.append(sum_30)
            delta_sum_60_values.append(sum_60)

        for offset in offsets:
            label = offset_labels[offset]
            row[f"gripper_value_{label}"] = sample_at_offset(values, close_idx, offset)
            row[f"action_delta_norm_{label}"] = sample_at_offset(delta_norms, close_idx, offset)
            x, y, z = xyz_at_offset(states, xyz_indices, close_idx, offset)
            row[f"right_ee_x_{label}"] = x
            row[f"right_ee_y_{label}"] = y
            row[f"right_ee_z_{label}"] = z
            distance = dist_at_offset(states, xyz_indices, tube_pick_point, close_idx, offset)
            row[f"dist_{label}"] = distance
            if offset == 0:
                row["dist_at_close"] = distance
                if distance != "":
                    dist_at_close_values.append(float(distance))

        if save_images:
            save_close_context_images(dataset, ep_idx, start, end, close_idx, offsets)

        rows.append(row)

    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = PROJECT_ROOT / "logs" / f"debug_dataset_close_spatial_stats_{timestamp}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved dataset close spatial CSV: {output_csv}")

    print("\nClose spatial summary:")
    print(f"  valid_episode_count={len(rows)}")
    if close_ratios:
        values_np = np.asarray(close_ratios, dtype=float)
        print(
            "  first_close_ratio "
            f"mean={np.mean(values_np):.4f} std={np.std(values_np):.4f} "
            f"min={np.min(values_np):.4f} max={np.max(values_np):.4f}"
        )
        print(f"  ratio < 0.3 count={int(np.sum(values_np < 0.3))}")
    else:
        print("  first_close_ratio: no close frames found")

    if dist_at_close_values:
        dist_np = np.asarray(dist_at_close_values, dtype=float)
        print(
            "  dist_at_close "
            f"mean={np.mean(dist_np):.4f} std={np.std(dist_np):.4f} "
            f"min={np.min(dist_np):.4f} max={np.max(dist_np):.4f}"
        )
        for threshold in (0.03, 0.05, 0.08):
            print(f"  dist_at_close > {threshold:.2f} count={int(np.sum(dist_np > threshold))}")
    else:
        print("  dist_at_close: unavailable")

    if delta_sum_30_values:
        print(f"  close_after action_delta_norm_sum_30 mean={np.mean(delta_sum_30_values):.6f}")
    if delta_sum_60_values:
        print(f"  close_after action_delta_norm_sum_60 mean={np.mean(delta_sum_60_values):.6f}")


def maybe_load_dataset(args: argparse.Namespace, train_section: dict[str, Any]):
    if torch is None:
        print("\nSKIP dataset diagnostics: torch is not installed in this Python environment.")
        return None

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_cfg = train_section.get("dataset", {})
    repo_id = args.repo_id or dataset_cfg.get("repo_id")
    root_raw = args.dataset_root or dataset_cfg.get("root")
    if not repo_id or not root_raw:
        print("\nSKIP dataset diagnostics: repo_id/root not provided.")
        return None
    root = Path(root_raw).expanduser()
    if not root.exists():
        print(f"\nSKIP dataset diagnostics: dataset root does not exist: {root}")
        return None
    return LeRobotDataset(str(repo_id), root=root, download_videos=False)


def run_offline_inference(
    dataset: Any,
    reason_cfg: Any,
    device_override: str | None,
    frame_index: int,
) -> None:
    print("\n== Offline checkpoint inference ==")
    if torch is None:
        print("SKIP inference: torch is not installed in this Python environment.")
        return
    pretrained_path = getattr(reason_cfg, "pretrained_path", None)
    if not pretrained_path:
        print("SKIP inference: reason config has no pretrained_path.")
        return
    pretrained = Path(str(pretrained_path)).expanduser()
    if not pretrained.exists():
        print(f"SKIP inference: pretrained_path does not exist: {pretrained}")
        return

    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.utils.constants import ACTION, OBS_PREFIX

    cfg = copy.deepcopy(reason_cfg)
    if device_override is not None:
        cfg.device = device_override
    elif str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    item = dataset[frame_index]
    obs = {key: value for key, value in item.items() if key.startswith(OBS_PREFIX)}
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    processed = preprocessor(obs)
    with torch.inference_mode():
        first_action = policy.select_action(processed)

    queued = list(policy._queues[ACTION])
    chunk_norm = torch.stack([first_action, *queued], dim=1) if queued else first_action.unsqueeze(1)
    chunk_unnorm = postprocessor(chunk_norm)

    action_names = action_names_from_dataset(dataset)
    gripper_indices = [i for i, name in enumerate(action_names) if "gripper" in name.lower()]
    head = min(8, chunk_norm.shape[1])
    print(f"frame_index: {frame_index}")
    print(f"policy action queue length after first pop: {len(queued)}")
    print(f"normalized chunk shape: {tuple(chunk_norm.shape)}")
    print(f"unnormalized chunk shape: {tuple(chunk_unnorm.shape)}")
    if gripper_indices:
        norm_np = chunk_norm[0, :head, gripper_indices].detach().cpu().numpy()
        unnorm_np = chunk_unnorm[0, :head, gripper_indices].detach().cpu().numpy()
        print(f"first {head} normalized gripper actions: {norm_np}")
        print(f"first {head} unnormalized gripper actions: {unnorm_np}")
    else:
        print(f"first {head} unnormalized actions: {chunk_unnorm[0, :head].detach().cpu().numpy()}")


def run_checkpoint_horizon_compare(
    dataset: Any,
    policy_config_path: Path,
    checkpoint_path: Path,
    episode_index: int | None,
    frame_index: int,
    gripper_dims: list[int] | None,
    device_override: str | None,
) -> None:
    print("\n== Checkpoint gripper horizon comparison ==")
    if torch is None:
        print("SKIP horizon comparison: torch is not installed in this Python environment.")
        return
    if not checkpoint_path.exists():
        print(f"SKIP horizon comparison: checkpoint path does not exist: {checkpoint_path}")
        return

    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import populate_queues
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, OBS_STATE

    cfg = build_policy_from_yaml("diffusion", policy_config_path, "reason", legacy_policy=None)
    cfg = copy.deepcopy(cfg)
    cfg.pretrained_path = str(checkpoint_path)
    if device_override is not None:
        cfg.device = device_override
    elif str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    records = dataset_episode_records(dataset)
    if episode_index is None:
        ep_idx = None
        ep_start = 0
        ep_end = len(dataset)
        global_index = frame_index
    else:
        matches = [record for record in records if record[0] == episode_index]
        if not matches:
            print(f"SKIP horizon comparison: episode_index {episode_index} not found.")
            return
        ep_idx, ep_start, ep_end = matches[0]
        global_index = ep_start + frame_index
        if global_index >= ep_end:
            print(
                f"SKIP horizon comparison: frame_index {frame_index} outside episode "
                f"{episode_index} length {ep_end - ep_start}."
            )
            return

    action_names = action_names_from_dataset(dataset)
    gripper_dims = normalize_dims(gripper_dims, len(action_names))
    grip_indices = gripper_dims or [i for i, name in enumerate(action_names) if "gripper" in name.lower()]
    if not grip_indices:
        print("SKIP horizon comparison: no gripper dims found/provided.")
        return

    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    item = dataset[global_index]
    obs = {key: value for key, value in item.items() if key.startswith(OBS_PREFIX)}
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    processed = preprocessor(obs)
    if ACTION in processed:
        processed.pop(ACTION)
    if policy.config.image_features:
        processed = dict(processed)
        processed[OBS_IMAGES] = torch.stack([processed[key] for key in policy.config.image_features], dim=-4)

    policy._queues = populate_queues(policy._queues, processed)
    batch = {key: torch.stack(list(policy._queues[key]), dim=1) for key in processed if key in policy._queues}

    with torch.inference_mode():
        global_cond = policy.diffusion._prepare_global_conditioning(batch)
        actions_norm = policy.diffusion.conditional_sample(batch[OBS_STATE].shape[0], global_cond=global_cond)
        actions_unnorm = postprocessor(actions_norm)

    expert_actions = load_key_range(dataset, "action", ep_start, ep_end)
    rel_index = global_index - ep_start
    current_expert = expert_actions[rel_index, grip_indices]
    future_hi = min(ep_end, global_index + int(cfg.horizon))
    future_expert = load_key_range(dataset, "action", global_index, future_hi)[:, grip_indices]

    print(f"policy_config: {policy_config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"episode_index: {ep_idx if episode_index is not None else 'global'}")
    print(f"frame_index: {frame_index} (global_index={global_index})")
    print(f"gripper dims: {[(idx, action_names[idx]) for idx in grip_indices]}")
    print(f"current expert gripper value: {current_expert}")
    print(f"future expert gripper values t..t+{len(future_expert)-1}:")
    print(future_expert)

    start = int(cfg.n_obs_steps) - 1
    end = start + int(cfg.n_action_steps)
    print("\nFull horizon comparison:")
    print("| horizon_idx | delta_index | EXEC | expert gripper at delta | predicted gripper |")
    print("| ---: | ---: | --- | --- | --- |")
    pred_np = actions_unnorm[0, :, grip_indices].detach().cpu().numpy()
    for horizon_idx, delta_idx in enumerate(cfg.action_delta_indices):
        expert_global = global_index + int(delta_idx)
        if ep_start <= expert_global < ep_end:
            expert_value = load_key_range(dataset, "action", expert_global, expert_global + 1)[0, grip_indices]
            expert_text = np.array2string(expert_value, precision=4)
        else:
            expert_text = "PAD/out-of-episode"
        marker = "EXEC" if start <= horizon_idx < end else ""
        pred_text = np.array2string(pred_np[horizon_idx], precision=4)
        print(f"| {horizon_idx} | {delta_idx} | {marker} | {expert_text} | {pred_text} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cfg",
        default=str(SCRIPTS_DIR / "config" / "train_cfg.yaml"),
        help="Path to scripts/config/train_cfg.yaml.",
    )
    parser.add_argument(
        "--record-cfg",
        default=str(SCRIPTS_DIR / "config" / "record_cfg.yaml"),
        help="Path to scripts/config/record_cfg.yaml.",
    )
    parser.add_argument(
        "--diffusion-train-config",
        default=str(SCRIPTS_DIR / "policy_config" / "diffusion_train_config.yaml"),
    )
    parser.add_argument(
        "--diffusion-reason-config",
        default=str(SCRIPTS_DIR / "policy_config" / "diffusion_reason_config.yaml"),
    )
    parser.add_argument("--dataset-root", default=None, help="Override training dataset root.")
    parser.add_argument("--repo-id", default=None, help="Override training dataset repo_id.")
    parser.add_argument("--max-episodes", "--num-episodes", dest="max_episodes", type=int, default=8)
    parser.add_argument(
        "--close-threshold",
        "--gripper-threshold",
        dest="close_threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument("--gripper-dims", default=None, help="Comma-separated gripper action dims.")
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated episode indices for dataset close spatial stats.",
    )
    parser.add_argument(
        "--gripper-dim",
        type=int,
        default=None,
        help="Right gripper action dim for dataset close spatial stats.",
    )
    parser.add_argument(
        "--ee-position-names",
        default=None,
        help="Comma-separated state feature names for right EE x,y,z.",
    )
    parser.add_argument(
        "--tube-pick-point",
        default=None,
        help="Optional pick point as 'x,y,z' in the same frame as right EE state.",
    )
    parser.add_argument(
        "--save-close-context-images",
        action="store_true",
        help="Save images around first close frames if dataset image keys are available.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for dataset close spatial stats.",
    )
    parser.add_argument("--static-eps", type=float, default=1e-4)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--offline-inference", action="store_true")
    parser.add_argument("--inference-frame", type=int, default=0)
    parser.add_argument(
        "--policy-config",
        default=None,
        help="Diffusion reason policy config for checkpoint horizon comparison.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint pretrained_model directory.")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override policy device for offline inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg_path = resolve_main_path(args.train_cfg)
    record_cfg_path = resolve_main_path(args.record_cfg)

    train_main = load_active_policy(train_cfg_path, "train", "train")
    record_main = load_active_policy(record_cfg_path, "record", "reason")
    train_diff_path, train_diff_cfg, reason_diff_path, reason_diff_cfg = load_explicit_diffusion_configs(args)

    print_config_summary(
        train_main,
        record_main,
        train_diff_path,
        train_diff_cfg,
        reason_diff_path,
        reason_diff_cfg,
    )

    train_section = train_main[0]
    dataset = maybe_load_dataset(args, train_section)
    gripper_dims = parse_int_list(args.gripper_dims)
    episode_indices = parse_int_list(args.episode_indices)
    ee_position_names = parse_name_list(args.ee_position_names)
    tube_pick_point = parse_float_xyz(args.tube_pick_point)
    spatial_gripper_dim = args.gripper_dim if args.gripper_dim is not None else (
        gripper_dims[0] if gripper_dims else None
    )
    if dataset is not None:
        print_dataset_diagnostics(
            dataset,
            max_episodes=args.max_episodes,
            close_threshold=args.close_threshold,
            static_eps=args.static_eps,
            context=args.context,
            gripper_dims=gripper_dims,
        )
        run_dataset_close_spatial_stats(
            dataset=dataset,
            max_episodes=args.max_episodes,
            episode_indices=episode_indices,
            gripper_dim=spatial_gripper_dim,
            gripper_threshold=args.close_threshold,
            ee_position_names=ee_position_names,
            tube_pick_point=tube_pick_point,
            save_images=args.save_close_context_images,
            output_csv=resolve_output_path(args.output_csv),
        )
        if args.offline_inference:
            run_offline_inference(dataset, reason_diff_cfg, args.device, args.inference_frame)
        if args.policy_config and args.checkpoint:
            run_checkpoint_horizon_compare(
                dataset=dataset,
                policy_config_path=resolve_main_path(args.policy_config),
                checkpoint_path=resolve_main_path(args.checkpoint),
                episode_index=args.episode_index,
                frame_index=args.frame_index if args.frame_index is not None else args.inference_frame,
                gripper_dims=gripper_dims,
                device_override=args.device,
            )
    elif args.offline_inference:
        print("\nSKIP inference: dataset is required to build policy features and sample observations.")
    elif args.policy_config and args.checkpoint:
        print("\nSKIP horizon comparison: dataset is required.")


if __name__ == "__main__":
    main()
