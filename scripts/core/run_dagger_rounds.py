#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUCCESS_POLICY_EXPLICIT = "explicit"
SUCCESS_POLICY_RECORDED_IS_SUCCESS = "recorded_is_success"
SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE = "allow_missing_for_smoke"
VALID_SUCCESS_POLICIES = {
    SUCCESS_POLICY_EXPLICIT,
    SUCCESS_POLICY_RECORDED_IS_SUCCESS,
    SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE,
}


@dataclass
class DAggerRoundsConfig:
    output_root: str | Path
    record_cfg_path: str | Path
    train_cfg_path: str | Path | None = None
    num_rounds: int | None = None
    episodes_per_round: int = 1
    seed_repo_id: str | None = None
    seed_dataset_path: str | Path | None = None
    seed_root: str | Path | None = None
    initial_pretrained_path: str | Path | None = None
    overwrite: bool = False
    train_round0_if_missing: bool = True
    export_min_segment_frames: int = 1
    min_episode_len_for_act: int | None = None
    pre_takeover_context: int = 0
    require_complete_expert_action: bool = True
    round_schedule: dict[str, Any] | None = None
    dagger_training: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    policy_backend: dict[str, Any] | None = None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_project_root() -> Path:
    return _default_scripts_dir().parent


def _default_dagger_rounds_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "dagger_rounds_cfg.yaml"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _clean_or_fail(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Path already exists: {path}")
    shutil.rmtree(path)


def _safe_repo_id(prefix: str, round_idx: int, suffix: str) -> str:
    return f"{prefix}_{suffix}_r{round_idx:03d}".replace("/", "_")


def _resolve_seed(cfg: DAggerRoundsConfig) -> tuple[str, Path]:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    seed_root = cfg.seed_dataset_path or cfg.seed_root
    if seed_root is not None:
        root = Path(seed_root)
    elif cfg.seed_repo_id is not None:
        root = HF_LEROBOT_HOME / cfg.seed_repo_id
    else:
        raise ValueError("One of seed_dataset_path, seed_root, or seed_repo_id is required.")

    repo_id = cfg.seed_repo_id or root.name
    return repo_id, root


def _require_checkpoint_exists(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} checkpoint path is required.")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint path does not exist or is not a directory: {path}")
    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} checkpoint is missing required file(s) {missing}: {path}")
    return path


def _validate_round_policy_descriptor(policy_cfg: dict[str, Any] | None) -> dict[str, Any]:
    from scripts.core.policy_config_utils import (
        DIFFUSION_POLICY_TYPES,
        get_default_policy_config_path,
        load_policy_yaml,
        normalize_policy_type,
        resolve_policy_config_path,
    )

    descriptor = copy.deepcopy(policy_cfg or {"type": "act"})
    raw_policy_type = str(descriptor.get("type", "act")).strip().lower()
    policy_type = normalize_policy_type(raw_policy_type)
    descriptor["type"] = policy_type

    if descriptor.get("reason_config_path") is None and descriptor.get("config_path") is not None:
        logging.warning(
            "[DEPRECATED] dagger_rounds.policy.config_path is treated as a legacy alias "
            "for dagger_rounds.policy.reason_config_path. Add train_config_path explicitly "
            "so DAgger rollout and training cannot share one yaml by accident."
        )
        descriptor["reason_config_path"] = descriptor["config_path"]

    descriptor.setdefault("reason_config_path", get_default_policy_config_path(policy_type, "reason"))
    descriptor.setdefault("train_config_path", get_default_policy_config_path(policy_type, "train"))

    reason_path = resolve_policy_config_path(
        descriptor,
        scripts_dir=_default_scripts_dir(),
        project_root=_default_project_root(),
        mode="reason",
    )
    train_path = resolve_policy_config_path(
        descriptor,
        scripts_dir=_default_scripts_dir(),
        project_root=_default_project_root(),
        mode="train",
    )
    load_policy_yaml(reason_path)
    load_policy_yaml(train_path)

    if raw_policy_type in DIFFUSION_POLICY_TYPES:
        raise NotImplementedError(
            "Diffusion policy is configured, but DAgger diffusion backend is not implemented yet. "
            f"Resolved diffusion reason config: {reason_path}; train config: {train_path}"
        )
    if policy_type != "act":
        raise ValueError(
            "dagger_rounds.policy.type must be one of: act | diffusion | dp | diffusion_policy. "
            f"Got: {raw_policy_type!r}"
        )
    return descriptor


def _apply_round_policy_descriptor(
    base_cfg: dict[str, Any],
    descriptor: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    if not descriptor:
        return base_cfg
    if mode not in {"train", "reason"}:
        raise ValueError(f"mode must be 'train' or 'reason'. Got: {mode!r}")

    policy = base_cfg.setdefault("policy", {})
    policy["type"] = descriptor["type"]
    config_key = "train_config_path" if mode == "train" else "reason_config_path"
    policy["config_path"] = descriptor[config_key]

    for stale_key in ("reason_config_path", "train_config_path"):
        policy.pop(stale_key, None)
    return base_cfg


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _clamp_steps(steps: int, min_steps: int | None, max_steps: int | None) -> int:
    if min_steps is not None:
        steps = max(int(min_steps), steps)
    if max_steps is not None:
        steps = min(int(max_steps), steps)
    return int(steps)


def _summary_containers(exported_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    summary = exported_summary or {}
    containers: list[dict[str, Any]] = []
    for value in (
        summary,
        summary.get("stats", {}),
        summary.get("exported_dataset", {}),
        summary.get("rules", {}),
    ):
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _first_available(
    exported_summary: dict[str, Any] | None,
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    for container in _summary_containers(exported_summary):
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
    return default


def _first_available_int(
    exported_summary: dict[str, Any] | None,
    keys: tuple[str, ...],
    default: int = 0,
) -> int:
    value = _first_available(exported_summary, keys)
    if value is None or value == "":
        return int(default)
    return int(value)


def _subtype_count(
    exported_summary: dict[str, Any] | None,
    table_name: str,
    subtype: str,
    default: int = 0,
) -> int:
    summary = exported_summary or {}
    for container in (summary, summary.get("stats", {})):
        if not isinstance(container, dict):
            continue
        subtype_table = container.get(table_name)
        if isinstance(subtype_table, dict) and subtype_table.get(subtype) is not None:
            return int(subtype_table[subtype])
    return int(default)


def _export_mode_from_summary(exported_summary: dict[str, Any] | None) -> str | None:
    value = _first_available(exported_summary, ("export_mode",))
    if value is None or value == "":
        return None
    return str(value).strip().lower()


def _normalize_optional_export_mode(export_mode: str | None) -> str | None:
    if export_mode is None:
        return None
    export_mode = str(export_mode).strip().lower()
    return export_mode or None


def _resolve_train_steps_export_mode(
    exported_summary: dict[str, Any] | None,
    configured_export_mode: str | None = None,
    round_id: int | None = None,
) -> str:
    summary_mode = _export_mode_from_summary(exported_summary)
    configured_mode = _normalize_optional_export_mode(configured_export_mode)
    if summary_mode and configured_mode and summary_mode != configured_mode:
        prefix = f"[round {round_id:03d}] " if round_id is not None else ""
        logging.warning(
            "%strain steps export_mode mismatch: summary=%s configured=%s; using summary.",
            prefix,
            summary_mode,
            configured_mode,
        )
    return summary_mode or configured_mode or "intervention_segments"


def _round_schedule_cfg(config: DAggerRoundsConfig) -> dict[str, Any]:
    return copy.deepcopy(config.round_schedule or {})


def _resolve_max_rounds(config: DAggerRoundsConfig, schedule_cfg: dict[str, Any]) -> int:
    max_rounds = schedule_cfg.get("max_rounds")
    if max_rounds is not None:
        return int(max_rounds)
    if config.num_rounds is None:
        raise ValueError("Either dagger_rounds.num_rounds or round_schedule.max_rounds is required.")
    return int(config.num_rounds)


def _train_steps_cfg(schedule_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not schedule_cfg:
        return {}
    train_steps = schedule_cfg.get("train_steps")
    if isinstance(train_steps, dict):
        return train_steps
    return schedule_cfg


def _early_stop_cfg(schedule_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not schedule_cfg:
        return {}
    early_stop = schedule_cfg.get("early_stop")
    if isinstance(early_stop, dict):
        return early_stop
    return {}


def _dagger_sampling_cfg(config: DAggerRoundsConfig) -> dict[str, Any]:
    dagger_training = copy.deepcopy(config.dagger_training or {})
    sampling = dagger_training.get("sampling", {})
    return copy.deepcopy(sampling if isinstance(sampling, dict) else {})


def _sampling_train_cfg(
    sampling_cfg: dict[str, Any],
    source_index_path: Path | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(sampling_cfg)
    cfg.setdefault("enabled", False)
    if source_index_path is not None:
        cfg["source_index_path"] = str(source_index_path)
    return cfg


def _policy_backend_export_cfg(round_cfg: dict[str, Any]) -> dict[str, Any]:
    export_cfg = (round_cfg.get("policy_backend") or {}).get("export", {})
    if not isinstance(export_cfg, dict):
        return {}
    return export_cfg


def _full_episode_export_enabled(round_cfg: dict[str, Any]) -> bool:
    export_cfg = _policy_backend_export_cfg(round_cfg)
    if not export_cfg:
        return False

    export_mode = str(export_cfg.get("export_mode", "intervention_segments")).lower()
    hybrid_cfg = export_cfg.get("hybrid") or {}
    if export_mode == "full_success_episode":
        return True
    return export_mode == "hybrid" and bool(
        hybrid_cfg.get("include_full_success_episode", True)
    )


def _full_episode_success_policy(round_cfg: dict[str, Any]) -> str:
    export_cfg = _policy_backend_export_cfg(round_cfg)
    full_episode_cfg = export_cfg.get("full_episode") or {}
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


def _exported_counts(exported_summary: dict[str, Any] | None) -> tuple[int, int]:
    exported_frames = _first_available_int(exported_summary, ("exported_frames", "total_frames"), default=0)
    exported_episodes = _first_available_int(
        exported_summary,
        ("exported_episodes", "total_episodes"),
        default=0,
    )
    return exported_frames, exported_episodes


def _export_metrics(export_summary: dict[str, Any], run_mix_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    exported_frames, exported_episodes = _exported_counts(export_summary)
    stats = export_summary.get("stats", {})
    run_mix_stats = run_mix_stats or {}
    expert_frame_ratio = stats.get("expert_frame_ratio", run_mix_stats.get("expert_frame_ratio", 0.0))
    intervention_count = stats.get("intervention_count", run_mix_stats.get("intervention_count", 0))
    return {
        "exported_frames": int(exported_frames),
        "exported_episodes": int(exported_episodes),
        "export_mode": _resolve_train_steps_export_mode(export_summary),
        "full_episode_count": _first_available_int(
            export_summary,
            ("full_episode_count", "full_episode_exported", "hybrid_full_episodes"),
            default=_subtype_count(export_summary, "subtype_episodes", "full_success_episode", 0),
        ),
        "full_episode_frames": _first_available_int(
            export_summary,
            ("full_episode_frames", "full_episode_frames_exported", "hybrid_full_frames"),
            default=_subtype_count(export_summary, "subtype_frames", "full_success_episode", 0),
        ),
        "recovery_frames": _first_available_int(
            export_summary,
            ("recovery_frames", "hybrid_recovery_frames", "intervention_frames", "intervention_segment_frames_exported"),
            default=_subtype_count(export_summary, "subtype_frames", "intervention_segment", 0),
        ),
        "recovery_segment_count": _first_available_int(
            export_summary,
            ("recovery_segment_count", "hybrid_recovery_segments", "intervention_segments_exported"),
            default=_subtype_count(export_summary, "subtype_episodes", "intervention_segment", 0),
        ),
        "expert_frame_ratio": float(expert_frame_ratio or 0.0),
        "intervention_count": int(intervention_count or 0),
        "skipped_short_segments": int(stats.get("skipped_short_segments", 0) or 0),
        "incomplete_expert_label_frames": stats.get("incomplete_expert_label_frames"),
        "success_policy": stats.get("success_policy"),
        "success_inferred_episodes": int(stats.get("success_inferred_episodes", 0) or 0),
        "success_explicit_true_episodes": int(stats.get("success_explicit_true_episodes", 0) or 0),
        "success_explicit_false_episodes": int(stats.get("success_explicit_false_episodes", 0) or 0),
        "skipped_explicit_failure": int(stats.get("skipped_explicit_failure", 0) or 0),
    }


def resolve_effective_train_step_mode(export_mode: str, train_steps_cfg: dict[str, Any]) -> str:
    configured_mode = str(train_steps_cfg.get("mode", "fixed")).strip().lower()
    if configured_mode != "auto_by_export_mode":
        return configured_mode

    export_mode = str(export_mode or "").strip().lower()
    if export_mode == "intervention_segments":
        return str(train_steps_cfg.get("intervention_segments_mode") or "hybrid").strip().lower()
    if export_mode == "full_success_episode":
        return str(train_steps_cfg.get("full_success_episode_mode") or "by_exported_episodes").strip().lower()
    if export_mode == "hybrid":
        return str(train_steps_cfg.get("hybrid_mode") or "by_export_subtype").strip().lower()
    raise ValueError(
        "round_schedule.train_steps.mode=auto_by_export_mode requires export_mode "
        "to be one of: intervention_segments, full_success_episode, hybrid. "
        f"Got: {export_mode!r}."
    )


def _subtype_train_step_counts(exported_summary: dict[str, Any]) -> dict[str, int]:
    full_episode_count = _first_available_int(
        exported_summary,
        ("full_episode_count", "full_episode_exported", "hybrid_full_episodes"),
        default=_subtype_count(exported_summary, "subtype_episodes", "full_success_episode", 0),
    )
    recovery_segment_count = _first_available_int(
        exported_summary,
        ("recovery_segment_count", "hybrid_recovery_segments", "intervention_segments_exported"),
        default=_subtype_count(exported_summary, "subtype_episodes", "intervention_segment", 0),
    )
    recovery_frames = _first_available_int(
        exported_summary,
        ("recovery_frames", "hybrid_recovery_frames", "intervention_frames", "intervention_segment_frames_exported"),
        default=_subtype_count(exported_summary, "subtype_frames", "intervention_segment", 0),
    )
    return {
        "full_episode_count": full_episode_count,
        "recovery_segment_count": recovery_segment_count,
        "recovery_frames": recovery_frames,
    }


def resolve_train_steps_by_mode(
    effective_mode: str,
    exported_summary: dict[str, Any],
    train_steps_cfg: dict[str, Any],
    base_steps: int,
    round_id: int,
) -> tuple[int, dict[str, Any]]:
    effective_mode = str(effective_mode).strip().lower()
    exported_frames, exported_episodes = _exported_counts(exported_summary)
    components: dict[str, Any] = {
        "exported_frames": int(exported_frames),
        "exported_episodes": int(exported_episodes),
    }
    if exported_frames <= 0 or exported_episodes <= 0:
        components.update(
            {
                "skipped_empty_export": True,
                "raw_steps": 0,
                "clamped_steps": 0,
            }
        )
        return 0, components

    fixed_steps = _optional_int(train_steps_cfg.get("fixed_steps"))
    min_steps = _optional_int(train_steps_cfg.get("min_steps"))
    max_steps = _optional_int(train_steps_cfg.get("max_steps"))
    round1_steps = _optional_int(train_steps_cfg.get("round1_steps"))
    decay_per_round = _optional_float(train_steps_cfg.get("decay_per_round"))
    if decay_per_round is None:
        decay_per_round = 1.0
    if decay_per_round <= 0:
        raise ValueError("round_schedule.train_steps.decay_per_round must be > 0.")

    clamp_result = False
    if round_id == 1 and round1_steps is not None:
        resolved = int(round1_steps)
        clamp_result = True
        components.update({"override": "round1_steps", "round1_steps": int(round1_steps)})
    elif effective_mode == "fixed":
        resolved = fixed_steps if fixed_steps is not None else int(base_steps)
        clamp_result = decay_per_round != 1.0
        components.update({"fixed_steps": resolved})
    elif effective_mode == "by_exported_frames":
        steps_per_frame = int(train_steps_cfg.get("steps_per_exported_frame", 1))
        resolved = exported_frames * steps_per_frame
        clamp_result = True
        components.update(
            {
                "steps_per_exported_frame": steps_per_frame,
                "frame_steps": int(resolved),
            }
        )
    elif effective_mode == "by_exported_episodes":
        steps_per_episode = int(train_steps_cfg.get("steps_per_exported_episode", int(base_steps)))
        resolved = exported_episodes * steps_per_episode
        clamp_result = True
        components.update(
            {
                "steps_per_exported_episode": steps_per_episode,
                "episode_steps": int(resolved),
            }
        )
    elif effective_mode == "hybrid":
        steps_per_frame = int(train_steps_cfg.get("steps_per_exported_frame", 1))
        steps_per_episode = int(train_steps_cfg.get("steps_per_exported_episode", int(base_steps)))
        frame_steps = exported_frames * steps_per_frame
        episode_steps = exported_episodes * steps_per_episode
        resolved = max(frame_steps, episode_steps)
        clamp_result = True
        components.update(
            {
                "steps_per_exported_frame": steps_per_frame,
                "steps_per_exported_episode": steps_per_episode,
                "frame_steps": int(frame_steps),
                "episode_steps": int(episode_steps),
            }
        )
    elif effective_mode == "by_export_subtype":
        subtype_counts = _subtype_train_step_counts(exported_summary)
        full_episode_count = subtype_counts["full_episode_count"]
        recovery_segment_count = subtype_counts["recovery_segment_count"]
        recovery_frames = subtype_counts["recovery_frames"]
        if full_episode_count == 0 and recovery_segment_count == 0 and recovery_frames == 0:
            logging.warning(
                "by_export_subtype requested but subtype summary is missing; "
                "fallback to by_exported_episodes"
            )
            steps_per_episode = int(train_steps_cfg.get("steps_per_exported_episode", int(base_steps)))
            resolved = exported_episodes * steps_per_episode
            components.update(
                {
                    "fallback_mode": "by_exported_episodes",
                    "steps_per_exported_episode": steps_per_episode,
                    "episode_steps": int(resolved),
                }
            )
        else:
            steps_per_full_episode = int(
                train_steps_cfg.get(
                    "steps_per_full_episode",
                    train_steps_cfg.get("steps_per_exported_episode", int(base_steps)),
                )
            )
            steps_per_recovery_segment = int(train_steps_cfg.get("steps_per_recovery_segment", 0))
            steps_per_recovery_frame = int(
                train_steps_cfg.get(
                    "steps_per_recovery_frame",
                    train_steps_cfg.get("steps_per_exported_frame", 1),
                )
            )
            full_steps = full_episode_count * steps_per_full_episode
            recovery_steps = (
                recovery_segment_count * steps_per_recovery_segment
                + recovery_frames * steps_per_recovery_frame
            )
            resolved = full_steps + recovery_steps
            components.update(
                {
                    "full_episode_count": full_episode_count,
                    "steps_per_full_episode": steps_per_full_episode,
                    "full_steps": int(full_steps),
                    "recovery_segment_count": recovery_segment_count,
                    "steps_per_recovery_segment": steps_per_recovery_segment,
                    "recovery_frames": recovery_frames,
                    "steps_per_recovery_frame": steps_per_recovery_frame,
                    "recovery_steps": int(recovery_steps),
                }
            )
        clamp_result = True
    else:
        raise ValueError(
            "round_schedule.train_steps mode must be one of: fixed, by_exported_frames, "
            "by_exported_episodes, hybrid, auto_by_export_mode, by_export_subtype."
        )

    components["pre_decay_steps"] = int(resolved)
    if decay_per_round != 1.0:
        decay_factor = decay_per_round ** max(0, int(round_id) - 1)
        resolved = int(round(float(resolved) * decay_factor))
        clamp_result = True
        components["decay_per_round"] = float(decay_per_round)
        components["decay_factor"] = float(decay_factor)

    components["raw_steps"] = int(resolved)
    if clamp_result:
        resolved = _clamp_steps(int(resolved), min_steps, max_steps)
    components["clamped_steps"] = int(resolved)
    return int(resolved), components


def resolve_train_steps_schedule(
    round_id: int,
    exported_summary: dict[str, Any],
    base_train_steps: int,
    schedule_cfg: dict[str, Any] | None,
    export_mode: str | None = None,
) -> dict[str, Any]:
    """Resolve per-round train steps plus metadata for state/logging."""

    train_steps_cfg = _train_steps_cfg(schedule_cfg)
    configured_mode = str(train_steps_cfg.get("mode", "fixed")).strip().lower()
    resolved_export_mode = _resolve_train_steps_export_mode(exported_summary, export_mode, round_id)
    effective_mode = resolve_effective_train_step_mode(resolved_export_mode, train_steps_cfg)
    resolved_steps, components = resolve_train_steps_by_mode(
        effective_mode=effective_mode,
        exported_summary=exported_summary,
        train_steps_cfg=train_steps_cfg,
        base_steps=base_train_steps,
        round_id=round_id,
    )
    return {
        "train_steps_configured_mode": configured_mode,
        "train_steps_effective_mode": effective_mode,
        "export_mode": resolved_export_mode,
        "base_train_steps": int(base_train_steps),
        "resolved_train_steps": int(resolved_steps),
        "train_steps_components": components,
    }


def resolve_train_steps(
    round_id: int,
    exported_summary: dict[str, Any],
    base_train_steps: int,
    schedule_cfg: dict[str, Any] | None,
    export_mode: str | None = None,
) -> int:
    """Resolve per-round train steps for the supported round DAgger controller.

    This helper only schedules the round-based `robot-dagger` path. It is not
    related to the removed offline DAgger trainer/pipeline.
    """

    schedule = resolve_train_steps_schedule(
        round_id=round_id,
        exported_summary=exported_summary,
        base_train_steps=base_train_steps,
        schedule_cfg=schedule_cfg,
        export_mode=export_mode,
    )
    return int(schedule["resolved_train_steps"])


def _format_train_steps_components(components: dict[str, Any]) -> str:
    preferred_keys = (
        "full_episode_count",
        "full_steps",
        "recovery_segment_count",
        "recovery_frames",
        "recovery_steps",
        "frame_steps",
        "episode_steps",
        "fallback_mode",
        "raw_steps",
        "clamped_steps",
    )
    parts = [f"{key}={components[key]}" for key in preferred_keys if key in components]
    return ", ".join(parts) if parts else str(components)


def _dagger_rounds_with_current(
    round_history: list[dict[str, Any]],
    current_round_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rounds = [state for state in round_history if int(state.get("round_id", 0) or 0) > 0]
    rounds.append(current_round_state)
    return rounds


def _tail_count(rounds: list[dict[str, Any]], predicate) -> int:
    count = 0
    for state in reversed(rounds):
        if predicate(state):
            count += 1
        else:
            break
    return count


def should_stop_dagger(
    round_history: list[dict[str, Any]],
    current_round_state: dict[str, Any],
    schedule_cfg: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Evaluate early stopping for the supported round DAgger controller only."""

    early_stop_cfg = _early_stop_cfg(schedule_cfg)
    if not bool(early_stop_cfg.get("enabled", False)):
        return False, None

    exported_frames = int(current_round_state.get("exported_frames", 0) or 0)
    exported_episodes = int(current_round_state.get("exported_episodes", 0) or 0)
    intervention_count = int(current_round_state.get("intervention_count", 0) or 0)

    patience = max(1, int(early_stop_cfg.get("patience", 1) or 1))
    rounds = _dagger_rounds_with_current(round_history, current_round_state)

    if bool(early_stop_cfg.get("stop_if_no_new_data", True)):
        no_data_count = _tail_count(
            rounds,
            lambda state: int(state.get("exported_frames", 0) or 0) <= 0
            or int(state.get("exported_episodes", 0) or 0) <= 0,
        )
        if no_data_count >= patience:
            if intervention_count == 0 and exported_frames <= 0:
                return (
                    True,
                    f"no exported DAgger data for {no_data_count} consecutive round(s): "
                    "intervention_count=0 and exported_frames=0 "
                    "(policy may already be good, or this round had no effective takeover)",
                )
            return (
                True,
                f"no exported DAgger data for {no_data_count} consecutive round(s): "
                f"exported_frames={exported_frames}, exported_episodes={exported_episodes}",
            )

    if bool(early_stop_cfg.get("stop_if_export_too_small", True)):
        min_exported_frames = int(early_stop_cfg.get("min_exported_frames", 1) or 0)
        min_exported_episodes = int(early_stop_cfg.get("min_exported_episodes", 1) or 0)
        small_count = _tail_count(
            rounds,
            lambda state: int(state.get("exported_frames", 0) or 0) < min_exported_frames
            or int(state.get("exported_episodes", 0) or 0) < min_exported_episodes,
        )
        if small_count >= patience:
            return (
                True,
                f"export too small for {small_count} consecutive round(s): "
                f"min_exported_frames={min_exported_frames}, "
                f"min_exported_episodes={min_exported_episodes}",
            )

    if bool(early_stop_cfg.get("stop_if_low_expert_ratio", False)):
        min_expert_frame_ratio = float(early_stop_cfg.get("min_expert_frame_ratio", 0.0) or 0.0)
        low_ratio_count = _tail_count(
            rounds,
            lambda state: float(state.get("expert_frame_ratio", 0.0) or 0.0) < min_expert_frame_ratio,
        )
        if low_ratio_count >= patience:
            return (
                True,
                f"expert_frame_ratio too low for {low_ratio_count} consecutive round(s): "
                f"min_expert_frame_ratio={min_expert_frame_ratio}",
            )

    return False, None


def _make_record_section(
    base_record_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path,
    episodes_per_round: int,
    repo_prefix: str,
    policy_runner,
    round_cfg: dict[str, Any],
) -> dict[str, Any]:
    record_section = copy.deepcopy(base_record_cfg)
    raw_root = round_dir / "raw_run_mix"
    raw_repo_id = _safe_repo_id(repo_prefix, round_idx, "raw_run_mix")

    record_section["repo_id"] = raw_repo_id
    record_section["dataset_name"] = raw_repo_id
    record_section["dataset_root"] = str(raw_root)
    record_section["dataset_path"] = str(raw_root)
    record_section["run_mode"] = "run_mix"
    record_section.setdefault("task", {})
    record_section["task"]["num_episodes"] = episodes_per_round
    record_section["task"]["resume"] = False
    if _full_episode_export_enabled(round_cfg):
        success_policy = _full_episode_success_policy(round_cfg)
        record_section["task"]["success_policy"] = success_policy
        record_section["task"]["annotate_success"] = success_policy == SUCCESS_POLICY_EXPLICIT
    record_section.setdefault("storage", {})
    record_section["storage"]["push_to_hub"] = False
    return policy_runner.prepare_record_cfg(record_section, checkpoint_path, round_cfg)


def _train_round(
    trainer,
    base_train_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path | None,
    aggregated_repo_id: str,
    aggregated_root: Path,
    seed_repo_id: str,
    seed_root: Path,
    train_steps: int | None = None,
    dagger_sampling: dict[str, Any] | None = None,
) -> Path:
    train_output_dir = round_dir / "train"
    train_section = trainer.prepare_train_cfg(
        base_train_cfg=base_train_cfg,
        aggregated_dataset_path=aggregated_root,
        repo_id=aggregated_repo_id,
        checkpoint_in=checkpoint_path,
        output_dir=train_output_dir,
        round_id=round_idx,
        resolved_steps=train_steps,
        seed_repo_id=seed_repo_id,
        seed_root=seed_root,
        dagger_sampling=dagger_sampling,
    )
    logging.info(
        "[round %03d] train output: %s steps=%s",
        round_idx,
        train_output_dir,
        train_section.get("steps"),
    )
    return Path(trainer.train(train_section))


def run_dagger_rounds(config: DAggerRoundsConfig | dict[str, Any]) -> dict[str, Any]:
    from scripts.core.dagger_backends import make_policy_backend
    from scripts.core.dagger_sampling import (
        format_sampling_stats,
        source_index_path_for_dataset,
        summarize_sampling_for_dataset_root,
        write_aggregated_source_index,
    )
    from scripts.core.run_dagger_export import assert_dataset_roots_schema_compatible
    from scripts.core.run_record import RecordConfig, run_record

    if isinstance(config, dict):
        config = DAggerRoundsConfig(**config)
    round_cfg = copy.deepcopy(config.__dict__)
    round_policy_descriptor = _validate_round_policy_descriptor(config.policy)
    policy_backend = make_policy_backend(config.policy_backend)
    policy_backend_info = policy_backend.metadata()

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_repo_id, seed_root = _resolve_seed(config)
    repo_prefix = seed_repo_id.replace("/", "_")
    base_record_cfg = _load_yaml(config.record_cfg_path)["record"]
    train_cfg_path = config.train_cfg_path or policy_backend.trainer.train_cfg_path
    if train_cfg_path is None:
        raise ValueError("dagger_rounds.train_cfg_path or policy_backend.trainer.train_cfg_path is required.")
    base_train_cfg = _load_yaml(train_cfg_path)["train"]
    _apply_round_policy_descriptor(base_record_cfg, round_policy_descriptor, mode="reason")
    _apply_round_policy_descriptor(base_train_cfg, round_policy_descriptor, mode="train")
    round_schedule_cfg = _round_schedule_cfg(config)
    max_rounds = _resolve_max_rounds(config, round_schedule_cfg)
    base_train_steps = int(base_train_cfg.get("steps", 10_000))
    train_steps_cfg = _train_steps_cfg(round_schedule_cfg)
    train_steps_mode = str(train_steps_cfg.get("mode", "fixed"))
    warm_start_from_previous = bool(train_steps_cfg.get("warm_start_from_previous", True))
    dagger_sampling_cfg = _dagger_sampling_cfg(config)

    current_checkpoint = Path(config.initial_pretrained_path) if config.initial_pretrained_path else None
    previous_aggregated_repo_id = seed_repo_id
    previous_aggregated_root = seed_root
    rounds: list[dict[str, Any]] = []
    stop_state: dict[str, Any] = {
        "stopped_early": False,
        "stop_round": None,
        "reason": None,
    }

    if current_checkpoint is None:
        if not config.train_round0_if_missing:
            raise ValueError("initial_pretrained_path is required when train_round0_if_missing=False.")
        round0_dir = output_root / "round_000"
        _clean_or_fail(round0_dir, config.overwrite)
        round0_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[round 000] training initial %s policy from seed dataset: %s", policy_backend.backend_type, seed_root)
        current_checkpoint = _train_round(
            trainer=policy_backend.trainer,
            base_train_cfg=base_train_cfg,
            round_idx=0,
            round_dir=round0_dir,
            checkpoint_path=None,
            aggregated_repo_id=seed_repo_id,
            aggregated_root=seed_root,
            seed_repo_id=seed_repo_id,
            seed_root=seed_root,
        )
        round0_state = {
            "round_id": 0,
            "round": 0,
            "mode": "seed_train",
            "raw_dataset_path": None,
            "exported_dataset_path": None,
            "aggregated_dataset_path": str(seed_root),
            "seed_dataset": {"repo_id": seed_repo_id, "root": str(seed_root)},
            "train_output": str(round0_dir / "train"),
            "train_output_dir": str(round0_dir / "train"),
            "checkpoint": str(current_checkpoint),
            "selected_checkpoint_path": str(current_checkpoint),
            "policy_backend": policy_backend_info,
        }
        _write_json(round0_dir / "round_state.json", round0_state)
        rounds.append(round0_state)
    else:
        current_checkpoint = _require_checkpoint_exists(current_checkpoint, "initial")
        logging.info("[round 000] reuse initial checkpoint: %s", current_checkpoint)

    for round_idx in range(1, max_rounds + 1):
        round_dir = output_root / f"round_{round_idx:03d}"
        _clean_or_fail(round_dir, config.overwrite)
        round_dir.mkdir(parents=True, exist_ok=True)

        logging.info("========== [DAgger round %03d] ==========", round_idx)
        previous_state_path = output_root / f"round_{round_idx - 1:03d}" / "round_state.json"
        if previous_state_path.is_file():
            previous_state = _load_json(previous_state_path)
            previous_checkpoint = Path(previous_state["selected_checkpoint_path"])
            _require_checkpoint_exists(previous_checkpoint, f"round {round_idx - 1:03d} selected")
            if previous_checkpoint != current_checkpoint:
                raise RuntimeError(
                    f"Checkpoint handoff mismatch before round {round_idx:03d}: "
                    f"state file has {previous_checkpoint}, controller has {current_checkpoint}"
                )
            logging.info(
                "[round %03d] previous selected checkpoint from metadata: %s",
                round_idx,
                previous_checkpoint,
            )
        current_checkpoint = _require_checkpoint_exists(current_checkpoint, f"round {round_idx:03d} input")
        logging.info("[round %03d] checkpoint: %s", round_idx, current_checkpoint)

        record_section = _make_record_section(
            base_record_cfg=base_record_cfg,
            round_idx=round_idx,
            round_dir=round_dir,
            checkpoint_path=current_checkpoint,
            episodes_per_round=int(config.episodes_per_round),
            repo_prefix=repo_prefix,
            policy_runner=policy_backend.runner,
            round_cfg=round_cfg,
        )
        record_result = run_record(
            RecordConfig(
                record_section,
                config_source_name=str(config.record_cfg_path),
            )
        )
        raw_repo_id = record_result["dataset_name"]
        raw_root = Path(record_result["dataset_root"])
        logging.info("[round %03d] raw run_mix logs: %s", round_idx, raw_root)

        exported_root = round_dir / "exported_dagger_dataset"
        exported_repo_id = _safe_repo_id(repo_prefix, round_idx, "dagger_export")
        export_section = policy_backend.export_profile.prepare_export_cfg(
            {
                "raw_repo_id": raw_repo_id,
                "raw_root": raw_root,
                "output_repo_id": exported_repo_id,
                "output_root": exported_root,
                "seed_repo_id": seed_repo_id,
                "seed_root": seed_root,
                "min_segment_frames": int(config.export_min_segment_frames),
                "overwrite": config.overwrite,
            },
            base_train_cfg,
            round_cfg,
        )
        export_summary = policy_backend.export_profile.export(export_section)
        logging.info("[round %03d] exported dagger dataset: %s", round_idx, exported_root)
        configured_export_mode = (
            export_section.get("export_mode")
            if isinstance(export_section, dict)
            else None
        ) or _policy_backend_export_cfg(round_cfg).get("export_mode")

        run_mix_stats = record_result.get("run_mix_stats", {})
        export_metrics = _export_metrics(export_summary, run_mix_stats)
        exported_frames = export_metrics["exported_frames"]
        exported_episodes = export_metrics["exported_episodes"]
        expert_frame_ratio = export_metrics["expert_frame_ratio"]
        intervention_count = export_metrics["intervention_count"]
        logging.info(
            "[round %03d] exported frames=%d, episodes=%d, expert_ratio=%.4f, interventions=%d",
            round_idx,
            exported_frames,
            exported_episodes,
            expert_frame_ratio,
            intervention_count,
        )

        train_steps_schedule = resolve_train_steps_schedule(
            round_id=round_idx,
            exported_summary=export_summary,
            base_train_steps=base_train_steps,
            schedule_cfg=round_schedule_cfg,
            export_mode=configured_export_mode,
        )
        resolved_train_steps = int(train_steps_schedule["resolved_train_steps"])
        train_steps_effective_mode = str(train_steps_schedule["train_steps_effective_mode"])
        train_steps_export_mode = str(train_steps_schedule["export_mode"])
        train_steps_components = train_steps_schedule["train_steps_components"]
        logging.info(
            "[round %03d] train steps mode: configured=%s effective=%s export_mode=%s",
            round_idx,
            train_steps_mode,
            train_steps_effective_mode,
            train_steps_export_mode,
        )
        logging.info(
            "[round %03d] train steps resolved: base=%d, exported_frames=%d, "
            "exported_episodes=%d, resolved=%d",
            round_idx,
            base_train_steps,
            exported_frames,
            exported_episodes,
            resolved_train_steps,
        )
        logging.info(
            "[round %03d] train steps components: %s",
            round_idx,
            _format_train_steps_components(train_steps_components),
        )
        logging.info("[round %03d] resolved train steps=%d", round_idx, resolved_train_steps)

        skipped_train = exported_frames <= 0 or exported_episodes <= 0
        skipped_train_reason = None
        train_checkpoint_in = None
        source_index_path: Path | None = None
        dagger_sampling_summary: dict[str, Any] = {
            "requested_enabled": bool(dagger_sampling_cfg.get("enabled", False)),
            "sampler_enabled": False,
            "strategy": dagger_sampling_cfg.get("strategy", "source_weighted"),
            "dagger_sample_ratio": dagger_sampling_cfg.get("dagger_sample_ratio"),
            "replacement": dagger_sampling_cfg.get("replacement", True),
            "source_index_path": None,
            "seed_samples": None,
            "dagger_samples": None,
            "estimated_dagger_ratio": None,
            "disabled_reason": "train_not_started",
        }
        if skipped_train:
            skipped_train_reason = (
                f"empty exported DAgger dataset: exported_frames={exported_frames}, "
                f"exported_episodes={exported_episodes}"
            )
            logging.warning(
                "[round %03d] %s; skipping aggregate/train and keeping previous checkpoint.",
                round_idx,
                skipped_train_reason,
            )
            aggregated_repo_id = previous_aggregated_repo_id
            aggregated_root = previous_aggregated_root
            previous_source_index_path = source_index_path_for_dataset(aggregated_root)
            source_index_path = previous_source_index_path if previous_source_index_path.is_file() else None
            dagger_sampling_summary["source_index_path"] = str(source_index_path) if source_index_path else None
            dagger_sampling_summary["disabled_reason"] = "train_skipped"
            train_checkpoint = current_checkpoint
        else:
            from lerobot.datasets.aggregate import aggregate_datasets

            if resolved_train_steps <= 0:
                raise ValueError(
                    f"[round {round_idx:03d}] resolved_train_steps must be > 0 when export is non-empty; "
                    f"got {resolved_train_steps}."
                )
            aggregated_root = round_dir / "aggregated_dataset"
            aggregated_repo_id = _safe_repo_id(repo_prefix, round_idx, "aggregated")
            _clean_or_fail(aggregated_root, config.overwrite)
            logging.info(
                "[round %03d] aggregate previous=%s current=%s -> %s",
                round_idx,
                previous_aggregated_root,
                exported_root,
                aggregated_root,
            )
            assert_dataset_roots_schema_compatible(
                previous_aggregated_repo_id,
                previous_aggregated_root,
                exported_repo_id,
                exported_root,
            )
            aggregate_datasets(
                repo_ids=[previous_aggregated_repo_id, exported_repo_id],
                roots=[previous_aggregated_root, exported_root],
                aggr_repo_id=aggregated_repo_id,
                aggr_root=aggregated_root,
            )
            assert_dataset_roots_schema_compatible(
                seed_repo_id,
                seed_root,
                aggregated_repo_id,
                aggregated_root,
            )

            logging.info("[round %03d] aggregated dataset: %s", round_idx, aggregated_root)
            assert_dataset_roots_schema_compatible(seed_repo_id, seed_root, aggregated_repo_id, aggregated_root)
            previous_fallback_source = "seed" if round_idx == 1 else "seed_or_previous"
            source_index_path = write_aggregated_source_index(
                previous_root=previous_aggregated_root,
                current_root=exported_root,
                aggregated_root=aggregated_root,
                round_id=round_idx,
                export_mode=train_steps_export_mode,
                previous_fallback_source=previous_fallback_source,
            )
            dagger_sampling_train_cfg = _sampling_train_cfg(dagger_sampling_cfg, source_index_path)
            dagger_sampling_summary = summarize_sampling_for_dataset_root(
                dataset_root=aggregated_root,
                source_index_path=source_index_path,
                sampling_cfg=dagger_sampling_train_cfg,
            )
            logging.info("[round %03d] %s", round_idx, format_sampling_stats(dagger_sampling_summary))
            train_checkpoint_in = current_checkpoint if warm_start_from_previous else None
            train_checkpoint = _train_round(
                trainer=policy_backend.trainer,
                base_train_cfg=base_train_cfg,
                round_idx=round_idx,
                round_dir=round_dir,
                checkpoint_path=train_checkpoint_in,
                aggregated_repo_id=aggregated_repo_id,
                aggregated_root=aggregated_root,
                seed_repo_id=seed_repo_id,
                seed_root=seed_root,
                train_steps=resolved_train_steps,
                dagger_sampling=dagger_sampling_train_cfg,
            )
            logging.info("[round %03d] next checkpoint: %s", round_idx, train_checkpoint)

        round_state = {
            "round_id": round_idx,
            "round": round_idx,
            "checkpoint_in": str(current_checkpoint),
            "checkpoint_in_path": str(current_checkpoint),
            "train_checkpoint_in": str(train_checkpoint_in) if not skipped_train and train_checkpoint_in else None,
            "raw_dataset_path": str(raw_root),
            "exported_dataset_path": str(exported_root),
            "aggregated_dataset_path": str(aggregated_root),
            "dagger_source_index_path": str(source_index_path) if source_index_path else None,
            "train_output_dir": str(round_dir / "train"),
            "selected_checkpoint_path": str(train_checkpoint),
            "exported_frames": exported_frames,
            "exported_episodes": exported_episodes,
            "export_mode": train_steps_export_mode,
            "full_episode_count": export_metrics["full_episode_count"],
            "full_episode_frames": export_metrics["full_episode_frames"],
            "recovery_segment_count": export_metrics["recovery_segment_count"],
            "recovery_frames": export_metrics["recovery_frames"],
            "expert_frame_ratio": expert_frame_ratio,
            "intervention_count": intervention_count,
            "skipped_short_segments": export_metrics["skipped_short_segments"],
            "incomplete_expert_label_frames": export_metrics["incomplete_expert_label_frames"],
            "success_policy": export_metrics["success_policy"],
            "success_inferred_episodes": export_metrics["success_inferred_episodes"],
            "success_explicit_true_episodes": export_metrics["success_explicit_true_episodes"],
            "success_explicit_false_episodes": export_metrics["success_explicit_false_episodes"],
            "skipped_explicit_failure": export_metrics["skipped_explicit_failure"],
            "base_train_steps": base_train_steps,
            "resolved_train_steps": resolved_train_steps,
            "train_steps_mode": train_steps_mode,
            "train_steps_effective_mode": train_steps_effective_mode,
            "train_steps_components": train_steps_components,
            "skipped_train": skipped_train,
            "skipped_train_reason": skipped_train_reason,
            "early_stop_triggered": False,
            "early_stop_reason": None,
            "raw_run_mix": {"repo_id": raw_repo_id, "root": str(raw_root)},
            "exported_dagger": {
                "repo_id": exported_repo_id,
                "root": str(exported_root),
                "summary": export_summary,
            },
            "aggregated_dataset": {
                "repo_id": aggregated_repo_id,
                "root": str(aggregated_root),
                "source_index_path": str(source_index_path) if source_index_path else None,
            },
            "dagger_sampling": dagger_sampling_summary,
            "train_output": str(round_dir / "train"),
            "checkpoint_out": str(train_checkpoint),
            "policy_backend": policy_backend_info,
            "run_mix_stats": run_mix_stats,
            "schedule": {
                **train_steps_schedule,
                "resolved_train_steps": resolved_train_steps,
                "base_train_steps": base_train_steps,
                "train_steps_mode": train_steps_mode,
                "train_steps_configured_mode": train_steps_mode,
                "train_steps_effective_mode": train_steps_effective_mode,
                "export_mode": train_steps_export_mode,
                "train_steps_components": train_steps_components,
                "warm_start_from_previous": warm_start_from_previous,
                "early_stop_checked": True,
                "early_stop_triggered": False,
                "early_stop_reason": None,
            },
        }
        early_stop_triggered, early_stop_reason = should_stop_dagger(rounds, round_state, round_schedule_cfg)
        logging.info(
            "[round %03d] early stop check: triggered=%s, reason=%s",
            round_idx,
            early_stop_triggered,
            early_stop_reason,
        )
        round_state["early_stop_triggered"] = early_stop_triggered
        round_state["early_stop_reason"] = early_stop_reason
        round_state["schedule"]["early_stop_triggered"] = early_stop_triggered
        round_state["schedule"]["early_stop_reason"] = early_stop_reason
        _write_json(round_dir / "round_state.json", round_state)
        rounds.append(round_state)

        current_checkpoint = train_checkpoint
        previous_aggregated_repo_id = aggregated_repo_id
        previous_aggregated_root = aggregated_root

        if early_stop_triggered:
            stop_state = {
                "stopped_early": True,
                "stop_round": round_idx,
                "reason": early_stop_reason,
            }
            break

    final_state = {
        "output_root": str(output_root),
        "policy_backend": policy_backend_info,
        "seed_dataset": {"repo_id": seed_repo_id, "root": str(seed_root)},
        "final_checkpoint": str(current_checkpoint),
        "final_aggregated_dataset": {
            "repo_id": previous_aggregated_repo_id,
            "root": str(previous_aggregated_root),
        },
        "total_rounds_completed": sum(1 for state in rounds if int(state.get("round_id", 0) or 0) > 0),
        "stopped_early": bool(stop_state["stopped_early"]),
        "stop_reason": stop_state["reason"],
        "stop": stop_state,
        "round_schedule": round_schedule_cfg,
        "dagger_training": copy.deepcopy(config.dagger_training or {}),
        "round_schedules": [
            {
                "round": state.get("round_id"),
                "exported_frames": state.get("exported_frames"),
                "exported_episodes": state.get("exported_episodes"),
                "resolved_train_steps": state.get("resolved_train_steps"),
                "base_train_steps": state.get("base_train_steps"),
                "train_steps_mode": state.get("train_steps_mode"),
                "train_steps_effective_mode": state.get("train_steps_effective_mode"),
                "export_mode": state.get("export_mode"),
                "train_steps_components": state.get("train_steps_components"),
                "dagger_sampling": state.get("dagger_sampling"),
                "skipped_train": state.get("skipped_train"),
                "early_stop_triggered": state.get("early_stop_triggered"),
                "early_stop_reason": state.get("early_stop_reason"),
            }
            for state in rounds
            if int(state.get("round_id", 0) or 0) > 0
        ],
        "rounds": rounds,
    }
    _write_json(output_root / "dagger_rounds_state.json", final_state)
    return final_state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run staged DAgger collection/export/training rounds.")
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_dagger_rounds_cfg_path(),
        help="Path to dagger_rounds_cfg.yaml.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    cfg = _load_yaml(args.config_path)
    rounds_cfg = cfg.get("dagger_rounds", cfg)
    run_dagger_rounds(rounds_cfg)


if __name__ == "__main__":
    main()
