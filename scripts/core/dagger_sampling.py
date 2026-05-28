#!/usr/bin/env python

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_INDEX_FILENAME = "dagger_source_index.json"

DEFAULT_DAGGER_SOURCE_VALUES = (
    "dagger",
    "exported_dagger",
    "full_success_episode",
    "intervention_segment",
    "intervention_segments",
    "hybrid_full",
    "hybrid_recovery",
)

VALID_MISSING_SOURCE_POLICIES = {"disable_sampler", "treat_as_seed", "error"}
VALID_SAMPLING_STRATEGIES = {"none", "source_weighted"}


@dataclass
class SourceWeightedSamplerResult:
    sampler: Any | None
    weights: Any | None
    dagger_mask: Any | None
    stats: dict[str, Any]


def source_index_path_for_dataset(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / SOURCE_INDEX_FILENAME


def load_source_index(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read_dataset_info(dataset_root: str | Path) -> dict[str, Any]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


def dataset_counts(dataset_root: str | Path) -> dict[str, int]:
    info = _read_dataset_info(dataset_root)
    return {
        "total_frames": int(info.get("total_frames", 0) or 0),
        "total_episodes": int(info.get("total_episodes", 0) or 0),
    }


def _episode_frame_ranges(dataset_root: str | Path) -> list[tuple[int, int]]:
    from lerobot.datasets.utils import load_episodes

    episodes = load_episodes(Path(dataset_root))
    ranges: list[tuple[int, int]] = []
    for episode in episodes:
        ranges.append((int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
    return ranges


def _range_entry(
    start: int,
    end: int,
    source: str,
    *,
    round_id: int | None = None,
    export_mode: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"from": int(start), "to": int(end), "source": source}
    if round_id is not None:
        entry["round_id"] = int(round_id)
    if export_mode:
        entry["export_mode"] = str(export_mode)
    if role:
        entry["role"] = role
    return entry


def _episode_entries_from_ranges(
    frame_ranges: list[tuple[int, int]],
    episode_offset: int,
    frame_offset: int,
    source: str,
    *,
    round_id: int | None = None,
    export_mode: str | None = None,
    role: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    episodes: dict[str, dict[str, Any]] = {}
    episode_ranges: list[dict[str, Any]] = []
    for local_ep_idx, (start, end) in enumerate(frame_ranges):
        global_ep_idx = int(episode_offset + local_ep_idx)
        global_start = int(frame_offset + start)
        global_end = int(frame_offset + end)
        entry = _range_entry(
            global_start,
            global_end,
            source,
            round_id=round_id,
            export_mode=export_mode,
            role=role,
        )
        entry["episode_index"] = global_ep_idx
        episodes[str(global_ep_idx)] = dict(entry)
        episode_ranges.append(entry)
    return episodes, episode_ranges


def _fallback_entries(
    total_frames: int,
    total_episodes: int,
    source: str,
    *,
    round_id: int | None = None,
    export_mode: str | None = None,
    role: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_ranges = [_range_entry(0, total_frames, source, round_id=round_id, export_mode=export_mode, role=role)]
    episodes: dict[str, dict[str, Any]] = {}
    episode_ranges: list[dict[str, Any]] = []
    if total_episodes > 0:
        for ep_idx in range(total_episodes):
            entry = {"episode_index": ep_idx, "source": source}
            if round_id is not None:
                entry["round_id"] = int(round_id)
            if export_mode:
                entry["export_mode"] = str(export_mode)
            if role:
                entry["role"] = role
            episodes[str(ep_idx)] = entry
            episode_ranges.append(dict(entry))
    return episodes, episode_ranges, frame_ranges


def build_aggregated_source_index(
    *,
    previous_root: str | Path,
    current_root: str | Path,
    aggregated_root: str | Path,
    round_id: int,
    export_mode: str | None,
    previous_fallback_source: str = "seed_or_previous",
) -> dict[str, Any]:
    previous_root = Path(previous_root)
    current_root = Path(current_root)
    aggregated_root = Path(aggregated_root)

    previous_counts = dataset_counts(previous_root)
    current_counts = dataset_counts(current_root)
    aggregated_counts = dataset_counts(aggregated_root)
    previous_frames = previous_counts["total_frames"]
    previous_episodes = previous_counts["total_episodes"]
    current_frames = current_counts["total_frames"]

    previous_source_path = source_index_path_for_dataset(previous_root)
    episodes: dict[str, dict[str, Any]]
    episode_ranges: list[dict[str, Any]]
    frame_ranges: list[dict[str, Any]]

    if previous_source_path.is_file():
        previous_index = load_source_index(previous_source_path)
        episodes = {str(k): dict(v) for k, v in previous_index.get("episodes", {}).items()}
        episode_ranges = [dict(entry) for entry in previous_index.get("episode_ranges", [])]
        frame_ranges = [dict(entry) for entry in previous_index.get("frame_ranges", [])]
        inherited_source = "sidecar"
    else:
        try:
            previous_episode_ranges = _episode_frame_ranges(previous_root)
            episodes, episode_ranges = _episode_entries_from_ranges(
                previous_episode_ranges,
                episode_offset=0,
                frame_offset=0,
                source=previous_fallback_source,
                role="previous",
            )
            frame_ranges = [
                _range_entry(0, previous_frames, previous_fallback_source, role="previous")
            ]
        except Exception as exc:  # pragma: no cover - defensive fallback for odd legacy datasets
            logging.warning(
                "Could not read previous episode metadata for source index at %s: %s; "
                "falling back to one frame range.",
                previous_root,
                exc,
            )
            episodes, episode_ranges, frame_ranges = _fallback_entries(
                previous_frames,
                previous_episodes,
                previous_fallback_source,
                role="previous",
            )
        inherited_source = "fallback"

    if current_frames > 0:
        current_role = "current_dagger"
        frame_ranges.append(
            _range_entry(
                previous_frames,
                previous_frames + current_frames,
                "dagger",
                round_id=round_id,
                export_mode=export_mode,
                role=current_role,
            )
        )
        try:
            current_episode_ranges = _episode_frame_ranges(current_root)
            current_episodes_map, current_episode_ranges_out = _episode_entries_from_ranges(
                current_episode_ranges,
                episode_offset=previous_episodes,
                frame_offset=previous_frames,
                source="dagger",
                round_id=round_id,
                export_mode=export_mode,
                role=current_role,
            )
            episodes.update(current_episodes_map)
            episode_ranges.extend(current_episode_ranges_out)
        except Exception as exc:  # pragma: no cover - defensive fallback for odd legacy datasets
            logging.warning(
                "Could not read current exported episode metadata for source index at %s: %s; "
                "recording frame range only.",
                current_root,
                exc,
            )

    source_index = {
        "version": 1,
        "dataset_root": str(aggregated_root),
        "round_id": int(round_id),
        "previous_root": str(previous_root),
        "current_root": str(current_root),
        "previous_source_index": str(previous_source_path) if previous_source_path.is_file() else None,
        "previous_inherited_from": inherited_source,
        "export_mode": export_mode,
        "total_frames": aggregated_counts["total_frames"],
        "total_episodes": aggregated_counts["total_episodes"],
        "episodes": episodes,
        "episode_ranges": episode_ranges,
        "frame_ranges": frame_ranges,
    }
    return source_index


def write_aggregated_source_index(
    *,
    previous_root: str | Path,
    current_root: str | Path,
    aggregated_root: str | Path,
    round_id: int,
    export_mode: str | None,
    previous_fallback_source: str = "seed_or_previous",
) -> Path:
    source_index = build_aggregated_source_index(
        previous_root=previous_root,
        current_root=current_root,
        aggregated_root=aggregated_root,
        round_id=round_id,
        export_mode=export_mode,
        previous_fallback_source=previous_fallback_source,
    )
    output_path = source_index_path_for_dataset(aggregated_root)
    _write_json(output_path, source_index)
    logging.info(
        "Wrote DAgger source index: %s (frames=%s episodes=%s)",
        output_path,
        source_index.get("total_frames"),
        source_index.get("total_episodes"),
    )
    return output_path


def normalize_sampling_config(sampling_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(sampling_cfg or {})
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["strategy"] = str(cfg.get("strategy", "none" if not cfg["enabled"] else "source_weighted"))
    cfg["strategy"] = cfg["strategy"].strip().lower()
    if cfg["strategy"] not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"dagger sampling strategy must be one of {sorted(VALID_SAMPLING_STRATEGIES)}; "
            f"got {cfg['strategy']!r}"
        )

    ratio = float(cfg.get("dagger_sample_ratio", 0.0) or 0.0)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"dagger_sample_ratio must be in [0, 1], got {ratio}")
    cfg["dagger_sample_ratio"] = ratio

    on_missing_source = str(cfg.get("on_missing_source", "disable_sampler")).strip().lower()
    if on_missing_source not in VALID_MISSING_SOURCE_POLICIES:
        raise ValueError(
            f"on_missing_source must be one of {sorted(VALID_MISSING_SOURCE_POLICIES)}; "
            f"got {on_missing_source!r}"
        )
    cfg["on_missing_source"] = on_missing_source
    cfg["replacement"] = bool(cfg.get("replacement", True))
    cfg["source_field_policy"] = str(cfg.get("source_field_policy", "metadata_or_sidecar"))
    cfg["dagger_source_values"] = list(cfg.get("dagger_source_values") or DEFAULT_DAGGER_SOURCE_VALUES)
    return cfg


def _disabled_result(cfg: dict[str, Any], reason: str, *, num_samples: int = 0) -> SourceWeightedSamplerResult:
    stats = {
        "requested_enabled": bool(cfg.get("enabled", False)),
        "sampler_enabled": False,
        "strategy": cfg.get("strategy", "none"),
        "dagger_sample_ratio": float(cfg.get("dagger_sample_ratio", 0.0) or 0.0),
        "replacement": bool(cfg.get("replacement", True)),
        "num_samples": int(num_samples),
        "seed_samples": None,
        "dagger_samples": None,
        "missing_source_samples": None,
        "seed_per_sample_weight": None,
        "dagger_per_sample_weight": None,
        "estimated_dagger_ratio": None,
        "replacement_warning": None,
        "disabled_reason": reason,
    }
    return SourceWeightedSamplerResult(None, None, None, stats)


def _handle_missing_source(cfg: dict[str, Any], reason: str, *, num_samples: int) -> SourceWeightedSamplerResult:
    if cfg["on_missing_source"] == "error":
        raise ValueError(reason)
    if cfg["on_missing_source"] == "treat_as_seed":
        logging.warning("%s; treating all unrecognized samples as seed and disabling sampler.", reason)
        result = _disabled_result(cfg, f"{reason}; treated_as_seed", num_samples=num_samples)
        result.stats["seed_samples"] = int(num_samples)
        result.stats["dagger_samples"] = 0
        result.stats["missing_source_samples"] = int(num_samples)
        return result
    logging.warning("%s; DAgger source-aware sampler disabled.", reason)
    return _disabled_result(cfg, reason, num_samples=num_samples)


def _source_value_is_dagger(entry: dict[str, Any], dagger_values: set[str]) -> bool:
    for key in ("source", "source_detail", "export_mode", "role"):
        value = entry.get(key)
        if value is not None and str(value).strip().lower() in dagger_values:
            return True
    return False


def _ranges_from_source_index(source_index: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = source_index.get("frame_ranges")
    if isinstance(ranges, list) and ranges:
        return [dict(entry) for entry in ranges]

    episodes = source_index.get("episodes", {})
    out: list[dict[str, Any]] = []
    if isinstance(episodes, dict):
        for entry in episodes.values():
            if not isinstance(entry, dict):
                continue
            if "from" in entry and "to" in entry:
                out.append(dict(entry))
    return out


def build_source_weighted_sampler_from_index(
    *,
    num_samples: int,
    source_index: dict[str, Any] | None,
    sampling_cfg: dict[str, Any] | None,
) -> SourceWeightedSamplerResult:
    cfg = normalize_sampling_config(sampling_cfg)
    if not cfg["enabled"] or cfg["strategy"] == "none":
        return _disabled_result(cfg, "disabled_by_config", num_samples=num_samples)

    if source_index is None:
        return _handle_missing_source(cfg, "DAgger source index is missing", num_samples=num_samples)

    ranges = _ranges_from_source_index(source_index)
    if not ranges:
        return _handle_missing_source(
            cfg,
            "DAgger source index contains no frame_ranges or episode ranges",
            num_samples=num_samples,
        )

    dagger_values = {str(value).strip().lower() for value in cfg["dagger_source_values"]}
    import torch
    from torch.utils.data import WeightedRandomSampler

    dagger_mask = torch.zeros(num_samples, dtype=torch.bool)
    assigned_mask = torch.zeros(num_samples, dtype=torch.bool)

    for entry in ranges:
        try:
            start = max(0, int(entry["from"]))
            end = min(num_samples, int(entry["to"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        is_dagger = _source_value_is_dagger(entry, dagger_values)
        dagger_mask[start:end] = is_dagger
        assigned_mask[start:end] = True

    missing_count = int((~assigned_mask).sum().item())
    if missing_count:
        reason = f"DAgger source index leaves {missing_count} of {num_samples} samples unassigned"
        if cfg["on_missing_source"] == "error":
            raise ValueError(reason)
        if cfg["on_missing_source"] == "disable_sampler":
            return _handle_missing_source(cfg, reason, num_samples=num_samples)
        logging.warning("%s; treating unassigned samples as seed.", reason)

    dagger_count = int(dagger_mask.sum().item())
    seed_count = int(num_samples - dagger_count)
    ratio = float(cfg["dagger_sample_ratio"])

    if dagger_count == 0:
        return _handle_missing_source(
            cfg,
            "DAgger source-aware sampler found zero DAgger samples",
            num_samples=num_samples,
        )

    weights = torch.zeros(num_samples, dtype=torch.double)
    if seed_count == 0:
        logging.warning("DAgger source-aware sampler found zero seed samples; all samples are DAgger.")
        weights[dagger_mask] = 1.0 / dagger_count
        seed_weight = None
        dagger_weight = 1.0 / dagger_count
    else:
        seed_weight = (1.0 - ratio) / seed_count
        dagger_weight = ratio / dagger_count
        weights[~dagger_mask] = seed_weight
        weights[dagger_mask] = dagger_weight

    replacement_warning = None
    if not cfg["replacement"]:
        replacement_warning = (
            "replacement=false cannot oversample scarce DAgger samples across a full epoch; "
            "the requested ratio mainly affects sample order."
        )
        logging.warning("DAgger source-aware sampler: %s", replacement_warning)

    total_weight = float(weights.sum().item())
    estimated_dagger_ratio = float(weights[dagger_mask].sum().item() / total_weight) if total_weight else 0.0
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=bool(cfg["replacement"]),
    )
    stats = {
        "requested_enabled": True,
        "sampler_enabled": True,
        "strategy": cfg["strategy"],
        "dagger_sample_ratio": ratio,
        "replacement": bool(cfg["replacement"]),
        "num_samples": int(num_samples),
        "seed_samples": seed_count,
        "dagger_samples": dagger_count,
        "missing_source_samples": missing_count,
        "seed_per_sample_weight": seed_weight,
        "dagger_per_sample_weight": dagger_weight,
        "estimated_dagger_ratio": estimated_dagger_ratio,
        "replacement_warning": replacement_warning,
        "disabled_reason": None,
    }
    return SourceWeightedSamplerResult(sampler, weights, dagger_mask, stats)


def build_source_weighted_sampler_for_dataset(
    dataset: Any,
    sampling_cfg: dict[str, Any] | None,
) -> SourceWeightedSamplerResult:
    cfg = normalize_sampling_config(sampling_cfg)
    source_index_path = cfg.get("source_index_path")
    if not source_index_path:
        dataset_root = getattr(dataset, "root", None)
        if dataset_root is not None:
            source_index_path = source_index_path_for_dataset(dataset_root)
    if source_index_path:
        cfg["source_index_path"] = str(source_index_path)

    source_index = None
    if source_index_path and Path(source_index_path).is_file():
        source_index = load_source_index(source_index_path)
    elif cfg["enabled"]:
        return _handle_missing_source(
            cfg,
            f"DAgger source index not found: {source_index_path}",
            num_samples=len(dataset),
        )

    result = build_source_weighted_sampler_from_index(
        num_samples=len(dataset),
        source_index=source_index,
        sampling_cfg=cfg,
    )
    result.stats["source_index_path"] = str(source_index_path) if source_index_path else None
    return result


def summarize_sampling_for_dataset_root(
    *,
    dataset_root: str | Path,
    source_index_path: str | Path | None,
    sampling_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = normalize_sampling_config(sampling_cfg)
    if source_index_path is not None:
        cfg["source_index_path"] = str(source_index_path)
    source_index = None
    if source_index_path is not None and Path(source_index_path).is_file():
        source_index = load_source_index(source_index_path)
    result = build_source_weighted_sampler_from_index(
        num_samples=dataset_counts(dataset_root)["total_frames"],
        source_index=source_index,
        sampling_cfg=cfg,
    )
    result.stats["source_index_path"] = str(source_index_path) if source_index_path else None
    return result.stats


def format_sampling_stats(stats: dict[str, Any]) -> str:
    return (
        "DAgger source-aware sampling: "
        f"enabled={stats.get('sampler_enabled')} "
        f"strategy={stats.get('strategy')} "
        f"seed_samples={stats.get('seed_samples')} "
        f"dagger_samples={stats.get('dagger_samples')} "
        f"target_dagger_ratio={stats.get('dagger_sample_ratio')} "
        f"estimated_dagger_ratio={stats.get('estimated_dagger_ratio')} "
        f"replacement={stats.get('replacement')} "
        f"source_index={stats.get('source_index_path')} "
        f"disabled_reason={stats.get('disabled_reason')}"
    )
