#!/usr/bin/env python
"""Merge multiple local LeRobot datasets into one materialized dataset.

The source datasets must share a compatible LeRobot schema. By default this
tool uses LeRobot's parquet/video aggregate path, which avoids decoding and
re-encoding every video frame. It falls back to the slower frame writer path
when a partial episode selection is requested.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:  # Keep --help/import usable outside the LeRobot runtime environment.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    np = None

try:  # pragma: no cover - depends on local env
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    torch = None

try:  # pragma: no cover - depends on local env
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import DEFAULT_FEATURES
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    LeRobotDataset = None
    LeRobotDatasetMetadata = None
    DEFAULT_FEATURES = {
        "timestamp": {},
        "frame_index": {},
        "episode_index": {},
        "index": {},
        "task_index": {},
    }

try:  # pragma: no cover - depends on local env
    from lerobot.datasets.aggregate import aggregate_datasets
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on local env
    aggregate_datasets = None


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    root: Path
    episodes: list[int] | None = None
    max_episodes: int | None = None


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict) or "merge_datasets" not in loaded:
        raise ValueError(f"Config must contain a top-level `merge_datasets` mapping: {path}")
    cfg = loaded["merge_datasets"]
    if not isinstance(cfg, dict):
        raise ValueError("`merge_datasets` must be a mapping.")
    return cfg


def _require_lerobot() -> None:
    if LeRobotDataset is None:
        raise SystemExit(
            "This tool needs lerobot installed/importable. Run it in the same "
            "environment you use for robot-record/robot-train."
        )
    if np is None:
        raise SystemExit("This tool needs numpy. Run it in the LeRobot environment.")


def _as_path_or_none(value: str | Path | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _resolve_for_safety(left)
    right = _resolve_for_safety(right)
    return left == right or left in right.parents or right in left.parents


def _repo_id_from_name(name: str, prefix: str | None) -> str:
    if "/" in name or not prefix:
        return name
    return f"{prefix.strip('/')}/{name.strip('/')}"


def _normalize_source_specs(cfg: dict[str, Any]) -> list[SourceSpec]:
    source_cfg = cfg.get("source") or {}
    parent_dir = _as_path_or_none(
        source_cfg.get("parent_dir") or source_cfg.get("common_dir") or source_cfg.get("root")
    )
    if parent_dir is None:
        raise ValueError("source.parent_dir is required.")

    datasets = source_cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("source.datasets must be a non-empty list.")

    default_prefix = source_cfg.get("repo_id_prefix")
    specs: list[SourceSpec] = []
    for item in datasets:
        episodes = None
        max_episodes = source_cfg.get("max_episodes")
        if isinstance(item, str):
            name = item
            repo_id = _repo_id_from_name(name, default_prefix)
            root = parent_dir / name
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("dataset_name") or "").strip()
            if not name and not item.get("root"):
                raise ValueError(f"Dataset mapping must contain `name` or `root`: {item}")
            repo_id = str(item.get("repo_id") or _repo_id_from_name(name, default_prefix))
            root = _as_path_or_none(item.get("root")) or parent_dir / name
            if item.get("episodes") is not None:
                episodes = [int(ep) for ep in item["episodes"]]
            if item.get("max_episodes") is not None:
                max_episodes = int(item["max_episodes"])
        else:
            raise ValueError(f"source.datasets items must be strings or mappings, got: {item!r}")

        if max_episodes is not None and episodes is not None:
            episodes = episodes[: int(max_episodes)]

        specs.append(
            SourceSpec(
                name=name or root.name,
                repo_id=repo_id,
                root=root.expanduser(),
                episodes=episodes,
                max_episodes=None if max_episodes is None else int(max_episodes),
            )
        )
    return specs


def _output_repo_id(output_cfg: dict[str, Any], dataset_name: str, source_cfg: dict[str, Any]) -> str:
    if output_cfg.get("repo_id"):
        return str(output_cfg["repo_id"])
    prefix = output_cfg.get("repo_id_prefix") or source_cfg.get("repo_id_prefix")
    return _repo_id_from_name(dataset_name, prefix)


def _output_root(cfg: dict[str, Any]) -> Path:
    output_cfg = cfg.get("output") or {}
    explicit_root = _as_path_or_none(output_cfg.get("root"))
    if explicit_root is not None:
        return explicit_root
    dataset_name = str(output_cfg.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError("output.dataset_name is required when output.root is not set.")
    parent_dir = _as_path_or_none(output_cfg.get("parent_dir"))
    if parent_dir is None:
        raise ValueError("output.parent_dir is required when output.root is not set.")
    return parent_dir / dataset_name


def _selected_episodes(dataset: Any, spec: SourceSpec) -> list[int]:
    metadata = _metadata(dataset)
    if spec.episodes is None:
        episodes = list(range(int(metadata.total_episodes)))
    else:
        episodes = list(spec.episodes)

    if spec.max_episodes is not None:
        episodes = episodes[: spec.max_episodes]

    total = int(metadata.total_episodes)
    bad = [ep for ep in episodes if ep < 0 or ep >= total]
    if bad:
        raise ValueError(f"{spec.name} has invalid episode indices {bad}; total_episodes={total}")
    return episodes


def _features_without_defaults(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key not in DEFAULT_FEATURES}


def _json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _metadata(dataset_or_metadata: Any) -> Any:
    return getattr(dataset_or_metadata, "meta", dataset_or_metadata)


def _total_episodes(dataset_or_metadata: Any) -> int:
    return int(_metadata(dataset_or_metadata).total_episodes)


def _total_frames(dataset_or_metadata: Any) -> int:
    return int(_metadata(dataset_or_metadata).total_frames)


def _robot_type(dataset_or_metadata: Any) -> Any:
    return _metadata(dataset_or_metadata).info.get("robot_type")


def _validate_compatible_sources(
    reference: Any,
    candidate: Any,
    *,
    reference_name: str,
    candidate_name: str,
    strict_schema: bool,
    strict_robot_type: bool,
) -> None:
    reference_meta = _metadata(reference)
    candidate_meta = _metadata(candidate)

    if int(candidate_meta.fps) != int(reference_meta.fps):
        raise ValueError(
            f"FPS mismatch: {reference_name} fps={reference_meta.fps}, "
            f"{candidate_name} fps={candidate_meta.fps}"
        )

    if strict_robot_type:
        reference_robot = _robot_type(reference_meta)
        candidate_robot = _robot_type(candidate_meta)
        if candidate_robot != reference_robot:
            raise ValueError(
                f"robot_type mismatch: {reference_name}={reference_robot!r}, "
                f"{candidate_name}={candidate_robot!r}"
            )

    if strict_schema:
        ref_features = _features_without_defaults(reference_meta.features)
        candidate_features = _features_without_defaults(candidate_meta.features)
        if _json_signature(candidate_features) != _json_signature(ref_features):
            ref_keys = set(ref_features)
            candidate_keys = set(candidate_features)
            raise ValueError(
                "Feature schema mismatch between source datasets.\n"
                f"Reference: {reference_name}\n"
                f"Candidate: {candidate_name}\n"
                f"Only in reference: {sorted(ref_keys - candidate_keys)}\n"
                f"Only in candidate: {sorted(candidate_keys - ref_keys)}"
            )


def _assert_output_is_separate(output_root: Path, sources: list[SourceSpec]) -> None:
    for spec in sources:
        if _paths_overlap(output_root, spec.root):
            raise ValueError(
                "Refusing to merge in place: output.root overlaps a source dataset.\n"
                f"source={_resolve_for_safety(spec.root)}\n"
                f"output={_resolve_for_safety(output_root)}"
            )


def _to_numpy(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if np is not None and isinstance(value, np.ndarray):
        return value
    return value


def _restore_image_layout(value: Any, feature: dict[str, Any]) -> Any:
    array = _to_numpy(value)
    if np is None or not isinstance(array, np.ndarray):
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


def _frame_from_source_item(source: Any, item: dict[str, Any]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key, feature in source.features.items():
        if key in DEFAULT_FEATURES:
            continue
        if feature["dtype"] in {"image", "video"}:
            frame[key] = _restore_image_layout(item[key], feature)
        else:
            frame[key] = _to_numpy(item[key])
    frame["task"] = item["task"]
    return frame


def _episode_bounds(dataset: Any, ep_idx: int) -> tuple[int, int]:
    ep = _metadata(dataset).episodes[int(ep_idx)]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def _write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    summary_path = output_root / "meta" / "merge_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fast_aggregate_skip_reason(
    *,
    cfg: dict[str, Any],
    output_cfg: dict[str, Any],
    source_specs: list[SourceSpec],
    strict_schema: bool,
    strict_robot_type: bool,
) -> str | None:
    fast_merge = bool(output_cfg.get("fast_merge", cfg.get("fast_merge", True)))
    if not fast_merge:
        return "fast_merge=false"
    if aggregate_datasets is None or LeRobotDatasetMetadata is None:
        return "lerobot.datasets.aggregate is not available"
    if not strict_schema:
        return "source.strict_schema=false"
    if not strict_robot_type:
        return "source.strict_robot_type=false"
    for spec in source_specs:
        if spec.episodes is not None:
            return f"{spec.name} selects explicit episodes"
        if spec.max_episodes is not None:
            return f"{spec.name} uses max_episodes"
    return None


def _load_sources(
    source_specs: list[SourceSpec],
    *,
    download_videos: bool,
    metadata_only: bool,
) -> list[tuple[SourceSpec, Any, list[int]]]:
    loaded_sources: list[tuple[SourceSpec, Any, list[int]]] = []
    for spec in source_specs:
        if not spec.root.exists():
            raise FileNotFoundError(f"Source dataset root does not exist: {spec.root}")

        if metadata_only:
            if LeRobotDatasetMetadata is None:
                raise RuntimeError("LeRobotDatasetMetadata is not available in this LeRobot environment.")
            dataset_or_meta = LeRobotDatasetMetadata(spec.repo_id, root=spec.root)
        else:
            dataset_or_meta = LeRobotDataset(spec.repo_id, root=spec.root, download_videos=download_videos)

        episodes = _selected_episodes(dataset_or_meta, spec)
        loaded_sources.append((spec, dataset_or_meta, episodes))
        logger.info(
            "[SOURCE] %s root=%s episodes=%d/%d frames=%d",
            spec.repo_id,
            spec.root,
            len(episodes),
            _total_episodes(dataset_or_meta),
            _total_frames(dataset_or_meta),
        )
    return loaded_sources


def _run_fast_aggregate(
    *,
    source_specs: list[SourceSpec],
    output_repo_id: str,
    output_root: Path,
    output_cfg: dict[str, Any],
) -> None:
    logger.info("[FAST] Using lerobot.datasets.aggregate.aggregate_datasets")
    aggregate_datasets(
        repo_ids=[spec.repo_id for spec in source_specs],
        roots=[spec.root for spec in source_specs],
        aggr_repo_id=output_repo_id,
        aggr_root=output_root,
        data_files_size_in_mb=output_cfg.get("data_files_size_in_mb"),
        video_files_size_in_mb=output_cfg.get("video_files_size_in_mb"),
        chunk_size=output_cfg.get("chunk_size") or output_cfg.get("chunks_size"),
    )


def _run_frame_rewrite_merge(
    *,
    source_specs: list[SourceSpec],
    loaded_metadata: list[tuple[SourceSpec, Any, list[int]]],
    output_repo_id: str,
    output_root: Path,
    output_cfg: dict[str, Any],
    download_videos: bool,
) -> int:
    logger.info("[SLOW] Falling back to frame rewrite via LeRobotDataset.add_frame/save_episode")
    reference = loaded_metadata[0][1]
    loaded_datasets = _load_sources(source_specs, download_videos=download_videos, metadata_only=False)
    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_root,
        fps=_metadata(reference).fps,
        features=_features_without_defaults(_metadata(reference).features),
        robot_type=_robot_type(reference),
        use_videos=len(_metadata(reference).video_keys) > 0,
        image_writer_processes=int(output_cfg.get("image_writer_processes", 0)),
        image_writer_threads=int(output_cfg.get("image_writer_threads", 4)),
        batch_encoding_size=int(output_cfg.get("batch_encoding_size", 1)),
    )

    written_frames = 0
    try:
        for spec, dataset, episodes in loaded_datasets:
            for ep_idx in episodes:
                start, end = _episode_bounds(dataset, ep_idx)
                logger.info(
                    "[MERGE] %s episode=%d -> output_episode=%d frames=%d",
                    spec.name,
                    ep_idx,
                    output.meta.total_episodes,
                    end - start,
                )
                for source_idx in range(start, end):
                    item = dataset[int(source_idx)]
                    output.add_frame(_frame_from_source_item(dataset, item))
                    written_frames += 1
                output.save_episode()
    finally:
        output.finalize()

    return written_frames


def merge_lerobot_datasets(cfg: dict[str, Any]) -> dict[str, Any]:
    _require_lerobot()

    source_specs = _normalize_source_specs(cfg)
    output_cfg = cfg.get("output") or {}
    output_name = str(output_cfg.get("dataset_name") or output_cfg.get("repo_id") or "").strip()
    if not output_name and not output_cfg.get("root"):
        raise ValueError("output.dataset_name or output.root is required.")

    output_root = _output_root(cfg)
    output_repo_id = _output_repo_id(output_cfg, output_name or output_root.name, cfg.get("source") or {})
    dry_run = bool(cfg.get("dry_run", False))
    strict_schema = bool((cfg.get("source") or {}).get("strict_schema", True))
    strict_robot_type = bool((cfg.get("source") or {}).get("strict_robot_type", True))
    download_videos = bool((cfg.get("source") or {}).get("download_videos", True))
    fast_skip_reason = _fast_aggregate_skip_reason(
        cfg=cfg,
        output_cfg=output_cfg,
        source_specs=source_specs,
        strict_schema=strict_schema,
        strict_robot_type=strict_robot_type,
    )

    _assert_output_is_separate(output_root, source_specs)
    if output_root.exists() and not dry_run:
        if output_cfg.get("overwrite", False):
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                "Set output.overwrite=true or pass --overwrite to replace it."
            )

    loaded_sources = _load_sources(
        source_specs,
        download_videos=download_videos,
        metadata_only=LeRobotDatasetMetadata is not None,
    )

    reference_spec, reference, _ = loaded_sources[0]
    for spec, dataset, _ in loaded_sources[1:]:
        _validate_compatible_sources(
            reference,
            dataset,
            reference_name=reference_spec.name,
            candidate_name=spec.name,
            strict_schema=strict_schema,
            strict_robot_type=strict_robot_type,
        )

    total_input_frames = 0
    total_output_episodes = 0
    source_summaries: list[dict[str, Any]] = []
    for spec, dataset, episodes in loaded_sources:
        frame_count = 0
        for ep_idx in episodes:
            start, end = _episode_bounds(dataset, ep_idx)
            frame_count += end - start
        total_input_frames += frame_count
        total_output_episodes += len(episodes)
        source_summaries.append(
            {
                "name": spec.name,
                "repo_id": spec.repo_id,
                "root": str(spec.root),
                "episodes": episodes,
                "frames": frame_count,
            }
        )

    summary = {
        "output_repo_id": output_repo_id,
        "output_root": str(output_root),
        "fps": int(_metadata(reference).fps),
        "robot_type": _robot_type(reference),
        "source_count": len(loaded_sources),
        "total_episodes": total_output_episodes,
        "total_frames": total_input_frames,
        "sources": source_summaries,
        "dry_run": dry_run,
        "merge_mode": "fast_aggregate" if fast_skip_reason is None else "frame_rewrite",
    }
    if fast_skip_reason is not None:
        summary["fast_aggregate_skip_reason"] = fast_skip_reason

    if dry_run:
        if fast_skip_reason is not None:
            logger.info("[FAST] Not used: %s", fast_skip_reason)
        logger.info(
            "[DRY-RUN] Would write %d episodes / %d frames to %s using %s",
            total_output_episodes,
            total_input_frames,
            output_root,
            summary["merge_mode"],
        )
        return summary

    if fast_skip_reason is None:
        _run_fast_aggregate(
            source_specs=source_specs,
            output_repo_id=output_repo_id,
            output_root=output_root,
            output_cfg=output_cfg,
        )
        written_frames = total_input_frames
    else:
        logger.info("[FAST] Not used: %s", fast_skip_reason)
        written_frames = _run_frame_rewrite_merge(
            source_specs=source_specs,
            loaded_metadata=loaded_sources,
            output_repo_id=output_repo_id,
            output_root=output_root,
            output_cfg=output_cfg,
            download_videos=download_videos,
        )

    summary["written_frames"] = written_frames
    summary["written_episodes"] = total_output_episodes
    _write_summary(output_root, summary)
    logger.info(
        "[DONE] output=%s root=%s episodes=%d frames=%d",
        output_repo_id,
        output_root,
        summary["written_episodes"],
        written_frames,
    )
    return summary


def main() -> None:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "merge_dataset_cfg.yaml"
    parser = argparse.ArgumentParser(description="Merge multiple local LeRobot datasets into one dataset.")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--dry-run", action="store_true", help="Only validate and report; do not write output.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite=true.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes per source dataset.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.overwrite:
        cfg.setdefault("output", {})["overwrite"] = True
    if args.max_episodes is not None:
        cfg.setdefault("source", {})["max_episodes"] = int(args.max_episodes)

    merge_lerobot_datasets(cfg)


if __name__ == "__main__":
    main()
