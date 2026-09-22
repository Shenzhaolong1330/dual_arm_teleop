#!/usr/bin/env python3
"""Fast, staged LeRobot v3 static-action cleaner.

This is the video-streaming alternative to :mod:`preprocess_dataset`.  It
uses exactly the same keep-mask helpers, but it does *not* call
``LeRobotDataset.__getitem__`` for every retained frame.  Instead it:

1. reads only parquet action/state columns to build the keep mask;
2. rewrites the retained numeric rows and their contiguous indices;
3. filters each source video once with FFmpeg's ``select`` filter; and
4. writes matching v3 episode/video metadata.

Filtering arbitrary AV1 frames still requires one decode/re-encode pass, but
it avoids Python frame-by-frame seeks and temporary PNG files.  Output is
assembled in ``<output-root>.fast-staging`` and atomically renamed only after
the parquet/video frame counts agree.  Raw datasets are read-only.

The config is deliberately compatible with ``preprocess_dataset``.  The YAML
may use either ``preprocess_dataset:`` or ``fast_preprocess_dataset:`` as its
top-level key and may contain the same ``jobs`` list.  Add the optional
``fast_stream`` block, for example::

    fast_stream:
      video_workers: 3
      ffmpeg_threads: 2
      verify_video_frames: true
      stats_mode: recompute_numeric_copy_visual

``stats_mode`` is explicit: ``recompute_numeric_copy_visual`` retains source
image statistics; ``recompute_visual_and_full_decode`` decodes every output
frame, checks PTS, and recomputes image statistics using up to 16 retained
images per episode. Numeric statistics are always recomputed. Set
``single_video_per_camera: true`` to concatenate filtered chunks once, without
re-encoding, into one full daily video per camera. The policy and verification
results are recorded in ``meta/fast_preprocess_manifest.json``.

The implementation supports source video files split across multiple v3
chunk/file paths.  Each output video keeps the corresponding path and gets a
new, contiguous local timeline; metadata maps every retained episode to the
correct output segment.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dataset_integrity import verify_all_videos_and_stats

# Keep the mask semantics in one place.  Running this file directly adds its
# directory to sys.path, so this import works without packaging scripts/tools.
from preprocess_dataset import (  # type: ignore
    validate_gripper_smoothing,
    _action_motion_mask,
    _deep_merge,
    _gripper_event_mask,
    _motion_mask,
    _smooth_actions,
    _trim_static_runs,
)

try:
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.utils import write_stats
    from lerobot.datasets.video_utils import get_video_info
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "The project's LeRobot environment is required. Run with the same "
        "Python environment used by scripts/tools/preprocess_dataset.py."
    ) from exc


LOGGER = logging.getLogger("fast_preprocess_dataset")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass(frozen=True)
class VideoSegment:
    """One physical source/output video file for one camera."""

    key: str
    chunk_index: int
    file_index: int
    source_path: Path

    @property
    def id(self) -> tuple[str, int, int]:
        return self.key, self.chunk_index, self.file_index


@dataclass
class EpisodePlan:
    source_episode_index: int
    output_episode_index: int
    source_start: int
    source_end: int
    keep_indices: np.ndarray
    smoothed_actions: np.ndarray
    task_names: list[str]
    gripper_protected_frames: int = 0
    gripper_restored_frames: int = 0
    video_ranges: dict[tuple[str, int, int], tuple[int, int]] = field(default_factory=dict)
    output_video_ranges: dict[tuple[str, int, int], tuple[int, int]] = field(default_factory=dict)

    @property
    def source_length(self) -> int:
        return self.source_end - self.source_start

    @property
    def output_length(self) -> int:
        return int(self.keep_indices.size)


@dataclass
class SegmentPlan:
    segment: VideoSegment
    source_frame_count: int
    keep: np.ndarray
    output_frame_count: int = 0

    def false_runs(self) -> list[tuple[int, int]]:
        """Inclusive source-frame intervals not present in the output."""
        false = ~self.keep
        if not false.any():
            return []
        padded = np.concatenate(([False], false, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        # Changes alternate false->true and true->false in the padded signal.
        return [(int(changes[i]), int(changes[i + 1] - 1)) for i in range(0, len(changes), 2)]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg = raw.get("fast_preprocess_dataset", raw.get("preprocess_dataset"))
    if not isinstance(cfg, dict):
        raise ValueError(
            "Config requires a top-level 'fast_preprocess_dataset' or 'preprocess_dataset' mapping."
        )
    return cfg


def _expand_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = cfg.get("jobs")
    if jobs is None:
        return [copy.deepcopy(cfg)]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    base = {key: value for key, value in cfg.items() if key != "jobs"}
    expanded: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping")
        merged = _deep_merge(base, job)
        if "source" not in merged or "output" not in merged:
            raise ValueError(f"jobs[{index}] must set source and output")
        merged["_job_name"] = str(job.get("name") or merged["output"].get("repo_id") or index)
        expanded.append(merged)
    return expanded


def _path(value: str | Path | None, label: str) -> Path:
    if not value:
        raise ValueError(f"Missing {label}")
    return Path(value).expanduser().resolve(strict=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _episode_metadata(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted((source_root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {source_root}")
    # Reading rows rather than a datasets.Dataset intentionally avoids creating
    # Hugging Face cache locks beside the source data.
    table = pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _resolve_pattern(root: Path, pattern: str, **kwargs: Any) -> Path:
    return root / pattern.format(**kwargs)


def _segment_from_row(
    source_root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    video_key: str,
) -> VideoSegment:
    chunk = int(row[f"videos/{video_key}/chunk_index"])
    file = int(row[f"videos/{video_key}/file_index"])
    path = _resolve_pattern(
        source_root,
        str(info["video_path"]),
        video_key=video_key,
        chunk_index=chunk,
        file_index=file,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Video metadata points to a missing file: {path}")
    return VideoSegment(video_key, chunk, file, path)


def _ffprobe_frame_count(path: Path) -> int:
    """Read the encoded packet/frame count without decoding pixel data."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    stream = streams[0]
    for key in ("nb_read_packets", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A"):
            count = int(value)
            if count >= 0:
                return count
    raise RuntimeError(f"ffprobe could not determine a frame count for {path}")


def _as_matrix(column: pa.ChunkedArray | pa.Array, dtype: np.dtype) -> np.ndarray:
    values = column.to_pylist()
    out = np.asarray(values, dtype=dtype)
    if out.ndim == 1:
        out = out[:, None]
    return out


class _DataFileCache:
    def __init__(self, source_root: Path, info: dict[str, Any]) -> None:
        self.source_root = source_root
        self.pattern = str(info["data_path"])
        self._tables: dict[tuple[int, int], tuple[pa.Table, int]] = {}
        self._range_tables: list[tuple[Path, pa.Table, int, int]] | None = None

    def _discover_range_tables(self) -> list[tuple[Path, pa.Table, int, int]]:
        """Find physical parquet ranges when aggregate metadata is stale.

        The project's older fast aggregate outputs occasionally retain a source
        ``data/file_index`` in episode metadata after physically concatenating
        all rows into file-000.  LeRobot's global loader masks that mismatch,
        but a direct parquet reader must resolve it explicitly.  This fallback
        is read-only and still refuses any non-contiguous/ambiguous range.
        """
        if self._range_tables is None:
            discovered: list[tuple[Path, pa.Table, int, int]] = []
            for path in sorted((self.source_root / "data").rglob("*.parquet")):
                table = pq.read_table(path)
                if table.num_rows == 0 or "index" not in table.column_names:
                    continue
                first = int(table["index"][0].as_py())
                last_exclusive = int(table["index"][table.num_rows - 1].as_py()) + 1
                discovered.append((path, table, first, last_exclusive))
            self._range_tables = discovered
        return self._range_tables

    def episode_table(self, row: dict[str, Any]) -> pa.Table:
        chunk = int(row["data/chunk_index"])
        file = int(row["data/file_index"])
        key = (chunk, file)
        if key not in self._tables:
            path = _resolve_pattern(
                self.source_root,
                self.pattern,
                chunk_index=chunk,
                file_index=file,
            )
            if path.is_file():
                table = pq.read_table(path)
                if table.num_rows == 0:
                    raise ValueError(f"Data parquet is empty: {path}")
                if "index" not in table.column_names:
                    raise ValueError(f"Data parquet has no index column: {path}")
                first_index = int(table["index"][0].as_py())
                self._tables[key] = (table, first_index)
            else:
                required_start = int(row["dataset_from_index"])
                required_end = int(row["dataset_to_index"])
                matches = [
                    (candidate, first)
                    for _, candidate, first, last in self._discover_range_tables()
                    if required_start >= first and required_end <= last
                ]
                if len(matches) != 1:
                    raise FileNotFoundError(
                        "Episode metadata points to a missing parquet and no unique physical fallback contains "
                        f"its global range [{required_start}, {required_end}): {path}"
                    )
                table, first_index = matches[0]
                LOGGER.warning(
                    "[metadata-repair] episode %s references missing %s; using physical parquet with index range instead",
                    row["episode_index"],
                    path.name,
                )
                self._tables[key] = (table, first_index)
        table, first_index = self._tables[key]
        start = int(row["dataset_from_index"]) - first_index
        length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        if start < 0 or start + length > table.num_rows:
            raise ValueError(
                "Episode data range does not fit its referenced parquet file: "
                f"episode={row['episode_index']} start={start} length={length} rows={table.num_rows}"
            )
        result = table.slice(start, length)
        index_values = result["index"].to_numpy(zero_copy_only=False)
        expected = np.arange(int(row["dataset_from_index"]), int(row["dataset_to_index"]), dtype=index_values.dtype)
        if not np.array_equal(index_values, expected):
            raise ValueError(f"Non-contiguous source index values in episode {row['episode_index']}")
        return result


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        raise ValueError(f"Missing required data column '{name}'")
    field = table.schema.field(name)
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type) or pa.types.is_fixed_size_list(field.type):
        array = pa.array(values.tolist(), type=field.type)
    else:
        array = pa.array(values, type=field.type)
    return table.set_column(table.schema.get_field_index(name), name, array)


def _output_episode_table(
    source_episode: pa.Table,
    plan: EpisodePlan,
    output_global_start: int,
    fps: float,
) -> pa.Table:
    selected = source_episode.take(pa.array(plan.keep_indices, type=pa.int64()))
    length = plan.output_length
    selected = _replace_column(selected, "action", plan.smoothed_actions[plan.keep_indices])
    selected = _replace_column(selected, "timestamp", np.arange(length, dtype=np.float32) / np.float32(fps))
    selected = _replace_column(selected, "frame_index", np.arange(length, dtype=np.int64))
    selected = _replace_column(
        selected,
        "episode_index",
        np.full(length, plan.output_episode_index, dtype=np.int64),
    )
    selected = _replace_column(selected, "index", np.arange(output_global_start, output_global_start + length, dtype=np.int64))
    return selected


def _numeric_episode_stats(table: pa.Table, info: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, feature in info["features"].items():
        if key not in table.column_names or feature.get("dtype") in {"video", "image", "string"}:
            continue
        values = _as_matrix(table[key], np.dtype(feature.get("dtype", "float32")))
        # All numeric v3 fields are frame-wise.  The existing LeRobot helper
        # yields exactly the representation expected by stats.json.
        stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
    return stats


def _difficulty_label(repo_id: str, source_root: Path) -> str:
    text = f"{repo_id} {source_root.name}".lower()
    return "hard" if re.search(r"(^|[_-])hard([_-]|$)", text) else "nonhard"


def _build_filter_script(path: Path, false_runs: Iterable[tuple[int, int]], fps: float) -> None:
    escaped_runs = [f"between(n\\,{start}\\,{end})" for start, end in false_runs]
    if escaped_runs:
        select = "select='not(" + "+".join(escaped_runs) + ")'"
        text = f"{select},setpts=N/({fps:g}*TB)\n"
    else:
        text = f"setpts=N/({fps:g}*TB)\n"
    path.write_text(text, encoding="utf-8")


def _run_ffmpeg_segment(
    segment_plan: SegmentPlan,
    stage_root: Path,
    video_pattern: str,
    fps: float,
    encoding: dict[str, Any],
    work_root: Path,
    ffmpeg_threads: int,
) -> tuple[VideoSegment, Path]:
    segment = segment_plan.segment
    output_path = _resolve_pattern(
        stage_root,
        video_pattern,
        video_key=segment.key,
        chunk_index=segment.chunk_index,
        file_index=segment.file_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_path = work_root / f"filter_{segment.key.replace('.', '_')}_{segment.chunk_index:03d}_{segment.file_index:03d}.txt"
    _build_filter_script(filter_path, segment_plan.false_runs(), fps)

    requested_codec = str(encoding.get("codec", "h264")).lower()
    codec_map = {"h264": "libx264", "libx264": "libx264", "hevc": "libx265", "libx265": "libx265"}
    codec = codec_map.get(requested_codec, requested_codec)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        str(encoding.get("loglevel", "error")),
        "-y",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
        "-i",
        str(segment.source_path),
        "-map",
        "0:v:0",
        "-filter_script:v",
        str(filter_path),
        # This host's ffmpeg predates -fps_mode.  setpts makes selected output
        # CFR anyway; -vsync 0 prevents a second duplicate/drop policy.
        "-vsync",
        "0",
        "-an",
        "-c:v",
        codec,
        "-pix_fmt",
        str(encoding.get("pixel_format", "yuv420p")),
        "-g",
        str(int(encoding.get("gop", round(fps)))),
        "-threads",
        str(max(1, int(ffmpeg_threads))),
    ]
    if "preset" in encoding:
        command.extend(["-preset", str(encoding["preset"])])
    if "crf" in encoding:
        command.extend(["-crf", str(encoding["crf"])])
    if "cq" in encoding:
        command.extend(["-cq", str(encoding["cq"])])
    if bool(encoding.get("faststart", True)):
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_path))

    LOGGER.info(
        "[video] %s chunk=%03d file=%03d frames %d -> %d",
        segment.key,
        segment.chunk_index,
        segment.file_index,
        segment_plan.source_frame_count,
        segment_plan.output_frame_count,
    )
    subprocess.run(command, check=True)
    return segment, output_path


def _source_video_info_after_output(info: dict[str, Any], output_paths: dict[tuple[str, int, int], Path]) -> None:
    for video_key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        candidates = [path for (key, _, _), path in output_paths.items() if key == video_key]
        if not candidates:
            raise RuntimeError(f"No output video generated for {video_key}")
        feature["info"] = get_video_info(sorted(candidates)[0])


def _copy_tasks(source_root: Path, stage_root: Path) -> None:
    source = source_root / "meta" / "tasks.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Missing task metadata: {source}")
    target = stage_root / "meta" / "tasks.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_episode_metadata(rows: list[dict[str, Any]], stage_root: Path) -> None:
    if not rows:
        raise ValueError("Cannot write zero episodes")
    path = stage_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def _verify_stage(
    stage_root: Path,
    info: dict[str, Any],
    segment_plans: dict[tuple[str, int, int], SegmentPlan],
    expected_frames: int,
    verify_video_frames: bool,
) -> dict[tuple[str, int, int], Path]:
    data_paths = sorted((stage_root / "data").rglob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in data_paths)
    if rows != expected_frames:
        raise RuntimeError(f"Output parquet rows {rows} != expected kept frames {expected_frames}")

    output_paths: dict[tuple[str, int, int], Path] = {}
    for identifier, plan in segment_plans.items():
        segment = plan.segment
        path = _resolve_pattern(
            stage_root,
            str(info["video_path"]),
            video_key=segment.key,
            chunk_index=segment.chunk_index,
            file_index=segment.file_index,
        )
        if not path.is_file():
            raise RuntimeError(f"Missing staged video {path}")
        if verify_video_frames:
            actual = _ffprobe_frame_count(path)
            if actual != plan.output_frame_count:
                raise RuntimeError(
                    f"Video frame mismatch for {path}: actual {actual}, expected {plan.output_frame_count}"
                )
        output_paths[identifier] = path
    return output_paths


def _validate_source_video_coverage(segment_plans: dict[tuple[str, int, int], SegmentPlan]) -> None:
    for plan in segment_plans.values():
        # False here would silently retain an unreferenced frame or drop a
        # referenced one.  The explicit coverage check protects alignment.
        if plan.keep.shape != (plan.source_frame_count,):
            raise AssertionError("segment keep vector has the wrong length")
        if plan.output_frame_count != int(plan.keep.sum()):
            raise AssertionError("segment kept frame count is inconsistent")
        if plan.output_frame_count == 0:
            raise ValueError(f"All video frames would be removed from {plan.segment.source_path}")


def _plan_job(cfg: dict[str, Any], max_episodes: int | None) -> tuple[
    Path,
    Path,
    dict[str, Any],
    list[dict[str, Any]],
    list[EpisodePlan],
    dict[tuple[str, int, int], SegmentPlan],
]:
    source_cfg = cfg["source"]
    output_cfg = cfg["output"]
    source_root = _path(source_cfg.get("root"), "source.root")
    output_root = _path(output_cfg.get("root"), "output.root")
    if re.search(r'wrong[ _-]*cam', source_root.name, re.IGNORECASE):
        raise ValueError(f'Refusing wrong_cam input: {source_root}')
    if not (source_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {source_root}")
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Output root must be separate from the source root")

    info = _read_json(source_root / "meta" / "info.json")
    validate_gripper_smoothing(info, cfg)
    fps = float(info["fps"])
    rows = _episode_metadata(source_root)
    requested_episodes = source_cfg.get("episodes")
    if requested_episodes is not None:
        allowed = {int(value) for value in requested_episodes}
        rows = [row for row in rows if int(row["episode_index"]) in allowed]
    source_max = source_cfg.get("max_episodes")
    if source_max is not None:
        rows = rows[: int(source_max)]
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("No source episodes selected")

    action_names = [str(value) for value in info["features"]["action"].get("names") or []]
    state_feature = info["features"].get("observation.state", {})
    state_names = [str(value) for value in state_feature.get("names") or []]
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("fast_preprocess_dataset requires at least one video feature")

    data_cache = _DataFileCache(source_root, info)
    segment_plans: dict[tuple[str, int, int], SegmentPlan] = {}
    episode_plans: list[EpisodePlan] = []

    # We need every source video length before allocating its exact keep vector.
    segments: dict[tuple[str, int, int], VideoSegment] = {}
    for row in rows:
        for video_key in video_keys:
            segment = _segment_from_row(source_root, info, row, video_key)
            segments[segment.id] = segment
    for identifier, segment in segments.items():
        count = _ffprobe_frame_count(segment.source_path)
        segment_plans[identifier] = SegmentPlan(segment=segment, source_frame_count=count, keep=np.zeros(count, dtype=bool))

    for output_episode_index, row in enumerate(rows):
        source_table = data_cache.episode_table(row)
        timestamps = source_table['timestamp'].to_numpy()
        if not np.allclose(timestamps, np.arange(len(timestamps)) / fps, atol=1e-4, rtol=0):
            raise ValueError(f'Nonuniform source frame timestamps in episode {row["episode_index"]}')
        actions = _as_matrix(source_table["action"], np.float32)
        states = (
            _as_matrix(source_table["observation.state"], np.float32)
            if "observation.state" in source_table.column_names
            else None
        )
        smoothed = _smooth_actions(actions, action_names, cfg)
        gripper_keep = _gripper_event_mask(actions, action_names, cfg, states, state_names)
        motion, motion_source = _motion_mask(smoothed, action_names, states, state_names, fps, cfg)
        keep_mask = _trim_static_runs(motion | gripper_keep, cfg)
        cartesian_only_keep = _trim_static_runs(motion, cfg)
        assert not (gripper_keep & ~keep_mask).any(), "A protected gripper frame would be deleted"
        assert not (cartesian_only_keep & ~keep_mask).any(), "Gripper protection must only add retained frames"
        keep_indices = np.flatnonzero(keep_mask).astype(np.int64)
        if keep_indices.size == 0:
            raise ValueError(f"Episode {row['episode_index']} would have zero retained frames")

        plan = EpisodePlan(
            source_episode_index=int(row["episode_index"]),
            output_episode_index=output_episode_index,
            source_start=int(row["dataset_from_index"]),
            source_end=int(row["dataset_to_index"]),
            keep_indices=keep_indices,
            smoothed_actions=smoothed,
            task_names=[str(value) for value in row.get("tasks", [])],
            gripper_protected_frames=int(gripper_keep.sum()),
            gripper_restored_frames=int((keep_mask & ~cartesian_only_keep).sum()),
        )
        for video_key in video_keys:
            segment = _segment_from_row(source_root, info, row, video_key)
            identifier = segment.id
            source_start_frame = int(round(float(row[f"videos/{video_key}/from_timestamp"]) * fps))
            source_end_frame = source_start_frame + plan.source_length
            segment_plan = segment_plans[identifier]
            if source_start_frame < 0 or source_end_frame > segment_plan.source_frame_count:
                raise ValueError(
                    "Episode/video metadata exceeds the physical video length: "
                    f"episode={row['episode_index']} video={segment.source_path} "
                    f"frames=[{source_start_frame}, {source_end_frame}) total={segment_plan.source_frame_count}"
                )
            # Check timestamp metadata agrees with its parquet frame range.  A
            # full sequential filter has no seek ambiguity, so this is the key
            # mapping that guarantees video and parquet row N share a source N.
            nominal_end = float(row[f"videos/{video_key}/to_timestamp"])
            if abs((source_end_frame / fps) - nominal_end) > (1.5 / fps):
                raise ValueError(
                    f"Inconsistent video timestamps in episode {row['episode_index']} for {video_key}"
                )
            if segment_plan.keep[source_start_frame:source_end_frame].any():
                raise ValueError(
                    f"Overlapping source video episode ranges in {segment.source_path}; refusing unsafe rewrite"
                )
            segment_plan.keep[source_start_frame + keep_indices] = True
            plan.video_ranges[identifier] = (source_start_frame, source_end_frame)

        episode_plans.append(plan)
        LOGGER.info(
            "[plan] ep %d -> %d: %d -> %d (%.1f%% kept), motion=%s",
            plan.source_episode_index,
            plan.output_episode_index,
            plan.source_length,
            plan.output_length,
            100.0 * plan.output_length / max(1, plan.source_length),
            motion_source,
        )

    # Every source video used by selected episodes must be completely covered.
    # This makes a full-day cleanup safe: any unexpected gaps become an error,
    # rather than silently becoming extra frames with no parquet row.
    for identifier, plan in segment_plans.items():
        referenced = np.zeros(plan.source_frame_count, dtype=bool)
        for episode in episode_plans:
            if identifier in episode.video_ranges:
                start, end = episode.video_ranges[identifier]
                referenced[start:end] = True
        if not referenced.all():
            missing = int((~referenced).sum())
            raise ValueError(
                f"Selected episodes do not cover all {plan.source_frame_count} frames of {plan.segment.source_path} "
                f"({missing} unreferenced). Use the full daily dataset; subset materialization is intentionally refused."
            )
        plan.output_frame_count = int(plan.keep.sum())
    _validate_source_video_coverage(segment_plans)

    # Output per-file video timestamps after select+setpts are contiguous from
    # zero.  Each output episode keeps a separate mapping for each camera file.
    segment_cursor: dict[tuple[str, int, int], int] = defaultdict(int)
    for episode in episode_plans:
        for identifier in episode.video_ranges:
            start = segment_cursor[identifier]
            end = start + episode.output_length
            episode.output_video_ranges[identifier] = (start, end)
            segment_cursor[identifier] = end
    for identifier, plan in segment_plans.items():
        if segment_cursor[identifier] != plan.output_frame_count:
            raise AssertionError(f"Output video timeline mismatch for {plan.segment.source_path}")

    return source_root, output_root, info, rows, episode_plans, segment_plans


def _materialize_job(cfg: dict[str, Any], max_episodes: int | None, dry_run: bool) -> dict[str, Any]:
    source_root, output_root, source_info, rows, plans, segment_plans = _plan_job(cfg, max_episodes)
    output_cfg = cfg["output"]
    fast_cfg = cfg.get("fast_stream", {}) or {}
    input_frames = sum(plan.source_length for plan in plans)
    output_frames = sum(plan.output_length for plan in plans)
    fps = float(source_info["fps"])
    repo_id = str(output_cfg.get("repo_id") or output_root.name)
    manifest: dict[str, Any] = {
        "format": "lerobot-fast-preprocess-v1",
        "source_root": str(source_root),
        "source_repo_id": str(cfg["source"].get("repo_id", source_root.name)),
        "output_root": str(output_root),
        "output_repo_id": repo_id,
        "difficulty": _difficulty_label(str(cfg["source"].get("repo_id", source_root.name)), source_root),
        "fps": fps,
        "episodes": len(plans),
        "input_frames": input_frames,
        "output_frames": output_frames,
        "input_duration_s": input_frames / fps,
        "output_duration_s": output_frames / fps,
        "removed_frames": input_frames - output_frames,
        "removed_duration_s": (input_frames - output_frames) / fps,
        "gripper_protection": {
            "protected_frames": sum(plan.gripper_protected_frames for plan in plans),
            "restored_vs_cartesian_only_frames": sum(plan.gripper_restored_frames for plan in plans),
            "all_protected_frames_retained": True,
        },
        "stats_mode": str(fast_cfg.get("stats_mode", "recompute_numeric_copy_visual")),
        "cleaning_rules": {key: copy.deepcopy(cfg.get(key, {})) for key in ('static_trim', 'gripper_events', 'action_smoothing')},
        "episode_frame_counts": [
            {
                "source_episode_index": plan.source_episode_index,
                "output_episode_index": plan.output_episode_index,
                "input_frames": plan.source_length,
                "output_frames": plan.output_length,
                "gripper_protected_frames": plan.gripper_protected_frames,
                "gripper_restored_frames": plan.gripper_restored_frames,
            }
            for plan in plans
        ],
        "videos": [
            {
                "key": plan.segment.key,
                "chunk_index": plan.segment.chunk_index,
                "file_index": plan.segment.file_index,
                "input_frames": plan.source_frame_count,
                "output_frames": plan.output_frame_count,
                "removed_intervals": len(plan.false_runs()),
            }
            for plan in sorted(segment_plans.values(), key=lambda value: value.segment.id)
        ],
    }
    LOGGER.info(
        "[summary] episodes=%d frames=%d -> %d duration=%.3fs -> %.3fs difficulty=%s",
        len(plans),
        input_frames,
        output_frames,
        input_frames / fps,
        output_frames / fps,
        manifest["difficulty"],
    )
    if dry_run:
        return manifest

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to modify existing output {output_root}. Choose a new output.root/repo_id."
        )
    stage_root = output_root.with_name(output_root.name + ".fast-staging")
    if stage_root.exists():
        raise FileExistsError(
            f"Staging directory already exists: {stage_root}. Inspect or remove it deliberately before rerunning."
        )
    stage_root.mkdir(parents=True)
    work_root = stage_root / ".fast_preprocess_work"
    work_root.mkdir()

    try:
        # Filter each independent video segment in parallel.  The default is
        # conservative enough for CPU hosts; users can tune it in YAML.
        workers = max(1, int(fast_cfg.get("video_workers", 3)))
        ffmpeg_threads = max(1, int(fast_cfg.get("ffmpeg_threads", 2)))
        encoding = copy.deepcopy(output_cfg.get("video_encoding") or {})
        encoding.setdefault("codec", "h264")
        encoding.setdefault("preset", "veryfast")
        encoding.setdefault("crf", 23)
        encoding.setdefault("gop", round(fps))

        output_paths: dict[tuple[str, int, int], Path] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_ffmpeg_segment,
                    segment_plan,
                    stage_root,
                    str(source_info["video_path"]),
                    fps,
                    encoding,
                    work_root,
                    ffmpeg_threads,
                ): identifier
                for identifier, segment_plan in segment_plans.items()
            }
            for future in as_completed(futures):
                identifier = futures[future]
                _, path = future.result()
                output_paths[identifier] = path

        # Numeric parquet rows and episode metadata are constructed after video
        # success.  A failure before this point leaves only a clearly named
        # staging directory and never changes the desired output location.
        data_cache = _DataFileCache(source_root, source_info)
        data_path = stage_root / "data" / "chunk-000" / "file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        writer: pq.ParquetWriter | None = None
        global_start = 0
        per_episode_numeric_stats: list[dict[str, dict[str, np.ndarray]]] = []
        episode_rows: list[dict[str, Any]] = []
        try:
            for row, plan in zip(rows, plans, strict=True):
                source_table = data_cache.episode_table(row)
                output_table = _output_episode_table(source_table, plan, global_start, fps)
                if writer is None:
                    writer = pq.ParquetWriter(data_path, output_table.schema, compression="snappy")
                writer.write_table(output_table)
                numeric_stats = _numeric_episode_stats(output_table, source_info)
                per_episode_numeric_stats.append(numeric_stats)

                ep_row: dict[str, Any] = {
                    "episode_index": plan.output_episode_index,
                    "tasks": plan.task_names,
                    "length": plan.output_length,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                    "dataset_from_index": global_start,
                    "dataset_to_index": global_start + plan.output_length,
                }
                for identifier, (start_frame, end_frame) in plan.output_video_ranges.items():
                    segment = segment_plans[identifier].segment
                    ep_row[f"videos/{segment.key}/chunk_index"] = segment.chunk_index
                    ep_row[f"videos/{segment.key}/file_index"] = segment.file_index
                    ep_row[f"videos/{segment.key}/from_timestamp"] = start_frame / fps
                    ep_row[f"videos/{segment.key}/to_timestamp"] = end_frame / fps
                episode_rows.append(ep_row)
                global_start += plan.output_length
        finally:
            if writer is not None:
                writer.close()

        if global_start != output_frames:
            raise AssertionError(f"Wrote {global_start} numeric rows, expected {output_frames}")
        _copy_tasks(source_root, stage_root)

        if fast_cfg.get('single_video_per_camera', False):
            # Concat once per camera, after all filtering, without another
            # lossy encoding pass. Each daily camera has one playable file.
            from lerobot.datasets.video_utils import concatenate_video_files
            combined_plans = {}
            output_paths = {}
            for key in [k for k, f in source_info['features'].items() if f['dtype'] == 'video']:
                paths = sorted({stage_root / source_info['video_path'].format(
                    video_key=key, chunk_index=r[f'videos/{key}/chunk_index'],
                    file_index=r[f'videos/{key}/file_index']) for r in episode_rows})
                target = stage_root / source_info['video_path'].format(video_key=key, chunk_index=0, file_index=0)
                if len(paths) > 1:
                    joined = work_root / (key + '.mp4')
                    concatenate_video_files(paths, joined)
                    for path in paths:
                        path.unlink()
                    joined.replace(target)
                cursor = 0
                for row in episode_rows:
                    row[f'videos/{key}/chunk_index'] = 0
                    row[f'videos/{key}/file_index'] = 0
                    row[f'videos/{key}/from_timestamp'] = cursor / fps
                    cursor += row['length']
                    row[f'videos/{key}/to_timestamp'] = cursor / fps
                seg = VideoSegment(key, 0, 0, target)
                combined_plans[seg.id] = SegmentPlan(seg, cursor, np.ones(cursor, dtype=bool), cursor)
                output_paths[seg.id] = target
            segment_plans = combined_plans

        output_info = copy.deepcopy(source_info)
        output_info["total_episodes"] = len(plans)
        output_info["total_frames"] = output_frames
        output_info["total_tasks"] = int(source_info.get("total_tasks", 0))
        output_info["splits"] = {"train": f"0:{len(plans)}"}
        _source_video_info_after_output(output_info, output_paths)
        _write_json(output_info, stage_root / "meta" / "info.json")

        stats_mode = str(fast_cfg.get("stats_mode", "recompute_numeric_copy_visual"))
        if stats_mode not in ("recompute_numeric_copy_visual", "recompute_visual_and_full_decode"):
            raise ValueError(
                "stats_mode must be recompute_numeric_copy_visual or recompute_visual_and_full_decode"
            )
        if stats_mode == 'recompute_visual_and_full_decode':
            LOGGER.info('[verify] Full video decode, PTS checks, and retained-image statistics')
            decoded_reports, visual_stats = verify_all_videos_and_stats(stage_root, output_info, episode_rows)
            for decoded in decoded_reports:
                decoded['path'] = str(output_root / Path(decoded['path']).relative_to(stage_root))
            manifest['full_decode_verification'] = decoded_reports
            manifest['visual_stats_sampling'] = '16 evenly spaced retained frames per episode, spatial stride 2'
            for row, numeric in zip(episode_rows, per_episode_numeric_stats, strict=True):
                numeric.update(visual_stats[row['episode_index']])
        for row, stats in zip(episode_rows, per_episode_numeric_stats, strict=True):
            for feature, values in stats.items():
                for stat, value in values.items():
                    row[f'stats/{feature}/{stat}'] = np.asarray(value).tolist()
        _write_episode_metadata(episode_rows, stage_root)
        source_stats_path = source_root / "meta" / "stats.json"
        output_stats = _read_json(source_stats_path) if source_stats_path.is_file() else {}
        numeric_stats = aggregate_stats(per_episode_numeric_stats)
        output_stats.update({key: value for key, value in numeric_stats.items()})
        write_stats(output_stats, stage_root)

        manifest["verification"] = {
            "parquet_rows": output_frames,
            "video_frame_counts": bool(fast_cfg.get("verify_video_frames", True)),
            "source_metadata_episodes": len(rows),
            "output_metadata_episodes": len(episode_rows),
        }
        verified_paths = _verify_stage(
            stage_root,
            output_info,
            segment_plans,
            output_frames,
            bool(fast_cfg.get("verify_video_frames", True)),
        )
        manifest["verification"]["videos"] = {
            f"{key}/{chunk:03d}/{file:03d}": str(output_root / path.relative_to(stage_root))
            for (key, chunk, file), path in sorted(verified_paths.items())
        }
        _write_json(manifest, stage_root / "meta" / "fast_preprocess_manifest.json")

        # The filter scripts are implementation debris, not part of a dataset.
        shutil.rmtree(work_root)
        os.replace(stage_root, output_root)
        LOGGER.info("[DONE] %s", output_root)
        return manifest
    except Exception:
        LOGGER.exception("[FAILED] Staging output kept for inspection: %s", stage_root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast FFmpeg-streaming LeRobot static-action cleaner")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML using preprocess_dataset or fast_preprocess_dataset schema",
    )
    parser.add_argument("--job", action="append", help="Run only this named job; may be repeated")
    parser.add_argument("--dry-run", action="store_true", help="Compute exact masks/counts without writing videos")
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Planning-only subset cap. Materializing a subset is refused to protect video coverage.",
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    jobs = _expand_jobs(cfg)
    if args.job:
        wanted = set(args.job)
        jobs = [job for job in jobs if job.get("_job_name") in wanted]
        missing = wanted - {job.get("_job_name") for job in jobs}
        if missing:
            raise ValueError(f"Unknown job(s): {sorted(missing)}")

    for number, job in enumerate(jobs, start=1):
        LOGGER.info("[JOB %d/%d] %s", number, len(jobs), job.get("_job_name", "unnamed"))
        manifest = _materialize_job(job, args.max_episodes, args.dry_run)
        if args.dry_run:
            LOGGER.info("[DRY-RUN] %s", json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
