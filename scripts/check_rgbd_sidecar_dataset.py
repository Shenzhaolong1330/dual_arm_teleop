#!/usr/bin/env python
"""Check LeRobot RGB-D/IR sidecars for schema, timing, and frozen frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME


CAMERAS = ("head", "left_wrist", "right_wrist")
MODALITIES = ("depth", "left_ir", "right_ir")
SYNC_KEYS = (
    "global_frame_index",
    "robot_timestamp",
    "head_rgbd_timestamp",
    "left_wrist_rgbd_timestamp",
    "right_wrist_rgbd_timestamp",
)
DEFAULT_MAX_CONSECUTIVE_IDENTICAL = 4


@dataclass
class ContentStats:
    episode: int
    camera: str
    modality: str
    total_frames: int = 0
    adjacent_duplicates: int = 0
    reused_count: int = 0
    unmarked_duplicates: int = 0
    first_unmarked_duplicate: int | None = None
    longest_run_start: int = 0
    longest_run_end: int = 0
    _current_run_start: int = 0
    _current_run_length: int = 0
    _previous_digest: bytes | None = None
    _previous_frame: np.ndarray | None = None
    _unique_digests: set[bytes] = field(default_factory=set)

    @property
    def unique_frames(self) -> int:
        return len(self._unique_digests)

    @property
    def longest_run_length(self) -> int:
        if self.total_frames == 0:
            return 0
        return self.longest_run_end - self.longest_run_start + 1

    def add(self, frame: np.ndarray, reused: bool, local_index: int) -> None:
        contiguous = np.ascontiguousarray(frame)
        digest = hashlib.sha256(memoryview(contiguous).cast("B")).digest()
        identical = (
            self._previous_digest == digest
            and self._previous_frame is not None
            and np.array_equal(contiguous, self._previous_frame)
        )

        self.total_frames += 1
        self.reused_count += int(reused)
        self._unique_digests.add(digest)
        if identical:
            self.adjacent_duplicates += 1
            self._current_run_length += 1
            if not reused:
                self.unmarked_duplicates += 1
                if self.first_unmarked_duplicate is None:
                    self.first_unmarked_duplicate = local_index
        else:
            self._current_run_start = local_index
            self._current_run_length = 1

        if self._current_run_length > self.longest_run_length:
            self.longest_run_start = self._current_run_start
            self.longest_run_end = local_index

        self._previous_digest = digest
        self._previous_frame = contiguous.copy()


@dataclass
class CheckReport:
    dataset_root: Path
    episode_frame_counts: dict[int, int]
    content_stats: dict[tuple[int, str, str], ContentStats]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


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


def _selected_modalities(save_depth: bool, save_ir: bool) -> tuple[str, ...]:
    modalities: list[str] = []
    if save_depth:
        modalities.append("depth")
    if save_ir:
        modalities.extend(("left_ir", "right_ir"))
    return tuple(modalities)


def _require_features(features: dict[str, dict], keys: list[str] | tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in features]
    if missing:
        raise AssertionError(f"Missing required feature(s): {missing}")


def _resolve_latest_dataset(root: Path) -> Path:
    if (root / "meta" / "info.json").is_file():
        return root
    candidates = sorted(
        root.rglob("meta/info.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found under {root}")
    return candidates[0].parents[1]


def resolve_dataset_root(root: Path | None, repo_id: str | None) -> Path:
    if root is not None and (root / "meta" / "info.json").is_file():
        return root

    search_root = root if root is not None else Path(HF_LEROBOT_HOME)
    if repo_id is not None:
        candidate = search_root / repo_id
        if (candidate / "meta" / "info.json").is_file():
            return candidate
        raise FileNotFoundError(
            f"LeRobot dataset repo_id={repo_id!r} not found under {search_root}"
        )
    return _resolve_latest_dataset(search_root)


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def _data_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}")
    return files


def _unwrap_extension(array: pa.Array) -> pa.Array:
    return array.storage if isinstance(array, pa.ExtensionArray) else array


def _column_to_frames(
    column: pa.Array, expected_shape: tuple[int, ...], expected_dtype: np.dtype, key: str
) -> np.ndarray:
    current = _unwrap_extension(column)
    for dimension in expected_shape:
        current = _unwrap_extension(current)
        if pa.types.is_list(current.type) or pa.types.is_large_list(current.type):
            offsets = current.offsets.to_numpy(zero_copy_only=False)
            lengths = np.diff(offsets)
            if lengths.size and not np.all(lengths == dimension):
                raise AssertionError(
                    f"{key} has inconsistent nested shape; expected dimension {dimension}, "
                    f"observed lengths={np.unique(lengths).tolist()}"
                )
            start = int(offsets[0])
            stop = int(offsets[-1])
            current = current.values.slice(start, stop - start)
        elif pa.types.is_fixed_size_list(current.type):
            if current.type.list_size != dimension:
                raise AssertionError(
                    f"{key} has dimension {current.type.list_size}, expected {dimension}"
                )
            start = int(current.offset) * dimension
            current = current.values.slice(start, len(current) * dimension)
        else:
            raise AssertionError(
                f"{key} has type {current.type} before expected shape {expected_shape} was exhausted"
            )

    current = _unwrap_extension(current)
    values = current.to_numpy(zero_copy_only=False)
    expected_size = len(column) * int(np.prod(expected_shape, dtype=np.int64))
    if values.size != expected_size:
        raise AssertionError(
            f"{key} contains {values.size} values, expected {expected_size} for "
            f"{len(column)} frame(s) of shape {expected_shape}"
        )
    frames = np.asarray(values).reshape((len(column), *expected_shape))
    if frames.dtype != expected_dtype:
        raise AssertionError(f"{key} has dtype {frames.dtype}, expected {expected_dtype}")
    return frames


def _column_numpy(batch: pa.RecordBatch, key: str, dtype: np.dtype) -> np.ndarray:
    index = batch.schema.get_field_index(key)
    if index < 0:
        raise AssertionError(f"Parquet batch is missing required column: {key}")
    column = _unwrap_extension(batch.column(index))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _check_monotonic(
    previous_sync: dict[tuple[int, str], float],
    errors: list[str],
    episode: int,
    key: str,
    value: float,
    local_index: int,
) -> None:
    state_key = (episode, key)
    previous = previous_sync.get(state_key)
    if previous is not None:
        valid = value > previous if key == "global_frame_index" else value >= previous
        if not valid:
            relation = "strictly increasing" if key == "global_frame_index" else "nondecreasing"
            errors.append(
                f"episode={episode} {key} is not {relation} at local_frame={local_index}: "
                f"previous={previous} current={value}"
            )
    previous_sync[state_key] = value


def inspect_dataset(
    dataset_root: Path,
    *,
    rgb_camera_name_mode: str = "rgb",
    save_depth: bool = True,
    save_ir: bool = True,
    max_consecutive_identical: int = DEFAULT_MAX_CONSECUTIVE_IDENTICAL,
    batch_size: int = 16,
) -> CheckReport:
    if max_consecutive_identical < 1:
        raise ValueError("max_consecutive_identical must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    info = _load_info(dataset_root)
    features = info.get("features", {})
    modalities = _selected_modalities(save_depth, save_ir)
    sidecar_keys = _sidecar_keys(save_depth=save_depth, save_ir=save_ir)
    reused_keys = [f"{camera}_rgbd_reused" for camera in CAMERAS]
    required_keys = [
        *_rgb_keys(rgb_camera_name_mode),
        *sidecar_keys,
        *SYNC_KEYS,
        *reused_keys,
    ]
    _require_features(features, required_keys)

    expected_arrays = {
        key: (
            tuple(int(value) for value in features[key]["shape"]),
            np.dtype(features[key]["dtype"]),
        )
        for key in sidecar_keys
    }
    parquet_columns = ["episode_index", *SYNC_KEYS, *reused_keys, *sidecar_keys]
    content_stats: dict[tuple[int, str, str], ContentStats] = {}
    episode_frame_counts: dict[int, int] = {}
    previous_sync: dict[tuple[int, str], float] = {}
    errors: list[str] = []

    for file_path in _data_files(dataset_root):
        parquet_file = pq.ParquetFile(file_path)
        missing_columns = [key for key in parquet_columns if key not in parquet_file.schema_arrow.names]
        if missing_columns:
            errors.append(
                f"{file_path.relative_to(dataset_root)} is missing required column(s): {missing_columns}"
            )
            continue

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=parquet_columns):
            episodes = _column_numpy(batch, "episode_index", np.dtype(np.int64))
            sync_values = {
                key: _column_numpy(batch, key, np.dtype(np.float64)) for key in SYNC_KEYS
            }
            reused_values = {
                camera: _column_numpy(batch, f"{camera}_rgbd_reused", np.dtype(bool))
                for camera in CAMERAS
            }
            frame_values: dict[tuple[str, str], np.ndarray] = {}
            for camera in CAMERAS:
                for modality in modalities:
                    key = f"sidecar.{camera}_{modality}"
                    column_index = batch.schema.get_field_index(key)
                    expected_shape, expected_dtype = expected_arrays[key]
                    frame_values[(camera, modality)] = _column_to_frames(
                        batch.column(column_index), expected_shape, expected_dtype, key
                    )

            for row_index, episode_value in enumerate(episodes):
                episode = int(episode_value)
                local_index = episode_frame_counts.get(episode, 0)
                episode_frame_counts[episode] = local_index + 1
                for key in SYNC_KEYS:
                    _check_monotonic(
                        previous_sync,
                        errors,
                        episode,
                        key,
                        float(sync_values[key][row_index]),
                        local_index,
                    )
                for camera in CAMERAS:
                    reused = bool(reused_values[camera][row_index])
                    for modality in modalities:
                        stats_key = (episode, camera, modality)
                        stats = content_stats.setdefault(
                            stats_key, ContentStats(episode, camera, modality)
                        )
                        stats.add(frame_values[(camera, modality)][row_index], reused, local_index)

    if not episode_frame_counts:
        errors.append("Dataset has no frames.")

    for stats in content_stats.values():
        if stats.unmarked_duplicates:
            errors.append(
                f"episode={stats.episode} camera={stats.camera} modality={stats.modality} "
                f"has {stats.unmarked_duplicates} adjacent identical frame(s) with "
                f"rgbd_reused=false; first at local_frame={stats.first_unmarked_duplicate}"
            )
        if stats.longest_run_length > max_consecutive_identical:
            errors.append(
                f"episode={stats.episode} camera={stats.camera} modality={stats.modality} "
                f"is frozen from local_frame={stats.longest_run_start} through "
                f"{stats.longest_run_end} ({stats.longest_run_length} identical frames), "
                f"exceeding max_consecutive_identical={max_consecutive_identical}"
            )

    expected_total_frames = info.get("total_frames")
    actual_total_frames = sum(episode_frame_counts.values())
    if expected_total_frames is not None and int(expected_total_frames) != actual_total_frames:
        errors.append(
            f"meta total_frames={expected_total_frames} does not match parquet rows={actual_total_frames}"
        )
    expected_total_episodes = info.get("total_episodes")
    if expected_total_episodes is not None and int(expected_total_episodes) != len(episode_frame_counts):
        errors.append(
            f"meta total_episodes={expected_total_episodes} does not match "
            f"parquet episodes={len(episode_frame_counts)}"
        )

    return CheckReport(dataset_root, episode_frame_counts, content_stats, errors)


def print_report(report: CheckReport, max_consecutive_identical: int) -> None:
    print(f"dataset_root={report.dataset_root}")
    print(
        f"num_episodes={len(report.episode_frame_counts)} "
        f"num_frames={sum(report.episode_frame_counts.values())} "
        f"max_consecutive_identical={max_consecutive_identical}"
    )
    for episode, frame_count in sorted(report.episode_frame_counts.items()):
        print(f"episode={episode} frames={frame_count}")
        for camera in CAMERAS:
            for modality in MODALITIES:
                stats = report.content_stats.get((episode, camera, modality))
                if stats is None:
                    continue
                print(
                    f"  camera={camera} modality={modality} total_frames={stats.total_frames} "
                    f"unique_frames={stats.unique_frames} "
                    f"adjacent_duplicates={stats.adjacent_duplicates} "
                    f"longest_identical={stats.longest_run_start}-{stats.longest_run_end}"
                    f"({stats.longest_run_length}) reused_true={stats.reused_count} "
                    f"unmarked_duplicates={stats.unmarked_duplicates}"
                )

    if report.errors:
        sys.stdout.flush()
        print(f"RGB-D/IR sidecar check FAILED with {len(report.errors)} error(s):", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
    else:
        print("RGB-D/IR sidecar check passed.")


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
    parser.add_argument(
        "--max-consecutive-identical",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_IDENTICAL,
        help=(
            "Maximum allowed run length for pixel-identical frames when repeats are marked "
            f"reused (default: {DEFAULT_MAX_CONSECUTIVE_IDENTICAL}). Unmarked repeats always fail."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Raw Parquet rows processed per batch (default: 16).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.root, args.repo_id)
    report = inspect_dataset(
        dataset_root,
        rgb_camera_name_mode=args.rgb_camera_name_mode,
        save_depth=not args.no_depth,
        save_ir=not args.no_ir,
        max_consecutive_identical=args.max_consecutive_identical,
        batch_size=args.batch_size,
    )
    print_report(report, args.max_consecutive_identical)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
