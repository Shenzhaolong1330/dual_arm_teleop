import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check_rgbd_sidecar_dataset import CAMERAS, inspect_dataset


SHAPE = (2, 3)
MODALITIES = ("depth", "left_ir", "right_ir")


def _features() -> dict[str, dict]:
    features: dict[str, dict] = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "global_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "robot_timestamp": {"dtype": "float64", "shape": [1], "names": None},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}_rgb"] = {
            "dtype": "video",
            "shape": [3, *SHAPE],
            "names": ["channels", "height", "width"],
        }
        features[f"{camera}_rgbd_timestamp"] = {
            "dtype": "float64",
            "shape": [1],
            "names": None,
        }
        features[f"{camera}_rgbd_reused"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        for modality in MODALITIES:
            features[f"sidecar.{camera}_{modality}"] = {
                "dtype": "uint16" if modality == "depth" else "uint8",
                "shape": list(SHAPE),
                "names": ["height", "width"],
            }
    return features


def _sidecar_array(values: list[int], dtype: np.dtype) -> pa.Array:
    frames = [np.full(SHAPE, value, dtype=dtype).tolist() for value in values]
    scalar_type = pa.uint16() if dtype == np.dtype(np.uint16) else pa.uint8()
    return pa.array(frames, type=pa.list_(pa.list_(scalar_type)))


def _write_dataset(
    root: Path, episode_values: list[list[int]], episode_reused: list[list[bool]]
) -> Path:
    root.mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "data" / "chunk-000").mkdir(parents=True)

    total_frames = sum(len(values) for values in episode_values)
    info = {
        "total_episodes": len(episode_values),
        "total_frames": total_frames,
        "fps": 30,
        "features": _features(),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    values = [value for episode in episode_values for value in episode]
    reused = [value for episode in episode_reused for value in episode]
    episode_indices = [
        episode for episode, frames in enumerate(episode_values) for _ in frames
    ]
    global_indices = list(range(total_frames))
    data: dict[str, pa.Array] = {
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "global_frame_index": pa.array(global_indices, type=pa.int64()),
        "robot_timestamp": pa.array(
            [1000.0 + index for index in global_indices], type=pa.float64()
        ),
    }
    for camera in CAMERAS:
        data[f"{camera}_rgbd_timestamp"] = pa.array(
            [2000.0 + index for index in global_indices], type=pa.float64()
        )
        data[f"{camera}_rgbd_reused"] = pa.array(reused, type=pa.bool_())
        for modality in MODALITIES:
            dtype = np.dtype(np.uint16 if modality == "depth" else np.uint8)
            data[f"sidecar.{camera}_{modality}"] = _sidecar_array(values, dtype)

    pq.write_table(pa.table(data), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_checker_rejects_frozen_unmarked_sidecars(tmp_path):
    root = _write_dataset(
        tmp_path / "frozen",
        [[0, 1, 2, 2, 2, 2, 2]],
        [[False, False, False, False, False, False, False]],
    )

    report = inspect_dataset(root)

    assert not report.ok
    stats = report.content_stats[(0, "head", "depth")]
    assert stats.total_frames == 7
    assert stats.unique_frames == 3
    assert stats.adjacent_duplicates == 4
    assert stats.longest_run_length == 5
    assert stats.unmarked_duplicates == 4
    assert any("is frozen from local_frame=2 through 6" in error for error in report.errors)


def test_checker_allows_changing_frames_and_resets_episode_boundaries(tmp_path):
    root = _write_dataset(
        tmp_path / "changing",
        [[0, 1, 9], [9, 10, 11]],
        [[False, False, False], [False, False, False]],
    )

    report = inspect_dataset(root)

    assert report.ok
    assert report.episode_frame_counts == {0: 3, 1: 3}
    assert report.content_stats[(0, "head", "depth")].adjacent_duplicates == 0
    assert report.content_stats[(1, "head", "depth")].adjacent_duplicates == 0


def test_checker_allows_short_repeats_when_reused_is_true(tmp_path):
    root = _write_dataset(
        tmp_path / "marked_reuse",
        [[0, 1, 1, 2]],
        [[False, False, True, False]],
    )

    report = inspect_dataset(root)

    assert report.ok
    stats = report.content_stats[(0, "right_wrist", "right_ir")]
    assert stats.adjacent_duplicates == 1
    assert stats.reused_count == 1
    assert stats.unmarked_duplicates == 0
    assert stats.longest_run_length == 2


def test_checker_rejects_long_freeze_even_when_reused_is_true(tmp_path):
    root = _write_dataset(
        tmp_path / "marked_freeze",
        [[3, 3, 3, 3, 3]],
        [[False, True, True, True, True]],
    )

    report = inspect_dataset(root)

    assert not report.ok
    assert not any("rgbd_reused=false" in error for error in report.errors)
    assert any("exceeding max_consecutive_identical=4" in error for error in report.errors)
