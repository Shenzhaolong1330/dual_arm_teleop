import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.check_rgbd_sidecar_dataset import inspect_dataset
from scripts.core.rgbd_zarr_sidecar import (
    CAMERAS,
    CALIBRATION_RELATIVE_PATH,
    EPISODE_ENDS_PATH,
    MANIFEST_RELATIVE_PATH,
    RgbdSidecarError,
    ZarrSidecarReader,
    ZarrSidecarWriter,
)
from scripts.tools import export_ffs_stereo_pair, export_rgbd_sidecar_preview
from scripts.tools.rgbd_sidecar_export_utils import RgbdSidecarSource


SHAPE = (2, 3)


def _calibration(root: Path) -> Path:
    streams = {
        name: {"height": SHAPE[0], "width": SHAPE[1]}
        for name in ("depth", "infrared1", "infrared2")
    }
    payload = {
        "schema_version": 1,
        "cameras": {
            f"{camera}_rgb": {"streams": streams}
            for camera in CAMERAS
        },
    }
    path = root / CALIBRATION_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _features() -> dict[str, dict]:
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
        "global_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "robot_timestamp": {"dtype": "float64", "shape": (1,), "names": None},
    }
    for camera in CAMERAS:
        features[f"{camera}_rgbd_timestamp"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        }
        features[f"{camera}_rgbd_reused"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    return features


def _dataset(root: Path) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id="tests/rgbd_zarr",
        fps=30,
        features=_features(),
        robot_type="flexiv_dual_arm",
        root=root,
        use_videos=False,
    )
    dataset.meta.metadata_buffer_size = 1
    _calibration(root)
    return dataset


def _writer(root: Path, **overrides) -> ZarrSidecarWriter:
    kwargs = {
        "height": SHAPE[0],
        "width": SHAPE[1],
        "chunk_frames": 2,
        "queue_capacity_frames": 8,
        "compressor": {
            "codec": "blosc",
            "cname": "lz4",
            "clevel": 1,
            "shuffle": "bitshuffle",
        },
    }
    kwargs.update(overrides)
    return ZarrSidecarWriter(root, **kwargs)


def _raw_frame(
    value: int,
    *,
    reused: bool = False,
    global_index: int | None = None,
    rgbd_timestamp: float | None = None,
) -> dict:
    global_index = value if global_index is None else global_index
    observation = {
        "state": float(value),
        "global_frame_index": global_index,
        "robot_timestamp": 1000.0 + value,
    }
    for camera_index, camera in enumerate(CAMERAS):
        observation[f"sidecar.{camera}_depth"] = np.full(
            SHAPE, value + camera_index, dtype=np.uint16
        )
        observation[f"sidecar.{camera}_left_ir"] = np.full(
            SHAPE, value + camera_index, dtype=np.uint8
        )
        observation[f"sidecar.{camera}_right_ir"] = np.full(
            SHAPE, value + camera_index + 1, dtype=np.uint8
        )
        observation[f"{camera}_rgbd_timestamp"] = (
            2000.0 + value if rgbd_timestamp is None else rgbd_timestamp
        )
        observation[f"{camera}_rgbd_reused"] = reused
    return observation


def _main_frame(observation: dict, value: int) -> dict:
    frame = {
        "observation.state": np.array([value], dtype=np.float32),
        "action": np.array([value + 0.5], dtype=np.float32),
        "global_frame_index": np.array([observation["global_frame_index"]], dtype=np.int64),
        "robot_timestamp": np.array([observation["robot_timestamp"]], dtype=np.float64),
        "task": "synthetic",
    }
    for camera in CAMERAS:
        frame[f"{camera}_rgbd_timestamp"] = np.array(
            [observation[f"{camera}_rgbd_timestamp"]], dtype=np.float64
        )
        frame[f"{camera}_rgbd_reused"] = np.array(
            [observation[f"{camera}_rgbd_reused"]], dtype=np.bool_
        )
    return frame


def _add(dataset: LeRobotDataset, writer: ZarrSidecarWriter, value: int, **kwargs) -> dict:
    observation = _raw_frame(value, **kwargs)
    frame = _main_frame(observation, value)
    writer.add_frame(observation=observation, frame=frame)
    dataset.add_frame(frame)
    return observation


def _commit(dataset: LeRobotDataset, writer: ZarrSidecarWriter) -> None:
    writer.prepare_episode(int(dataset.episode_buffer["size"]))
    dataset.save_episode()
    dataset.seal_episode_writers()
    writer.commit_episode(
        info_total_frames=dataset.meta.total_frames,
        info_total_episodes=dataset.meta.total_episodes,
    )


def _finish(dataset: LeRobotDataset, writer: ZarrSidecarWriter) -> None:
    dataset.finalize()
    writer.finalize(
        info_total_frames=dataset.meta.total_frames,
        info_total_episodes=dataset.meta.total_episodes,
    )


def _add_rgb_feature_declarations(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for camera in CAMERAS:
        info["features"][f"observation.images.{camera}_rgb"] = {
            "dtype": "video",
            "shape": [3, *SHAPE],
            "names": ["channels", "height", "width"],
        }
    info_path.write_text(json.dumps(info, sort_keys=True) + "\n")


def test_streaming_snapshots_arrays_and_excludes_them_from_parquet_stats(tmp_path):
    root = tmp_path / "dataset"
    dataset = _dataset(root)
    writer = _writer(root)

    first = _add(dataset, writer, 0)
    _add(dataset, writer, 1)
    first["sidecar.head_depth"][:] = 999

    deadline = time.monotonic() + 3.0
    physical = zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="r")
    while physical["data/head/depth"].shape[0] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert physical["data/head/depth"].shape[0] == 2
    assert np.array_equal(physical["data/head/depth"][0], np.zeros(SHAPE, dtype=np.uint16))
    assert int(dataset.episode_buffer["size"]) == 2
    assert not any(key.startswith("sidecar.") for key in dataset.features)
    assert not any(key.startswith("sidecar.") for key in dataset.episode_buffer)

    _commit(dataset, writer)
    _finish(dataset, writer)

    parquet = pq.ParquetFile(next((root / "data").rglob("*.parquet")))
    assert not any(key.startswith("sidecar.") for key in parquet.schema_arrow.names)
    assert not any(key.startswith("sidecar.") for key in dataset.meta.stats)
    reader = ZarrSidecarReader(root)
    assert reader.committed_frames == 2
    for camera_index, camera in enumerate(CAMERAS):
        raw = reader.frame(1, camera)
        assert raw["depth"].dtype == np.uint16
        assert raw["left_ir"].dtype == np.uint8
        assert raw["right_ir"].dtype == np.uint8
        assert np.array_equal(
            raw["depth"], np.full(SHAPE, 1 + camera_index, dtype=np.uint16)
        )
        assert np.array_equal(
            raw["left_ir"], np.full(SHAPE, 1 + camera_index, dtype=np.uint8)
        )
        assert np.array_equal(
            raw["right_ir"], np.full(SHAPE, 2 + camera_index, dtype=np.uint8)
        )


def test_two_episode_commit_and_rerecord_rollback(tmp_path):
    root = tmp_path / "dataset"
    dataset = _dataset(root)
    writer = _writer(root)

    _add(dataset, writer, 0)
    _add(dataset, writer, 1)
    _commit(dataset, writer)

    _add(dataset, writer, 2)
    writer.rollback_episode()
    dataset.clear_episode_buffer(delete_images=False)
    assert writer.active_frames == 0
    group = zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="r")
    assert group["meta/index"].shape == (2,)

    _add(dataset, writer, 3, global_index=4, rgbd_timestamp=10.0)
    _add(dataset, writer, 4, global_index=7, rgbd_timestamp=11.0)
    _commit(dataset, writer)
    _finish(dataset, writer)

    reader = ZarrSidecarReader(root)
    assert reader.committed_episodes == 2
    assert np.array_equal(reader.array(EPISODE_ENDS_PATH)[:], np.array([2, 4]))
    assert np.array_equal(reader.array("/meta/global_frame_index")[:], np.array([0, 1, 4, 7]))


def test_incomplete_reader_rejected_and_resume_truncates_uncommitted_tail(tmp_path):
    root = tmp_path / "dataset"
    dataset = _dataset(root)
    writer = _writer(root)
    _add(dataset, writer, 0)
    _commit(dataset, writer)
    _add(dataset, writer, 1)
    writer.drain()
    assert zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="r")[
        "meta/index"
    ].shape == (2,)
    writer.abort("synthetic interruption")
    dataset.finalize()

    with pytest.raises(RgbdSidecarError, match="not readable"):
        ZarrSidecarReader(root)

    resumed = _writer(root, resume=True)
    assert resumed.committed_frames == 1
    assert resumed.active_frames == 0
    assert zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="r")[
        "meta/index"
    ].shape == (1,)
    resumed.abort("end resume test")


def test_resume_rejects_main_parquet_count_disagreement(tmp_path):
    root = tmp_path / "dataset"
    dataset = _dataset(root)
    writer = _writer(root)
    _add(dataset, writer, 0)
    _commit(dataset, writer)
    writer.abort("closed before synthetic mismatch")
    dataset.meta.info["total_frames"] = 2
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 2
    info_path.write_text(json.dumps(info) + "\n")

    with pytest.raises(RgbdSidecarError, match="Cannot safely resume"):
        _writer(root, resume=True)


def test_background_writer_and_queue_failures_propagate(tmp_path):
    root = tmp_path / "writer_failure"
    dataset = _dataset(root)
    writer = _writer(root, chunk_frames=1, queue_capacity_frames=2)
    writer._group["meta/index"].resize((1,))
    _add(dataset, writer, 0)
    with pytest.raises(RgbdSidecarError, match="background writer failed"):
        writer.drain()
    writer.abort("synthetic writer failure", corrupt=True)
    assert json.loads((root / MANIFEST_RELATIVE_PATH).read_text())["status"] == "corrupt"

    root = tmp_path / "queue_failure"
    dataset = _dataset(root)
    writer = _writer(root, chunk_frames=1, queue_capacity_frames=1)
    entered = __import__("threading").Event()
    release = __import__("threading").Event()
    original_flush = writer._flush_records

    def blocked_flush(records):
        entered.set()
        assert release.wait(timeout=3.0)
        original_flush(records)

    writer._flush_records = blocked_flush
    _add(dataset, writer, 0)
    assert entered.wait(timeout=3.0)
    _add(dataset, writer, 1)
    with pytest.raises(RgbdSidecarError, match="queue is full"):
        _add(dataset, writer, 2)
    release.set()
    writer.abort("synthetic queue full", corrupt=True)


def test_checker_zarr_freeze_join_and_manifest_failures(tmp_path):
    root = tmp_path / "frozen"
    dataset = _dataset(root)
    writer = _writer(root)
    for index, value in enumerate((0, 1, 1)):
        _add(dataset, writer, value, reused=False, global_index=index)
    _commit(dataset, writer)
    _finish(dataset, writer)
    _add_rgb_feature_declarations(root)

    report = inspect_dataset(root, batch_size=2)
    assert report.storage == "zarr"
    assert not report.ok
    assert report.content_stats[(0, "head", "depth")].unmarked_duplicates == 1
    assert any("rgbd_reused=false" in error for error in report.errors)

    manifest = json.loads((root / MANIFEST_RELATIVE_PATH).read_text())
    manifest["calibration"]["sha256"] = "0" * 64
    (root / MANIFEST_RELATIVE_PATH).write_text(json.dumps(manifest) + "\n")
    report = inspect_dataset(root)
    assert not report.ok
    assert any("Calibration SHA-256 mismatch" in error for error in report.errors)


def test_reader_rejects_array_length_and_checker_rejects_join_mismatch(tmp_path):
    root = tmp_path / "length_mismatch"
    dataset = _dataset(root)
    writer = _writer(root)
    _add(dataset, writer, 0)
    _commit(dataset, writer)
    _finish(dataset, writer)
    _add_rgb_feature_declarations(root)

    group = zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="a")
    group["data/head/depth"].resize((2, *SHAPE))
    with pytest.raises(RgbdSidecarError, match="shape"):
        ZarrSidecarReader(root)

    root = tmp_path / "join_mismatch"
    dataset = _dataset(root)
    writer = _writer(root)
    _add(dataset, writer, 0)
    _commit(dataset, writer)
    _finish(dataset, writer)
    _add_rgb_feature_declarations(root)
    group = zarr.open_group(str(root / "sidecars" / "realsense.zarr"), mode="a")
    group["meta/global_frame_index"][0] = 99
    report = inspect_dataset(root)
    assert not report.ok
    assert any("Parquet/Zarr join mismatch for global_frame_index" in error for error in report.errors)


class _RawRows:
    def __init__(self, row):
        self.row = row

    def with_format(self, _format):
        return self

    def __getitem__(self, _index):
        return self.row


class _FakeDataset:
    def __init__(self, root: Path, row: dict, *, zarr_storage: bool):
        self.root = root
        self.repo_id = "tests/fake"
        self.hf_dataset = _RawRows(row)
        self.features = {
            "observation.images.head_rgb": {
                "dtype": "video",
                "shape": [3, *SHAPE],
                "names": ["channels", "height", "width"],
            }
        }
        if not zarr_storage:
            self.features.update(
                {
                    "sidecar.head_depth": {"dtype": "uint16", "shape": SHAPE, "names": None},
                    "sidecar.head_left_ir": {"dtype": "uint8", "shape": SHAPE, "names": None},
                    "sidecar.head_right_ir": {"dtype": "uint8", "shape": SHAPE, "names": None},
                }
            )
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [1]}
        )
        self.num_episodes = 1

    def _ensure_hf_dataset_loaded(self):
        return None

    def __getitem__(self, _index):
        return {
            "observation.images.head_rgb": np.zeros((3, *SHAPE), dtype=np.uint8)
        }


def test_zarr_preview_and_lossless_unrectified_ffs_export(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    dataset = _dataset(root)
    writer = _writer(root)
    _add(dataset, writer, 5)
    _commit(dataset, writer)
    _finish(dataset, writer)
    _add_rgb_feature_declarations(root)
    row = {
        "global_frame_index": 5,
        "robot_timestamp": 1005.0,
        "head_rgbd_timestamp": 2005.0,
        "head_rgbd_reused": False,
    }
    fake = _FakeDataset(root, row, zarr_storage=True)
    output = tmp_path / "exports"

    monkeypatch.setattr(export_rgbd_sidecar_preview, "resolve_dataset", lambda *_: fake)
    monkeypatch.setattr(
        sys,
        "argv",
        ["preview", "--root", str(root), "--episode", "0", "--frame-index", "0", "--output-dir", str(output)],
    )
    export_rgbd_sidecar_preview.main()

    monkeypatch.setattr(export_ffs_stereo_pair, "resolve_dataset", lambda *_: fake)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ffs", "--root", str(root), "--episode", "0", "--frame-index", "0", "--output-dir", str(output)],
    )
    export_ffs_stereo_pair.main()

    left_path = next(output.glob("*_left_ir.png"))
    right_path = next(output.glob("*_right_ir.png"))
    assert np.array_equal(np.asarray(Image.open(left_path)), np.full(SHAPE, 5, dtype=np.uint8))
    assert np.array_equal(np.asarray(Image.open(right_path)), np.full(SHAPE, 6, dtype=np.uint8))
    metadata = json.loads(next(output.glob("*_metadata.json")).read_text())
    assert metadata["sidecar_storage"] == "zarr"
    assert "no rectification" in metadata["note"]


def test_legacy_parquet_source_remains_supported(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    row = {
        "sidecar.head_depth": np.full(SHAPE, 7, dtype=np.uint16),
        "sidecar.head_left_ir": np.full(SHAPE, 8, dtype=np.uint8),
        "sidecar.head_right_ir": np.full(SHAPE, 9, dtype=np.uint8),
        "head_rgbd_timestamp": 1.25,
        "head_rgbd_reused": True,
    }
    fake = _FakeDataset(root, row, zarr_storage=False)
    source = RgbdSidecarSource(fake)
    source.require("head", ("depth", "left_ir", "right_ir"))
    frame = source.frame(0, "head")
    assert source.storage == "parquet"
    assert np.array_equal(frame["depth"], row["sidecar.head_depth"])
    assert frame["rgbd_timestamp"] == 1.25
    assert frame["rgbd_reused"] is True
