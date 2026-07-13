#!/usr/bin/env python
"""Transactional Zarr v2 storage for Flexiv RealSense raw depth/IR frames.

The JSON manifest is the commit ledger. Zarr arrays may contain an uncommitted
tail while an episode is active, but readers must expose only the committed
prefix recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_NAME = "lerobot_realsense_raw_sidecar"
SCHEMA_VERSION = 1
STORAGE_NAME = "zarr_v2"
DEFAULT_RELATIVE_PATH = "sidecars/realsense.zarr"
MANIFEST_RELATIVE_PATH = "meta/rgbd_sidecar.json"
CALIBRATION_RELATIVE_PATH = "meta/realsense_calibration.json"
CAMERAS = ("head", "left_wrist", "right_wrist")
MODALITIES = ("depth", "left_ir", "right_ir")
TIME_META_PATHS = (
    "/meta/index",
    "/meta/episode_index",
    "/meta/frame_index",
    "/meta/global_frame_index",
    "/meta/robot_timestamp",
)
EPISODE_ENDS_PATH = "/meta/episode_ends"
VALID_STATUSES = {"in_progress", "complete", "incomplete", "corrupt"}


class RgbdSidecarError(RuntimeError):
    """Raised when the raw sidecar cannot safely continue or be read."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _import_zarr_v2():
    try:
        import zarr  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Zarr raw sidecars require zarr>=2.12,<3. Install the dual_arm_teleop dependencies."
        ) from exc
    major = int(str(zarr.__version__).split(".", 1)[0])
    if major != 2:
        raise RgbdSidecarError(
            f"Raw sidecars require Zarr v2, but imported zarr {zarr.__version__}. "
            "Install zarr>=2.12,<3."
        )
    return zarr


def _build_compressor(config: Mapping[str, Any]):
    codec = str(config.get("codec", "blosc")).strip().lower()
    if codec in {"none", "null", "uncompressed"}:
        return None
    if codec != "blosc":
        raise ValueError(f"Unsupported RGB-D sidecar compressor codec: {codec!r}")
    try:
        from numcodecs import Blosc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("The blosc sidecar compressor requires numcodecs.") from exc

    shuffle_name = str(config.get("shuffle", "bitshuffle")).strip().lower()
    shuffle_values = {
        "none": Blosc.NOSHUFFLE,
        "noshuffle": Blosc.NOSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "bitshuffle": Blosc.BITSHUFFLE,
    }
    if shuffle_name not in shuffle_values:
        raise ValueError(
            "rgbd_sidecar_zarr.compressor.shuffle must be one of "
            f"{sorted(shuffle_values)}, got {shuffle_name!r}"
        )
    return Blosc(
        cname=str(config.get("cname", "lz4")),
        clevel=int(config.get("clevel", 1)),
        shuffle=shuffle_values[shuffle_name],
        blocksize=int(config.get("blocksize", 0)),
    )


def _safe_sidecar_path(dataset_root: Path, relative_path: str) -> Path:
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise RgbdSidecarError(f"Sidecar relative_path must be relative, got {relative_path!r}")
    root = dataset_root.resolve()
    resolved = (root / raw_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise RgbdSidecarError(f"Sidecar relative_path escapes the dataset root: {relative_path!r}")
    return resolved


def _base_camera_name(name: str) -> str:
    for suffix in ("_rgb", "_image"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def _selected_modalities(save_depth: bool, save_ir: bool) -> tuple[str, ...]:
    values: list[str] = []
    if save_depth:
        values.append("depth")
    if save_ir:
        values.extend(("left_ir", "right_ir"))
    return tuple(values)


def _time_array_specs(
    cameras: tuple[str, ...],
    modalities: tuple[str, ...],
    height: int,
    width: int,
) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for camera in cameras:
        for modality in modalities:
            dtype = np.dtype(np.uint16 if modality == "depth" else np.uint8)
            specs[f"/data/{camera}/{modality}"] = (dtype, (height, width))
        specs[f"/data/{camera}/rgbd_timestamp"] = (np.dtype(np.float64), ())
        specs[f"/data/{camera}/rgbd_reused"] = (np.dtype(bool), ())
    specs.update(
        {
            "/meta/index": (np.dtype(np.int64), ()),
            "/meta/episode_index": (np.dtype(np.int64), ()),
            "/meta/frame_index": (np.dtype(np.int64), ()),
            "/meta/global_frame_index": (np.dtype(np.int64), ()),
            "/meta/robot_timestamp": (np.dtype(np.float64), ()),
        }
    )
    return specs


def parquet_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").rglob("*.parquet"))


def parquet_row_count(dataset_root: Path) -> int:
    files = parquet_files(dataset_root)
    if not files:
        return 0
    return sum(int(pq.read_metadata(path).num_rows) for path in files)


def _unwrap_arrow(array: pa.Array) -> pa.Array:
    return array.storage if isinstance(array, pa.ExtensionArray) else array


def _arrow_numpy(batch: pa.RecordBatch, key: str, dtype: np.dtype) -> np.ndarray:
    column_index = batch.schema.get_field_index(key)
    if column_index < 0:
        raise RgbdSidecarError(f"Parquet batch is missing required join column: {key}")
    column = _unwrap_arrow(batch.column(column_index))
    valid_type = (
        pa.types.is_int64(column.type)
        if dtype == np.dtype(np.int64)
        else pa.types.is_float64(column.type)
        if dtype == np.dtype(np.float64)
        else pa.types.is_boolean(column.type)
        if dtype == np.dtype(bool)
        else False
    )
    if not valid_type:
        raise RgbdSidecarError(
            f"Parquet join column {key} has type={column.type}, expected dtype={dtype}."
        )
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.dtype == object:
        values = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in values],
            dtype=dtype,
        )
    else:
        values = values.astype(dtype, copy=False)
    return values.reshape(-1)


def read_parquet_join_range(dataset_root: Path, start: int, end: int) -> dict[str, np.ndarray]:
    """Read scalar join columns for a physical row-ordinal interval."""

    if start < 0 or end < start:
        raise ValueError(f"Invalid parquet range [{start}, {end})")
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "global_frame_index",
        "robot_timestamp",
        *[f"{camera}_rgbd_timestamp" for camera in CAMERAS],
        *[f"{camera}_rgbd_reused" for camera in CAMERAS],
    ]
    dtypes = {
        "index": np.dtype(np.int64),
        "episode_index": np.dtype(np.int64),
        "frame_index": np.dtype(np.int64),
        "global_frame_index": np.dtype(np.int64),
        "robot_timestamp": np.dtype(np.float64),
        **{f"{camera}_rgbd_timestamp": np.dtype(np.float64) for camera in CAMERAS},
        **{f"{camera}_rgbd_reused": np.dtype(bool) for camera in CAMERAS},
    }
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in columns}
    physical_offset = 0
    for file_path in parquet_files(dataset_root):
        parquet_file = pq.ParquetFile(file_path)
        file_rows = int(parquet_file.metadata.num_rows)
        file_end = physical_offset + file_rows
        overlap_start = max(start, physical_offset)
        overlap_end = min(end, file_end)
        if overlap_start < overlap_end:
            missing = [key for key in columns if key not in parquet_file.schema_arrow.names]
            if missing:
                raise RgbdSidecarError(
                    f"{file_path.relative_to(dataset_root)} is missing join column(s): {missing}"
                )
            local_start = overlap_start - physical_offset
            local_end = overlap_end - physical_offset
            local_offset = 0
            for batch in parquet_file.iter_batches(batch_size=4096, columns=columns):
                batch_end = local_offset + len(batch)
                take_start = max(local_start, local_offset)
                take_end = min(local_end, batch_end)
                if take_start < take_end:
                    slice_start = take_start - local_offset
                    slice_length = take_end - take_start
                    sliced = batch.slice(slice_start, slice_length)
                    for key in columns:
                        pieces[key].append(_arrow_numpy(sliced, key, dtypes[key]))
                local_offset = batch_end
                if local_offset >= local_end:
                    break
        physical_offset = file_end

    result = {
        key: np.concatenate(values) if values else np.empty((0,), dtype=dtypes[key])
        for key, values in pieces.items()
    }
    expected = end - start
    if any(len(values) != expected for values in result.values()):
        observed = {key: len(values) for key, values in result.items()}
        raise RgbdSidecarError(
            f"Parquet join range [{start}, {end}) is incomplete: observed lengths={observed}"
        )
    return result


def _calibration_camera(calibration: Mapping[str, Any], camera: str) -> Mapping[str, Any]:
    cameras = calibration.get("cameras")
    if not isinstance(cameras, Mapping):
        raise RgbdSidecarError("RealSense calibration manifest has no cameras mapping.")
    for candidate in (camera, f"{camera}_rgb", f"{camera}_image"):
        payload = cameras.get(candidate)
        if isinstance(payload, Mapping):
            return payload
    raise RgbdSidecarError(f"Calibration manifest is missing logical camera {camera!r}.")


def validate_calibration_shapes(
    calibration: Mapping[str, Any],
    cameras: tuple[str, ...],
    modalities: tuple[str, ...],
    height: int,
    width: int,
) -> None:
    stream_names = {"depth": "depth", "left_ir": "infrared1", "right_ir": "infrared2"}
    for camera in cameras:
        camera_payload = _calibration_camera(calibration, camera)
        streams = camera_payload.get("streams")
        if not isinstance(streams, Mapping):
            raise RgbdSidecarError(f"Calibration for {camera} has no streams mapping.")
        for modality in modalities:
            stream_name = stream_names[modality]
            stream = streams.get(stream_name)
            if not isinstance(stream, Mapping):
                raise RgbdSidecarError(
                    f"Calibration for {camera} is missing stream profile {stream_name}."
                )
            observed = (int(stream.get("height", -1)), int(stream.get("width", -1)))
            if observed != (height, width):
                raise RgbdSidecarError(
                    f"Calibration {camera}/{stream_name} shape={observed}, expected {(height, width)}."
                )


def load_manifest(dataset_root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    manifest_path = dataset_root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing RGB-D sidecar manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RgbdSidecarError(f"Invalid RGB-D sidecar manifest: {manifest_path}: {exc}") from exc
    if manifest.get("schema_name") != SCHEMA_NAME:
        raise RgbdSidecarError(
            f"Unsupported RGB-D sidecar schema_name={manifest.get('schema_name')!r}."
        )
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RgbdSidecarError(
            f"Unsupported RGB-D sidecar schema_version={manifest.get('schema_version')!r}."
        )
    if manifest.get("storage") != STORAGE_NAME:
        raise RgbdSidecarError(f"Unsupported RGB-D sidecar storage={manifest.get('storage')!r}.")
    status = manifest.get("status")
    if status not in VALID_STATUSES:
        raise RgbdSidecarError(f"Invalid RGB-D sidecar status={status!r}.")
    if require_complete and status != "complete":
        raise RgbdSidecarError(
            f"RGB-D sidecar is not readable as a completed dataset: status={status!r}."
        )
    if not isinstance(manifest.get("arrays"), Mapping):
        raise RgbdSidecarError("RGB-D sidecar manifest has no arrays mapping.")
    _safe_sidecar_path(dataset_root, str(manifest.get("relative_path", "")))
    return manifest


class ZarrSidecarReader:
    """Strict committed-prefix reader used by the checker and exporters."""

    def __init__(self, dataset_root: Path, *, require_complete: bool = True):
        self.dataset_root = Path(dataset_root)
        self.manifest = load_manifest(self.dataset_root, require_complete=require_complete)
        self.committed_frames = int(self.manifest.get("committed_frames", -1))
        self.committed_episodes = int(self.manifest.get("committed_episodes", -1))
        if self.committed_frames < 0 or self.committed_episodes < 0:
            raise RgbdSidecarError("Manifest committed counts must be nonnegative integers.")

        calibration_rel = str(
            self.manifest.get("calibration", {}).get("relative_path", CALIBRATION_RELATIVE_PATH)
        )
        self.calibration_path = _safe_sidecar_path(self.dataset_root, calibration_rel)
        if not self.calibration_path.is_file():
            raise RgbdSidecarError(f"Missing calibration manifest: {self.calibration_path}")
        observed_hash = sha256_file(self.calibration_path)
        expected_hash = self.manifest.get("calibration", {}).get("sha256")
        if observed_hash != expected_hash:
            raise RgbdSidecarError(
                f"Calibration SHA-256 mismatch: expected={expected_hash} observed={observed_hash}"
            )
        self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))

        cameras = tuple(str(value) for value in self.manifest.get("cameras", []))
        modalities = tuple(str(value) for value in self.manifest.get("modalities", []))
        frame_shape = self.manifest.get("frame_shape", {})
        height = int(frame_shape.get("height", -1))
        width = int(frame_shape.get("width", -1))
        if set(cameras) != set(CAMERAS) or len(cameras) != len(CAMERAS):
            raise RgbdSidecarError(
                f"Manifest cameras={cameras}, expected exactly {CAMERAS}."
            )
        if not modalities or not set(modalities).issubset(MODALITIES):
            raise RgbdSidecarError(f"Invalid manifest modalities={modalities}.")
        if height <= 0 or width <= 0:
            raise RgbdSidecarError("Manifest frame_shape declaration is incomplete.")
        canonical_specs = _time_array_specs(cameras, modalities, height, width)
        canonical_paths = {*canonical_specs, EPISODE_ENDS_PATH}
        if set(self.manifest["arrays"]) != canonical_paths:
            raise RgbdSidecarError(
                "Manifest array path set violates the canonical schema: "
                f"manifest={sorted(self.manifest['arrays'])} expected={sorted(canonical_paths)}"
            )

        zarr = _import_zarr_v2()
        self.store_path = _safe_sidecar_path(
            self.dataset_root, str(self.manifest["relative_path"])
        )
        if not self.store_path.is_dir():
            raise RgbdSidecarError(f"Missing Zarr v2 sidecar store: {self.store_path}")
        self.group = zarr.open_group(str(self.store_path), mode="r")
        for key in ("schema_name", "schema_version", "storage", "relative_path"):
            manifest_value = self.manifest[key]
            attr_value = self.group.attrs.get(key)
            if attr_value != manifest_value:
                raise RgbdSidecarError(
                    f"Zarr root attr {key!r} conflicts with the manifest: "
                    f"manifest={manifest_value!r} zarr={attr_value!r}"
                )

        self.arrays: dict[str, Any] = {}
        for path, declared in self.manifest["arrays"].items():
            key = path.lstrip("/")
            if key not in self.group:
                raise RgbdSidecarError(f"Manifest array is missing from Zarr: {path}")
            array = self.group[key]
            self.arrays[path] = array
            declared_dtype = np.dtype(declared.get("dtype"))
            expected_dtype = (
                np.dtype(np.int64) if path == EPISODE_ENDS_PATH else canonical_specs[path][0]
            )
            expected_tail = () if path == EPISODE_ENDS_PATH else canonical_specs[path][1]
            if declared_dtype != expected_dtype:
                raise RgbdSidecarError(
                    f"Manifest {path} dtype={declared_dtype}, canonical dtype={expected_dtype}."
                )
            if np.dtype(array.dtype) != declared_dtype:
                raise RgbdSidecarError(
                    f"Zarr {path} dtype={array.dtype}, manifest declares {declared_dtype}."
                )
            declared_shape = tuple(int(value) for value in declared.get("shape", []))
            if declared_shape[1:] != expected_tail:
                raise RgbdSidecarError(
                    f"Manifest {path} tail shape={declared_shape[1:]}, canonical shape={expected_tail}."
                )
            declared_axis0 = self.committed_episodes if path == EPISODE_ENDS_PATH else self.committed_frames
            expected_declared_shape = (declared_axis0, *expected_tail)
            if declared_shape != expected_declared_shape:
                raise RgbdSidecarError(
                    f"Manifest {path} shape={declared_shape}, committed shape={expected_declared_shape}."
                )
            if tuple(array.shape) != declared_shape and require_complete:
                raise RgbdSidecarError(
                    f"Zarr {path} shape={array.shape}, manifest declares {declared_shape}."
                )
            declared_chunks = tuple(int(value) for value in declared.get("chunks", []))
            if tuple(array.chunks) != declared_chunks:
                raise RgbdSidecarError(
                    f"Zarr {path} chunks={array.chunks}, manifest declares {declared_chunks}."
                )
            observed_compressor = array.compressor.get_config() if array.compressor is not None else None
            if observed_compressor != declared.get("compressor"):
                raise RgbdSidecarError(
                    f"Zarr {path} compressor conflicts with the manifest: "
                    f"manifest={declared.get('compressor')} zarr={observed_compressor}"
                )

        validate_calibration_shapes(self.calibration, cameras, modalities, height, width)

        for path, array in self.arrays.items():
            expected_axis0 = self.committed_episodes if path == EPISODE_ENDS_PATH else self.committed_frames
            if require_complete and int(array.shape[0]) != expected_axis0:
                raise RgbdSidecarError(
                    f"Zarr {path} axis-0={array.shape[0]}, expected committed count {expected_axis0}."
                )
            if not require_complete and int(array.shape[0]) < expected_axis0:
                raise RgbdSidecarError(
                    f"Zarr {path} axis-0={array.shape[0]} is shorter than committed count {expected_axis0}."
                )

        episode_ends = np.asarray(self.arrays[EPISODE_ENDS_PATH][: self.committed_episodes])
        if len(episode_ends) != self.committed_episodes:
            raise RgbdSidecarError("episode_ends length does not match committed_episodes.")
        if len(episode_ends) and (
            np.any(np.diff(episode_ends) <= 0) or int(episode_ends[-1]) != self.committed_frames
        ):
            raise RgbdSidecarError(
                f"Invalid episode_ends={episode_ends.tolist()} for committed_frames={self.committed_frames}."
            )

        info_path = self.dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise RgbdSidecarError(f"Missing LeRobot info.json: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        if require_complete:
            info_frames = int(self.info.get("total_frames", -1))
            info_episodes = int(self.info.get("total_episodes", -1))
            rows = parquet_row_count(self.dataset_root)
            if (info_frames, rows) != (self.committed_frames, self.committed_frames):
                raise RgbdSidecarError(
                    "Committed frame count mismatch: "
                    f"manifest={self.committed_frames} info={info_frames} parquet_rows={rows}."
                )
            if info_episodes != self.committed_episodes:
                raise RgbdSidecarError(
                    "Committed episode count mismatch: "
                    f"manifest={self.committed_episodes} info={info_episodes}."
                )

    def array(self, path: str):
        try:
            return self.arrays[path]
        except KeyError as exc:
            raise RgbdSidecarError(f"RGB-D sidecar does not contain array {path}.") from exc

    def frame(self, row: int, camera: str) -> dict[str, Any]:
        if row < 0 or row >= self.committed_frames:
            raise IndexError(f"Sidecar row {row} is outside [0, {self.committed_frames}).")
        values: dict[str, Any] = {}
        for modality in MODALITIES:
            path = f"/data/{camera}/{modality}"
            if path in self.arrays:
                values[modality] = np.asarray(self.arrays[path][row])
        values["rgbd_timestamp"] = float(self.array(f"/data/{camera}/rgbd_timestamp")[row])
        values["rgbd_reused"] = bool(self.array(f"/data/{camera}/rgbd_reused")[row])
        return values


@dataclass
class _FrameRecord:
    values: dict[str, Any]


@dataclass
class _WriterCommand:
    kind: str
    value: int | None = None
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class ZarrSidecarWriter:
    """Bounded-queue background writer with manifest-led episode commits."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        cameras: tuple[str, ...] = CAMERAS,
        save_depth: bool = True,
        save_ir: bool = True,
        height: int,
        width: int,
        relative_path: str = DEFAULT_RELATIVE_PATH,
        chunk_frames: int = 8,
        queue_capacity_frames: int = 64,
        compressor: Mapping[str, Any] | None = None,
        resume: bool = False,
    ):
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        if queue_capacity_frames < chunk_frames:
            raise ValueError("queue_capacity_frames must be >= chunk_frames")
        normalized_cameras = tuple(_base_camera_name(str(value)) for value in cameras)
        if len(set(normalized_cameras)) != len(normalized_cameras):
            raise ValueError(f"Camera names are not unique after normalization: {cameras}")
        if set(normalized_cameras) != set(CAMERAS):
            raise ValueError(
                f"Flexiv RealSense sidecars require cameras={CAMERAS}, got {normalized_cameras}."
            )

        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / MANIFEST_RELATIVE_PATH
        self.relative_path = relative_path
        self.store_path = _safe_sidecar_path(self.dataset_root, relative_path)
        self.cameras = normalized_cameras
        self.modalities = _selected_modalities(save_depth, save_ir)
        if not self.modalities:
            raise ValueError("Zarr sidecar requires save_depth_sidecar and/or save_ir_sidecar.")
        self.height = int(height)
        self.width = int(width)
        self.chunk_frames = int(chunk_frames)
        self.queue_capacity_frames = int(queue_capacity_frames)
        self.compressor_config = dict(compressor or {})
        self._compressor = _build_compressor(self.compressor_config)
        self._specs = _time_array_specs(
            self.cameras, self.modalities, self.height, self.width
        )
        self._queue: queue.Queue[_FrameRecord | _WriterCommand] = queue.Queue(
            maxsize=self.queue_capacity_frames
        )
        self._worker_error: BaseException | None = None
        self._closed = False
        self._max_queue_depth = 0
        self._written_frames = 0
        self._active_frames = 0
        self._last_global_frame_index: int | None = None
        self._last_camera_timestamps: dict[str, float] = {}

        calibration_path = self.dataset_root / CALIBRATION_RELATIVE_PATH
        if not calibration_path.is_file():
            raise RgbdSidecarError(
                f"Zarr sidecar requires calibration before writer startup: {calibration_path}"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        validate_calibration_shapes(
            calibration, self.cameras, self.modalities, self.height, self.width
        )
        self._calibration_sha256 = sha256_file(calibration_path)

        if resume:
            self._resume_existing()
        else:
            self._create_new()

        self._thread = threading.Thread(
            target=self._worker_main,
            name="rgbd_zarr_sidecar_writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def committed_frames(self) -> int:
        return int(self._manifest["committed_frames"])

    @property
    def committed_episodes(self) -> int:
        return int(self._manifest["committed_episodes"])

    @property
    def active_frames(self) -> int:
        return self._active_frames

    @property
    def max_queue_depth(self) -> int:
        return self._max_queue_depth

    @property
    def next_global_frame_index(self) -> int:
        return 0 if self._last_global_frame_index is None else self._last_global_frame_index + 1

    def _array_manifest_entry(self, array: Any, committed: int) -> dict[str, Any]:
        shape = [committed, *[int(value) for value in array.shape[1:]]]
        return {
            "dtype": np.dtype(array.dtype).name,
            "shape": shape,
            "chunks": [int(value) for value in array.chunks],
            "compressor": array.compressor.get_config() if array.compressor is not None else None,
        }

    def _manifest_arrays(self, committed_frames: int, committed_episodes: int) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for path in (*self._specs.keys(), EPISODE_ENDS_PATH):
            array = self._group[path.lstrip("/")]
            committed = committed_episodes if path == EPISODE_ENDS_PATH else committed_frames
            entries[path] = self._array_manifest_entry(array, committed)
        return entries

    def _create_new(self) -> None:
        if self.manifest_path.exists() or self.store_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing RGB-D sidecar files under {self.dataset_root}."
            )
        zarr = _import_zarr_v2()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._group = zarr.open_group(str(self.store_path), mode="w")
        self._group.attrs.update(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "storage": STORAGE_NAME,
                "relative_path": self.relative_path,
            }
        )
        for path, (dtype, tail_shape) in self._specs.items():
            chunks = (self.chunk_frames, *tail_shape)
            self._group.create_dataset(
                path.lstrip("/"),
                shape=(0, *tail_shape),
                chunks=chunks,
                dtype=dtype,
                compressor=self._compressor,
                overwrite=False,
            )
        self._group.create_dataset(
            EPISODE_ENDS_PATH.lstrip("/"),
            shape=(0,),
            chunks=(max(128, self.chunk_frames),),
            dtype=np.int64,
            compressor=self._compressor,
            overwrite=False,
        )
        now = utc_now_iso()
        self._manifest = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "storage": STORAGE_NAME,
            "relative_path": self.relative_path,
            "status": "in_progress",
            "committed_frames": 0,
            "committed_episodes": 0,
            "row_semantics": (
                "Zarr row ordinal i joins LeRobot Parquet index == i; readers must also verify "
                "episode_index, frame_index, global_frame_index, robot/per-camera timestamps, and reused."
            ),
            "commit_semantics": (
                "Only the manifest committed prefix is readable. Physical Zarr tails beyond it are "
                "uncommitted and are truncated on explicit resume or rerecord."
            ),
            "depth_units": (
                "Native RealSense uint16 depth units; values are not multiplied by depth_scale_m_per_unit."
            ),
            "cameras": list(self.cameras),
            "modalities": list(self.modalities),
            "frame_shape": {"height": self.height, "width": self.width},
            "arrays": self._manifest_arrays(0, 0),
            "calibration": {
                "relative_path": CALIBRATION_RELATIVE_PATH,
                "sha256": self._calibration_sha256,
            },
            "writer_config": {
                "chunk_frames": self.chunk_frames,
                "queue_capacity_frames": self.queue_capacity_frames,
                "compressor": self.compressor_config,
                "max_queue_depth": 0,
            },
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        atomic_write_json(self.manifest_path, self._manifest)

    def _resume_existing(self) -> None:
        manifest = load_manifest(self.dataset_root, require_complete=False)
        if manifest["status"] == "corrupt":
            raise RgbdSidecarError(
                "Refusing to resume a sidecar marked corrupt. Preserve it for diagnosis and start a new dataset."
            )
        if manifest["relative_path"] != self.relative_path:
            raise RgbdSidecarError(
                f"Resume relative_path mismatch: manifest={manifest['relative_path']!r} "
                f"config={self.relative_path!r}"
            )
        if tuple(manifest.get("cameras", [])) != self.cameras:
            raise RgbdSidecarError("Resume camera list conflicts with the existing manifest.")
        if tuple(manifest.get("modalities", [])) != self.modalities:
            raise RgbdSidecarError("Resume modality list conflicts with the existing manifest.")
        if manifest.get("frame_shape") != {"height": self.height, "width": self.width}:
            raise RgbdSidecarError("Resume frame shape conflicts with the existing manifest.")
        existing_writer_config = manifest.get("writer_config", {})
        if int(existing_writer_config.get("chunk_frames", -1)) != self.chunk_frames:
            raise RgbdSidecarError("Resume chunk_frames conflicts with the existing manifest.")
        if existing_writer_config.get("compressor") != self.compressor_config:
            raise RgbdSidecarError("Resume compressor config conflicts with the existing manifest.")
        if manifest.get("calibration", {}).get("sha256") != self._calibration_sha256:
            raise RgbdSidecarError("Resume calibration SHA-256 conflicts with the existing manifest.")

        committed_frames = int(manifest["committed_frames"])
        committed_episodes = int(manifest["committed_episodes"])
        info = json.loads((self.dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
        info_frames = int(info.get("total_frames", -1))
        info_episodes = int(info.get("total_episodes", -1))
        rows = parquet_row_count(self.dataset_root)
        if (info_frames, rows) != (committed_frames, committed_frames):
            raise RgbdSidecarError(
                "Cannot safely resume: the LeRobot committed prefix disagrees with the sidecar ledger. "
                f"manifest committed_frames={committed_frames}, info.total_frames={info_frames}, "
                f"parquet_rows={rows}. Do not guess a commit; preserve the dataset for recovery diagnosis."
            )
        if info_episodes != committed_episodes:
            raise RgbdSidecarError(
                "Cannot safely resume: manifest committed_episodes="
                f"{committed_episodes}, info.total_episodes={info_episodes}."
            )

        zarr = _import_zarr_v2()
        if not self.store_path.is_dir():
            raise RgbdSidecarError(f"Cannot resume: missing Zarr v2 store {self.store_path}.")
        self._group = zarr.open_group(str(self.store_path), mode="a")
        for key in ("schema_name", "schema_version", "storage", "relative_path"):
            if self._group.attrs.get(key) != manifest[key]:
                raise RgbdSidecarError(f"Resume Zarr root attr conflict for {key!r}.")
        expected_paths = {*self._specs, EPISODE_ENDS_PATH}
        if set(manifest["arrays"]) != expected_paths:
            raise RgbdSidecarError(
                "Resume array path set conflicts with the expected schema: "
                f"manifest={sorted(manifest['arrays'])} expected={sorted(expected_paths)}"
            )
        for path in expected_paths:
            key = path.lstrip("/")
            if key not in self._group:
                raise RgbdSidecarError(f"Resume Zarr store is missing {path}.")
            array = self._group[key]
            declared = manifest["arrays"][path]
            expected_dtype = (
                np.dtype(np.int64) if path == EPISODE_ENDS_PATH else self._specs[path][0]
            )
            if np.dtype(declared["dtype"]) != expected_dtype or np.dtype(array.dtype) != expected_dtype:
                raise RgbdSidecarError(f"Resume dtype mismatch for {path}.")
            expected_tail = () if path == EPISODE_ENDS_PATH else self._specs[path][1]
            if tuple(array.shape[1:]) != expected_tail:
                raise RgbdSidecarError(
                    f"Resume shape mismatch for {path}: {array.shape[1:]} != {expected_tail}."
                )
            if tuple(array.chunks) != tuple(declared["chunks"]):
                raise RgbdSidecarError(f"Resume chunks mismatch for {path}.")
            observed_compressor = array.compressor.get_config() if array.compressor is not None else None
            if observed_compressor != declared.get("compressor"):
                raise RgbdSidecarError(f"Resume compressor mismatch for {path}.")
            prefix = committed_episodes if path == EPISODE_ENDS_PATH else committed_frames
            if tuple(int(value) for value in declared.get("shape", [])) != (prefix, *expected_tail):
                raise RgbdSidecarError(f"Resume committed manifest shape mismatch for {path}.")
            if int(array.shape[0]) < prefix:
                raise RgbdSidecarError(
                    f"Cannot resume: {path} length={array.shape[0]} is shorter than committed prefix={prefix}."
                )
            if int(array.shape[0]) > prefix:
                array.resize((prefix, *array.shape[1:]))

        self._manifest = manifest
        self._written_frames = committed_frames
        if committed_frames:
            self._last_global_frame_index = int(
                self._group["meta/global_frame_index"][committed_frames - 1]
            )
        self._validate_parquet_join(0, committed_frames)
        self._manifest["status"] = "in_progress"
        self._manifest["updated_at"] = utc_now_iso()
        self._manifest["completed_at"] = None
        self._manifest["resumed_at"] = self._manifest["updated_at"]
        self._manifest["arrays"] = self._manifest_arrays(committed_frames, committed_episodes)
        atomic_write_json(self.manifest_path, self._manifest)

    def _check_health(self) -> None:
        if self._closed:
            raise RgbdSidecarError("RGB-D sidecar writer is closed.")
        if self._worker_error is not None:
            raise RgbdSidecarError(
                f"RGB-D sidecar background writer failed: {self._worker_error}"
            ) from self._worker_error
        if hasattr(self, "_thread") and not self._thread.is_alive():
            raise RgbdSidecarError("RGB-D sidecar background writer stopped unexpectedly.")

    @staticmethod
    def _scalar(value: Any, key: str) -> Any:
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise RgbdSidecarError(f"{key} must contain exactly one scalar, got shape={array.shape}.")
        return array[0].item()

    def _snapshot_array(self, value: Any, dtype: np.dtype, shape: tuple[int, ...], key: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != dtype:
            raise RgbdSidecarError(f"{key} dtype={array.dtype}, expected {dtype}.")
        if tuple(array.shape) != shape:
            raise RgbdSidecarError(f"{key} shape={array.shape}, expected {shape}.")
        return np.array(array, copy=True, order="C")

    def add_frame(self, *, observation: Mapping[str, Any], frame: Mapping[str, Any]) -> None:
        """Snapshot and enqueue one frame immediately before LeRobot buffers it."""

        self._check_health()
        row = self.committed_frames + self._active_frames
        episode = self.committed_episodes
        frame_index = self._active_frames
        values: dict[str, Any] = {
            "/meta/index": np.int64(row),
            "/meta/episode_index": np.int64(episode),
            "/meta/frame_index": np.int64(frame_index),
        }

        for key, path, dtype in (
            ("global_frame_index", "/meta/global_frame_index", np.int64),
            ("robot_timestamp", "/meta/robot_timestamp", np.float64),
        ):
            if key not in observation or key not in frame:
                raise RgbdSidecarError(f"Zarr sidecar join requires scalar field {key!r}.")
            raw_value = self._scalar(observation[key], key)
            parquet_value = self._scalar(frame[key], key)
            if raw_value != parquet_value:
                raise RgbdSidecarError(
                    f"Sidecar/Parquet join value differs before enqueue for {key}: "
                    f"observation={raw_value} frame={parquet_value}"
                )
            values[path] = dtype(raw_value)

        global_index = int(values["/meta/global_frame_index"])
        if self._last_global_frame_index is not None and global_index <= self._last_global_frame_index:
            raise RgbdSidecarError(
                "global_frame_index must be strictly increasing (gaps are allowed): "
                f"previous={self._last_global_frame_index} current={global_index}"
            )

        for camera in self.cameras:
            for modality in self.modalities:
                observation_key = f"sidecar.{camera}_{modality}"
                if observation_key not in observation:
                    raise RgbdSidecarError(f"Observation is missing {observation_key}.")
                dtype = np.dtype(np.uint16 if modality == "depth" else np.uint8)
                values[f"/data/{camera}/{modality}"] = self._snapshot_array(
                    observation[observation_key], dtype, (self.height, self.width), observation_key
                )

            timestamp_key = f"{camera}_rgbd_timestamp"
            reused_key = f"{camera}_rgbd_reused"
            if timestamp_key not in observation or timestamp_key not in frame:
                raise RgbdSidecarError(f"Zarr sidecar join requires {timestamp_key}.")
            if reused_key not in observation or reused_key not in frame:
                raise RgbdSidecarError(f"Zarr sidecar join requires {reused_key}.")
            timestamp = float(self._scalar(observation[timestamp_key], timestamp_key))
            parquet_timestamp = float(self._scalar(frame[timestamp_key], timestamp_key))
            reused = bool(self._scalar(observation[reused_key], reused_key))
            parquet_reused = bool(self._scalar(frame[reused_key], reused_key))
            if timestamp != parquet_timestamp or reused != parquet_reused:
                raise RgbdSidecarError(
                    f"Sidecar/Parquet join value differs before enqueue for camera={camera}."
                )
            previous_timestamp = self._last_camera_timestamps.get(camera)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise RgbdSidecarError(
                    f"{timestamp_key} must be nondecreasing: previous={previous_timestamp} current={timestamp}"
                )
            values[f"/data/{camera}/rgbd_timestamp"] = np.float64(timestamp)
            values[f"/data/{camera}/rgbd_reused"] = np.bool_(reused)

        try:
            self._queue.put_nowait(_FrameRecord(values))
        except queue.Full as exc:
            raise RgbdSidecarError(
                "RGB-D sidecar queue is full; refusing to drop a frame. "
                f"capacity={self.queue_capacity_frames} active_episode_frames={self._active_frames}"
            ) from exc
        self._active_frames += 1
        self._last_global_frame_index = global_index
        for camera in self.cameras:
            self._last_camera_timestamps[camera] = float(
                values[f"/data/{camera}/rgbd_timestamp"]
            )
        self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        self._check_health()

    def _flush_records(self, records: list[_FrameRecord]) -> None:
        if not records:
            return
        start = self._written_frames
        end = start + len(records)
        for path, (dtype, tail_shape) in self._specs.items():
            array = self._group[path.lstrip("/")]
            if int(array.shape[0]) != start:
                raise RgbdSidecarError(
                    f"Writer order mismatch for {path}: length={array.shape[0]} expected={start}."
                )
            array.resize((end, *tail_shape))
            if tail_shape:
                batch = np.stack([record.values[path] for record in records], axis=0)
            else:
                batch = np.asarray([record.values[path] for record in records], dtype=dtype)
            if batch.dtype != dtype or tuple(batch.shape) != (len(records), *tail_shape):
                raise RgbdSidecarError(
                    f"Writer batch mismatch for {path}: dtype={batch.dtype} shape={batch.shape}."
                )
            array[start:end] = batch
        self._written_frames = end

    def _worker_main(self) -> None:
        buffered: list[_FrameRecord] = []
        current_command: _WriterCommand | None = None
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, _FrameRecord):
                    buffered.append(item)
                    if len(buffered) >= self.chunk_frames:
                        self._flush_records(buffered)
                        buffered.clear()
                    continue

                current_command = item
                if buffered:
                    self._flush_records(buffered)
                    buffered.clear()
                if item.kind == "drain":
                    pass
                elif item.kind == "truncate":
                    target = int(item.value or 0)
                    for path in self._specs:
                        array = self._group[path.lstrip("/")]
                        if int(array.shape[0]) < target:
                            raise RgbdSidecarError(
                                f"Cannot truncate {path} to {target}; current length={array.shape[0]}."
                            )
                        array.resize((target, *array.shape[1:]))
                    episode_ends = self._group[EPISODE_ENDS_PATH.lstrip("/")]
                    episode_ends.resize((self.committed_episodes,))
                    self._written_frames = target
                elif item.kind == "episode_end":
                    episode_end = int(item.value or 0)
                    array = self._group[EPISODE_ENDS_PATH.lstrip("/")]
                    expected_length = self.committed_episodes
                    if int(array.shape[0]) != expected_length:
                        raise RgbdSidecarError(
                            f"episode_ends length={array.shape[0]}, expected={expected_length}."
                        )
                    array.resize((expected_length + 1,))
                    array[expected_length] = episode_end
                elif item.kind == "stop":
                    item.event.set()
                    return
                else:
                    raise RgbdSidecarError(f"Unknown RGB-D writer command: {item.kind}")
                item.event.set()
                current_command = None
        except BaseException as exc:  # noqa: BLE001
            self._worker_error = exc
            if current_command is not None:
                current_command.error = exc
                current_command.event.set()

    def _command(self, kind: str, value: int | None = None) -> None:
        self._check_health()
        command = _WriterCommand(kind=kind, value=value)
        while True:
            self._check_health()
            try:
                self._queue.put(command, timeout=0.1)
                break
            except queue.Full:
                continue
        while not command.event.wait(timeout=0.1):
            self._check_health()
        if command.error is not None:
            raise RgbdSidecarError(
                f"RGB-D sidecar command {kind!r} failed: {command.error}"
            ) from command.error
        if kind != "stop":
            self._check_health()

    def drain(self) -> None:
        self._command("drain")

    def prepare_episode(self, expected_frames: int) -> None:
        if expected_frames != self._active_frames:
            raise RgbdSidecarError(
                f"Episode frame mismatch before drain: LeRobot={expected_frames} sidecar={self._active_frames}."
            )
        if expected_frames <= 0:
            raise RgbdSidecarError("Cannot commit an empty RGB-D sidecar episode.")
        self.drain()
        expected_length = self.committed_frames + expected_frames
        if self._written_frames != expected_length:
            raise RgbdSidecarError(
                f"Sidecar drain ended at {self._written_frames}, expected {expected_length}."
            )
        for path in self._specs:
            length = int(self._group[path.lstrip("/")].shape[0])
            if length != expected_length:
                raise RgbdSidecarError(
                    f"Sidecar drain length mismatch for {path}: {length} != {expected_length}."
                )

    def _validate_parquet_join(self, start: int, end: int) -> None:
        parquet = read_parquet_join_range(self.dataset_root, start, end)
        zarr_values = {
            "index": np.asarray(self._group["meta/index"][start:end]),
            "episode_index": np.asarray(self._group["meta/episode_index"][start:end]),
            "frame_index": np.asarray(self._group["meta/frame_index"][start:end]),
            "global_frame_index": np.asarray(self._group["meta/global_frame_index"][start:end]),
            "robot_timestamp": np.asarray(self._group["meta/robot_timestamp"][start:end]),
            **{
                f"{camera}_rgbd_timestamp": np.asarray(
                    self._group[f"data/{camera}/rgbd_timestamp"][start:end]
                )
                for camera in self.cameras
            },
            **{
                f"{camera}_rgbd_reused": np.asarray(
                    self._group[f"data/{camera}/rgbd_reused"][start:end]
                )
                for camera in self.cameras
            },
        }
        expected_index = np.arange(start, end, dtype=np.int64)
        if not np.array_equal(parquet["index"], expected_index):
            raise RgbdSidecarError(
                f"Parquet index does not equal row ordinal over [{start}, {end})."
            )
        for key, values in zarr_values.items():
            if not np.array_equal(parquet[key], values):
                raise RgbdSidecarError(
                    f"Parquet/Zarr join mismatch for {key} over rows [{start}, {end})."
                )

    def commit_episode(self, *, info_total_frames: int, info_total_episodes: int) -> None:
        candidate_frames = self.committed_frames + self._active_frames
        candidate_episodes = self.committed_episodes + 1
        if int(info_total_frames) != candidate_frames or int(info_total_episodes) != candidate_episodes:
            raise RgbdSidecarError(
                "LeRobot metadata was not durably advanced to the sidecar candidate prefix: "
                f"candidate=({candidate_frames} frames, {candidate_episodes} episodes) "
                f"info=({info_total_frames} frames, {info_total_episodes} episodes)."
            )
        rows = parquet_row_count(self.dataset_root)
        if rows != candidate_frames:
            raise RgbdSidecarError(
                f"Parquet row count={rows}, expected candidate committed_frames={candidate_frames}."
            )
        self._validate_parquet_join(self.committed_frames, candidate_frames)
        self._command("episode_end", candidate_frames)

        self._manifest["committed_frames"] = candidate_frames
        self._manifest["committed_episodes"] = candidate_episodes
        self._manifest["arrays"] = self._manifest_arrays(candidate_frames, candidate_episodes)
        self._manifest["writer_config"]["max_queue_depth"] = self._max_queue_depth
        self._manifest["updated_at"] = utc_now_iso()
        atomic_write_json(self.manifest_path, self._manifest)
        self._active_frames = 0
        self._last_camera_timestamps.clear()

    def rollback_episode(self) -> None:
        self.drain()
        self._command("truncate", self.committed_frames)
        self._active_frames = 0
        if self.committed_frames:
            self._last_global_frame_index = int(
                self._group["meta/global_frame_index"][self.committed_frames - 1]
            )
        else:
            self._last_global_frame_index = None
        self._last_camera_timestamps.clear()

    def finalize(self, *, info_total_frames: int, info_total_episodes: int) -> None:
        if self._active_frames:
            raise RgbdSidecarError(
                f"Cannot finalize with {self._active_frames} uncommitted sidecar frame(s)."
            )
        if self.committed_frames == 0 or self.committed_episodes == 0:
            raise RgbdSidecarError("Refusing to mark an empty RGB-D sidecar complete.")
        self.drain()
        if (
            int(info_total_frames) != self.committed_frames
            or int(info_total_episodes) != self.committed_episodes
            or parquet_row_count(self.dataset_root) != self.committed_frames
        ):
            raise RgbdSidecarError(
                "Cannot mark sidecar complete because manifest/info/Parquet counts disagree."
            )
        for path in self._specs:
            if int(self._group[path.lstrip("/")].shape[0]) != self.committed_frames:
                raise RgbdSidecarError(f"Cannot finalize: {path} length mismatch.")
        if int(self._group[EPISODE_ENDS_PATH.lstrip("/")].shape[0]) != self.committed_episodes:
            raise RgbdSidecarError("Cannot finalize: episode_ends length mismatch.")
        self._command("stop")
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RgbdSidecarError("RGB-D sidecar writer did not stop during finalize.")
        self._manifest["status"] = "complete"
        self._manifest["arrays"] = self._manifest_arrays(
            self.committed_frames, self.committed_episodes
        )
        self._manifest["writer_config"]["max_queue_depth"] = self._max_queue_depth
        self._manifest["updated_at"] = utc_now_iso()
        self._manifest["completed_at"] = self._manifest["updated_at"]
        atomic_write_json(self.manifest_path, self._manifest)
        self._closed = True

    def abort(self, reason: str, *, corrupt: bool = False) -> None:
        if self._closed:
            return
        worker_failed = self._worker_error is not None
        if hasattr(self, "_thread") and self._thread.is_alive() and not worker_failed:
            try:
                self._command("stop")
                self._thread.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                worker_failed = True
        self._manifest["status"] = "corrupt" if corrupt or worker_failed else "incomplete"
        self._manifest["failure_reason"] = str(reason)
        self._manifest["updated_at"] = utc_now_iso()
        self._manifest["writer_config"]["max_queue_depth"] = self._max_queue_depth
        atomic_write_json(self.manifest_path, self._manifest)
        self._closed = True
