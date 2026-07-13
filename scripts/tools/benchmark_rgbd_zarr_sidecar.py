#!/usr/bin/env python
"""Hardware-free sustained benchmark for the raw RealSense Zarr v2 sidecar."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.core.rgbd_zarr_sidecar import CAMERAS, ZarrSidecarWriter


RAW_INPUT_MIB_S_640X480_30FPS = 105.46875


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


def _write_calibration(root: Path, height: int, width: int) -> None:
    streams = {
        stream: {"height": height, "width": width}
        for stream in ("depth", "infrared1", "infrared2")
    }
    payload = {
        "schema_version": 1,
        "cameras": {f"{camera}_rgb": {"streams": streams} for camera in CAMERAS},
    }
    path = root / "meta" / "realsense_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _pool(height: int, width: int, frames: int, seed: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    values: list[dict[str, np.ndarray]] = []
    for _ in range(frames):
        frame: dict[str, np.ndarray] = {}
        for camera in CAMERAS:
            frame[f"sidecar.{camera}_depth"] = rng.integers(
                0, 65536, size=(height, width), dtype=np.uint16
            )
            frame[f"sidecar.{camera}_left_ir"] = rng.integers(
                0, 256, size=(height, width), dtype=np.uint8
            )
            frame[f"sidecar.{camera}_right_ir"] = rng.integers(
                0, 256, size=(height, width), dtype=np.uint8
            )
        values.append(frame)
    return values


def run_benchmark(args: argparse.Namespace) -> dict:
    temporary = args.root is None
    root = Path(args.root) if args.root is not None else Path(tempfile.mkdtemp(prefix="rgbd_zarr_bench_"))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Benchmark root must be empty: {root}")
    if temporary:
        # LeRobotDataset.create intentionally requires a non-existent root.
        root.rmdir()
    dataset = LeRobotDataset.create(
        repo_id="benchmark/rgbd_zarr",
        fps=int(args.fps),
        features=_features(),
        robot_type="flexiv_dual_arm",
        root=root,
        use_videos=False,
    )
    dataset.meta.metadata_buffer_size = 1
    _write_calibration(root, args.height, args.width)
    writer = ZarrSidecarWriter(
        root,
        height=args.height,
        width=args.width,
        chunk_frames=args.chunk_frames,
        queue_capacity_frames=args.queue_capacity_frames,
        compressor={
            "codec": "blosc",
            "cname": args.cname,
            "clevel": args.clevel,
            "shuffle": args.shuffle,
        },
    )
    source_pool = _pool(
        args.height,
        args.width,
        max(args.chunk_frames, args.pool_frames),
        args.seed,
    )
    bytes_per_frame = len(CAMERAS) * args.height * args.width * (2 + 1 + 1)
    raw_mib = bytes_per_frame * args.frames / (1024**2)
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    started = time.perf_counter()
    next_deadline = started
    for index in range(args.frames):
        if args.fps > 0:
            next_deadline = started + index / args.fps
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        sensor = source_pool[index % len(source_pool)]
        observation = {
            **sensor,
            "state": float(index),
            "global_frame_index": index,
            "robot_timestamp": 1000.0 + index / max(args.fps, 1),
        }
        frame = {
            "observation.state": np.array([index], dtype=np.float32),
            "action": np.array([index], dtype=np.float32),
            "global_frame_index": np.array([index], dtype=np.int64),
            "robot_timestamp": np.array([observation["robot_timestamp"]], dtype=np.float64),
            "task": "synthetic benchmark",
        }
        for camera in CAMERAS:
            timestamp = 2000.0 + index / max(args.fps, 1)
            observation[f"{camera}_rgbd_timestamp"] = timestamp
            observation[f"{camera}_rgbd_reused"] = False
            frame[f"{camera}_rgbd_timestamp"] = np.array([timestamp], dtype=np.float64)
            frame[f"{camera}_rgbd_reused"] = np.array([False], dtype=np.bool_)
        writer.add_frame(observation=observation, frame=frame)
        dataset.add_frame(frame)

    commit_started = time.perf_counter()
    writer.prepare_episode(int(dataset.episode_buffer["size"]))
    write_elapsed_s = time.perf_counter() - started
    dataset.save_episode()
    dataset.seal_episode_writers()
    writer.commit_episode(
        info_total_frames=dataset.meta.total_frames,
        info_total_episodes=dataset.meta.total_episodes,
    )
    episode_commit_s = time.perf_counter() - commit_started

    finalize_started = time.perf_counter()
    dataset.finalize()
    writer.finalize(
        info_total_frames=dataset.meta.total_frames,
        info_total_episodes=dataset.meta.total_episodes,
    )
    finalize_s = time.perf_counter() - finalize_started
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result = {
        "root": str(root),
        "frames": args.frames,
        "frame_shape": [args.height, args.width],
        "target_fps": args.fps,
        "raw_input_mib": raw_mib,
        "write_elapsed_s": write_elapsed_s,
        "write_mib_s": raw_mib / write_elapsed_s,
        "required_mib_s_640x480_30fps": RAW_INPUT_MIB_S_640X480_30FPS,
        "keeps_up_with_640x480_30fps": raw_mib / write_elapsed_s >= RAW_INPUT_MIB_S_640X480_30FPS,
        "max_queue_depth_frames": writer.max_queue_depth,
        "queue_capacity_frames": args.queue_capacity_frames,
        "chunk_frames": args.chunk_frames,
        "compressor": {
            "codec": "blosc",
            "cname": args.cname,
            "clevel": args.clevel,
            "shuffle": args.shuffle,
        },
        "peak_rss_mib": peak_rss_mib,
        "peak_rss_increase_mib": max(0.0, peak_rss_mib - rss_before_kib / 1024.0),
        "episode_commit_s": episode_commit_s,
        "finalize_s": finalize_s,
    }
    if temporary and not args.keep_root:
        shutil.rmtree(root)
        result["root_removed"] = True
    else:
        result["root_removed"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Optional empty output root.")
    parser.add_argument("--keep-root", action="store_true", help="Keep a temporary benchmark dataset.")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=float, default=30.0, help="Producer rate; <=0 disables pacing.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--queue-capacity-frames", type=int, default=64)
    parser.add_argument("--pool-frames", type=int, default=8)
    parser.add_argument("--cname", default="lz4")
    parser.add_argument("--clevel", type=int, default=1)
    parser.add_argument("--shuffle", choices=("none", "shuffle", "bitshuffle"), default="bitshuffle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.frames < 1 or args.width < 1 or args.height < 1:
        parser.error("frames, width, and height must be positive")
    return args


def main() -> None:
    result = run_benchmark(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
