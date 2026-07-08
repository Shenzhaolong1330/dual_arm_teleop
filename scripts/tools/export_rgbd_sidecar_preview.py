#!/usr/bin/env python
"""Export RGB/depth/IR preview PNGs for one LeRobot RGB-D sidecar frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.tools.rgbd_sidecar_export_utils import (
    CAMERAS,
    camera_rgb_key,
    default_output_dir,
    depth_to_colormap,
    global_index_for_episode_frame,
    grayscale_to_uint8,
    require_keys,
    resolve_dataset,
    rgb_to_uint8_hwc,
    sanitize_name,
    scalar_or_none,
    sidecar_keys,
    write_json,
    write_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="LeRobot dataset repo_id.")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root, or a base root containing repo_id.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to export.")
    parser.add_argument("--frame-index", type=int, required=True, help="Frame index within the episode.")
    parser.add_argument("--camera", choices=CAMERAS, default="head", help="Camera to export.")
    parser.add_argument(
        "--rgb-camera-name-mode",
        choices=("rgb", "legacy_image"),
        default="rgb",
        help="RGB field naming used by the dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for exported files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args.repo_id, args.root)
    dataset._ensure_hf_dataset_loaded()

    rgb_key = camera_rgb_key(args.camera, args.rgb_camera_name_mode)
    sidecar = sidecar_keys(args.camera)
    require_keys(dataset.features, [rgb_key, sidecar["depth"], sidecar["left_ir"], sidecar["right_ir"]])

    dataset_index = global_index_for_episode_frame(dataset, args.episode, args.frame_index)
    sample = dataset[dataset_index]
    raw_sample = dataset.hf_dataset[dataset_index]

    out_dir = args.output_dir or default_output_dir(dataset, "rgbd_sidecar_preview")
    prefix = (
        f"{sanitize_name(dataset.repo_id)}"
        f"_ep{args.episode:06d}_frame{args.frame_index:06d}_{args.camera}"
    )

    paths = {
        "rgb": out_dir / f"{prefix}_rgb.png",
        "depth_colormap": out_dir / f"{prefix}_depth_colormap.png",
        "left_ir": out_dir / f"{prefix}_left_ir.png",
        "right_ir": out_dir / f"{prefix}_right_ir.png",
        "metadata": out_dir / f"{prefix}_metadata.json",
    }

    write_png(paths["rgb"], rgb_to_uint8_hwc(sample[rgb_key]))
    write_png(paths["depth_colormap"], depth_to_colormap(raw_sample[sidecar["depth"]]))
    write_png(paths["left_ir"], grayscale_to_uint8(raw_sample[sidecar["left_ir"]]))
    write_png(paths["right_ir"], grayscale_to_uint8(raw_sample[sidecar["right_ir"]]))

    metadata = {
        "repo_id": dataset.repo_id,
        "dataset_root": str(dataset.root),
        "episode_index": args.episode,
        "episode_frame_index": args.frame_index,
        "dataset_index": dataset_index,
        "camera": args.camera,
        "rgb_key": rgb_key,
        "depth_key": sidecar["depth"],
        "left_ir_key": sidecar["left_ir"],
        "right_ir_key": sidecar["right_ir"],
        "global_frame_index": scalar_or_none(raw_sample.get("global_frame_index")),
        "robot_timestamp": scalar_or_none(raw_sample.get("robot_timestamp")),
        "rgbd_timestamp": scalar_or_none(raw_sample.get(f"{args.camera}_rgbd_timestamp")),
        "rgbd_reused": scalar_or_none(raw_sample.get(f"{args.camera}_rgbd_reused")),
        "outputs": {name: str(path) for name, path in paths.items() if name != "metadata"},
    }
    write_json(paths["metadata"], metadata)

    print("Exported RGB-D sidecar preview:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
