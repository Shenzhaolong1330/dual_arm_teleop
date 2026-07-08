#!/usr/bin/env python
"""Export one left/right IR PNG pair for Fast-FoundationStereo demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.tools.rgbd_sidecar_export_utils import (
    CAMERAS,
    default_output_dir,
    global_index_for_episode_frame,
    grayscale_to_uint8,
    require_keys,
    resolve_dataset,
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
    parser.add_argument("--camera", choices=CAMERAS, default="head", help="Camera whose IR pair is exported.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for exported files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args.repo_id, args.root)
    dataset._ensure_hf_dataset_loaded()

    sidecar = sidecar_keys(args.camera)
    require_keys(dataset.features, [sidecar["left_ir"], sidecar["right_ir"]])

    dataset_index = global_index_for_episode_frame(dataset, args.episode, args.frame_index)
    raw_sample = dataset.hf_dataset[dataset_index]

    out_dir = args.output_dir or default_output_dir(dataset, "ffs_stereo_pair")
    prefix = (
        f"{sanitize_name(dataset.repo_id)}"
        f"_ep{args.episode:06d}_frame{args.frame_index:06d}_{args.camera}"
    )
    left_path = out_dir / f"{prefix}_left_ir.png"
    right_path = out_dir / f"{prefix}_right_ir.png"
    metadata_path = out_dir / f"{prefix}_metadata.json"

    write_png(left_path, grayscale_to_uint8(raw_sample[sidecar["left_ir"]]))
    write_png(right_path, grayscale_to_uint8(raw_sample[sidecar["right_ir"]]))

    metadata = {
        "repo_id": dataset.repo_id,
        "dataset_root": str(dataset.root),
        "episode_index": args.episode,
        "episode_frame_index": args.frame_index,
        "dataset_index": dataset_index,
        "camera": args.camera,
        "left_ir_key": sidecar["left_ir"],
        "right_ir_key": sidecar["right_ir"],
        "left_ir_png": str(left_path),
        "right_ir_png": str(right_path),
        "global_frame_index": scalar_or_none(raw_sample.get("global_frame_index")),
        "robot_timestamp": scalar_or_none(raw_sample.get("robot_timestamp")),
        "rgbd_timestamp": scalar_or_none(raw_sample.get(f"{args.camera}_rgbd_timestamp")),
        "rgbd_reused": scalar_or_none(raw_sample.get(f"{args.camera}_rgbd_reused")),
        "note": "IR images are exported as stored sidecar frames; no rectification is applied here.",
    }
    write_json(metadata_path, metadata)

    print("Exported FFS stereo pair:")
    print(f"  left_ir: {left_path}")
    print(f"  right_ir: {right_path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
