"""Copy PRIMEU v2.1 to native LeRobot v3 without changing numerical labels/videos."""
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from examples.primeu.contract import load_dataset_contract
import copy
import hashlib
import json
from pathlib import Path
import shutil

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, flatten_dict,
    write_episodes, write_info, write_stats, write_tasks,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vector_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite numerical data")
    return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0),
                q01=np.quantile(x, .01, axis=0), q99=np.quantile(x, .99, axis=0),
                count=np.array([len(x)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="YAML with source and output; relative paths resolve beside the YAML")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--video-source-root", type=Path,
                   help="Explicit original v2.1 dataset root when reduced-data video links are broken")
    args = p.parse_args()
    config = {}
    if args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text()) or {}
        if not isinstance(config, dict) or set(config) - {"source", "output", "video_source_root"}:
            raise ValueError("Preparation YAML accepts source, output, and optional video_source_root only")
    paths = {}
    for name in ("source", "output", "video_source_root"):
        value = getattr(args, name)
        if value is None:
            if not config.get(name):
                if name == "video_source_root":
                    paths[name] = None
                    continue
                p.error(f"Specify --{name} or {name} in --config")
            value = Path(config[name]).expanduser()
            if not value.is_absolute():
                value = args.config.resolve().parent / value
        paths[name] = value.expanduser().resolve()
    convert(paths["source"], paths["output"], video_source_root=paths["video_source_root"])


def episode_path(root, info, episode_index, *, camera=None):
    values = {"episode_chunk": episode_index // info["chunks_size"], "episode_index": episode_index}
    if camera is not None:
        values["video_key"] = camera
    return root / info["video_path" if camera is not None else "data_path"].format(**values)


def verify_video_origin_table(table, original, episode_index):
    """A video override is valid only for the same recorded physical trajectory."""
    # Derived datasets can legitimately relabel task text/index; physical frame
    # identity must still match exactly. Annotation differences are reported.
    columns = ("observation.state", "frame_index", "timestamp", "episode_index")
    if len(table) != len(original):
        raise ValueError(f"Episode {episode_index}: video-origin row count mismatch")
    for key in columns:
        if key not in original.column_names or not np.array_equal(
            np.asarray(table[key].to_pylist()), np.asarray(original[key].to_pylist())
        ):
            raise ValueError(f"Episode {episode_index}: video-origin {key} differs from source; refusing guessed alignment")


def convert(source, output, video_source_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    temporary = output.with_name(output.name + ".partial")
    if output == source or output.is_relative_to(source) or output.exists() or temporary.exists():
        raise ValueError("Use a new output directory outside the source; output/partial must not exist")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("This converter expects PRIMEU LeRobot v2.1")
    episodes = [json.loads(l) for l in (source / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l) for l in (source / "meta/tasks.jsonl").read_text().splitlines()]
    episodes.sort(key=lambda e: e["episode_index"])
    ids = [e["episode_index"] for e in episodes]
    if len(ids) != len(set(ids)) or len(ids) != info["total_episodes"]:
        raise ValueError("Duplicate/missing episode metadata")
    if sorted(t["task_index"] for t in tasks) != list(range(len(tasks))):
        raise ValueError("Task indices must be contiguous")
    task_map = {t["task_index"]: t["task"] for t in tasks}
    modality = json.loads((source / "meta/modality.json").read_text())
    contract = load_dataset_contract(source)
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    dimensions = {"observation.state": contract.state_dim, "action": contract.action_dim}
    video_root = Path(video_source_root).resolve() if video_source_root is not None else source
    video_info = info
    video_metadata_hashes = {}
    original_tasks = task_map
    if video_source_root is not None:
        if output == video_root or output.is_relative_to(video_root):
            raise ValueError("Output must be outside the original video dataset")
        video_info = json.loads((video_root / "meta/info.json").read_text())
        if video_info["codebase_version"] != "v2.1" or video_info["fps"] != info["fps"]:
            raise ValueError("Video source must be a v2.1 dataset with matching FPS")
        original_episodes = [json.loads(line) for line in (video_root / "meta/episodes.jsonl").read_text().splitlines()]
        original_episode_map = {episode["episode_index"]: episode for episode in original_episodes}
        if len(original_episode_map) != len(original_episodes):
            raise ValueError("Duplicate video-origin episode indices")
        original_tasks = {item["task_index"]: item["task"] for item in
                          [json.loads(line) for line in (video_root / "meta/tasks.jsonl").read_text().splitlines()]}
        for episode in episodes:
            original = original_episode_map.get(episode["episode_index"])
            if original is None or original["length"] != episode["length"]:
                raise ValueError(f"Episode {episode['episode_index']}: video-origin metadata length mismatch")
        for camera in cameras:
            feature = video_info["features"].get(camera, {})
            if feature.get("dtype") != "video" or feature.get("shape") != info["features"][camera]["shape"]:
                raise ValueError(f"Video-origin camera metadata mismatch: {camera}")
        video_metadata_hashes = {name: sha256(video_root / name) for name in
                                 ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl")}
    # Fail on stale video links before creating an output directory or writing rows.
    for episode in episodes:
        old_id = episode["episode_index"]
        paths = [episode_path(source, info, old_id)]
        if video_source_root is not None:
            paths.append(episode_path(video_root, video_info, old_id))
        paths.extend(episode_path(video_root, video_info, old_id, camera=camera) for camera in cameras)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing source file (possibly a stale symlink): {path}. "
                                        "Set video_source_root explicitly to the matching original dataset if needed.")
    temporary.mkdir(parents=True)
    source_hashes = {str(f.relative_to(source)): sha256(f) for f in (source / "meta").iterdir() if f.is_file()}
    rows, reports, image_stats = [], [], []
    vectors = {"observation.state": [], "action": []}
    offset = 0
    for new_id, ep in enumerate(episodes):
        old_id = ep["episode_index"]
        src = episode_path(source, info, old_id)
        original_hash = sha256(src)
        table = pq.read_table(src)
        video_original_path, video_original_hash = None, None
        task_alignment = None
        if video_source_root is not None:
            video_original_path = episode_path(video_root, video_info, old_id)
            video_original_hash = sha256(video_original_path)
            original_video_table = pq.read_table(video_original_path)
            verify_video_origin_table(table, original_video_table, old_id)
            source_task_values = table["task_index"].to_pylist()
            origin_task_values = original_video_table["task_index"].to_pylist()
            if any(index not in original_tasks for index in origin_task_values):
                raise ValueError(f"Episode {old_id}: invalid task index in original video dataset")
            source_task_texts = [task_map[index] for index in source_task_values]
            origin_task_texts = [original_tasks[index] for index in origin_task_values]
            task_alignment = {
                "task_index_values_equal": source_task_values == origin_task_values,
                "task_text_values_equal": source_task_texts == origin_task_texts,
                "source_task_index_sha256": hashlib.sha256(json.dumps(source_task_values).encode()).hexdigest(),
                "video_origin_task_index_sha256": hashlib.sha256(json.dumps(origin_task_values).encode()).hexdigest(),
                "source_task_texts": sorted(set(source_task_texts)),
                "video_origin_task_texts": sorted(set(origin_task_texts)),
            }
            del original_video_table
        n = len(table)
        if n != ep["length"] or not np.all(table["episode_index"].to_numpy() == old_id):
            raise ValueError(f"Episode {old_id}: length/index mismatch")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(n)):
            raise ValueError(f"Episode {old_id}: non-contiguous frame_index")
        if not np.array_equal(table["index"].to_numpy(), np.arange(offset, offset+n)):
            raise ValueError(f"Episode {old_id}: non-contiguous global index")
        recorded_ts = table["timestamp"].to_numpy()
        if not np.isfinite(recorded_ts).all() or recorded_ts[0] != 0 or np.any(np.diff(recorded_ts) <= 0):
            raise ValueError(f"Episode {old_id}: invalid recording timestamps")
        video_ts = (np.arange(n) / info["fps"]).astype(np.float32)
        task_ids = np.unique(table["task_index"].to_numpy()).tolist()
        actual_tasks = [task_map[i] for i in task_ids]
        if set(actual_tasks) != set(ep["tasks"]):
            raise ValueError(f"Episode {old_id}: task text mismatch")
        stats = {}
        for key in vectors:
            x = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if x.shape != (n, dimensions[key]):
                raise ValueError(f"Episode {old_id}: {key} shape {x.shape}")
            stats[key] = vector_stats(x)
            vectors[key].append(x)
        changed = table.set_column(table.schema.get_field_index("episode_index"), table.schema.field("episode_index"),
                                   pa.array(np.full(n, new_id), type=pa.int64()))
        if "source_timestamp" in table.column_names:
            raise ValueError("source_timestamp already exists")
        changed = changed.append_column("source_timestamp", table["timestamp"])
        changed = changed.set_column(changed.schema.get_field_index("timestamp"), table.schema.field("timestamp"),
                                     pa.array(video_ts, type=pa.float32()))
        dest = temporary / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=new_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(changed, dest)
        loaded = pq.read_table(dest)
        for key in table.column_names:
            if not loaded[key].equals(changed[key]):
                raise ValueError(f"Parquet roundtrip changed {key}")
            if key not in ("episode_index", "timestamp") and not loaded[key].equals(table[key]):
                raise ValueError(f"Unexpected numerical change: {key}")
        if not loaded["source_timestamp"].equals(table["timestamp"]):
            raise ValueError("Original timestamps were not preserved")
        row = dict(episode_index=new_id, tasks=actual_tasks, length=n,
                   dataset_from_index=offset, dataset_to_index=offset+n)
        row.update({"data/chunk_index": 0, "data/file_index": new_id,
                    "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        video_reports = []
        for camera in cameras:
            video = episode_path(video_root, video_info, old_id, camera=camera)
            target = temporary / DEFAULT_VIDEO_PATH.format(video_key=camera, chunk_index=0, file_index=new_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(video)
            shutil.copy2(video, target)
            if sha256(target) != digest or sha256(video) != digest:
                raise ValueError(f"Video copy mismatch: {video}")
            decoder = VideoDecoder(str(target), num_ffmpeg_threads=1)
            if len(decoder) != n:
                raise ValueError(f"{video}: video has {len(decoder)} frames, parquet has {n}")
            indices = sorted(set([0, n//2, n-1]))
            sampled = decoder.get_frames_at(indices=indices)
            if not np.allclose(sampled.pts_seconds.numpy(), np.array(indices)/info["fps"], atol=1e-4):
                raise ValueError(f"Video time alignment mismatch: {video}")
            pixels = sampled.data.numpy()
            stride = max(1, max(pixels.shape[-2:]) // 150)
            x = pixels[:, :, ::stride, ::stride].transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float64) / 255
            stats[camera] = {name: getattr(x, name)(axis=0).reshape(3, 1, 1)
                             for name in ("min", "max", "mean", "std")}
            stats[camera]["count"] = np.array([len(indices)])
            row.update({f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": new_id,
                        f"videos/{camera}/from_timestamp": 0., f"videos/{camera}/to_timestamp": n/info["fps"]})
            video_reports.append(dict(camera=camera, sha256=digest, decoded_indices=indices))
            del decoder
        image_stats.append({k: stats[k] for k in cameras})
        row.update({k: np.asarray(v).tolist() for k, v in flatten_dict({"stats": stats}).items()})
        rows.append(row)
        if sha256(src) != original_hash:
            raise ValueError(f"Source changed while converting: {src}")
        if video_original_path is not None and sha256(video_original_path) != video_original_hash:
            raise ValueError(f"Video-origin Parquet changed while converting: {video_original_path}")
        reports.append(dict(source_episode=old_id, episode_index=new_id, frames=n,
                            timestamp_alignment_max_seconds=float(np.max(np.abs(recorded_ts-video_ts))),
                            source_parquet_sha256=original_hash,
                            video_origin_parquet_sha256=video_original_hash,
                            video_origin_task_alignment=task_alignment, videos=video_reports))
        offset += n
        print(f"Converted {new_id+1}/{len(episodes)} episodes; {offset} frames", flush=True)
    if offset != info["total_frames"]:
        raise ValueError("Total frame count mismatch")
    combined = {k: vector_stats(np.concatenate(v)) for k, v in vectors.items()}
    combined.update(aggregate_stats(image_stats))
    new_info = copy.deepcopy(info)
    new_info.update(codebase_version="v3.0", data_path=DEFAULT_DATA_PATH, video_path=DEFAULT_VIDEO_PATH,
                    fps=int(info["fps"]), data_files_size_in_mb=100, video_files_size_in_mb=500,
                    splits={"train": f"0:{len(episodes)}"})
    new_info["features"]["source_timestamp"] = dict(dtype="float32", shape=[1], names=None)
    for key in ("total_chunks", "total_videos"):
        new_info.pop(key, None)
    for key, feature in new_info["features"].items():
        feature.setdefault("names", None)
        if feature["dtype"] != "video":
            feature["fps"] = new_info["fps"]
    for key, group in (("observation.state", "state"), ("action", "action")):
        names = [None] * dimensions[key]
        for name, bounds in modality[group].items():
            for i in range(bounds["start"], bounds["end"]):
                if names[i] is not None:
                    raise ValueError("Overlapping modality ranges")
                names[i] = f"{name}_{i-bounds['start']}"
        if any(n is None for n in names):
            raise ValueError("Incomplete modality ranges")
        new_info["features"][key]["names"] = names
    write_info(new_info, temporary)
    write_stats(combined, temporary)
    write_tasks(pd.DataFrame({"task_index": [t["task_index"] for t in tasks]},
                            index=[t["task"] for t in tasks]), temporary)
    write_episodes(datasets.Dataset.from_list(rows), temporary)
    metadata_assets = ["modality.json", *contract.basis_files.values()]
    if (source / "meta/action_layout.json").exists():
        metadata_assets.append("action_layout.json")
    for name in metadata_assets:
        shutil.copy2(source / "meta" / name, temporary / "meta" / name)
    output_contract = load_dataset_contract(temporary)
    (temporary / "meta/primeu_contract.json").write_text(json.dumps(output_contract.to_dict(), indent=2) + "\n")
    for name, digest in source_hashes.items():
        if sha256(source / name) != digest:
            raise ValueError(f"Source metadata changed: {name}")
    for name, digest in video_metadata_hashes.items():
        if sha256(video_root / name) != digest:
            raise ValueError(f"Video-origin metadata changed: {name}")
    report = dict(source=str(source), output=str(output), episodes=len(rows), frames=offset,
                  video_source_root=str(video_root), video_source_override=video_source_root is not None,
                  video_origin_metadata_sha256=video_metadata_hashes,
                  source_task_map=task_map, video_origin_task_map=original_tasks,
                  video_origin_task_annotation_policy="Preserve source labels; physical identity does not require original task text/index equality",
                  video_origin_annotation_mismatch_episodes=[item["source_episode"] for item in reports
                      if item["video_origin_task_alignment"] is not None and
                      (not item["video_origin_task_alignment"]["task_index_values_equal"] or
                       not item["video_origin_task_alignment"]["task_text_values_equal"])],
                  video_origin_alignment=("exact observation.state/frame_index/timestamp/episode_index equality"
                                          if video_source_root is not None else "source dataset videos"),
                  changed_columns=["episode_index", "timestamp"], added_columns=["source_timestamp"],
                  timestamp_semantics="timestamp=frame_index/fps for CFR video; original timestamp retained in source_timestamp",
                  original_metadata_sha256=source_hashes,
                  action_semantics="stored labels unchanged; no action-minus-state transform",
                  action_dim=contract.action_dim, state_dim=contract.state_dim,
                  basis_sha256=contract.basis_sha256,
                  output_contract_sha256=output_contract.contract_sha256,
                  deployment_hold_policy="Unpredicted joints use first valid observation after reset, not source reference poses",
                  vector_statistics="all original frames once, float64 population std and exact quantiles",
                  image_statistics="first/middle/last frame per episode, strided pixel sample; training uses ImageNet mean/std",
                  verified_video_copies=len(rows)*len(cameras), episode_mapping=reports)
    (temporary / "meta/primeu_conversion_report.json").write_text(json.dumps(report, indent=2)+"\n")
    temporary.rename(output)
    print(json.dumps({k: report[k] for k in ("output", "episodes", "frames", "verified_video_copies")}, indent=2))


if __name__ == "__main__":
    main()
