#!/usr/bin/env python3
"""Relabel legacy feedback-width actions on closed plateaus; never edit the source."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path.home() / ".cache/huggingface/lerobot/franka_dual_arm/insert_tube_rack_all_E1984"
SEMANTICS_KEY = "gripper_action_semantics"
SEMANTICS = "heuristic_closed_plateau_zero_v1"
GRIPPER_NAMES = ("left_gripper_width", "right_gripper_width")
QUANTILES = {"q01": .01, "q10": .1, "q50": .5, "q90": .9, "q99": .99}
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Parameters:
    open_width: float = .085
    open_tolerance: float = .001
    stable_seconds: float = .5
    stable_range: float = .0005
    closing_drop: float = .002
    opening_rise: float = .002

    def window_frames(self, fps: float) -> int:
        values = asdict(self)
        if not all(math.isfinite(v) for v in (*values.values(), fps)):
            raise ValueError("Parameters and fps must be finite")
        if fps <= 0 or any(values[k] <= 0 for k in values if k != "open_tolerance"):
            raise ValueError("fps, widths, duration and motion thresholds must be positive")
        if not 0 <= self.open_tolerance < self.open_width:
            raise ValueError("open_tolerance must be in [0, open_width)")
        if self.stable_range >= min(self.closing_drop, self.opening_rise):
            raise ValueError("stable_range must be smaller than closing_drop and opening_rise")
        return max(2, math.ceil(self.stable_seconds * fps))


def closed_plateau_mask(widths: np.ndarray, fps: float, params: Parameters) -> np.ndarray:
    """One hand, one episode, original actions only; intervals use exclusive ends.

    Confirm closing with a cumulative drop from the running peak. Confirm opening
    with a cumulative rise from the running trough, backdating the closing end to
    the last occurrence of that trough. Within those spans, mark the union of
    qualifying fixed-duration stable windows (including the first window's frames).
    """
    raw = np.asarray(widths)
    if raw.ndim != 1 or raw.dtype.kind != "f" or not np.isfinite(raw).all():
        raise ValueError("Gripper widths must be a finite one-dimensional float array")
    window = params.window_frames(fps)
    mask = np.zeros(len(raw), dtype=bool)
    if len(raw) < window:
        return mask
    x = raw.astype(np.float64)
    # Compare the strict cutoff in the stored dtype: float32(0.084) is NOT < 84 mm.
    cutoff = float(np.asarray(params.open_width - params.open_tolerance, dtype=raw.dtype))
    epsilon = np.finfo(raw.dtype).eps * params.open_width * 2
    peak, peak_i = x[0], 0
    start = None
    trough, trough_i = x[0], 0
    spans = []
    for i in range(1, len(x)):
        value = x[i]
        if start is None:
            if value >= peak:
                peak, peak_i = value, i
            elif peak - value >= params.closing_drop - epsilon:
                start = peak_i + 1
                trough, trough_i = value, i
        elif value <= trough:
            trough, trough_i = value, i
        elif value - trough >= params.opening_rise - epsilon:
            spans.append((start, trough_i + 1))
            start = None
            # The confirming sample is the new peak: earlier samples since the
            # trough were below the confirmation threshold.
            peak, peak_i = value, i
    if start is not None:
        spans.append((start, len(x)))

    for start, end in spans:
        if end - start < window:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(x[start:end], window)
        hi, lo = windows.max(axis=1), windows.min(axis=1)
        starts = np.flatnonzero((hi < cutoff) & (hi - lo <= params.stable_range + epsilon))
        # Difference-array union avoids repeatedly writing long overlapping windows.
        coverage = np.zeros(end - start + 1, dtype=np.int64)
        np.add.at(coverage, starts, 1)
        np.add.at(coverage, starts + window, -1)
        mask[start:end] = np.cumsum(coverage[:-1]) > 0
    return mask


def intervals(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def action_matrix(column: pa.ChunkedArray, dim: int) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count or not (pa.types.is_fixed_size_list(array.type) or pa.types.is_list(array.type)):
        raise ValueError("Action must be a non-null list column")
    if array.values.null_count or len(array.values) != len(array) * dim:
        raise ValueError("Invalid action vector lengths or null components")
    if pa.types.is_list(array.type) and not np.all(np.diff(array.offsets.to_numpy()) == dim):
        raise ValueError("Action vector lengths do not match the schema")
    values = array.values.to_numpy().reshape(-1, dim)
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("All actions must be finite floating-point values")
    return values


def action_stats(actions: np.ndarray) -> dict:
    """Exact numeric statistics in LeRobot's JSON/episode metadata representation."""
    values = actions.astype(np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot compute action statistics for empty/nonfinite data")
    result = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
              "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
              "count": [len(values)]}
    qs = np.quantile(values, list(QUANTILES.values()), axis=0)
    result.update({key: value.tolist() for key, value in zip(QUANTILES, qs)})
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source must be self-contained, without symlinks: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(root))] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def validate_paths(source: Path, output: Path) -> None:
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-overlapping directories")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; refusing overwrite: {output}")
    stage = output.with_name(output.name + ".relabel-staging")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f"Staging directory already exists; inspect it first: {stage}")


def plan_relabel(source: Path, params: Parameters) -> dict:
    info = read_json(source / "meta/info.json")
    if info.get(SEMANTICS_KEY) is not None:
        raise ValueError("Input must be an unmarked legacy dataset; refusing to relabel command/inferred targets")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("This tool supports LeRobot v3.0 Parquet datasets")
    fps = float(info["fps"])
    window = params.window_frames(fps)
    feature = info["features"]["action"]
    names = feature.get("names") or []
    if len(names) != len(set(names)) or list(feature["shape"]) != [len(names)]:
        raise ValueError("Invalid/duplicate action feature names or shape")
    if any(name not in names for name in GRIPPER_NAMES):
        raise ValueError("Missing left/right_gripper_width action fields")
    columns = [names.index(name) for name in GRIPPER_NAMES]
    total = int(info["total_frames"])
    data_files, pieces, ep_ids, frame_ids = [], [], [], []
    cursor = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["action", "index", "episode_index", "frame_index"])
        n = len(table)
        if not n or not np.array_equal(table["index"].to_numpy(), np.arange(cursor, cursor + n)):
            raise ValueError(f"Non-contiguous global indices in {path}")
        pieces.append(action_matrix(table["action"], len(names)))
        ep_ids.append(table["episode_index"].to_numpy())
        frame_ids.append(table["frame_index"].to_numpy())
        data_files.append((path.relative_to(source), cursor, cursor + n))
        cursor += n
    if cursor != total or not pieces:
        raise ValueError("Parquet frame count does not match info.json")
    actions = np.concatenate(pieces)
    episode_ids, frame_ids = np.concatenate(ep_ids), np.concatenate(frame_ids)
    metadata = [(path.relative_to(source), pq.read_table(path))
                for path in sorted((source / "meta/episodes").rglob("*.parquet"))]
    rows = sorted([row for _, table in metadata for row in table.to_pylist()], key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(int(info["total_episodes"]))):
        raise ValueError("Episode metadata indices must be unique and contiguous from zero")
    relabeled = actions.copy()
    changes, stats, counts, changed_eps = [], {}, dict.fromkeys(GRIPPER_NAMES, 0), {k: set() for k in GRIPPER_NAMES}
    cursor = 0
    for row in rows:
        ep = int(row["episode_index"])
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if start != cursor or not start < end <= total or row["length"] != end - start:
            raise ValueError(f"Invalid episode bounds/length for episode {ep}")
        if not np.all(episode_ids[start:end] == ep) or not np.array_equal(frame_ids[start:end], np.arange(end - start)):
            raise ValueError(f"Episode/frame indices disagree with metadata for episode {ep}")
        reference = Path(info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        if not any(path == reference and a <= start < b for path, a, b in data_files):
            raise ValueError(f"Episode {ep} references the wrong physical data file")
        cursor = end
        for name, col in zip(GRIPPER_NAMES, columns):
            widths = actions[start:end, col]
            mask = closed_plateau_mask(widths, fps, params) & (widths != 0)
            relabeled[start:end, col][mask] = 0
            counts[name] += int(mask.sum())
            if mask.any():
                changed_eps[name].add(ep)
            for a, b in intervals(mask):
                changes.append({"episode_index": ep, "gripper": name,
                                "frame_start": int(a), "frame_end_exclusive": int(b),
                                "index_start": start + int(a), "index_end_exclusive": start + int(b),
                                "start_seconds": float(a / fps), "end_seconds_exclusive": float(b / fps),
                                "original_min_m": float(widths[a:b].min()), "original_max_m": float(widths[a:b].max()),
                                "new_target_m": 0.0, "modified_frames": int(b - a)})
        stats[ep] = action_stats(relabeled[start:end])
        if (ep + 1) % 250 == 0:
            LOG.info("Analyzed %d/%d episodes", ep + 1, len(rows))
    if cursor != total:
        raise ValueError("Episode metadata does not cover all data frames")
    global_stats = read_json(source / "meta/stats.json")
    unknown_stats = set(global_stats.get("action", {})) - set(stats[0])
    if unknown_stats:
        raise ValueError(f"Unsupported action statistics: {sorted(unknown_stats)}")
    global_stats["action"] = action_stats(relabeled)
    report = {"algorithm": SEMANTICS, "created_at": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "source_semantics": info.get(SEMANTICS_KEY),
              "parameters": asdict(params), "fps": fps, "stable_frames": window,
              "total_episodes": len(rows), "total_frames": total,
              "modified_frames_per_gripper": counts,
              "modified_episodes_per_gripper": {k: len(v) for k, v in changed_eps.items()},
              "modified_episodes_any_gripper": len(set.union(*changed_eps.values())),
              "modified_frames_any_gripper": int(np.any(actions[:, columns] != relabeled[:, columns], axis=1).sum()),
              "interval_count": len(changes), "intervals": changes,
              "statistics_method": "float64 population mean/std and exact numpy quantiles",
              "limitation": "Heuristic closing intent; original teleoperation commands cannot be recovered."}
    return dict(info=info, actions=relabeled, data_files=data_files, metadata=metadata,
                episode_stats=stats, global_stats=global_stats, report=report, columns=columns)


def replace_action(table: pa.Table, values: np.ndarray) -> pa.Table:
    field = table.schema.field("action")
    flat = pa.array(values.reshape(-1), type=field.type.value_type)
    if pa.types.is_fixed_size_list(field.type):
        array = pa.FixedSizeListArray.from_arrays(flat, type=field.type)
    else:
        offsets = pa.array(np.arange(0, values.size + 1, values.shape[1]), type=pa.int32())
        array = pa.ListArray.from_arrays(offsets, flat, type=field.type)
    return table.set_column(table.schema.get_field_index("action"), field, array)


def updated_metadata(table: pa.Table, stats: dict) -> pa.Table:
    episode_ids = table["episode_index"].to_pylist()
    keys = set(next(iter(stats.values())))
    existing_keys = {key.removeprefix("stats/action/") for key in table.column_names if key.startswith("stats/action/")}
    if existing_keys - keys:
        raise ValueError(f"Unsupported episode action statistics: {existing_keys - keys}")
    for key in sorted(keys):
        name = f"stats/action/{key}"
        values = [stats[int(ep)][key] for ep in episode_ids]
        if name in table.column_names:
            field = table.schema.field(name)
            table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
        else:
            dtype = pa.list_(pa.int64() if key == "count" else pa.float64())
            table = table.append_column(name, pa.array(values, type=dtype))
    return table


def verify_output(source: Path, stage: Path, plan: dict, source_inventory: dict) -> dict:
    modified_files = {str(p) for p, _, _ in plan["data_files"]} | {str(p) for p, _ in plan["metadata"]}
    modified_files |= {"meta/info.json", "meta/stats.json"}
    output_inventory = inventory(stage)
    if set(output_inventory) != set(source_inventory):
        raise ValueError("Output file inventory differs from source")
    for name, entry in source_inventory.items():
        if name not in modified_files and output_inventory[name] != entry:
            raise ValueError(f"Unmodified file differs: {name}")
        if (source / name).stat().st_ino == (stage / name).stat().st_ino and (source / name).stat().st_dev == (stage / name).stat().st_dev:
            raise ValueError(f"Output must not hard-link source files: {name}")
    for path, start, end in plan["data_files"]:
        original, output = pq.read_table(source / path), pq.read_table(stage / path)
        if not original.schema.equals(output.schema, check_metadata=True) or len(original) != len(output):
            raise ValueError(f"Schema/row count changed: {path}")
        for name in original.column_names:
            if name != "action" and not original[name].equals(output[name]):
                raise ValueError(f"Unexpected modification of {name} in {path}")
        actual = action_matrix(output["action"], plan["actions"].shape[1])
        expected = plan["actions"][start:end]
        if not np.array_equal(actual, expected):
            raise ValueError(f"Action write verification failed: {path}")
        before = action_matrix(original["action"], actual.shape[1])
        arm_cols = [i for i in range(actual.shape[1]) if i not in plan["columns"]]
        if not np.array_equal(before[:, arm_cols], actual[:, arm_cols]):
            raise ValueError(f"Non-gripper actions changed: {path}")
    for path, original in plan["metadata"]:
        output = pq.read_table(stage / path)
        if not output.equals(updated_metadata(original, plan["episode_stats"])):
            raise ValueError(f"Episode metadata verification failed: {path}")
    expected_info = dict(plan["info"], **{SEMANTICS_KEY: SEMANTICS})
    if read_json(stage / "meta/info.json") != expected_info or read_json(stage / "meta/stats.json") != plan["global_stats"]:
        raise ValueError("Output info/statistics verification failed")
    if inventory(source) != source_inventory:
        raise ValueError("Source changed during relabeling; refusing to publish")
    return {"source_unchanged": True, "other_columns_unchanged": True,
            "episode_boundaries_unchanged": True, "videos_sha256_identical": True,
            "independent_copies": True, "action_statistics_verified": True}


def relabel_dataset(source: Path, output: Path, params: Parameters, dry_run: bool = False) -> dict:
    if output.expanduser().is_symlink():
        raise FileExistsError(f"Output must not be an existing symlink: {output}")
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    validate_paths(source, output)
    LOG.info("Checking source %s", source)
    before = inventory(source)
    plan = plan_relabel(source, params)
    report = plan["report"]
    report.update(output=str(output), dry_run=dry_run)
    if dry_run:
        if inventory(source) != before:
            raise ValueError("Source changed during analysis")
        return report
    # Validate metadata transformations before creating the staging directory.
    meta_tables = [(p, updated_metadata(t, plan["episode_stats"])) for p, t in plan["metadata"]]
    stage = output.with_name(output.name + ".relabel-staging")
    validate_paths(source, output)
    LOG.info("Copying dataset to %s (no links, no video encoding)", stage)
    try:
        shutil.copytree(source, stage, copy_function=shutil.copy2)
        for path, start, end in plan["data_files"]:
            LOG.info("Rewriting %s", path)
            table = replace_action(pq.read_table(source / path), plan["actions"][start:end])
            pq.write_table(table, stage / path, compression="snappy")
        for path, table in meta_tables:
            pq.write_table(table, stage / path, compression="snappy")
        write_json(stage / "meta/info.json", dict(plan["info"], **{SEMANTICS_KEY: SEMANTICS}))
        write_json(stage / "meta/stats.json", plan["global_stats"])
        LOG.info("Verifying numeric columns, metadata, copied files and source SHA-256 hashes")
        report["verification"] = verify_output(source, stage, plan, before)
        report["source_files"] = before
        write_json(stage / "meta/gripper_relabel_report.json", report)
        if output.exists():
            raise FileExistsError(f"Output appeared during processing: {output}")
        stage.rename(output)
    except BaseException:
        LOG.error("Failed; staging directory retained for inspection: %s", stage)
        raise
    LOG.info("Published %s", output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="Default: sibling <source>_gripper_relabel_v01")
    parser.add_argument("--dry-run", action="store_true", help="Analyze every episode without writing files")
    defaults = Parameters()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value,
                            help=f"{'Seconds' if name == 'stable_seconds' else 'Metres'}; default {value}")
    args = parser.parse_args()
    params = Parameters(**{key: getattr(args, key) for key in asdict(defaults)})
    output = args.output or args.source.with_name(args.source.name + "_gripper_relabel_v01")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = relabel_dataset(args.source, output, params, args.dry_run)
    summary = {key: value for key, value in report.items() if key not in {"intervals", "source_files"}}
    summary["example_intervals"] = report["intervals"][:6]
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
