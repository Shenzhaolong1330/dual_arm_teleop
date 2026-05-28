#!/usr/bin/env python3
"""Merge one LeRobot task label into another inside a local v3 dataset.

This updates the task table, data parquet task_index values, episode metadata
task labels, task_index episode stats, info.json total_tasks, and stats.json
task_index statistics. It does not touch observation/action tensors.

Example:
  python scripts/tools/merge_lerobot_tasks.py \
      --dataset-root /path/to/dataset \
      --source-task "pick up small vials and insert them in the empty rack" \
      --target-task "pick up small vials and place them in the empty rack" \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TASKS_PATH = Path("meta/tasks.parquet")
DEFAULT_INFO_PATH = Path("meta/info.json")
DEFAULT_STATS_PATH = Path("meta/stats.json")
DEFAULT_EPISODES_DIR = Path("meta/episodes")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


@dataclass
class PlannedChange:
    path: Path
    description: str


def require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "This script needs pandas and a parquet engine such as pyarrow. "
            "Run it in the same conda environment you use for LeRobot."
        ) from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one task prompt into another in a local LeRobot dataset."
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--repo-id", help="LeRobot repo id relative to HF_LEROBOT_HOME.")
    location.add_argument("--dataset-root", type=Path, help="Path to the local dataset root.")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        help="Override LeRobot dataset home. Defaults like LeRobot: HF_LEROBOT_HOME or HF_HOME/lerobot.",
    )
    parser.add_argument("--source-task", help="Task prompt to merge/remove.")
    parser.add_argument("--target-task", required=True, help="Task prompt to keep.")
    parser.add_argument(
        "--repair-single-task",
        action="store_true",
        help=(
            "Recover an already partially merged dataset by forcing the task table, "
            "data task_index, episode tasks, info.json, and task_index stats to one target task."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing.")
    parser.set_defaults(backup=True)
    parser.add_argument("--backup", dest="backup", action="store_true", help="Back up changed files.")
    parser.add_argument("--no-backup", dest="backup", action="store_false", help="Disable backups.")
    return parser.parse_args()


def default_lerobot_home() -> Path:
    if os.getenv("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "lerobot"
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        path = args.dataset_root.expanduser()
        if path.name == "tasks.parquet":
            return path.parent.parent.resolve()
        return path.resolve()

    lerobot_home = args.lerobot_home.expanduser() if args.lerobot_home else default_lerobot_home()
    return (lerobot_home / args.repo_id).resolve()


def backup_file(path: Path, timestamp: str, dry_run: bool) -> Path:
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.bak.{timestamp}.{suffix}")
        suffix += 1
    if dry_run:
        print(f"[dry-run] Would backup {path} -> {backup_path}")
        return backup_path
    shutil.copy2(path, backup_path)
    return backup_path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tasks(dataset_root: Path, pd: Any):
    tasks_path = dataset_root / DEFAULT_TASKS_PATH
    if not tasks_path.exists():
        raise SystemExit(f"Cannot find tasks table: {tasks_path}")
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        raise SystemExit(f"tasks.parquet is missing required column 'task_index': {tasks_path}")
    return tasks


def build_task_mapping(tasks, source_task: str, target_task: str) -> tuple[Any, dict[int, int], int, int]:
    if source_task == target_task:
        raise SystemExit("--source-task and --target-task must be different.")
    if source_task not in tasks.index:
        raise SystemExit(f'Source task not found in tasks.parquet: "{source_task}"')
    if target_task not in tasks.index:
        raise SystemExit(f'Target task not found in tasks.parquet: "{target_task}"')

    source_idx = int(tasks.loc[source_task, "task_index"])
    target_idx = int(tasks.loc[target_task, "task_index"])
    remaining_task_names = [task for task in tasks.index.tolist() if task != source_task]
    updated_tasks = tasks.loc[remaining_task_names].copy()
    updated_tasks["task_index"] = list(range(len(updated_tasks)))

    task_to_new_idx = {task: int(updated_tasks.loc[task, "task_index"]) for task in updated_tasks.index}
    old_idx_to_task = {int(row.task_index): task for task, row in tasks.iterrows()}
    old_to_new_idx: dict[int, int] = {}
    for old_idx, task in old_idx_to_task.items():
        if task == source_task:
            old_to_new_idx[old_idx] = task_to_new_idx[target_task]
        else:
            old_to_new_idx[old_idx] = task_to_new_idx[task]

    return updated_tasks, old_to_new_idx, source_idx, target_idx


def build_single_task_mapping(tasks, target_task: str) -> tuple[Any, dict[int, int]]:
    if target_task not in tasks.index:
        raise SystemExit(f'Target task not found in tasks.parquet: "{target_task}"')

    updated_tasks = tasks.loc[[target_task]].copy()
    updated_tasks["task_index"] = [0]
    old_to_new_idx = {int(row.task_index): 0 for _, row in tasks.iterrows()}
    return updated_tasks, old_to_new_idx


def normalize_tasks_value(
    value: Any,
    source_task: str | None,
    target_task: str,
    force_target_task: bool = False,
) -> tuple[list[str], bool]:
    original = value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, tuple):
        value = list(value)
    elif value is None:
        value = []
    elif not isinstance(value, list):
        value = [str(value)]

    if force_target_task:
        replaced = [target_task]
    else:
        replaced = [target_task if item == source_task else item for item in value]
    deduped = []
    seen = set()
    for item in replaced:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    comparable_original = original.tolist() if hasattr(original, "tolist") else original
    return deduped, force_target_task or deduped != comparable_original


def list_parquets(dataset_root: Path, relative_dir: Path) -> list[Path]:
    directory = dataset_root / relative_dir
    if not directory.exists():
        return []
    return sorted(directory.glob("*/*.parquet"))


def compute_task_index_stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        raise ValueError("Cannot compute task_index stats from an empty dataset.")

    values = values.astype(float)
    stats: dict[str, list[float]] = {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(values.size)],
    }
    for key, quantile in DEFAULT_QUANTILES.items():
        stats[key] = [float(np.quantile(values, quantile))]
    return stats


def shape_like(old_value: Any, new_value: float | int) -> Any:
    if hasattr(old_value, "tolist"):
        old_value = old_value.tolist()
    if isinstance(old_value, list):
        return [new_value]
    return new_value


def update_task_index_stats_columns(row: Any, task_idx: int) -> Any:
    for column in row.index:
        if not column.startswith("stats/task_index/"):
            continue
        stat_name = column.rsplit("/", 1)[-1]
        old_value = row[column]
        if stat_name == "std":
            row[column] = shape_like(old_value, 0.0)
        elif stat_name == "count":
            row[column] = old_value
        else:
            row[column] = shape_like(old_value, task_idx)
    return row


def plan_changes(
    dataset_root: Path,
    source_task: str | None,
    target_task: str,
    pd: Any,
    repair_single_task: bool,
) -> tuple[list[PlannedChange], Any, dict[int, int], list[Path], list[Path]]:
    tasks = load_tasks(dataset_root, pd)
    if repair_single_task:
        updated_tasks, old_to_new_idx = build_single_task_mapping(tasks, target_task)
        task_table_description = f'force a single target task "{target_task}" with task_index 0'
    else:
        if source_task is None:
            raise SystemExit("--source-task is required unless --repair-single-task is used.")
        updated_tasks, old_to_new_idx, source_idx, target_idx = build_task_mapping(
            tasks, source_task, target_task
        )
        task_table_description = f"remove source task index {source_idx} and keep target task index {target_idx}"

    changes = [
        PlannedChange(
            dataset_root / DEFAULT_TASKS_PATH,
            task_table_description,
        )
    ]

    data_files = list_parquets(dataset_root, DEFAULT_DATA_DIR)
    episode_files = list_parquets(dataset_root, DEFAULT_EPISODES_DIR)

    for path in data_files:
        changes.append(PlannedChange(path, f"remap task_index with {old_to_new_idx}"))
    for path in episode_files:
        if repair_single_task:
            changes.append(PlannedChange(path, f'force all episode tasks to "{target_task}"'))
        else:
            changes.append(PlannedChange(path, f'replace task "{source_task}" with "{target_task}"'))

    info_path = dataset_root / DEFAULT_INFO_PATH
    if info_path.exists():
        changes.append(PlannedChange(info_path, f"set total_tasks to {len(updated_tasks)}"))

    stats_path = dataset_root / DEFAULT_STATS_PATH
    if stats_path.exists():
        changes.append(PlannedChange(stats_path, "recompute stats.json task_index summary"))

    return changes, updated_tasks, old_to_new_idx, data_files, episode_files


def apply_tasks_table(dataset_root: Path, updated_tasks: Any, timestamp: str, dry_run: bool, backup: bool) -> None:
    path = dataset_root / DEFAULT_TASKS_PATH
    if dry_run:
        print(f"[dry-run] Would write tasks table with {len(updated_tasks)} task(s): {path}")
        return
    if backup:
        print(f"Backup: {backup_file(path, timestamp, dry_run=False)}")
    updated_tasks.to_parquet(path)
    print(f"Updated tasks table: {path}")


def apply_data_files(
    data_files: list[Path],
    old_to_new_idx: dict[int, int],
    pd: Any,
    timestamp: str,
    dry_run: bool,
    backup: bool,
    map_unmapped_to: int | None = None,
) -> np.ndarray:
    all_task_indices = []
    total_changed = 0

    for path in data_files:
        df = pd.read_parquet(path)
        if "task_index" not in df.columns:
            continue
        old_series = df["task_index"].copy()
        df["task_index"] = df["task_index"].map(
            lambda value: old_to_new_idx.get(
                int(value),
                map_unmapped_to if map_unmapped_to is not None else int(value),
            )
        )
        changed = int((old_series != df["task_index"]).sum())
        total_changed += changed
        all_task_indices.append(df["task_index"].to_numpy())

        if changed == 0:
            continue
        if dry_run:
            print(f"[dry-run] Would update {changed} rows in data parquet: {path}")
            continue
        if backup:
            print(f"Backup: {backup_file(path, timestamp, dry_run=False)}")
        df.to_parquet(path)
        print(f"Updated data parquet: {path} ({changed} rows)")

    print(f"Data task_index rows changed: {total_changed}")
    if not all_task_indices:
        raise SystemExit("No data parquet task_index values were found.")
    return np.concatenate(all_task_indices)


def apply_episode_files(
    episode_files: list[Path],
    task_to_idx: dict[str, int],
    source_task: str | None,
    target_task: str,
    pd: Any,
    timestamp: str,
    dry_run: bool,
    backup: bool,
    force_target_task: bool = False,
) -> None:
    changed_files = 0
    changed_rows = 0

    for path in episode_files:
        df = pd.read_parquet(path)
        if "tasks" not in df.columns:
            continue

        changed = False
        updated_tasks = []
        for value in df["tasks"].tolist():
            new_tasks, value_changed = normalize_tasks_value(
                value,
                source_task,
                target_task,
                force_target_task=force_target_task,
            )
            updated_tasks.append(new_tasks)
            changed = changed or value_changed

        if not changed:
            continue

        df["tasks"] = updated_tasks
        for idx, row in df.iterrows():
            task_indices = [task_to_idx[task] for task in row["tasks"] if task in task_to_idx]
            if len(set(task_indices)) == 1:
                df.loc[idx] = update_task_index_stats_columns(row, task_indices[0])

        changed_files += 1
        changed_rows += len(df)
        if dry_run:
            print(f"[dry-run] Would update episode metadata: {path}")
            continue
        if backup:
            print(f"Backup: {backup_file(path, timestamp, dry_run=False)}")
        df.to_parquet(path)
        print(f"Updated episode metadata: {path}")

    print(f"Episode metadata files changed: {changed_files}; rows touched: {changed_rows}")


def apply_info_json(dataset_root: Path, total_tasks: int, timestamp: str, dry_run: bool, backup: bool) -> None:
    path = dataset_root / DEFAULT_INFO_PATH
    if not path.exists():
        return
    info = read_json(path)
    old_total_tasks = info.get("total_tasks")
    if old_total_tasks == total_tasks:
        return
    if dry_run:
        print(f"[dry-run] Would update {path}: total_tasks {old_total_tasks} -> {total_tasks}")
        return
    if backup:
        print(f"Backup: {backup_file(path, timestamp, dry_run=False)}")
    info["total_tasks"] = total_tasks
    write_json(path, info)
    print(f"Updated info.json: total_tasks {old_total_tasks} -> {total_tasks}")


def apply_stats_json(
    dataset_root: Path,
    task_index_values: np.ndarray,
    timestamp: str,
    dry_run: bool,
    backup: bool,
) -> None:
    path = dataset_root / DEFAULT_STATS_PATH
    if not path.exists():
        return
    stats = read_json(path)
    if "task_index" not in stats:
        return
    new_stats = compute_task_index_stats(task_index_values)
    if dry_run:
        print(f"[dry-run] Would update stats.json task_index: {path}")
        print(f"[dry-run] New task_index stats: {new_stats}")
        return
    if backup:
        print(f"Backup: {backup_file(path, timestamp, dry_run=False)}")
    stats["task_index"] = new_stats
    write_json(path, stats)
    print(f"Updated stats.json task_index summary: {path}")


def validate_dataset_loads(dataset_root: Path, repo_id: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    src_dir = repo_root / "src"
    if src_dir.exists():
        import sys

        sys.path.insert(0, str(src_dir))

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    validation_repo_id = repo_id or f"{dataset_root.parent.name}/{dataset_root.name}"
    meta = LeRobotDatasetMetadata(validation_repo_id, root=dataset_root)
    if meta.total_tasks != len(meta.tasks):
        raise RuntimeError(f"total_tasks={meta.total_tasks} but tasks table has {len(meta.tasks)} rows")
    print(f"Validated metadata load: total_tasks={meta.total_tasks}, tasks={list(meta.tasks.index)}")


def main() -> None:
    args = parse_args()
    pd = require_pandas()
    dataset_root = resolve_dataset_root(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    changes, updated_tasks, old_to_new_idx, data_files, episode_files = plan_changes(
        dataset_root=dataset_root,
        source_task=args.source_task,
        target_task=args.target_task,
        pd=pd,
        repair_single_task=args.repair_single_task,
    )

    task_to_idx = {task: int(row.task_index) for task, row in updated_tasks.iterrows()}

    print(f"Dataset root: {dataset_root}")
    if args.source_task is not None:
        print(f'Source task: "{args.source_task}"')
    print(f'Target task: "{args.target_task}"')
    if args.repair_single_task:
        print("Mode: repair single task")
    print(f"Task index mapping: {old_to_new_idx}")
    print("Planned changes:")
    for change in changes:
        print(f"  - {change.path}: {change.description}")

    apply_tasks_table(dataset_root, updated_tasks, timestamp, args.dry_run, args.backup)
    task_index_values = apply_data_files(
        data_files,
        old_to_new_idx,
        pd,
        timestamp,
        args.dry_run,
        args.backup,
        map_unmapped_to=0 if args.repair_single_task else None,
    )
    apply_episode_files(
        episode_files,
        task_to_idx,
        args.source_task,
        args.target_task,
        pd,
        timestamp,
        args.dry_run,
        args.backup,
        force_target_task=args.repair_single_task,
    )
    apply_info_json(dataset_root, len(updated_tasks), timestamp, args.dry_run, args.backup)
    apply_stats_json(dataset_root, task_index_values, timestamp, args.dry_run, args.backup)

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    validate_dataset_loads(dataset_root, args.repo_id)
    print("Done.")


if __name__ == "__main__":
    main()
