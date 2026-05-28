#!/usr/bin/env python3
"""Rename a task prompt in a local LeRobot dataset.

This updates:
  - meta/tasks.parquet
  - meta/episodes/*/*.parquet, if their "tasks" column contains the old prompt
  - dataset_info.txt next to the dataset folder, if present

Example:
  python scripts/tools/rename_lerobot_task.py \
      --repo-id nero_task3_step1/2mL_empty_20260509_v01 \
      --old-task "pick up small vials and place them in the empty rack" \
      --new-task "pick up big vials and place them in the empty rack"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TASKS_PATH = Path("meta/tasks.parquet")
DEFAULT_EPISODES_DIR = Path("meta/episodes")
pd = None


def require_pandas() -> Any:
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as pandas_module
    except ImportError as exc:
        raise SystemExit(
            "This script needs pandas and a parquet engine such as pyarrow. "
            "Run it in the same conda environment you use for LeRobot recording."
        ) from exc
    pd = pandas_module
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename a task prompt in a local LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument(
        "--repo-id",
        help=(
            "LeRobot repo id relative to HF_LEROBOT_HOME, e.g. "
            "nero_task3_step1/2mL_empty_20260509_v01."
        ),
    )
    location.add_argument(
        "--dataset-root",
        type=Path,
        help="Path to the dataset folder, or directly to meta/tasks.parquet.",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        help="Override LeRobot dataset home. Defaults like LeRobot: HF_LEROBOT_HOME or HF_HOME/lerobot.",
    )
    parser.add_argument(
        "--old-task",
        help="Task prompt to replace. If omitted, it is inferred when the dataset has exactly one task.",
    )
    parser.add_argument(
        "--new-task",
        required=True,
        help="Replacement task prompt.",
    )
    parser.add_argument(
        "--skip-episode-metadata",
        action="store_true",
        help="Only update meta/tasks.parquet and leave meta/episodes parquet files unchanged.",
    )
    parser.add_argument(
        "--skip-dataset-info",
        action="store_true",
        help="Do not update dataset_info.txt next to the dataset folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing files.",
    )
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

    lerobot_home = (args.lerobot_home.expanduser() if args.lerobot_home else default_lerobot_home())
    return (lerobot_home / args.repo_id).resolve()


def backup_file(path: Path, timestamp: str, dry_run: bool) -> Optional[Path]:
    backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
    if dry_run:
        print(f"[dry-run] Would backup {path} -> {backup_path}")
        return backup_path
    shutil.copy2(path, backup_path)
    return backup_path


def infer_task_from_tasks_df(tasks: pd.DataFrame) -> str:
    index_values = [value for value in tasks.index.tolist() if isinstance(value, str)]
    if len(index_values) == 1 and "task" not in tasks.columns:
        return index_values[0]

    if "task" in tasks.columns:
        unique_tasks = [value for value in tasks["task"].dropna().unique().tolist() if isinstance(value, str)]
        if len(unique_tasks) == 1:
            return unique_tasks[0]

    raise ValueError(
        "--old-task was not provided and the script could not infer a single task from meta/tasks.parquet."
    )


def rename_tasks_table(tasks: pd.DataFrame, old_task: str, new_task: str) -> tuple[pd.DataFrame, bool]:
    if old_task == new_task:
        return tasks.copy(), False

    updated = tasks.copy()
    changed = False

    if old_task in updated.index:
        if new_task in updated.index:
            raise ValueError(
                f'Task "{new_task}" already exists in tasks.parquet; refusing to create duplicate task rows.'
            )
        updated = updated.rename(index={old_task: new_task})
        changed = True

    if "task" in updated.columns:
        old_mask = updated["task"] == old_task
        if old_mask.any():
            if (updated["task"] == new_task).any():
                raise ValueError(
                    f'Task "{new_task}" already exists in tasks.parquet; refusing to create duplicate task rows.'
                )
            updated.loc[old_mask, "task"] = new_task
            changed = True

    return updated, changed


def replace_task_value(value: Any, old_task: str, new_task: str) -> tuple[Any, bool]:
    if isinstance(value, str):
        return (new_task, True) if value == old_task else (value, False)

    sequence: Any = value
    if hasattr(value, "tolist"):
        sequence = value.tolist()

    if isinstance(sequence, tuple):
        sequence = list(sequence)

    if isinstance(sequence, list):
        replaced = [new_task if item == old_task else item for item in sequence]
        return replaced, replaced != sequence

    return value, False


def iter_episode_parquets(dataset_root: Path) -> Iterable[Path]:
    episodes_dir = dataset_root / DEFAULT_EPISODES_DIR
    if not episodes_dir.exists():
        return []
    return sorted(episodes_dir.glob("*/*.parquet"))


def update_episode_metadata(
    dataset_root: Path,
    old_task: str,
    new_task: str,
    timestamp: str,
    dry_run: bool,
) -> list[Path]:
    changed_files: list[Path] = []

    for parquet_path in iter_episode_parquets(dataset_root):
        episodes = pd.read_parquet(parquet_path)
        if "tasks" not in episodes.columns:
            continue

        changed = False
        new_values = []
        for value in episodes["tasks"].tolist():
            new_value, value_changed = replace_task_value(value, old_task, new_task)
            new_values.append(new_value)
            changed = changed or value_changed

        if not changed:
            continue

        changed_files.append(parquet_path)
        if dry_run:
            print(f"[dry-run] Would update episode metadata: {parquet_path}")
            continue

        backup_file(parquet_path, timestamp, dry_run=False)
        episodes["tasks"] = new_values
        episodes.to_parquet(parquet_path)

    return changed_files


def update_dataset_info(
    dataset_root: Path,
    repo_id: Optional[str],
    old_task: str,
    new_task: str,
    timestamp: str,
    dry_run: bool,
) -> bool:
    info_path = dataset_root.parent / "dataset_info.txt"
    if not info_path.exists():
        return False

    dataset_folder = dataset_root.name
    name_re = re.compile(r'name="([^"]+)"')
    task_re = re.compile(r'task="([^"]*)"')

    lines = info_path.read_text().splitlines(keepends=True)
    changed = False
    updated_lines = []

    for line in lines:
        name_match = name_re.search(line)
        task_match = task_re.search(line)
        if not name_match or not task_match:
            updated_lines.append(line)
            continue

        name = name_match.group(1)
        task = task_match.group(1)
        is_target_dataset = (
            (repo_id is not None and name == repo_id)
            or name == dataset_folder
            or name.endswith(f"/{dataset_folder}")
        )

        if is_target_dataset and task == old_task:
            line = task_re.sub(lambda _: f'task="{new_task}"', line, count=1)
            changed = True

        updated_lines.append(line)

    if changed and not dry_run:
        backup_file(info_path, timestamp, dry_run=False)
        info_path.write_text("".join(updated_lines))
    elif changed:
        print(f"[dry-run] Would update dataset_info.txt: {info_path}")

    return changed


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    dataset_root = resolve_dataset_root(args)
    tasks_path = dataset_root / DEFAULT_TASKS_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not tasks_path.exists():
        raise SystemExit(f"Cannot find tasks.parquet: {tasks_path}")

    tasks = pandas.read_parquet(tasks_path)
    old_task = args.old_task or infer_task_from_tasks_df(tasks)
    new_task = args.new_task

    updated_tasks, tasks_changed = rename_tasks_table(tasks, old_task, new_task)
    episode_files = []
    dataset_info_changed = False

    print(f"Dataset root: {dataset_root}")
    print(f'Old task: "{old_task}"')
    print(f'New task: "{new_task}"')

    if tasks_changed:
        if args.dry_run:
            print(f"[dry-run] Would update tasks table: {tasks_path}")
        else:
            backup = backup_file(tasks_path, timestamp, dry_run=False)
            updated_tasks.to_parquet(tasks_path)
            print(f"Updated tasks table: {tasks_path}")
            print(f"Backup: {backup}")
    else:
        print("No matching task found in meta/tasks.parquet.")

    if not args.skip_episode_metadata:
        episode_files = update_episode_metadata(dataset_root, old_task, new_task, timestamp, args.dry_run)
        if episode_files:
            print(f"Updated episode metadata files: {len(episode_files)}")
        else:
            print("No episode metadata files needed changes.")

    if not args.skip_dataset_info:
        dataset_info_changed = update_dataset_info(
            dataset_root=dataset_root,
            repo_id=args.repo_id,
            old_task=old_task,
            new_task=new_task,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        if dataset_info_changed:
            print("Updated dataset_info.txt.")

    if not tasks_changed and not episode_files and not dataset_info_changed:
        raise SystemExit("Nothing changed. Check --old-task and dataset path.")

    if args.dry_run:
        print("Dry run complete; no files were written.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
