import argparse
import os
from pathlib import Path
import re
import shutil
from datetime import datetime

import yaml

try:  # Keep --help usable outside the LeRobot runtime environment.
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    HF_LEROBOT_HOME = None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "record_cfg.yaml"


def _default_lerobot_home() -> Path:
    if HF_LEROBOT_HOME is not None:
        return Path(HF_LEROBOT_HOME)
    if os.getenv("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "lerobot"
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def _load_repo_id(cfg_path: Path) -> str:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    record_cfg = cfg.get("record", cfg)
    repo_id = record_cfg.get("repo_id")
    if not repo_id:
        raise ValueError(f"Cannot find record.repo_id in config: {cfg_path}")
    return str(repo_id)


def clean_dataset_info(config_path: Path | None = None, lerobot_home: Path | None = None):
    # ====== [LOAD CONFIG] ======
    cfg_path = config_path or _default_config_path()
    repo_id = _load_repo_id(cfg_path)
    user_name = repo_id.split("/", 1)[0]

    # ====== [DEFINE PATHS] ======
    base_path = (lerobot_home or _default_lerobot_home()) / user_name
    info_file = base_path / "dataset_info.txt"

    if not info_file.exists():
        print(f"====== [ERROR] dataset_info.txt not found at {info_file} ======")
        return

    # ====== [CREATE BACKUP BEFORE MODIFICATION] ======
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = base_path / "dataset_info_backup" / f"dataset_info_backup_{timestamp}.txt"
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info_file, backup_file)
    print(f"====== [BACKUP] Created backup file: {backup_file} ======")

    # ====== [READ EXISTING FOLDERS] ======
    existing_folders = {p.name for p in base_path.iterdir() if p.is_dir()}
    print(f"====== [INFO] Found {len(existing_folders)} existing dataset folders ======")

    # ====== [READ INFO FILE LINES] ======
    with open(info_file, "r") as f:
        lines = f.readlines()

    kept_lines = []
    removed_lines = []

    # filter lines: only keep folders that exist
    for line in lines:
        match = re.search(r'name="([^"]+)"', line)
        if match:
            full_name = match.group(1)
            folder_name = full_name.split("/", 1)[1] if "/" in full_name else full_name
            if folder_name in existing_folders:
                kept_lines.append(line)
            else:
                removed_lines.append(line)
        else:
            kept_lines.append(line)

    # ====== [UPDATE record_id] ======
    # ====== [UPDATE record_id as string] ======
    updated_lines = []
    for idx, line in enumerate(kept_lines, start=1):
        line = re.sub(r'record_id="[^"]*"', f'record_id="{idx}"', line)
        updated_lines.append(line)


    # ====== [WRITE CLEAN FILE BACK] ======
    with open(info_file, "w") as f:
        f.writelines(updated_lines)

    # ====== [REPORT RESULTS] ======
    print("====== [CLEANUP COMPLETE] ======")
    print(f"Kept {len(updated_lines)} lines, removed {len(removed_lines)} invalid entries.")
    print(f"Backup saved at: {backup_file}")
    if removed_lines:
        print("Removed entries:")
        for rl in removed_lines:
            print(" -", rl.strip())



def main():
    parser = argparse.ArgumentParser(description="Clean stale entries from local dataset_info.txt.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to record_cfg.yaml used to infer the dataset owner folder.",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=None,
        help="Override LeRobot dataset home. Defaults like LeRobot: HF_LEROBOT_HOME or HF_HOME/lerobot.",
    )
    args = parser.parse_args()
    clean_dataset_info(args.config, args.lerobot_home)


if __name__ == "__main__":
    main()
