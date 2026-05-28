#!/usr/bin/env python3
"""Safely inspect and patch explicit local LeRobot dataset metadata fields.

This tool is intentionally conservative. It patches only clearly identified
``repo_id`` and description-like metadata fields in JSON, JSONL, simple YAML,
and dataset-card Markdown metadata. It never rewrites data parquet files,
episode indices, task indices, stats, or feature schemas.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


DESCRIPTION_KEYS = {"description", "dataset_description"}
REPO_ID_KEYS = {"repo_id"}
PATCHABLE_KEYS = DESCRIPTION_KEYS | REPO_ID_KEYS
REQUIRED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


@dataclass
class FieldOccurrence:
    path: Path
    key_path: str
    key: str
    value: Any
    source: str


@dataclass
class Issue:
    severity: str
    message: str


@dataclass
class Change:
    path: Path
    key_path: str
    old: Any
    new: Any
    source: str


@dataclass
class PlannedPatch:
    path: Path
    new_text: str
    changes: list[Change] = field(default_factory=list)


@dataclass
class ScanReport:
    root: Path
    info: dict[str, Any] | None = None
    info_error: str | None = None
    repo_id_occurrences: list[FieldOccurrence] = field(default_factory=list)
    description_occurrences: list[FieldOccurrence] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    tasks_count: int | None = None
    tasks_error: str | None = None
    episode_files: list[Path] = field(default_factory=list)
    episode_rows: int | None = None
    episode_index_count: int | None = None
    episode_length_sum: int | None = None
    data_files: list[Path] = field(default_factory=list)
    data_rows: int | None = None
    covered_data_files: set[Path] = field(default_factory=set)
    covered_video_files: set[Path] = field(default_factory=set)
    dataset_info_entries: list[dict[str, str]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and patch explicit metadata fields in a local LeRobot dataset.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset root.")
    parser.add_argument("--description", help="New dataset description metadata value.")
    parser.add_argument("--repo-id", help="New repo_id metadata value. Only explicit existing fields are patched.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned edits without writing files.")
    parser.set_defaults(backup=True)
    parser.add_argument("--backup", dest="backup", action="store_true", help="Back up files before writing.")
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Disable backups before writing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat metadata warnings, missing fields, and parse issues as fatal.",
    )
    parser.add_argument("--check-only", action="store_true", help="Only scan consistency; do not patch.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_text(data: Any) -> str:
    return json.dumps(data, indent=4, ensure_ascii=False) + "\n"


def display_value(value: Any, limit: int = 180) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def add_issue(report: ScanReport, severity: str, message: str) -> None:
    if any(issue.severity == severity and issue.message == message for issue in report.issues):
        return
    report.issues.append(Issue(severity=severity, message=message))


def import_pandas(report: ScanReport | None = None):
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        if report is not None:
            add_issue(
                report,
                "warning",
                "pandas is unavailable; parquet task/episode metadata checks were skipped.",
            )
        logging.debug("pandas import failed: %s", exc)
        return None
    return pd


def import_pyarrow_parquet(report: ScanReport | None = None):
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        if report is not None:
            add_issue(
                report,
                "warning",
                "pyarrow is unavailable; parquet path and row-count checks were skipped.",
            )
        logging.debug("pyarrow import failed: %s", exc)
        return None
    return pq


def json_key_path(parent: str, key: str) -> str:
    if not parent:
        return f"$.{key}"
    return f"{parent}.{key}"


def collect_json_fields(data: Any, path: Path, source: str, parent: str = "$") -> list[FieldOccurrence]:
    occurrences: list[FieldOccurrence] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_path = json_key_path(parent, key)
            if key in PATCHABLE_KEYS:
                occurrences.append(FieldOccurrence(path, key_path, key, value, source))
            occurrences.extend(collect_json_fields(value, path, source, key_path))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            occurrences.extend(collect_json_fields(value, path, source, f"{parent}[{idx}]"))
    return occurrences


def patch_json_value(
    data: Any,
    path: Path,
    source: str,
    new_description: str | None,
    new_repo_id: str | None,
    parent: str = "$",
) -> tuple[Any, list[Change]]:
    changes: list[Change] = []
    if isinstance(data, dict):
        updated = {}
        for key, value in data.items():
            key_path = json_key_path(parent, key)
            replacement = None
            should_replace = False
            if key in DESCRIPTION_KEYS and new_description is not None:
                replacement = new_description
                should_replace = value != replacement
            elif key in REPO_ID_KEYS and new_repo_id is not None:
                replacement = new_repo_id
                should_replace = value != replacement

            if should_replace:
                updated[key] = replacement
                changes.append(Change(path, key_path, value, replacement, source))
                continue

            child, child_changes = patch_json_value(
                value, path, source, new_description, new_repo_id, key_path
            )
            updated[key] = child
            changes.extend(child_changes)
        return updated, changes

    if isinstance(data, list):
        updated_list = []
        for idx, value in enumerate(data):
            child, child_changes = patch_json_value(
                value, path, source, new_description, new_repo_id, f"{parent}[{idx}]"
            )
            updated_list.append(child)
            changes.extend(child_changes)
        return updated_list, changes

    return data, changes


def candidate_json_files(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for path in [
        root / "meta" / "info.json",
        root / "meta" / "modality.json",
        root / "dataset_info.json",
    ]:
        if path.exists():
            candidates.add(path)

    for base in [root, root / "meta"]:
        if base.exists():
            for path in base.glob("*.json"):
                if path.name == "stats.json":
                    continue
                candidates.add(path)
    return sorted(candidates)


def candidate_jsonl_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    meta = root / "meta"
    if meta.exists():
        paths.update(meta.glob("*.jsonl"))
    return sorted(paths)


def candidate_yaml_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for base in [root, root / "meta"]:
        if base.exists():
            paths.update(base.glob("*.yaml"))
            paths.update(base.glob("*.yml"))
    return sorted(paths)


def candidate_markdown_files(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    return [path for path in paths if path.exists()]


def scan_explicit_metadata_fields(root: Path, report: ScanReport) -> None:
    for path in candidate_json_files(root):
        try:
            data = read_json(path)
        except Exception as exc:
            add_issue(report, "warning", f"Could not parse JSON metadata {path}: {exc}")
            continue
        for occurrence in collect_json_fields(data, path, "JSON"):
            if occurrence.key in REPO_ID_KEYS:
                report.repo_id_occurrences.append(occurrence)
            elif occurrence.key in DESCRIPTION_KEYS:
                report.description_occurrences.append(occurrence)

    for path in candidate_jsonl_files(root):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    for occurrence in collect_json_fields(data, path, "JSONL", parent=f"$[{line_no}]"):
                        if occurrence.key in REPO_ID_KEYS:
                            report.repo_id_occurrences.append(occurrence)
                        elif occurrence.key in DESCRIPTION_KEYS:
                            report.description_occurrences.append(occurrence)
        except Exception as exc:
            add_issue(report, "warning", f"Could not parse JSONL metadata {path}: {exc}")

    for path in candidate_yaml_files(root):
        scan_simple_yaml_fields(path, report, source="YAML")

    for path in candidate_markdown_files(root):
        scan_markdown_fields(path, report)


YAML_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(?P<value>.*?)(?P<comment>\s+#.*)?$")


def parse_simple_yaml_scalar(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def quote_yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def scan_simple_yaml_fields(path: Path, report: ScanReport, source: str, key_prefix: str = "$") -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        add_issue(report, "warning", f"Could not read YAML metadata {path}: {exc}")
        return

    for line_no, line in enumerate(lines, start=1):
        match = YAML_KEY_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        if key not in PATCHABLE_KEYS:
            continue
        occurrence = FieldOccurrence(
            path=path,
            key_path=f"{key_prefix}.{key}@line{line_no}",
            key=key,
            value=parse_simple_yaml_scalar(match.group("value")),
            source=source,
        )
        if key in REPO_ID_KEYS:
            report.repo_id_occurrences.append(occurrence)
        else:
            report.description_occurrences.append(occurrence)


def scan_markdown_fields(path: Path, report: ScanReport) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        add_issue(report, "warning", f"Could not read Markdown metadata {path}: {exc}")
        return

    front_matter = split_front_matter(text)
    if front_matter is not None:
        _, body_start = front_matter
        fm_text = text[:body_start]
        temp_path = Path(f"{path}#front_matter")
        temp_report = ScanReport(root=report.root)
        scan_simple_yaml_text(fm_text, temp_path, temp_report, "Markdown front matter", "$.front_matter")
        report.repo_id_occurrences.extend(temp_report.repo_id_occurrences)
        report.description_occurrences.extend(temp_report.description_occurrences)

    section = extract_dataset_description_section(text)
    if section is not None:
        old_value, _, _ = section
        report.description_occurrences.append(
            FieldOccurrence(
                path=path,
                key_path="markdown.section.Dataset Description",
                key="dataset_description",
                value=old_value,
                source="Markdown",
            )
        )


def scan_simple_yaml_text(
    text: str,
    path: Path,
    report: ScanReport,
    source: str,
    key_prefix: str,
) -> None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = YAML_KEY_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        if key not in PATCHABLE_KEYS:
            continue
        occurrence = FieldOccurrence(
            path=path,
            key_path=f"{key_prefix}.{key}@line{line_no}",
            key=key,
            value=parse_simple_yaml_scalar(match.group("value")),
            source=source,
        )
        if key in REPO_ID_KEYS:
            report.repo_id_occurrences.append(occurrence)
        else:
            report.description_occurrences.append(occurrence)


def split_front_matter(text: str) -> tuple[int, int] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    line_end = text.find("\n", end + 4)
    if line_end == -1:
        line_end = len(text)
    else:
        line_end += 1
    return end, line_end


def extract_dataset_description_section(text: str) -> tuple[str, int, int] | None:
    lines = text.splitlines(keepends=True)
    start = None
    for idx, line in enumerate(lines):
        if re.match(r"^##\s+Dataset Description\s*$", line.strip()):
            start = idx + 1
            break
    if start is None:
        return None

    end = len(lines)
    for idx in range(start, len(lines)):
        if re.match(r"^##\s+", lines[idx].strip()):
            end = idx
            break

    return "".join(lines[start:end]).strip(), start, end


def load_info(root: Path, report: ScanReport) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        report.info_error = f"Missing {info_path}"
        add_issue(report, "error", report.info_error)
        return
    try:
        report.info = read_json(info_path)
    except Exception as exc:
        report.info_error = str(exc)
        add_issue(report, "error", f"Could not parse {info_path}: {exc}")
        return

    features = report.info.get("features")
    if not isinstance(features, dict) or not features:
        add_issue(report, "warning", "meta/info.json has missing or empty features schema.")
    else:
        missing_required = sorted(REQUIRED_FEATURES - set(features))
        if missing_required:
            add_issue(
                report,
                "warning",
                f"features schema is missing required LeRobot fields: {missing_required}",
            )


def load_tasks(root: Path, report: ScanReport) -> None:
    tasks_parquet = root / "meta" / "tasks.parquet"
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    pd = import_pandas(report)

    if tasks_parquet.exists() and pd is None:
        return

    if tasks_parquet.exists() and pd is not None:
        try:
            tasks_df = pd.read_parquet(tasks_parquet)
            report.tasks_count = len(tasks_df)
            if len(tasks_df.index) > 0:
                report.tasks = [str(value) for value in tasks_df.index.tolist()]
            elif "task" in tasks_df.columns:
                report.tasks = [str(value) for value in tasks_df["task"].dropna().tolist()]
            return
        except Exception as exc:
            report.tasks_error = str(exc)
            add_issue(report, "warning", f"Could not read {tasks_parquet}: {exc}")

    if tasks_jsonl.exists():
        tasks: list[str] = []
        try:
            with tasks_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for key in ["task", "name", "description"]:
                        if key in row:
                            tasks.append(str(row[key]))
                            break
            report.tasks = tasks
            report.tasks_count = len(tasks)
            return
        except Exception as exc:
            report.tasks_error = str(exc)
            add_issue(report, "warning", f"Could not read {tasks_jsonl}: {exc}")

    add_issue(report, "warning", "No tasks metadata found at meta/tasks.parquet or meta/tasks.jsonl.")


def format_indexed_path(template: str, chunk_index: int, file_index: int, video_key: str | None = None) -> Path:
    kwargs = {"chunk_index": int(chunk_index), "file_index": int(file_index)}
    if video_key is not None:
        kwargs["video_key"] = video_key
    return Path(template.format(**kwargs))


def read_parquet_columns(path: Path, columns: list[str], report: ScanReport):
    pd = import_pandas(report)
    if pd is None:
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        add_issue(report, "warning", f"Could not read parquet metadata {path}: {exc}")
        return None


def parquet_schema_names(path: Path, report: ScanReport) -> list[str]:
    pq = import_pyarrow_parquet(report)
    if pq is None:
        return []
    try:
        return list(pq.read_schema(path).names)
    except Exception as exc:
        add_issue(report, "warning", f"Could not read parquet schema {path}: {exc}")
        return []


def scan_episode_metadata(root: Path, report: ScanReport) -> None:
    episodes_dir = root / "meta" / "episodes"
    jsonl_path = root / "meta" / "episodes.jsonl"
    report.episode_files = sorted(episodes_dir.glob("*/*.parquet")) if episodes_dir.exists() else []
    data_path_template = (report.info or {}).get("data_path") or DEFAULT_DATA_PATH
    video_path_template = (report.info or {}).get("video_path") or DEFAULT_VIDEO_PATH
    video_keys = [
        key
        for key, feature in (report.info or {}).get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]

    if report.episode_files:
        if import_pyarrow_parquet(report) is None or import_pandas(report) is None:
            return

        all_episode_indices: set[int] = set()
        episode_rows = 0
        length_sum = 0
        covered_data: set[Path] = set()
        covered_videos: set[Path] = set()

        for path in report.episode_files:
            names = parquet_schema_names(path, report)
            wanted = [
                "episode_index",
                "length",
                "dataset_from_index",
                "dataset_to_index",
                "data/chunk_index",
                "data/file_index",
            ]
            for video_key in video_keys:
                wanted.extend([f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"])
            columns = [name for name in wanted if name in names]
            if not columns:
                continue
            df = read_parquet_columns(path, columns, report)
            if df is None:
                continue
            episode_rows += len(df)
            if "episode_index" in df:
                all_episode_indices.update(int(value) for value in df["episode_index"].dropna().tolist())
            if "length" in df:
                length_sum += int(df["length"].fillna(0).sum())
            if "data/chunk_index" in df and "data/file_index" in df:
                for chunk_idx, file_idx in zip(df["data/chunk_index"], df["data/file_index"], strict=False):
                    covered_data.add(format_indexed_path(data_path_template, chunk_idx, file_idx))
            for video_key in video_keys:
                chunk_col = f"videos/{video_key}/chunk_index"
                file_col = f"videos/{video_key}/file_index"
                if chunk_col in df and file_col in df and video_path_template:
                    for chunk_idx, file_idx in zip(df[chunk_col], df[file_col], strict=False):
                        covered_videos.add(
                            format_indexed_path(video_path_template, chunk_idx, file_idx, video_key)
                        )

        report.episode_rows = episode_rows
        report.episode_index_count = len(all_episode_indices)
        report.episode_length_sum = length_sum
        report.covered_data_files = covered_data
        report.covered_video_files = covered_videos

        if all_episode_indices:
            expected_indices = set(range(max(all_episode_indices) + 1))
            missing = sorted(expected_indices - all_episode_indices)
            if missing:
                add_issue(report, "warning", f"episode_index is not contiguous; missing indices: {missing[:20]}")
        return

    if jsonl_path.exists():
        try:
            rows = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            report.episode_rows = len(rows)
            indices = {int(row["episode_index"]) for row in rows if "episode_index" in row}
            report.episode_index_count = len(indices)
            report.episode_length_sum = sum(int(row.get("length", 0)) for row in rows)
            return
        except Exception as exc:
            add_issue(report, "warning", f"Could not read legacy episodes metadata {jsonl_path}: {exc}")

    add_issue(report, "warning", "No episode metadata found at meta/episodes/*/*.parquet or meta/episodes.jsonl.")


def scan_data_files(root: Path, report: ScanReport) -> None:
    report.data_files = sorted((root / "data").glob("*/*.parquet")) if (root / "data").exists() else []
    pq = import_pyarrow_parquet(report)
    if pq is None:
        return

    total_rows = 0
    for path in report.data_files:
        try:
            total_rows += int(pq.read_metadata(path).num_rows)
        except Exception as exc:
            add_issue(report, "warning", f"Could not read data parquet metadata {path}: {exc}")
    report.data_rows = total_rows


def scan_dataset_info_txt(root: Path, report: ScanReport) -> None:
    info_txt = root.parent / "dataset_info.txt"
    if not info_txt.exists():
        return

    name_re = re.compile(r'name="([^"]+)"')
    task_re = re.compile(r'task="([^"]*)"')
    try:
        for line in info_txt.read_text(encoding="utf-8").splitlines():
            name_match = name_re.search(line)
            task_match = task_re.search(line)
            if not name_match:
                continue
            name = name_match.group(1)
            task = task_match.group(1) if task_match else ""
            if name == root.name or name.endswith(f"/{root.name}"):
                report.dataset_info_entries.append({"name": name, "task": task})
    except Exception as exc:
        add_issue(report, "warning", f"Could not read adjacent dataset_info.txt: {exc}")


def check_consistency(report: ScanReport) -> None:
    info = report.info or {}
    root = report.root

    repo_values = [str(item.value) for item in report.repo_id_occurrences]
    if len(set(repo_values)) > 1:
        add_issue(report, "warning", f"Multiple explicit repo_id values found: {sorted(set(repo_values))}")
    for repo_id in set(repo_values):
        basename = repo_id.rsplit("/", 1)[-1]
        if basename and basename != root.name:
            add_issue(
                report,
                "warning",
                f"dataset root basename '{root.name}' does not match repo_id basename '{basename}'.",
            )

    description_values = [str(item.value) for item in report.description_occurrences]
    if len(set(description_values)) > 1:
        add_issue(
            report,
            "warning",
            f"Multiple description values found: {sorted(set(description_values))}",
        )
    if not description_values:
        add_issue(report, "warning", "No explicit description metadata field was found.")

    total_tasks = info.get("total_tasks")
    if total_tasks is not None and report.tasks_count is not None and int(total_tasks) != report.tasks_count:
        add_issue(
            report,
            "warning",
            f"info total_tasks={total_tasks} but tasks metadata contains {report.tasks_count} tasks.",
        )

    total_episodes = info.get("total_episodes")
    if total_episodes is not None and report.episode_index_count is not None:
        if int(total_episodes) != report.episode_index_count:
            add_issue(
                report,
                "warning",
                f"info total_episodes={total_episodes} but episode metadata covers {report.episode_index_count} episode indices.",
            )

    total_frames = info.get("total_frames")
    if total_frames is not None and report.episode_length_sum is not None:
        if int(total_frames) != int(report.episode_length_sum):
            add_issue(
                report,
                "warning",
                f"info total_frames={total_frames} but sum(meta episode length)={report.episode_length_sum}.",
            )
    if total_frames is not None and report.data_rows is not None and int(total_frames) != report.data_rows:
        add_issue(
            report,
            "warning",
            f"info total_frames={total_frames} but data parquet row count={report.data_rows}.",
        )

    actual_data_rel = {path.relative_to(root) for path in report.data_files}
    missing_data = sorted(path for path in report.covered_data_files if not (root / path).exists())
    extra_data = sorted(actual_data_rel - report.covered_data_files) if report.covered_data_files else []
    if missing_data:
        add_issue(
            report,
            "warning",
            f"episode metadata points to missing data parquet files: {[str(path) for path in missing_data[:20]]}",
        )
    if extra_data:
        add_issue(
            report,
            "warning",
            f"data parquet files exist but are not referenced by episode metadata: {[str(path) for path in extra_data[:20]]}",
        )

    missing_videos = sorted(path for path in report.covered_video_files if not (root / path).exists())
    if missing_videos:
        add_issue(
            report,
            "warning",
            f"episode metadata points to missing video files: {[str(path) for path in missing_videos[:20]]}",
        )


def scan_dataset(root: Path) -> ScanReport:
    report = ScanReport(root=root)
    if not root.exists():
        add_issue(report, "error", f"Dataset root does not exist: {root}")
        return report
    if not root.is_dir():
        add_issue(report, "error", f"Dataset root is not a directory: {root}")
        return report

    load_info(root, report)
    scan_explicit_metadata_fields(root, report)
    load_tasks(root, report)
    scan_episode_metadata(root, report)
    scan_data_files(root, report)
    scan_dataset_info_txt(root, report)
    check_consistency(report)
    return report


def print_occurrences(title: str, occurrences: list[FieldOccurrence]) -> None:
    print(f"{title}:")
    if not occurrences:
        print("  (none)")
        return
    for item in occurrences:
        print(f"  - {item.path}: {item.key_path} = {display_value(item.value)} [{item.source}]")


def print_scan_report(report: ScanReport) -> None:
    print("== LeRobot dataset metadata scan ==")
    print(f"dataset_root: {report.root}")
    info = report.info or {}
    if report.info_error:
        print(f"info: ERROR: {report.info_error}")
    else:
        print(f"codebase_version: {info.get('codebase_version')}")
        print(f"fps: {info.get('fps')}")
        print(f"robot_type: {info.get('robot_type')}")
        print(f"total_episodes: {info.get('total_episodes')}")
        print(f"total_frames: {info.get('total_frames')}")
        print(f"total_tasks: {info.get('total_tasks')}")
        print(f"features_count: {len(info.get('features', {}) or {})}")
        print(f"features: {', '.join((info.get('features', {}) or {}).keys())}")
        print(f"data_path: {info.get('data_path')}")
        print(f"video_path: {info.get('video_path')}")

    print_occurrences("repo_id fields", report.repo_id_occurrences)
    print_occurrences("description fields", report.description_occurrences)

    print("tasks:")
    if report.tasks_count is None:
        print("  (unavailable)")
    else:
        print(f"  count: {report.tasks_count}")
        for task in report.tasks[:10]:
            print(f"  - {task}")
        if len(report.tasks) > 10:
            print(f"  ... {len(report.tasks) - 10} more")

    print("files:")
    print(f"  episode metadata parquets: {len(report.episode_files)}")
    print(f"  episode rows: {report.episode_rows}")
    print(f"  covered episode indices: {report.episode_index_count}")
    print(f"  sum episode lengths: {report.episode_length_sum}")
    print(f"  data parquets: {len(report.data_files)}")
    print(f"  data parquet rows: {report.data_rows}")
    print(f"  referenced data parquets: {len(report.covered_data_files)}")
    print(f"  referenced video files: {len(report.covered_video_files)}")

    if report.dataset_info_entries:
        print("adjacent dataset_info.txt entries:")
        for entry in report.dataset_info_entries:
            print(f"  - name={display_value(entry['name'])}, task={display_value(entry['task'])}")

    if report.issues:
        print("issues:")
        for issue in report.issues:
            print(f"  [{issue.severity}] {issue.message}")
    else:
        print("issues: none")


def plan_json_patch(path: Path, description: str | None, repo_id: str | None) -> PlannedPatch | None:
    data = read_json(path)
    updated, changes = patch_json_value(data, path, "JSON", description, repo_id)
    if not changes:
        return None
    return PlannedPatch(path=path, new_text=write_json_text(updated), changes=changes)


def plan_jsonl_patch(path: Path, description: str | None, repo_id: str | None) -> PlannedPatch | None:
    changes: list[Change] = []
    new_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                new_lines.append(line)
                continue
            data = json.loads(line)
            updated, line_changes = patch_json_value(
                data,
                path,
                "JSONL",
                description,
                repo_id,
                parent=f"$[{line_no}]",
            )
            changes.extend(line_changes)
            new_lines.append(json.dumps(updated, ensure_ascii=False) + "\n")
    if not changes:
        return None
    return PlannedPatch(path=path, new_text="".join(new_lines), changes=changes)


def patch_yaml_lines(
    lines: list[str],
    path: Path,
    source: str,
    description: str | None,
    repo_id: str | None,
    key_prefix: str = "$",
) -> tuple[list[str], list[Change]]:
    changes: list[Change] = []
    updated_lines: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        keep_newline = "\n" if line.endswith("\n") else ""
        raw_line = line[:-1] if keep_newline else line
        match = YAML_KEY_RE.match(raw_line)
        if not match:
            updated_lines.append(line)
            continue
        key = match.group("key")
        old_value = parse_simple_yaml_scalar(match.group("value"))
        replacement = None
        if key in DESCRIPTION_KEYS and description is not None:
            replacement = description
        elif key in REPO_ID_KEYS and repo_id is not None:
            replacement = repo_id

        if replacement is None or old_value == replacement:
            updated_lines.append(line)
            continue

        comment = match.group("comment") or ""
        new_line = f"{match.group('indent')}{key}: {quote_yaml_string(replacement)}{comment}{keep_newline}"
        updated_lines.append(new_line)
        changes.append(
            Change(
                path=path,
                key_path=f"{key_prefix}.{key}@line{line_no}",
                old=old_value,
                new=replacement,
                source=source,
            )
        )
    return updated_lines, changes


def plan_yaml_patch(path: Path, description: str | None, repo_id: str | None) -> PlannedPatch | None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines, changes = patch_yaml_lines(lines, path, "YAML", description, repo_id)
    if not changes:
        return None
    return PlannedPatch(path=path, new_text="".join(updated_lines), changes=changes)


def plan_markdown_patch(path: Path, description: str | None, repo_id: str | None) -> PlannedPatch | None:
    text = path.read_text(encoding="utf-8")
    changes: list[Change] = []
    updated_text = text

    front_matter = split_front_matter(updated_text)
    if front_matter is not None:
        _, body_start = front_matter
        fm_text = updated_text[:body_start]
        body_text = updated_text[body_start:]
        fm_lines = fm_text.splitlines(keepends=True)
        updated_fm_lines, fm_changes = patch_yaml_lines(
            fm_lines,
            path,
            "Markdown front matter",
            description,
            repo_id,
            key_prefix="$.front_matter",
        )
        if fm_changes:
            updated_text = "".join(updated_fm_lines) + body_text
            changes.extend(fm_changes)

    if description is not None:
        section = extract_dataset_description_section(updated_text)
        if section is not None:
            old_value, start, end = section
            if old_value != description:
                lines = updated_text.splitlines(keepends=True)
                replacement = ["\n", f"{description.rstrip()}\n", "\n"]
                lines[start:end] = replacement
                updated_text = "".join(lines)
                changes.append(
                    Change(
                        path=path,
                        key_path="markdown.section.Dataset Description",
                        old=old_value,
                        new=description,
                        source="Markdown",
                    )
                )

    if not changes:
        return None
    return PlannedPatch(path=path, new_text=updated_text, changes=changes)


def plan_patches(root: Path, description: str | None, repo_id: str | None) -> list[PlannedPatch]:
    patches: list[PlannedPatch] = []

    for path in candidate_json_files(root):
        patch = plan_json_patch(path, description, repo_id)
        if patch is not None:
            patches.append(patch)

    for path in candidate_jsonl_files(root):
        patch = plan_jsonl_patch(path, description, repo_id)
        if patch is not None:
            patches.append(patch)

    for path in candidate_yaml_files(root):
        patch = plan_yaml_patch(path, description, repo_id)
        if patch is not None:
            patches.append(patch)

    for path in candidate_markdown_files(root):
        patch = plan_markdown_patch(path, description, repo_id)
        if patch is not None:
            patches.append(patch)

    return patches


def print_planned_patches(patches: list[PlannedPatch], dry_run: bool) -> None:
    label = "[dry-run] Would patch" if dry_run else "Will patch"
    if not patches:
        print(f"{label}: no files")
        return
    print(f"{label}:")
    for patch in patches:
        print(f"- {patch.path}")
        for change in patch.changes:
            print(
                f"  {change.source} {change.key_path}: "
                f"{display_value(change.old)} -> {display_value(change.new)}"
            )


def backup_file(path: Path, timestamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.bak.{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{timestamp}.{suffix}")
        suffix += 1
    shutil.copy2(path, candidate)
    return candidate


def apply_patches(patches: list[PlannedPatch], make_backup: bool) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for patch in patches:
        if make_backup:
            backup = backup_file(patch.path, timestamp)
            logging.info("Backup written: %s", backup)
        patch.path.write_text(patch.new_text, encoding="utf-8")
        logging.info("Patched: %s", patch.path)


def repo_id_values(report: ScanReport) -> list[str]:
    return [str(item.value) for item in report.repo_id_occurrences if item.value not in (None, "")]


def derive_repo_id_for_validation(root: Path, report: ScanReport, requested_repo_id: str | None) -> str:
    if requested_repo_id:
        return requested_repo_id
    values = repo_id_values(report)
    if values:
        return values[0]
    if root.parent.name:
        return f"{root.parent.name}/{root.name}"
    return root.name


def validate_with_lerobot(root: Path, repo_id: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    if dataset.meta.total_frames != metadata.total_frames:
        raise RuntimeError(
            f"LeRobotDataset total_frames={dataset.meta.total_frames} differs from metadata={metadata.total_frames}"
        )
    logging.info(
        "LeRobot validation loaded metadata and dataset: episodes=%s frames=%s",
        dataset.meta.total_episodes,
        len(dataset),
    )


def has_fatal_issues(report: ScanReport, strict: bool) -> bool:
    if strict:
        return bool(report.issues)
    return any(issue.severity == "error" for issue in report.issues)


def print_absent_patch_messages(
    report: ScanReport,
    patches: list[PlannedPatch],
    description: str | None,
    repo_id: str | None,
) -> None:
    repo_id_changed = any(
        change.key_path for patch in patches for change in patch.changes if "repo_id" in change.key_path
    )
    description_changed = any(
        change.key_path
        for patch in patches
        for change in patch.changes
        if "description" in change.key_path.lower()
    )

    if repo_id is not None and not repo_id_changed:
        if report.repo_id_occurrences:
            print("Explicit repo_id fields already match the requested value; no repo_id field was patched.")
        else:
            print("repo_id appears to be path-derived or absent from local metadata; no repo_id field was patched.")
    if description is not None and not description_changed:
        if report.description_occurrences:
            print("Description fields already match the requested value; no description field was patched.")
        else:
            print("No explicit description field was found; no description field was patched.")


def main() -> int:
    configure_logging()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if args.check_only and (args.description is not None or args.repo_id is not None):
        logging.warning("--check-only ignores --description and --repo-id.")

    report = scan_dataset(dataset_root)
    print_scan_report(report)

    if has_fatal_issues(report, args.strict):
        level = "strict metadata issue" if args.strict else "metadata error"
        logging.error("Stopping because %s was found.", level)
        return 2

    if args.check_only:
        return 0

    if args.description is None and args.repo_id is None:
        logging.info("No patch requested. Pass --description and/or --repo-id, or use --check-only.")
        return 0

    patches = plan_patches(dataset_root, args.description, args.repo_id)
    print_planned_patches(patches, dry_run=args.dry_run)
    print_absent_patch_messages(report, patches, args.description, args.repo_id)

    if not patches:
        return 0

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    apply_patches(patches, make_backup=args.backup)

    updated_report = scan_dataset(dataset_root)
    if has_fatal_issues(updated_report, args.strict):
        print_scan_report(updated_report)
        logging.error("Patched files were written, but post-patch metadata validation found issues.")
        return 3

    validation_repo_id = derive_repo_id_for_validation(dataset_root, updated_report, args.repo_id)
    try:
        validate_with_lerobot(dataset_root, validation_repo_id)
    except Exception as exc:
        logging.error("LeRobotDataset validation failed after patching: %s", exc)
        return 4

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
