#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml


EXPORT_MODES = ("intervention_segments", "full_success_episode", "hybrid")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _path_repo_id(path: Path, fallback: str) -> str:
    return path.name or fallback


def _safe_repo_id(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _clean_or_fail(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Path already exists: {path}")
    shutil.rmtree(path)


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    return path


def _require_checkpoint(path: Path) -> Path:
    _require_dir(path, "initial checkpoint")
    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"initial checkpoint is missing required file(s) {missing}: {path}")
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _configured_export_mode(dagger_rounds_cfg: dict[str, Any]) -> str:
    export_cfg = (dagger_rounds_cfg.get("policy_backend") or {}).get("export", {})
    if not isinstance(export_cfg, dict):
        export_cfg = {}
    mode = str(export_cfg.get("export_mode") or "intervention_segments").strip().lower()
    if mode not in EXPORT_MODES:
        raise ValueError(
            "dagger_rounds.policy_backend.export.export_mode must be one of: "
            f"{', '.join(EXPORT_MODES)}. Got: {mode!r}."
        )
    return mode


def _dagger_sampling_cfg(dagger_rounds_cfg: dict[str, Any]) -> dict[str, Any]:
    dagger_training = copy.deepcopy(dagger_rounds_cfg.get("dagger_training") or {})
    sampling = dagger_training.get("sampling", {})
    return copy.deepcopy(sampling if isinstance(sampling, dict) else {})


def _sampling_train_cfg(sampling_cfg: dict[str, Any], source_index_path: Path | None) -> dict[str, Any]:
    cfg = copy.deepcopy(sampling_cfg)
    cfg.setdefault("enabled", False)
    if source_index_path is not None:
        cfg["source_index_path"] = str(source_index_path)
    return cfg


def _resolve_modes(mode: str | None, dagger_rounds_cfg: dict[str, Any]) -> list[str]:
    mode = _configured_export_mode(dagger_rounds_cfg) if mode is None else mode.strip().lower()
    if mode == "all":
        return list(EXPORT_MODES)
    if mode not in EXPORT_MODES:
        raise ValueError(f"Unknown --mode `{mode}`. Expected one of: all, {', '.join(EXPORT_MODES)}")
    return [mode]


def _extract_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    section = cfg.get(key, cfg)
    if not isinstance(section, dict):
        raise ValueError(f"Config section `{key}` must be a mapping.")
    return copy.deepcopy(section)


def _build_arg_parser() -> argparse.ArgumentParser:
    default_dir = Path(__file__).resolve().parent.parent / "config"
    parser = argparse.ArgumentParser(
        description=(
            "Run ACT DAgger export+train experiments from an existing raw run_mix dataset: "
            "export -> aggregate -> train -> checkpoint."
        )
    )
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Existing raw run_mix dataset root.")
    parser.add_argument(
        "--base-dataset",
        type=Path,
        required=True,
        help="Seed or existing aggregated dataset root to aggregate with exported data.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        required=True,
        help="ACT pretrained_model directory used as the warm start for every mode.",
    )
    parser.add_argument("--train-cfg", type=Path, default=default_dir / "train_cfg.yaml")
    parser.add_argument("--dagger-cfg", type=Path, default=default_dir / "dagger_rounds_cfg.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("all", *EXPORT_MODES),
        help=(
            "Override dagger_rounds.policy_backend.export.export_mode. "
            "If omitted, the experiment uses the export_mode from --dagger-cfg."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            "Override train.steps for every mode. If omitted, steps are resolved from "
            "dagger_rounds.round_schedule.train_steps after export."
        ),
    )
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size for every mode.")
    parser.add_argument("--min-episode-len", type=int, help="Override export min_episode_len_for_act.")
    parser.add_argument("--pre-takeover-context", type=int, help="Override export pre_takeover_context.")
    parser.add_argument(
        "--require-success",
        dest="require_success",
        action="store_true",
        default=None,
        help="Require success for full_success_episode/hybrid full-episode export.",
    )
    parser.add_argument(
        "--no-require-success",
        dest="require_success",
        action="store_false",
        help="Do not require success for full_success_episode/hybrid full-episode export.",
    )
    parser.add_argument(
        "--allow-missing-success",
        action="store_true",
        help="Smoke-test only: allow full episode export without success validation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths/config without running.")
    return parser


def _experiment_config(args: argparse.Namespace, modes: list[str]) -> dict[str, Any]:
    return {
        "raw_dataset": str(args.raw_dataset),
        "base_dataset": str(args.base_dataset),
        "initial_checkpoint": str(args.initial_checkpoint),
        "train_cfg": str(args.train_cfg),
        "dagger_cfg": str(args.dagger_cfg),
        "output_root": str(args.output_root),
        "modes": modes,
        "mode_argument": args.mode,
        "mode_source": getattr(args, "mode_source", None),
        "configured_export_mode": getattr(args, "configured_export_mode", None),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "min_episode_len": args.min_episode_len,
        "pre_takeover_context": args.pre_takeover_context,
        "require_success": args.require_success,
        "allow_missing_success": bool(args.allow_missing_success),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "fairness": {
            "same_initial_checkpoint_for_every_mode": True,
            "same_base_dataset_for_every_mode": True,
            "mode_outputs_are_independent": True,
        },
    }


def _prepare_export_section(
    *,
    mode: str,
    mode_dir: Path,
    args: argparse.Namespace,
    policy_backend,
    base_train_cfg: dict[str, Any],
    dagger_rounds_cfg: dict[str, Any],
    raw_repo_id: str,
    base_repo_id: str,
) -> dict[str, Any]:
    export_section = policy_backend.export_profile.prepare_export_cfg(
        {
            "raw_repo_id": raw_repo_id,
            "raw_root": args.raw_dataset,
            "output_repo_id": _safe_repo_id(f"{base_repo_id}_{mode}_dagger_export"),
            "output_root": mode_dir / "exported_dataset",
            "seed_repo_id": base_repo_id,
            "seed_root": args.base_dataset,
            "min_segment_frames": int(dagger_rounds_cfg.get("export_min_segment_frames", 1) or 1),
            "overwrite": bool(args.overwrite),
        },
        base_train_cfg,
        dagger_rounds_cfg,
    )
    export_section["export_mode"] = mode
    if args.min_episode_len is not None:
        export_section["min_episode_len_for_act"] = int(args.min_episode_len)
    if args.pre_takeover_context is not None:
        export_section["pre_takeover_context"] = int(args.pre_takeover_context)

    full_episode_cfg = copy.deepcopy(export_section.get("full_episode") or {})
    if args.allow_missing_success:
        full_episode_cfg["require_success"] = False
    elif args.require_success is not None:
        full_episode_cfg["require_success"] = bool(args.require_success)
    export_section["full_episode"] = full_episode_cfg
    return export_section


def _prepare_train_cfg(
    *,
    mode: str,
    mode_dir: Path,
    args: argparse.Namespace,
    policy_backend,
    base_train_cfg: dict[str, Any],
    aggregated_repo_id: str,
    aggregated_root: Path,
    base_repo_id: str,
    resolved_steps: int,
    dagger_sampling: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_overrides: dict[str, Any] = {}
    if args.steps is not None:
        train_overrides["steps"] = int(args.steps)
    if args.batch_size is not None:
        train_overrides["batch_size"] = int(args.batch_size)

    train_cfg = policy_backend.trainer.prepare_train_cfg(
        base_train_cfg=base_train_cfg,
        aggregated_dataset_path=aggregated_root,
        repo_id=aggregated_repo_id,
        checkpoint_in=args.initial_checkpoint,
        output_dir=mode_dir / "train",
        round_id=1,
        resolved_steps=resolved_steps,
        seed_repo_id=base_repo_id,
        seed_root=args.base_dataset,
        dagger_sampling=dagger_sampling,
    )
    train_cfg["job_name"] = f"dagger_export_train_{mode}"
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
        train_cfg.setdefault("dagger", {}).setdefault("training", {})["batch_size"] = int(args.batch_size)
    return train_cfg, train_overrides


def _resolve_experiment_train_steps(
    *,
    mode: str,
    args: argparse.Namespace,
    base_train_cfg: dict[str, Any],
    dagger_rounds_cfg: dict[str, Any],
    export_summary: dict[str, Any],
) -> dict[str, Any]:
    base_train_steps = int(base_train_cfg.get("steps", 10_000))
    if args.steps is not None:
        resolved_steps = int(args.steps)
        return {
            "train_steps_source": "cli_override",
            "train_steps_configured_mode": "cli_override",
            "train_steps_effective_mode": "fixed",
            "export_mode": mode,
            "base_train_steps": base_train_steps,
            "resolved_train_steps": resolved_steps,
            "train_steps_components": {
                "cli_steps": resolved_steps,
                "raw_steps": resolved_steps,
                "clamped_steps": resolved_steps,
            },
        }

    from scripts.core.run_dagger_rounds import resolve_train_steps_schedule

    schedule = resolve_train_steps_schedule(
        round_id=1,
        exported_summary=export_summary,
        base_train_steps=base_train_steps,
        schedule_cfg=dagger_rounds_cfg.get("round_schedule", {}),
        export_mode=mode,
    )
    schedule["train_steps_source"] = "dagger_rounds.round_schedule.train_steps"
    return schedule


def _aggregate_summary(aggregated_repo_id: str, aggregated_root: Path) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(aggregated_repo_id, root=aggregated_root)
    return {
        "repo_id": aggregated_repo_id,
        "root": str(aggregated_root),
        "total_frames": int(meta.total_frames),
        "total_episodes": int(meta.total_episodes),
    }


def _mode_metrics(export_summary: dict[str, Any] | None) -> dict[str, Any]:
    export_summary = export_summary or {}
    exported_dataset = export_summary.get("exported_dataset", {})
    stats = export_summary.get("stats", {})
    return {
        "exported_frames": int(export_summary.get("exported_frames", exported_dataset.get("total_frames", 0)) or 0),
        "exported_episodes": int(
            export_summary.get("exported_episodes", exported_dataset.get("total_episodes", 0)) or 0
        ),
        "full_episode_frames": int(
            export_summary.get(
                "full_episode_frames",
                stats.get("full_episode_frames_exported", stats.get("hybrid_full_frames", 0)),
            )
            or 0
        ),
        "recovery_frames": int(
            export_summary.get(
                "recovery_frames",
                stats.get("intervention_segment_frames_exported", stats.get("hybrid_recovery_frames", 0)),
            )
            or 0
        ),
        "recovery_ratio": float(export_summary.get("hybrid_recovery_ratio", stats.get("hybrid_recovery_ratio", 0.0)) or 0.0),
        "skipped_short_segments": int(stats.get("skipped_short_segments", 0) or 0),
        "skipped_incomplete_expert_labels": stats.get("incomplete_expert_label_frames"),
        "skipped_no_success": int(stats.get("full_episode_skipped_no_success", 0) or 0),
        "skipped_reset": int(stats.get("full_episode_skipped_reset", 0) or 0),
    }


def run_one_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    policy_backend,
    base_train_cfg: dict[str, Any],
    dagger_rounds_cfg: dict[str, Any],
    raw_repo_id: str,
    base_repo_id: str,
) -> dict[str, Any]:
    mode_dir = args.output_root / mode
    state: dict[str, Any] = {
        "mode": mode,
        "raw_dataset_path": str(args.raw_dataset),
        "base_dataset_path": str(args.base_dataset),
        "initial_checkpoint": str(args.initial_checkpoint),
        "exported_dataset_path": str(mode_dir / "exported_dataset"),
        "aggregated_dataset_path": str(mode_dir / "aggregated_dataset"),
        "train_output_dir": str(mode_dir / "train"),
        "selected_checkpoint_path": None,
        "checkpoint_path": None,
        "export_summary": None,
        "aggregate_summary": None,
        "train_cfg_overrides": {},
        "skipped_train": False,
        "skipped_train_reason": None,
        "status": "pending",
        "warnings": [],
    }

    try:
        _clean_or_fail(mode_dir, args.overwrite)
        mode_dir.mkdir(parents=True, exist_ok=True)

        export_section = _prepare_export_section(
            mode=mode,
            mode_dir=mode_dir,
            args=args,
            policy_backend=policy_backend,
            base_train_cfg=base_train_cfg,
            dagger_rounds_cfg=dagger_rounds_cfg,
            raw_repo_id=raw_repo_id,
            base_repo_id=base_repo_id,
        )
        if args.allow_missing_success and mode in {"full_success_episode", "hybrid"}:
            state["warnings"].append(
                "WARNING: success field missing, exported without success validation"
            )

        state["export_cfg"] = _to_jsonable(export_section)
        logging.info("[%s] export raw -> %s", mode, export_section["output_root"])
        export_summary = policy_backend.export_profile.export(export_section)
        state["export_summary"] = _to_jsonable(export_summary)
        _write_json(mode_dir / "export_summary.json", state["export_summary"])

        metrics = _mode_metrics(export_summary)
        state.update(metrics)
        train_steps_schedule = _resolve_experiment_train_steps(
            mode=mode,
            args=args,
            base_train_cfg=base_train_cfg,
            dagger_rounds_cfg=dagger_rounds_cfg,
            export_summary=export_summary,
        )
        resolved_train_steps = int(train_steps_schedule["resolved_train_steps"])
        state["train_steps_schedule"] = _to_jsonable(train_steps_schedule)
        state["resolved_train_steps"] = resolved_train_steps
        state["train_steps_source"] = train_steps_schedule.get("train_steps_source")
        state["train_steps_effective_mode"] = train_steps_schedule.get("train_steps_effective_mode")
        logging.info(
            "[%s] train steps mode: source=%s configured=%s effective=%s export_mode=%s resolved=%d",
            mode,
            train_steps_schedule.get("train_steps_source"),
            train_steps_schedule.get("train_steps_configured_mode"),
            train_steps_schedule.get("train_steps_effective_mode"),
            train_steps_schedule.get("export_mode"),
            resolved_train_steps,
        )
        if metrics["exported_frames"] <= 0 or metrics["exported_episodes"] <= 0:
            state["skipped_train"] = True
            state["skipped_train_reason"] = (
                f"empty exported dataset: exported_frames={metrics['exported_frames']}, "
                f"exported_episodes={metrics['exported_episodes']}"
            )
            state["status"] = "empty_export"
            _write_json(mode_dir / "experiment_state.json", _to_jsonable(state))
            return state

        from lerobot.datasets.aggregate import aggregate_datasets
        from scripts.core.run_dagger_export import assert_dataset_roots_schema_compatible

        aggregated_root = mode_dir / "aggregated_dataset"
        aggregated_repo_id = _safe_repo_id(f"{base_repo_id}_{mode}_aggregated")
        assert_dataset_roots_schema_compatible(
            base_repo_id,
            args.base_dataset,
            export_section["output_repo_id"],
            export_section["output_root"],
        )
        logging.info("[%s] aggregate base + exported -> %s", mode, aggregated_root)
        aggregate_datasets(
            repo_ids=[base_repo_id, export_section["output_repo_id"]],
            roots=[args.base_dataset, export_section["output_root"]],
            aggr_repo_id=aggregated_repo_id,
            aggr_root=aggregated_root,
        )
        assert_dataset_roots_schema_compatible(base_repo_id, args.base_dataset, aggregated_repo_id, aggregated_root)
        state["aggregate_summary"] = _aggregate_summary(aggregated_repo_id, aggregated_root)
        _write_json(mode_dir / "aggregate_summary.json", _to_jsonable(state["aggregate_summary"]))

        from scripts.core.dagger_sampling import (
            format_sampling_stats,
            summarize_sampling_for_dataset_root,
            write_aggregated_source_index,
        )

        source_index_path = write_aggregated_source_index(
            previous_root=args.base_dataset,
            current_root=export_section["output_root"],
            aggregated_root=aggregated_root,
            round_id=1,
            export_mode=mode,
            previous_fallback_source="seed",
        )
        dagger_sampling_train_cfg = _sampling_train_cfg(
            _dagger_sampling_cfg(dagger_rounds_cfg),
            source_index_path,
        )
        dagger_sampling_summary = summarize_sampling_for_dataset_root(
            dataset_root=aggregated_root,
            source_index_path=source_index_path,
            sampling_cfg=dagger_sampling_train_cfg,
        )
        logging.info("[%s] %s", mode, format_sampling_stats(dagger_sampling_summary))
        state["dagger_source_index_path"] = str(source_index_path)
        state["dagger_sampling"] = _to_jsonable(dagger_sampling_summary)

        train_cfg, train_overrides = _prepare_train_cfg(
            mode=mode,
            mode_dir=mode_dir,
            args=args,
            policy_backend=policy_backend,
            base_train_cfg=base_train_cfg,
            aggregated_repo_id=aggregated_repo_id,
            aggregated_root=aggregated_root,
            base_repo_id=base_repo_id,
            resolved_steps=resolved_train_steps,
            dagger_sampling=dagger_sampling_train_cfg,
        )
        state["train_cfg_overrides"] = train_overrides
        _write_yaml(mode_dir / "train_cfg_resolved.yaml", _to_jsonable(train_cfg))

        logging.info("[%s] train ACT from %s", mode, args.initial_checkpoint)
        checkpoint_path = Path(policy_backend.trainer.train(train_cfg))
        state["selected_checkpoint_path"] = str(checkpoint_path)
        state["checkpoint_path"] = str(checkpoint_path)
        state["train_summary"] = {
            "train_output_dir": str(mode_dir / "train"),
            "selected_checkpoint_path": str(checkpoint_path),
            "initial_checkpoint": str(args.initial_checkpoint),
            "steps": int(train_cfg.get("steps", 0) or 0),
            "batch_size": int(train_cfg.get("batch_size", 0) or 0),
            "warm_start_from_same_initial_checkpoint": True,
            "train_steps_schedule": _to_jsonable(train_steps_schedule),
        }
        _write_json(mode_dir / "train_summary.json", _to_jsonable(state["train_summary"]))
        state["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - per-mode failures should not stop the comparison.
        state["status"] = "failed"
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        logging.exception("[%s] failed", mode)
    finally:
        _write_json(mode_dir / "experiment_state.json", _to_jsonable(state))

    return state


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _gripper_rows(mode_states: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for mode, state in mode_states.items():
        diag = ((state.get("export_summary") or {}).get("gripper_diagnostics") or {})
        if not diag.get("enabled"):
            rows.append(f"- `{mode}`: Gripper diagnostics skipped: gripper_action_indices not configured.")
            continue
        by_subtype = diag.get("by_subtype") or {}
        for subtype, stats in by_subtype.items():
            first_open = [value for value in stats.get("first_open_timesteps", []) if value is not None]
            if first_open:
                first_open_summary = (
                    f"mean={np.mean(first_open):.2f}, min={min(first_open)}, max={max(first_open)}"
                )
            else:
                first_open_summary = "none"
            rows.append(
                f"- `{mode}` / `{subtype}`: open_ratio={stats.get('open_ratio', 0):.4f}, "
                f"first_open={first_open_summary}"
            )
        for warning in diag.get("warnings", []):
            rows.append(f"  - WARNING: {warning}")
    return rows


def _data_quality_warnings(mode: str, state: dict[str, Any]) -> list[str]:
    warnings = list(state.get("warnings") or [])
    metrics = state
    if state.get("status") != "completed":
        reason = state.get("skipped_train_reason") or state.get("error") or state.get("status")
        warnings.append(f"{mode}: not completed ({reason})")
    if int(metrics.get("exported_frames", 0) or 0) <= 0:
        warnings.append(f"{mode}: exported frames = 0")
    if float(metrics.get("recovery_ratio", 0.0) or 0.0) > 0.5:
        warnings.append(f"{mode}: recovery ratio is high ({metrics.get('recovery_ratio')})")
    if int(metrics.get("skipped_short_segments", 0) or 0) > 0:
        warnings.append(f"{mode}: short segments skipped = {metrics.get('skipped_short_segments')}")
    incomplete = metrics.get("skipped_incomplete_expert_labels")
    if incomplete:
        warnings.append(f"{mode}: incomplete expert label frames = {incomplete}")
    if int(metrics.get("skipped_no_success", 0) or 0) > 0:
        warnings.append(f"{mode}: full episodes skipped for no success = {metrics.get('skipped_no_success')}")
    diag = ((state.get("export_summary") or {}).get("gripper_diagnostics") or {})
    warnings.extend(f"{mode}: {warning}" for warning in diag.get("warnings", []))
    return warnings


def write_comparison_report(summary: dict[str, Any], output_path: Path) -> None:
    setup = summary["setup"]
    mode_states: dict[str, dict[str, Any]] = summary.get("mode_states", summary["modes"])
    lines: list[str] = [
        "# DAgger Export+Train Experiment Report",
        "",
        "## 1. Experiment Setup",
        "",
        f"- raw dataset path: `{setup['raw_dataset_path']}`",
        f"- base dataset path: `{setup['base_dataset_path']}`",
        f"- initial checkpoint: `{setup['initial_checkpoint']}`",
        f"- train cfg: `{setup['train_cfg']}`",
        f"- dagger cfg: `{setup['dagger_cfg']}`",
        f"- export mode source: `{setup.get('mode_source')}`",
        f"- configured export mode: `{setup.get('configured_export_mode')}`",
        f"- train steps: `{setup.get('train_steps_description')}`",
        f"- output root: `{setup['output_root']}`",
        f"- timestamp: `{setup['timestamp']}`",
        "- fairness: every mode uses the same base dataset, train config overrides, and initial checkpoint.",
        "",
        "## 2. Mode Summary Table",
        "",
        "| mode | exported frames | exported episodes | train steps | train steps mode | full episode frames | recovery frames | recovery ratio | skipped short segments | skipped incomplete expert labels | skipped no success | skipped reset | aggregated dataset path | train output path | checkpoint path |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for mode, state in mode_states.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    _format_int(state.get("exported_frames")),
                    _format_int(state.get("exported_episodes")),
                    _format_int(state.get("resolved_train_steps")),
                    str(state.get("train_steps_effective_mode") or ""),
                    _format_int(state.get("full_episode_frames")),
                    _format_int(state.get("recovery_frames")),
                    _format_float(state.get("recovery_ratio")),
                    _format_int(state.get("skipped_short_segments")),
                    _format_int(state.get("skipped_incomplete_expert_labels")),
                    _format_int(state.get("skipped_no_success")),
                    _format_int(state.get("skipped_reset")),
                    f"`{state.get('aggregated_dataset_path')}`",
                    f"`{state.get('train_output_dir')}`",
                    f"`{state.get('checkpoint_path')}`" if state.get("checkpoint_path") else "`null`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## 3. Gripper / Release Timing Diagnostics", ""])
    lines.extend(_gripper_rows(mode_states) or ["Gripper diagnostics skipped: gripper_action_indices not configured."])

    lines.extend(["", "## 4. Data Quality Warnings", ""])
    warnings: list[str] = []
    for mode, state in mode_states.items():
        warnings.extend(_data_quality_warnings(mode, state))
    lines.extend([f"- {warning}" for warning in warnings] or ["- No data quality warnings generated."])

    lines.extend(
        [
            "",
            "## 5. Recommended Deployment Test Order",
            "",
            "1. base checkpoint",
            "2. full_success_episode checkpoint",
            "3. hybrid checkpoint",
            "4. intervention_segments checkpoint",
            "",
            "The current concern is skipping the insert phase and opening the gripper too early, so full/hybrid checkpoints should be tested before the short recovery-only checkpoint.",
            "",
            "## 6. Checkpoint Paths",
            "",
        ]
    )
    for mode in EXPORT_MODES:
        if mode in mode_states:
            lines.append(f"- {mode}: `{mode_states[mode].get('checkpoint_path')}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _comparison_summary(
    *,
    args: argparse.Namespace,
    modes: list[str],
    mode_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    setup = {
        "raw_dataset_path": str(args.raw_dataset),
        "base_dataset_path": str(args.base_dataset),
        "initial_checkpoint": str(args.initial_checkpoint),
        "train_cfg": str(args.train_cfg),
        "dagger_cfg": str(args.dagger_cfg),
        "output_root": str(args.output_root),
        "mode_argument": args.mode,
        "mode_source": getattr(args, "mode_source", None),
        "configured_export_mode": getattr(args, "configured_export_mode", None),
        "steps": args.steps,
        "train_steps_description": (
            f"cli --steps={int(args.steps)}"
            if args.steps is not None
            else "dagger_rounds.round_schedule.train_steps"
        ),
        "batch_size": args.batch_size,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "modes_requested": modes,
        "allow_missing_success": bool(args.allow_missing_success),
    }
    compact_modes = {}
    for mode, state in mode_states.items():
        compact_modes[mode] = {
            "status": state.get("status"),
            "exported_frames": int(state.get("exported_frames", 0) or 0),
            "exported_episodes": int(state.get("exported_episodes", 0) or 0),
            "checkpoint_path": state.get("checkpoint_path"),
            "skipped_train": bool(state.get("skipped_train", False)),
            "skipped_train_reason": state.get("skipped_train_reason"),
            "error": state.get("error"),
            "resolved_train_steps": int(state.get("resolved_train_steps", 0) or 0),
            "train_steps_source": state.get("train_steps_source"),
            "train_steps_effective_mode": state.get("train_steps_effective_mode"),
            "full_episode_frames": int(state.get("full_episode_frames", 0) or 0),
            "recovery_frames": int(state.get("recovery_frames", 0) or 0),
            "recovery_ratio": float(state.get("recovery_ratio", 0.0) or 0.0),
        }
    return {
        **setup,
        "setup": setup,
        "modes": compact_modes,
        "mode_states": mode_states,
    }


def _print_dry_run(args: argparse.Namespace, modes: list[str], dagger_rounds_cfg: dict[str, Any]) -> None:
    raw_repo_id = _path_repo_id(args.raw_dataset, "raw_run_mix")
    base_repo_id = _path_repo_id(args.base_dataset, "base_dataset")
    preview = _experiment_config(args, modes)
    preview["mode_outputs"] = {
        mode: {
            "exported_dataset": str(args.output_root / mode / "exported_dataset"),
            "aggregated_dataset": str(args.output_root / mode / "aggregated_dataset"),
            "train": str(args.output_root / mode / "train"),
            "raw_repo_id": raw_repo_id,
            "base_repo_id": base_repo_id,
        }
        for mode in modes
    }
    preview["policy_backend"] = dagger_rounds_cfg.get("policy_backend", {})
    preview["round_schedule"] = dagger_rounds_cfg.get("round_schedule", {})
    preview["dagger_training"] = dagger_rounds_cfg.get("dagger_training", {})
    preview["configured_export_mode"] = getattr(args, "configured_export_mode", None)
    preview["mode_source"] = getattr(args, "mode_source", None)
    preview["train_steps_source"] = (
        f"cli --steps={int(args.steps)}"
        if args.steps is not None
        else "dagger_rounds.round_schedule.train_steps after export"
    )
    print(yaml.safe_dump(_to_jsonable(preview), sort_keys=False))


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    args.raw_dataset = _require_dir(args.raw_dataset, "raw dataset")
    args.base_dataset = _require_dir(args.base_dataset, "base dataset")
    args.initial_checkpoint = _require_checkpoint(args.initial_checkpoint)
    args.train_cfg = Path(args.train_cfg)
    args.dagger_cfg = Path(args.dagger_cfg)

    dagger_cfg = _load_yaml(args.dagger_cfg)
    train_cfg = _load_yaml(args.train_cfg)
    dagger_rounds_cfg = _extract_section(dagger_cfg, "dagger_rounds")
    base_train_cfg = _extract_section(train_cfg, "train")
    args.configured_export_mode = _configured_export_mode(dagger_rounds_cfg)
    args.mode_source = (
        "cli --mode"
        if args.mode is not None
        else "dagger_rounds.policy_backend.export.export_mode"
    )
    modes = _resolve_modes(args.mode, dagger_rounds_cfg)
    logging.info(
        "[experiment] export modes=%s source=%s configured_export_mode=%s",
        modes,
        args.mode_source,
        args.configured_export_mode,
    )

    if args.dry_run:
        _print_dry_run(args, modes, dagger_rounds_cfg)
        return

    _clean_or_fail(args.output_root, args.overwrite)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_yaml(args.output_root / "experiment_config.yaml", _to_jsonable(_experiment_config(args, modes)))

    from scripts.core.dagger_backends import make_policy_backend

    policy_backend = make_policy_backend(dagger_rounds_cfg.get("policy_backend"))
    raw_repo_id = _path_repo_id(args.raw_dataset, "raw_run_mix")
    base_repo_id = _path_repo_id(args.base_dataset, "base_dataset")

    mode_states: dict[str, dict[str, Any]] = {}
    for mode in modes:
        logging.info("========== [export+train experiment: %s] ==========", mode)
        mode_states[mode] = run_one_mode(
            mode=mode,
            args=args,
            policy_backend=policy_backend,
            base_train_cfg=base_train_cfg,
            dagger_rounds_cfg=dagger_rounds_cfg,
            raw_repo_id=raw_repo_id,
            base_repo_id=base_repo_id,
        )

    summary = _comparison_summary(args=args, modes=modes, mode_states=mode_states)
    _write_json(args.output_root / "comparison_summary.json", _to_jsonable(summary))
    write_comparison_report(summary, args.output_root / "comparison_report.md")

    print("\nCheckpoint paths:")
    for mode, state in mode_states.items():
        print(f"{mode}: {state.get('checkpoint_path')}")
    print(f"\nReport: {args.output_root / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
