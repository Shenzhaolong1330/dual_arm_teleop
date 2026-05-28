#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION

from scripts.core.run_dagger_export import (
    _frame_role_at,
    _iter_export_segments,
    _segment_id_at,
    _to_bool,
    _to_int,
    _to_str,
    assert_lerobot_schema_compatible,
)


RAW_DAGGER_FIELDS = {
    "policy_action",
    "expert_action",
    "sent_action",
    "action_source",
    "is_expert",
    "intervention_segment_id",
    "frame_role",
    "expert_label_complete",
    "expert_action_missing",
}


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _repo_id_from_root(root: Path, fallback: str) -> str:
    return root.name or fallback


def _cfg_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def _resolve_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _shape_dtype(value: Any) -> str:
    array = _as_numpy(value)
    return f"shape={tuple(array.shape)}, dtype={array.dtype}"


def _flatten_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, dict):
        flat: list[str] = []
        for value in names.values():
            flat.extend(_flatten_names(value))
        return flat
    if isinstance(names, (list, tuple)):
        flat = []
        for value in names:
            flat.extend(_flatten_names(value))
        return flat
    return [str(names)]


def _episode_rows(dataset: LeRobotDataset) -> list[dict[str, Any]]:
    episodes = dataset.meta.episodes
    if hasattr(episodes, "iterrows"):
        return [row.to_dict() for _, row in episodes.iterrows()]
    return [dict(row) for row in episodes]


def _episode_lengths(dataset: LeRobotDataset) -> list[int]:
    lengths: list[int] = []
    for row in _episode_rows(dataset):
        if "dataset_from_index" in row and "dataset_to_index" in row:
            lengths.append(int(row["dataset_to_index"]) - int(row["dataset_from_index"]))
        elif "length" in row:
            lengths.append(int(row["length"]))
        else:
            raise KeyError(f"Cannot infer episode length from episode row keys: {sorted(row)}")
    return lengths


def _feature_summary(meta: LeRobotDatasetMetadata) -> str:
    return json.dumps(meta.features, indent=2, ensure_ascii=False)


def _print_header(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def _load_dataset(label: str, root: Path, repo_id: str | None) -> LeRobotDataset:
    repo_id = repo_id or _repo_id_from_root(root, label)
    print(f"[load] {label}: repo_id={repo_id}, root={root}")
    return LeRobotDataset(repo_id, root=root)


def inspect_basic_loading(dataset: LeRobotDataset) -> None:
    _print_header("1. LeRobotDataset loading")
    print(f"dataset length: {len(dataset)}")
    print(f"episode count: {dataset.num_episodes}")
    print(f"fps: {dataset.fps}")
    print(f"robot_type: {dataset.meta.robot_type}")
    print(f"features/schema:\n{_feature_summary(dataset.meta)}")

    if len(dataset) == 0:
        print("[ERROR] exported dataset is empty.")
        return

    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    for label, idx in zip(["dataset[0]", "dataset[len//2]", "dataset[-1]"], sample_indices, strict=False):
        item = dataset[idx]
        print(f"\n[{label} -> idx={idx}] keys={sorted(item.keys())}")
        if ACTION in item:
            print(f"  action: {_shape_dtype(item[ACTION])}")
        if "observation.state" in item:
            print(f"  observation.state: {_shape_dtype(item['observation.state'])}")
        for key in dataset.meta.camera_keys:
            if key in item:
                print(f"  {key}: {_shape_dtype(item[key])}")


def inspect_schema(seed: LeRobotDataset | None, exported: LeRobotDataset) -> list[str]:
    _print_header("2. Seed/export schema compatibility")
    risks: list[str] = []
    extra_raw = sorted(RAW_DAGGER_FIELDS & set(exported.meta.features))
    if extra_raw:
        risks.append(f"raw DAgger fields leaked into exported training schema: {extra_raw}")
        print(f"[ERROR] raw-only fields in exported schema: {extra_raw}")
    else:
        print("[OK] raw run_mix extra fields are not present in exported ACT training schema.")

    if seed is None:
        print("[SKIP] seed dataset not provided; cannot do hard seed/export schema compare.")
        return risks

    try:
        assert_lerobot_schema_compatible(
            seed.meta,
            exported.meta,
            reference_name="seed",
            candidate_name="exported",
        )
        print("[OK] seed/export schema match exactly: feature keys, dtype, shape, names, fps, robot_type.")
    except ValueError as exc:
        risks.append(str(exc))
        print(f"[ERROR] schema mismatch:\n{exc}")

    seed_action = seed.meta.features.get(ACTION)
    export_action = exported.meta.features.get(ACTION)
    print(f"seed action feature: {seed_action}")
    print(f"exported action feature: {export_action}")
    return risks


def inspect_export_summary(export_root: Path) -> list[str]:
    _print_header("3. Export summary")
    risks: list[str] = []
    summary = _load_json(export_root / "dagger_export_summary.json")
    if summary is None:
        risks.append("missing dagger_export_summary.json; cannot verify export rules/statistics from metadata")
        print("[WARN] dagger_export_summary.json not found.")
        return risks

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    rules = summary.get("rules", {})
    export_mode = rules.get("export_mode", "intervention_segments")
    label_source = rules.get("label_source")
    if rules.get("standard_action_field") != ACTION:
        risks.append("export summary does not confirm standard action field")
        print("[ERROR] export summary does not confirm standard action field.")
    elif export_mode == "intervention_segments" and label_source != "expert_action":
        risks.append("intervention export summary does not say action came from expert_action")
        print("[ERROR] intervention export summary does not confirm action = expert_action.")
    elif export_mode == "full_success_episode" and label_source != "sent_action":
        risks.append("full-success export summary does not say action came from sent_action")
        print("[ERROR] full-success export summary does not confirm action = sent_action.")
    elif export_mode == "hybrid" and label_source != "by_export_subtype":
        risks.append("hybrid export summary does not say action labels are chosen by subtype")
        print("[ERROR] hybrid export summary does not confirm subtype-specific action labels.")
    elif export_mode == "intervention_segments":
        print("[OK] export summary confirms ACT label source: action = expert_action.")
    else:
        print(f"[OK] export summary confirms {export_mode} action label source: {label_source}.")
    return risks


def inspect_actions(dataset: LeRobotDataset, sample_count: int, zero_tol: float) -> list[str]:
    _print_header("4. Action supervision audit")
    risks: list[str] = []
    features = dataset.meta.features
    if ACTION not in features:
        risks.append("exported dataset has no standard action feature")
        print("[ERROR] no standard `action` feature.")
        return risks

    raw_fields = sorted({"expert_action", "policy_action", "sent_action"} & set(features))
    if raw_fields:
        print(f"[WARN] exported dataset still contains raw action fields: {raw_fields}")
        for idx in np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int):
            item = dataset[int(idx)]
            if "expert_action" in item:
                same = np.allclose(_as_numpy(item[ACTION]), _as_numpy(item["expert_action"]))
                print(f"  idx={idx}: action == expert_action? {same}")
                if not same:
                    risks.append(f"sample idx={idx} has action != expert_action")
            if "policy_action" in item:
                same_policy = np.allclose(_as_numpy(item[ACTION]), _as_numpy(item["policy_action"]))
                print(f"  idx={idx}: action == policy_action? {same_policy}")
            if "sent_action" in item:
                same_sent = np.allclose(_as_numpy(item[ACTION]), _as_numpy(item["sent_action"]))
                print(f"  idx={idx}: action == sent_action? {same_sent}")
    else:
        print(
            "[OK] exported schema contains only standard `action`; "
            "policy_action/expert_action/sent_action were not carried into ACT training features."
        )
        print(
            "     Direct equality with expert_action is therefore verified from export code/summary, "
            "not by comparing fields inside the standardized dataset."
        )

    n = min(sample_count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    actions = np.stack([_as_numpy(dataset[int(idx)][ACTION]).reshape(-1) for idx in indices])
    action_feature = features[ACTION]
    action_names = _flatten_names(action_feature.get("names"))
    if not action_names or len(action_names) != actions.shape[1]:
        action_names = [f"action[{i}]" for i in range(actions.shape[1])]

    print(f"sampled frames: {n}")
    print(f"action shape from samples: {actions.shape}")
    print(f"action dtype from samples: {actions.dtype}")
    print(f"action feature shape: {action_feature.get('shape')}")
    print(f"action names/order: {action_names}")
    print(f"global min/max/mean/std: {actions.min():.6g} / {actions.max():.6g} / {actions.mean():.6g} / {actions.std():.6g}")

    zero_fraction = np.mean(np.abs(actions) <= zero_tol, axis=0)
    near_constant = np.std(actions, axis=0) <= zero_tol
    suspicious_zero_dims = [action_names[i] for i, frac in enumerate(zero_fraction) if frac >= 0.98]
    constant_dims = [action_names[i] for i, flag in enumerate(near_constant) if flag]
    print("per-dim zero fraction:")
    for name, frac, std in zip(action_names, zero_fraction, np.std(actions, axis=0), strict=False):
        print(f"  {name}: zero_fraction={frac:.3f}, std={std:.6g}")
    if suspicious_zero_dims:
        print(f"[WARN] dimensions that are zero in >=98% sampled frames: {suspicious_zero_dims}")
    if len(constant_dims) == actions.shape[1]:
        risks.append("all sampled action dimensions are nearly constant")
        print("[ERROR] all sampled action dimensions are nearly constant.")

    groups = {
        "left": [i for i, name in enumerate(action_names) if "left" in name.lower()],
        "right": [i for i, name in enumerate(action_names) if "right" in name.lower()],
        "gripper": [
            i
            for i, name in enumerate(action_names)
            if "grip" in name.lower() or "gripper" in name.lower()
        ],
    }
    for group, dims in groups.items():
        if not dims:
            print(f"[INFO] no action name matched group `{group}`; cannot classify {group} takeover samples by name.")
            continue
        group_std = np.std(actions[:, dims], axis=0)
        print(f"{group} dims: {[action_names[i] for i in dims]}, std={group_std.tolist()}")

    nongripper_dims = [i for i in range(actions.shape[1]) if i not in groups["gripper"]]
    if nongripper_dims and np.all(np.std(actions[:, nongripper_dims], axis=0) <= zero_tol):
        risks.append("non-gripper action dimensions are nearly constant in sampled exported data")
        print("[ERROR] non-gripper dimensions are nearly constant; check for gripper-only labels.")

    return risks


def inspect_act_chunks(
    dataset: LeRobotDataset,
    train_cfg: dict[str, Any],
    export_summary: dict[str, Any] | None,
) -> list[str]:
    _print_header("5. ACT chunk compatibility")
    risks: list[str] = []
    train = _cfg_section(train_cfg, "train")
    policy = _cfg_section(train, "policy") if "train" in train_cfg else _cfg_section(train_cfg, "policy")
    chunk_size = int(policy.get("chunk_size", 1))
    n_action_steps = policy.get("n_action_steps")
    temporal_ensemble_coeff = policy.get("temporal_ensemble_coeff")
    print(f"policy.chunk_size: {chunk_size}")
    print(f"policy.n_action_steps: {n_action_steps}")
    print(f"policy.temporal_ensemble_coeff: {temporal_ensemble_coeff}")

    lengths = np.asarray(_episode_lengths(dataset), dtype=int)
    if len(lengths) == 0:
        risks.append("no exported episodes")
        print("[ERROR] no exported episodes.")
        return risks
    print(f"episode count: {len(lengths)}")
    print(f"min/max/mean/median length: {lengths.min()} / {lengths.max()} / {lengths.mean():.2f} / {np.median(lengths):.2f}")
    bins = [0, 5, 10, 20, chunk_size, chunk_size * 2, chunk_size * 4, int(lengths.max()) + 1]
    bins = sorted(set(b for b in bins if b >= 0))
    hist, edges = np.histogram(lengths, bins=bins)
    print("episode length histogram:")
    for left, right, count in zip(edges[:-1], edges[1:], hist, strict=False):
        print(f"  [{int(left)}, {int(right)}): {int(count)}")

    short = int(np.sum(lengths < chunk_size))
    barely = int(np.sum((lengths >= chunk_size) & (lengths < chunk_size + max(1, int(n_action_steps or 1)))))
    print(f"episodes shorter than chunk_size: {short}")
    print(f"episodes only barely >= chunk_size: {barely}")
    if export_summary is not None:
        print(f"skipped_short_segments from export summary: {export_summary.get('stats', {}).get('skipped_short_segments')}")

    if short:
        risks.append(f"{short} exported episodes are shorter than ACT chunk_size={chunk_size}")
        print("[ERROR] at least one episode is shorter than ACT chunk_size.")
    if barely > max(1, len(lengths) // 2):
        risks.append("many exported episodes are only barely longer than chunk_size")
        print("[WARN] many episodes are only barely longer than chunk_size; training may run but have weak chunk diversity.")
    return risks


def inspect_raw_interventions(
    raw_root: Path | None,
    raw_repo_id: str | None,
    exported: LeRobotDataset,
    cfg: dict[str, Any],
) -> list[str]:
    _print_header("6. Raw intervention/export continuity")
    risks: list[str] = []
    if raw_root is None:
        print(
            "[SKIP] raw dataset not provided. This is expected if you only want ACT training schema checks, "
            "but frame_role/intervention_segment_id continuity cannot be proven from standardized exported data."
        )
        return risks

    raw_dataset = _load_dataset("raw", raw_root, raw_repo_id)
    export_cfg = _cfg_section(cfg, "dagger_export")
    export_mode = export_cfg.get("export_mode", "intervention_segments")
    if export_mode != "intervention_segments":
        print(
            f"[SKIP] raw intervention continuity strict check is for intervention_segments; "
            f"current export_mode={export_mode}."
        )
        return risks

    keep_frame_roles = set(export_cfg.get("keep_frame_roles", ["takeover_start", "recovery"]))
    pre_takeover_context = int(export_cfg.get("pre_takeover_context") or 0)
    require_complete = bool(export_cfg.get("require_complete_expert_action", True))
    min_segment_frames = int(export_cfg.get("min_segment_frames") or 1)
    min_episode_len_for_act = int(export_cfg.get("min_episode_len_for_act") or min_segment_frames)
    effective_min_len = max(min_segment_frames, min_episode_len_for_act)

    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    raw_columns = set(raw_hf_dataset.column_names)
    required_raw = {"frame_role", "intervention_segment_id", "expert_label_complete"}
    missing = sorted(required_raw - raw_columns)
    if missing:
        risks.append(f"raw dataset missing intervention audit columns: {missing}")
        print(f"[ERROR] raw dataset missing columns: {missing}")
        return risks

    segments = _iter_export_segments(
        raw_dataset=raw_dataset,
        keep_frame_roles=keep_frame_roles,
        pre_takeover_context=pre_takeover_context,
        require_complete_expert_action=require_complete,
    )
    kept_segments = [seg for seg in segments if len(seg["indices"]) >= effective_min_len]
    print(f"raw segments before length filter: {len(segments)}")
    print(f"raw segments kept by effective_min_len={effective_min_len}: {len(kept_segments)}")
    print(f"exported episodes: {exported.num_episodes}")
    if len(kept_segments) != exported.num_episodes:
        risks.append("kept raw segment count does not match exported episode count")
        print("[ERROR] kept raw segment count does not match exported episode count.")

    exported_lengths = _episode_lengths(exported)
    kept_lengths = [len(seg["indices"]) for seg in kept_segments]
    if kept_lengths and exported_lengths != kept_lengths:
        risks.append("exported episode lengths do not match kept raw segment lengths")
        print("[ERROR] exported episode lengths do not match kept raw segment lengths.")
        print(f"  exported first 20: {exported_lengths[:20]}")
        print(f"  raw kept first 20: {kept_lengths[:20]}")
    elif kept_lengths:
        print("[OK] exported episode lengths match kept raw continuous segment lengths.")

    context_prefix_count = 0
    multi_positive_segment_id = 0
    gap_count = 0
    role_counter: Counter[str] = Counter()
    segment_id_counter: Counter[int] = Counter()
    for seg in kept_segments:
        indices = seg["indices"]
        roles = [_frame_role_at(raw_hf_dataset, idx) for idx in indices]
        segment_ids = [_segment_id_at(raw_hf_dataset, idx, fallback=-1) for idx in indices]
        role_counter.update(roles)
        segment_id_counter.update(sid for sid in segment_ids if sid >= 0)
        if roles and roles[0] not in keep_frame_roles:
            context_prefix_count += 1
        positive_ids = {sid for sid in segment_ids if sid >= 0}
        if len(positive_ids) > 1:
            multi_positive_segment_id += 1
        if any(b != a + 1 for a, b in zip(indices[:-1], indices[1:], strict=False)):
            gap_count += 1

    print(f"frame_role distribution in kept raw-export segments: {dict(role_counter)}")
    print(f"intervention segment id distribution: {dict(segment_id_counter)}")
    print(f"segments with pre_takeover_context prefix: {context_prefix_count}")
    print(f"segments with multiple positive intervention_segment_id values: {multi_positive_segment_id}")
    print(f"segments with raw index gaps: {gap_count}")
    if multi_positive_segment_id:
        risks.append("some exported episodes contain multiple positive intervention_segment_id values")
        print("[ERROR] at least one exported episode may contain unrelated intervention ids.")
    if gap_count:
        risks.append("some exported episodes have raw index gaps")
        print("[ERROR] at least one exported episode has non-contiguous raw indices.")

    print("sample kept segment role/id sequences:")
    for i, seg in enumerate(kept_segments[:5]):
        indices = seg["indices"]
        roles = [_frame_role_at(raw_hf_dataset, idx) for idx in indices[:80]]
        ids = [_segment_id_at(raw_hf_dataset, idx, fallback=-1) for idx in indices[:80]]
        complete = [
            _to_bool(raw_hf_dataset["expert_label_complete"][idx])
            if "expert_label_complete" in raw_columns
            else None
            for idx in indices[:80]
        ]
        print(
            f"  segment[{i}] raw_episode={seg['raw_episode_index']} "
            f"intervention_id={seg['intervention_segment_id']} len={len(indices)}"
        )
        print(f"    roles={roles}")
        print(f"    ids={ids}")
        print(f"    expert_label_complete={complete}")

    action_sources = Counter()
    if "action_source" in raw_columns:
        for value in raw_hf_dataset["action_source"]:
            action_sources[_to_str(value)] += 1
        print(f"raw action_source distribution: {dict(action_sources)}")
    if "expert_label_complete" in raw_columns and "is_expert" in raw_columns:
        incomplete_expert = sum(
            1
            for is_expert, is_complete in zip(
                raw_hf_dataset["is_expert"],
                raw_hf_dataset["expert_label_complete"],
                strict=False,
            )
            if _to_bool(is_expert) and not _to_bool(is_complete)
        )
        print(f"raw expert frames with incomplete expert labels: {incomplete_expert}")
        if incomplete_expert:
            print("[INFO] incomplete expert labels should be dropped by export when require_complete_expert_action=true.")

    return risks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an exported ACT DAgger dataset before training.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config" / "dagger_rounds_cfg.yaml")
    parser.add_argument("--train-config", type=Path, default=Path(__file__).resolve().parent.parent / "config" / "train_cfg.yaml")
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--export-repo-id", type=str, default=None)
    parser.add_argument("--seed-root", type=Path, default=None)
    parser.add_argument("--seed-repo-id", type=str, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--raw-repo-id", type=str, default=None)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--zero-tol", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    train_cfg = _load_yaml(args.train_config)
    export_cfg = _cfg_section(cfg, "dagger_export")

    export_root = Path(_resolve_value(args.export_root, export_cfg.get("output_root")) or "")
    seed_root_value = _resolve_value(args.seed_root, export_cfg.get("seed_root"))
    raw_root_value = _resolve_value(args.raw_root, export_cfg.get("raw_root"))
    if not str(export_root):
        raise ValueError("Provide --export-root or set dagger_export.output_root in the config.")

    export_repo_id = _resolve_value(args.export_repo_id, export_cfg.get("output_repo_id"), _repo_id_from_root(export_root, "exported"))
    seed_repo_id = _resolve_value(args.seed_repo_id, export_cfg.get("seed_repo_id"))
    raw_repo_id = _resolve_value(args.raw_repo_id, export_cfg.get("raw_repo_id"))

    exported = _load_dataset("exported", export_root, export_repo_id)
    seed = None
    if seed_root_value:
        seed = _load_dataset("seed", Path(seed_root_value), seed_repo_id)

    risks: list[str] = []
    inspect_basic_loading(exported)
    risks.extend(inspect_schema(seed, exported))
    risks.extend(inspect_export_summary(export_root))
    summary = _load_json(export_root / "dagger_export_summary.json")
    risks.extend(inspect_actions(exported, sample_count=max(1, args.sample_count), zero_tol=args.zero_tol))
    risks.extend(inspect_act_chunks(exported, train_cfg=train_cfg, export_summary=summary))
    risks.extend(
        inspect_raw_interventions(
            raw_root=Path(raw_root_value) if raw_root_value else None,
            raw_repo_id=raw_repo_id,
            exported=exported,
            cfg=cfg,
        )
    )

    _print_header("7. Final verdict")
    if risks:
        print("基本可以但有风险 / 或不建议继续，取决于 ERROR 项是否涉及 schema/action/chunk。")
        print("Risks:")
        for risk in risks:
            print(f"  - {risk}")
    else:
        print("可以进入 ACT + round DAgger smoke test。")
        print("No blocking risks found by this audit script.")


if __name__ == "__main__":
    main()
