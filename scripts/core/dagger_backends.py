#!/usr/bin/env python

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.core.policy_config_utils import load_policy_yaml, resolve_policy_config_path


ONLY_ACT_BACKEND_MESSAGE = "Only ACT backend is implemented currently."
EXPORT_DAGGER_DATASET_KEYS = (
    "raw_repo_id",
    "raw_root",
    "output_repo_id",
    "output_root",
    "seed_repo_id",
    "seed_root",
    "keep_frame_roles",
    "min_segment_frames",
    "min_episode_len_for_act",
    "export_mode",
    "pre_takeover_context",
    "require_complete_expert_action",
    "full_episode",
    "intervention_segments",
    "hybrid",
    "gripper_action_indices",
    "gripper_open_threshold",
    "overwrite",
    "image_writer_processes",
    "image_writer_threads",
)


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_project_root() -> Path:
    return _default_scripts_dir().parent


def _act_chunk_size_from_train_cfg(train_cfg: dict[str, Any]) -> int:
    policy_cfg = train_cfg.get("policy", {})
    if not isinstance(policy_cfg, dict):
        raise ValueError("train.policy must be a mapping when resolving ACT chunk_size.")
    if policy_cfg.get("chunk_size") is not None:
        return int(policy_cfg["chunk_size"])

    policy_config_path = resolve_policy_config_path(
        policy_cfg,
        scripts_dir=_default_scripts_dir(),
        project_root=_default_project_root(),
        mode="train",
    )
    policy_yaml = load_policy_yaml(policy_config_path)
    if policy_yaml.get("chunk_size") is None:
        return 1
    return int(policy_yaml["chunk_size"])


class PolicyRunner:
    policy_type: str = "base"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = copy.deepcopy(cfg or {})

    def prepare_record_cfg(
        self,
        record_cfg: dict[str, Any],
        checkpoint_path: str | Path,
        round_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class DAggerExportProfile:
    profile_type: str = "base"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = copy.deepcopy(cfg or {})

    def prepare_export_cfg(
        self,
        export_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        round_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def export(self, export_cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class PolicyTrainer:
    trainer_type: str = "base"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = copy.deepcopy(cfg or {})

    @property
    def train_cfg_path(self) -> str | Path | None:
        return self.cfg.get("train_cfg_path")

    def prepare_train_cfg(
        self,
        base_train_cfg: dict[str, Any],
        aggregated_dataset_path: str | Path,
        repo_id: str,
        checkpoint_in: str | Path | None,
        output_dir: str | Path,
        round_id: int,
        resolved_steps: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def train(self, train_cfg: dict[str, Any]) -> str:
        raise NotImplementedError


class ACTPolicyRunner(PolicyRunner):
    policy_type = "act"

    def prepare_record_cfg(
        self,
        record_cfg: dict[str, Any],
        checkpoint_path: str | Path,
        round_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        record_section = copy.deepcopy(record_cfg)
        policy = record_section.setdefault("policy", {})

        if policy.get("type") == "act_dagger":
            policy["type"] = "act"
        policy.setdefault("type", "act")
        if policy.get("type") != "act":
            raise ValueError("ACTPolicyRunner requires record.policy.type to be ACT-compatible.")

        if "device" in self.cfg and self.cfg["device"] is not None:
            policy.setdefault("device", self.cfg["device"])

        # Checkpoint handoff is owned by the round controller; backend config
        # may document a placeholder but must not pin a checkpoint across rounds.
        policy["pretrained_path"] = str(checkpoint_path)
        return record_section


class ACTChunkExportProfile(DAggerExportProfile):
    profile_type = "act_chunk"

    def prepare_export_cfg(
        self,
        export_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        round_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        export_section = copy.deepcopy(export_cfg)

        # Priority: backend config -> legacy round-level config -> ACT chunk size.
        min_episode_len = self.cfg.get("min_episode_len")
        if min_episode_len is None:
            min_episode_len = round_cfg.get("min_episode_len_for_act")
        if min_episode_len is None:
            min_episode_len = _act_chunk_size_from_train_cfg(train_cfg)

        pre_takeover_context = (
            self.cfg["pre_takeover_context"]
            if "pre_takeover_context" in self.cfg
            else round_cfg.get("pre_takeover_context", 0)
        )
        require_complete_expert_action = (
            self.cfg["require_complete_expert_action"]
            if "require_complete_expert_action" in self.cfg
            else round_cfg.get("require_complete_expert_action", True)
        )
        export_mode = str(self.cfg.get("export_mode", "intervention_segments")).lower()
        keep_frame_roles = self.cfg.get("keep_frame_roles", ["takeover_start", "recovery"])
        full_episode_cfg = copy.deepcopy(self.cfg.get("full_episode") or {})
        intervention_segments_cfg = copy.deepcopy(self.cfg.get("intervention_segments") or {})
        hybrid_cfg = copy.deepcopy(self.cfg.get("hybrid") or {})

        intervention_segments_cfg.setdefault("keep_frame_roles", keep_frame_roles)
        intervention_segments_cfg.setdefault("label_source", "expert_action")
        intervention_segments_cfg.setdefault(
            "require_complete_expert_action",
            bool(require_complete_expert_action),
        )

        export_section["label_source"] = "expert_action"
        export_section["standard_action_field"] = "action"
        export_section["keep_frame_roles"] = tuple(keep_frame_roles)
        export_section["dropped_frame_roles"] = ["policy", "reset", "ignore"]
        export_section["min_episode_len_for_act"] = int(min_episode_len)
        export_section["export_mode"] = export_mode
        export_section["pre_takeover_context"] = int(pre_takeover_context or 0)
        export_section["require_complete_expert_action"] = bool(require_complete_expert_action)
        export_section["full_episode"] = full_episode_cfg
        export_section["intervention_segments"] = intervention_segments_cfg
        export_section["hybrid"] = hybrid_cfg
        if "gripper_action_indices" in self.cfg:
            export_section["gripper_action_indices"] = copy.deepcopy(self.cfg["gripper_action_indices"])
        if "gripper_open_threshold" in self.cfg:
            export_section["gripper_open_threshold"] = self.cfg["gripper_open_threshold"]
        return export_section

    def export(self, export_cfg: dict[str, Any]) -> dict[str, Any]:
        from scripts.core.run_dagger_export import export_dagger_dataset

        export_kwargs = {key: export_cfg[key] for key in EXPORT_DAGGER_DATASET_KEYS if key in export_cfg}
        return export_dagger_dataset(**export_kwargs)


class ACTTrainer(PolicyTrainer):
    trainer_type = "act"

    def prepare_train_cfg(
        self,
        base_train_cfg: dict[str, Any],
        aggregated_dataset_path: str | Path,
        repo_id: str,
        checkpoint_in: str | Path | None,
        output_dir: str | Path,
        round_id: int,
        resolved_steps: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        train_section = copy.deepcopy(base_train_cfg)
        train_section["output_dir"] = str(output_dir)
        train_section["job_name"] = f"act_dagger_round_{round_id:03d}"
        train_section["resume"] = False
        if resolved_steps is not None:
            train_section["steps"] = int(resolved_steps)

        train_section.setdefault("dataset", {})
        train_section["dataset"]["repo_id"] = repo_id
        train_section["dataset"]["root"] = str(aggregated_dataset_path)

        train_section.setdefault("policy", {})
        train_section["policy"]["type"] = "act_dagger"
        if checkpoint_in is None:
            train_section["policy"].pop("pretrained_path", None)
        else:
            train_section["policy"]["pretrained_path"] = str(checkpoint_in)

        seed_repo_id = kwargs.get("seed_repo_id") or base_train_cfg.get("dataset", {}).get("repo_id")
        seed_root = kwargs.get("seed_root") or base_train_cfg.get("dataset", {}).get("root")

        dagger_section = train_section.setdefault("dagger", {})
        dagger_dataset = dagger_section.setdefault("dataset", {})
        dagger_dataset.update(
            {
                "aggregated_repo_id": repo_id,
                "aggregated_root": str(aggregated_dataset_path),
                "seed_repo_id": seed_repo_id,
                "seed_root": str(seed_root) if seed_root is not None else None,
                "resume_aggregation": True,
                "copy_seed_if_missing": False,
            }
        )

        dagger_training = dagger_section.setdefault("training", {})
        dagger_training["rounds"] = 1
        dagger_training["steps_per_round"] = int(train_section.get("steps", 10_000))
        dagger_training.setdefault("batch_size", train_section.get("batch_size", 8))
        dagger_training.setdefault("num_workers", train_section.get("num_workers", 4))
        dagger_training.setdefault("log_freq", train_section.get("log_freq", 100))
        dagger_training["save_checkpoint"] = True
        train_section["save_checkpoint"] = True

        dagger_sampling = kwargs.get("dagger_sampling")
        if dagger_sampling is not None:
            train_section["dagger_sampling"] = copy.deepcopy(dagger_sampling)

        return train_section

    def train(self, train_cfg: dict[str, Any]) -> str:
        from scripts.utils.training_device import apply_cuda_visible_devices_from_train_cfg

        apply_cuda_visible_devices_from_train_cfg(train_cfg)

        from scripts.core.run_train import run_act_dagger_from_train_cfg

        run_act_dagger_from_train_cfg(train_cfg)
        return str(_resolve_latest_pretrained(Path(train_cfg["output_dir"])))


@dataclass
class PolicyBackend:
    backend_type: str
    runner: PolicyRunner
    export_profile: DAggerExportProfile
    trainer: PolicyTrainer

    def metadata(self) -> dict[str, str]:
        return {
            "type": self.backend_type,
            "runner": type(self.runner).__name__,
            "export_profile": type(self.export_profile).__name__,
            "trainer": type(self.trainer).__name__,
        }


def _resolve_latest_pretrained(train_output_dir: Path) -> Path:
    last = train_output_dir / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        return last
    raise FileNotFoundError(
        f"No selected training checkpoint found: {last}. "
        "Expected train output to contain checkpoints/last/pretrained_model."
    )


def _require_act(value: str) -> None:
    if value != "act":
        raise ValueError(ONLY_ACT_BACKEND_MESSAGE)


def make_policy_backend(cfg: dict[str, Any] | None = None) -> PolicyBackend:
    backend_cfg = copy.deepcopy(cfg or {})
    backend_type = str(backend_cfg.get("type", "act")).lower()
    _require_act(backend_type)

    runner_cfg = copy.deepcopy(backend_cfg.get("runner") or {})
    runner_type = str(runner_cfg.get("type", "act")).lower()
    _require_act(runner_type)

    export_cfg = copy.deepcopy(backend_cfg.get("export") or {})
    export_profile = str(export_cfg.get("profile", "act_chunk")).lower()
    if export_profile != "act_chunk":
        raise ValueError(ONLY_ACT_BACKEND_MESSAGE)

    trainer_cfg = copy.deepcopy(backend_cfg.get("trainer") or {})
    trainer_type = str(trainer_cfg.get("type", "act")).lower()
    _require_act(trainer_type)

    return PolicyBackend(
        backend_type="act",
        runner=ACTPolicyRunner(runner_cfg),
        export_profile=ACTChunkExportProfile(export_cfg),
        trainer=ACTTrainer(trainer_cfg),
    )
