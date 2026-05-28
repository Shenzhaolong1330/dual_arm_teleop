#!/usr/bin/env python

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TRAINING_DEVICES = {"auto", "cuda", "cpu"}
DEVICE_ENV_APPLIED_MARKER = "LEROBOT_TRAINING_DEVICE_CONFIG_APPLIED"


@dataclass
class GPUControlConfig:
    enabled: bool = True
    visible_devices: str | None = ""
    max_num_gpus: int | None = 1
    memory_fraction: float | None = None
    allow_tf32: bool = True
    cudnn_benchmark: bool = False
    empty_cache_on_start: bool = True
    log_gpu_info: bool = True
    fail_if_cuda_unavailable: bool = False

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None, *, legacy_defaults: bool = False) -> "GPUControlConfig":
        if legacy_defaults:
            return cls(
                visible_devices=None,
                max_num_gpus=None,
                memory_fraction=None,
                allow_tf32=True,
                cudnn_benchmark=True,
                empty_cache_on_start=False,
                log_gpu_info=True,
                fail_if_cuda_unavailable=False,
            )

        cfg = cfg or {}
        return cls(
            enabled=_as_bool(cfg.get("enabled", True), "training.gpu.enabled"),
            visible_devices=_normalize_visible_devices(cfg.get("visible_devices", "")),
            max_num_gpus=_optional_int(cfg.get("max_num_gpus", 1), "training.gpu.max_num_gpus"),
            memory_fraction=_optional_float(cfg.get("memory_fraction"), "training.gpu.memory_fraction"),
            allow_tf32=_as_bool(cfg.get("allow_tf32", True), "training.gpu.allow_tf32"),
            cudnn_benchmark=_as_bool(
                cfg.get("cudnn_benchmark", False), "training.gpu.cudnn_benchmark"
            ),
            empty_cache_on_start=_as_bool(
                cfg.get("empty_cache_on_start", True), "training.gpu.empty_cache_on_start"
            ),
            log_gpu_info=_as_bool(cfg.get("log_gpu_info", True), "training.gpu.log_gpu_info"),
            fail_if_cuda_unavailable=_as_bool(
                cfg.get("fail_if_cuda_unavailable", False),
                "training.gpu.fail_if_cuda_unavailable",
            ),
        )

    def validate(self) -> None:
        if self.max_num_gpus is not None and self.max_num_gpus < 0:
            raise ValueError("training.gpu.max_num_gpus must be >= 0 or null.")

        if self.memory_fraction is not None and not 0 < self.memory_fraction <= 1:
            raise ValueError("training.gpu.memory_fraction must be in (0, 1] or null.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "visible_devices": self.visible_devices if self.visible_devices is not None else "",
            "max_num_gpus": self.max_num_gpus,
            "memory_fraction": self.memory_fraction,
            "allow_tf32": self.allow_tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "empty_cache_on_start": self.empty_cache_on_start,
            "log_gpu_info": self.log_gpu_info,
            "fail_if_cuda_unavailable": self.fail_if_cuda_unavailable,
        }


@dataclass
class TrainingDeviceConfig:
    device: str | None = None
    gpu: GPUControlConfig = field(default_factory=GPUControlConfig)
    present_in_config: bool = False

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "TrainingDeviceConfig":
        if cfg is None:
            return cls(device=None, gpu=GPUControlConfig.from_mapping(None, legacy_defaults=True))
        if not isinstance(cfg, dict):
            raise ValueError("training must be a mapping when provided.")

        device = cfg.get("device")
        if device is not None:
            device = str(device).strip().lower()
            if device not in SUPPORTED_TRAINING_DEVICES:
                raise ValueError(
                    "training.device must be one of: auto, cuda, cpu. "
                    f"Got {device!r}."
                )

        gpu_cfg = GPUControlConfig.from_mapping(cfg.get("gpu"))
        gpu_cfg.validate()
        return cls(device=device, gpu=gpu_cfg, present_in_config=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "gpu": self.gpu.to_dict(),
        }


@dataclass
class EarlyCudaVisibleDevicesResult:
    requested_visible_devices: str | None
    final_visible_devices: str | None
    applied: bool
    warning: str | None = None


@dataclass
class TrainingDeviceState:
    requested_device: str
    policy_device: str | None
    final_device: Any
    cuda_available: bool
    cuda_visible_devices: str | None
    cuda_device_count: int
    gpu_names: list[str]
    memory_fraction: float | None
    allow_tf32: bool
    cudnn_benchmark: bool
    warnings: list[str] = field(default_factory=list)
    nvidia_smi_summary: str | None = None


def extract_train_section(cfg: dict[str, Any]) -> dict[str, Any]:
    train_section = cfg.get("train", cfg)
    if not isinstance(train_section, dict):
        raise ValueError("The train config must contain a mapping at the root or under `train`.")
    return train_section


def apply_cuda_visible_devices_from_config_path(config_path: str | Path | None) -> EarlyCudaVisibleDevicesResult:
    if os.environ.get(DEVICE_ENV_APPLIED_MARKER) == "1":
        return EarlyCudaVisibleDevicesResult(
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            applied=False,
        )

    if config_path is None:
        return EarlyCudaVisibleDevicesResult(None, None, applied=False)

    path = Path(config_path).expanduser()
    if not path.is_file():
        return EarlyCudaVisibleDevicesResult(
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            applied=False,
            warning=f"Config file not found while applying early CUDA visibility: {path}",
        )

    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    train_cfg = extract_train_section(raw_cfg)
    return apply_cuda_visible_devices_from_train_cfg(train_cfg)


def apply_cuda_visible_devices_from_train_cfg(train_cfg: dict[str, Any]) -> EarlyCudaVisibleDevicesResult:
    training_cfg = TrainingDeviceConfig.from_mapping(train_cfg.get("training"))
    return apply_cuda_visible_devices(training_cfg)


def apply_cuda_visible_devices(training_cfg: TrainingDeviceConfig) -> EarlyCudaVisibleDevicesResult:
    final_visible = resolve_cuda_visible_devices(training_cfg, os.environ.get("CUDA_VISIBLE_DEVICES"))
    warning = None
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    changed = final_visible is not None and previous_visible != final_visible

    if changed:
        if _torch_already_imported():
            warning = (
                "torch is already imported; changing CUDA_VISIBLE_DEVICES now may be too late "
                "if CUDA has already been initialized."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = final_visible
    os.environ[DEVICE_ENV_APPLIED_MARKER] = "1"

    return EarlyCudaVisibleDevicesResult(
        requested_visible_devices=training_cfg.gpu.visible_devices,
        final_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        applied=changed,
        warning=warning,
    )


def resolve_cuda_visible_devices(
    training_cfg: TrainingDeviceConfig,
    current_env: str | None,
) -> str | None:
    gpu_cfg = training_cfg.gpu
    gpu_cfg.validate()

    if not gpu_cfg.enabled or gpu_cfg.max_num_gpus == 0:
        return ""

    visible_devices = gpu_cfg.visible_devices
    if visible_devices is None:
        return None

    if visible_devices.strip() == "":
        if current_env is None or current_env.strip() == "":
            return None
        visible_devices = current_env

    devices = [part.strip() for part in visible_devices.split(",") if part.strip()]
    if gpu_cfg.max_num_gpus is not None:
        devices = devices[: gpu_cfg.max_num_gpus]
    return ",".join(devices)


def resolve_policy_device_for_config(
    train_cfg: dict[str, Any],
    training_cfg: TrainingDeviceConfig,
) -> str | None:
    policy_cfg = train_cfg.get("policy", {})
    if not isinstance(policy_cfg, dict):
        raise ValueError("train.policy must be a mapping.")

    policy_device = policy_cfg.get("device")
    if not training_cfg.gpu.enabled or training_cfg.device == "cpu":
        return "cpu"
    if training_cfg.device == "cuda":
        return "cuda"
    if policy_device is None:
        return None
    return str(policy_device)


def setup_training_device(
    training_cfg: TrainingDeviceConfig,
    *,
    policy_device: str | None,
) -> TrainingDeviceState:
    import torch

    warnings: list[str] = []
    early_result = apply_cuda_visible_devices(training_cfg)
    if early_result.warning:
        warnings.append(early_result.warning)

    requested_device = _resolve_requested_device(training_cfg, policy_device)
    requested_kind = requested_device.split(":", maxsplit=1)[0]
    if training_cfg.device is not None and policy_device is not None:
        normalized_policy_device = str(policy_device).strip().lower()
        if training_cfg.device != "auto" and normalized_policy_device != training_cfg.device:
            warnings.append(
                "Both training.device and policy.device are set; training.device takes priority. "
                f"training.device={training_cfg.device!r}, policy.device={normalized_policy_device!r}."
            )
        elif training_cfg.device == "auto":
            warnings.append(
                "Both training.device='auto' and policy.device are set; auto resolution takes priority."
            )

    if not training_cfg.gpu.enabled:
        final_device = torch.device("cpu")
        if requested_kind == "cuda":
            warnings.append("training.gpu.enabled=false forces CPU even though CUDA was requested.")
    elif requested_kind == "auto":
        final_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_kind == "cuda":
        if torch.cuda.is_available():
            final_device = torch.device("cuda")
        elif training_cfg.gpu.fail_if_cuda_unavailable:
            raise RuntimeError(
                "training.device='cuda' was requested, but CUDA is not available after applying "
                "the configured GPU visibility."
            )
        else:
            warnings.append("training.device='cuda' was requested, but CUDA is unavailable; falling back to CPU.")
            final_device = torch.device("cpu")
    elif requested_kind == "cpu":
        final_device = torch.device("cpu")
    else:
        final_device = torch.device(requested_device)

    cuda_available = False if not training_cfg.gpu.enabled else torch.cuda.is_available()

    if final_device.type == "cuda":
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            if training_cfg.gpu.fail_if_cuda_unavailable:
                raise RuntimeError("Resolved CUDA device, but torch.cuda.device_count() is 0.")
            warnings.append("Resolved CUDA device, but no CUDA devices are visible; falling back to CPU.")
            final_device = torch.device("cpu")

    if final_device.type == "cuda":
        if training_cfg.gpu.empty_cache_on_start:
            torch.cuda.empty_cache()

        if training_cfg.gpu.memory_fraction is not None:
            # This caps the PyTorch CUDA allocator per process. It does not throttle GPU compute utilization.
            for index in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(
                    training_cfg.gpu.memory_fraction,
                    device=torch.device("cuda", index),
                )

    torch.backends.cudnn.benchmark = training_cfg.gpu.cudnn_benchmark
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = training_cfg.gpu.allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = training_cfg.gpu.allow_tf32

    gpu_names: list[str] = []
    device_count = torch.cuda.device_count() if cuda_available else 0
    if training_cfg.gpu.log_gpu_info and cuda_available:
        for index in range(device_count):
            gpu_names.append(torch.cuda.get_device_name(index))

    return TrainingDeviceState(
        requested_device=requested_device,
        policy_device=policy_device,
        final_device=final_device,
        cuda_available=cuda_available,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_count=device_count,
        gpu_names=gpu_names,
        memory_fraction=training_cfg.gpu.memory_fraction if final_device.type == "cuda" else None,
        allow_tf32=training_cfg.gpu.allow_tf32,
        cudnn_benchmark=training_cfg.gpu.cudnn_benchmark,
        warnings=warnings,
        nvidia_smi_summary=_nvidia_smi_summary() if training_cfg.gpu.log_gpu_info else None,
    )


def log_training_device_state(state: TrainingDeviceState) -> None:
    import logging

    for warning in state.warnings:
        logging.warning(warning)

    logging.info("Training device setup:")
    logging.info("  requested device: %s", state.requested_device)
    logging.info("  policy config device: %s", state.policy_device)
    logging.info("  resolved device: %s", state.final_device.type)
    logging.info("  cuda available: %s", state.cuda_available)
    logging.info("  CUDA_VISIBLE_DEVICES: %s", _display_env_value(state.cuda_visible_devices))
    logging.info("  torch.cuda.device_count(): %s", state.cuda_device_count)
    if state.gpu_names:
        logging.info("  visible GPU names: %s", ", ".join(state.gpu_names))
    logging.info(
        "  memory_fraction: %s",
        state.memory_fraction if state.memory_fraction is not None else "disabled",
    )
    logging.info("  TF32 allowed: %s", state.allow_tf32)
    logging.info("  cudnn.benchmark: %s", state.cudnn_benchmark)
    logging.info(
        "  note: memory_fraction limits PyTorch CUDA memory allocation, not GPU compute utilization."
    )
    if state.nvidia_smi_summary:
        logging.info("  nvidia-smi: %s", state.nvidia_smi_summary)


def _resolve_requested_device(training_cfg: TrainingDeviceConfig, policy_device: str | None) -> str:
    if training_cfg.device is not None:
        return training_cfg.device
    if policy_device:
        return str(policy_device).strip().lower()
    return "auto"


def _normalize_visible_devices(value: Any) -> str | None:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(part).strip() for part in value)
    return str(value).strip()


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "~"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or null. Got {value!r}.") from exc


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "~"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float or null. Got {value!r}.") from exc


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean. Got {value!r}.")


def _torch_already_imported() -> bool:
    return "torch" in sys.modules


def _display_env_value(value: str | None) -> str:
    if value is None:
        return "<unset>"
    if value == "":
        return "<empty>"
    return value


def _nvidia_smi_summary() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return " | ".join(lines[:4]) if lines else None
