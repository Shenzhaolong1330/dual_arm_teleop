from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import yaml

ACT_POLICY_TYPES = {"act", "act_dagger"}
DIFFUSION_POLICY_TYPES = {"diffusion", "dp", "diffusion_policy"}
POLICY_CONFIG_META_KEYS = {"type", "config_path", "reason_config_path", "train_config_path", "name"}
POLICY_CONFIG_RUNTIME_OVERRIDE_KEYS = {"pretrained_path"}
POLICY_CONFIG_MODES = {"train", "reason"}
DEPRECATED_POLICY_CONFIG_FILENAMES = {
    "act_config.yaml": (
        "scripts/policy_config/act_train_config.yaml",
        "scripts/policy_config/act_reason_config.yaml",
    ),
    "diffusion_policy.yaml": (
        "scripts/policy_config/diffusion_train_config.yaml",
        "scripts/policy_config/diffusion_reason_config.yaml",
    ),
}


def normalize_policy_type(policy_type: str) -> str:
    policy_type = str(policy_type).strip().lower()
    if policy_type in ACT_POLICY_TYPES:
        return "act"
    if policy_type in DIFFUSION_POLICY_TYPES:
        return "diffusion"
    raise ValueError(
        f"No config for policy type: {policy_type}. "
        "Supported policy types: act | diffusion | dp | diffusion_policy"
    )


def _validate_policy_config_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in POLICY_CONFIG_MODES:
        raise ValueError(f"Policy config mode must be one of {sorted(POLICY_CONFIG_MODES)}. Got: {mode!r}")
    return mode


def get_default_policy_config_path(policy_type: str, mode: str) -> str:
    policy_type = normalize_policy_type(policy_type)
    mode = _validate_policy_config_mode(mode)
    if policy_type == "act":
        return f"scripts/policy_config/act_{mode}_config.yaml"
    return f"scripts/policy_config/diffusion_{mode}_config.yaml"


def default_policy_config_path(policy_type: str, mode: str = "reason") -> str:
    """Backward-compatible alias for callers that have not been mode-aware yet."""
    return get_default_policy_config_path(policy_type, mode)


def warn_if_deprecated_policy_config_path(path: Path) -> None:
    replacements = DEPRECATED_POLICY_CONFIG_FILENAMES.get(path.name)
    if replacements is None:
        return
    logging.warning(
        "[DEPRECATED] Policy config file %s has been split. Use %s for training "
        "and %s for record/reason/deploy configs.",
        path,
        replacements[0],
        replacements[1],
    )


def resolve_policy_config_path(
    policy_cfg: Dict[str, Any],
    scripts_dir: Path,
    project_root: Path,
    *,
    mode: str = "reason",
    config_path_key: str | None = None,
) -> Path:
    """Resolve policy config paths without depending on the process cwd.

    Relative paths are checked in this order:
    1. Project root, so `scripts/policy_config/act_reason_config.yaml` works
       from the lerobot_dual_arm_teleop project root.
    2. The scripts directory, so `policy_config/act_reason_config.yaml` also works.
    """
    mode = _validate_policy_config_mode(mode)
    policy_type = normalize_policy_type(policy_cfg.get("type", ""))

    if config_path_key is not None:
        raw_path = policy_cfg.get(config_path_key)
    elif mode == "train":
        raw_path = policy_cfg.get("train_config_path") or policy_cfg.get("config_path")
    else:
        raw_path = policy_cfg.get("reason_config_path") or policy_cfg.get("config_path")
    raw_path = raw_path or get_default_policy_config_path(policy_type, mode)

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        if path.is_file():
            resolved = path.resolve()
            warn_if_deprecated_policy_config_path(resolved)
            return resolved
        raise FileNotFoundError(f"Policy config file does not exist: {path}")

    candidates = [project_root / path, scripts_dir / path]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            warn_if_deprecated_policy_config_path(resolved)
            return resolved

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Policy config file '{raw_path}' was not found. Tried:\n{candidate_text}"
    )


def load_policy_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Policy config yaml must contain a mapping at the top level: {path}")
    return cfg


def _policy_label(policy_type: str) -> str:
    return "Diffusion" if policy_type == "diffusion" else "ACT"


def get_act_config_field_names() -> set[str]:
    from lerobot.policies import ACTConfig

    return {config_field.name for config_field in fields(ACTConfig)}


def get_diffusion_config_field_names() -> set[str]:
    from lerobot.policies import DiffusionConfig

    return {config_field.name for config_field in fields(DiffusionConfig)}


def _normalization_mode_from_config(value: Any) -> Any:
    from lerobot.configs.types import NormalizationMode

    if isinstance(value, NormalizationMode):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return NormalizationMode[text]
        except KeyError:
            try:
                return NormalizationMode(text)
            except ValueError as exc:
                valid = [mode.name for mode in NormalizationMode]
                raise ValueError(
                    f"Unknown normalization mode {value!r}. Expected one of: {valid}"
                ) from exc
    raise ValueError(
        "normalization_mapping values must be NormalizationMode names or strings. "
        f"Got {type(value).__name__}: {value!r}"
    )


def convert_normalization_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "`normalization_mapping` must be a mapping such as "
            "{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}."
        )
    return {str(key): _normalization_mode_from_config(mode) for key, mode in value.items()}


def _feature_type_from_config(value: Any) -> Any:
    from lerobot.configs.types import FeatureType

    if isinstance(value, FeatureType):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return FeatureType[text]
        except KeyError:
            try:
                return FeatureType(text)
            except ValueError as exc:
                valid = [feature_type.name for feature_type in FeatureType]
                raise ValueError(f"Unknown feature type {value!r}. Expected one of: {valid}") from exc
    raise ValueError(f"Feature type must be a string. Got {type(value).__name__}: {value!r}")


def _convert_policy_features(value: Any, field_name: str) -> dict[str, Any]:
    from lerobot.configs.types import PolicyFeature

    if not isinstance(value, dict):
        raise ValueError(f"`{field_name}` must be a mapping of feature names to feature specs.")

    converted: dict[str, Any] = {}
    for name, feature in value.items():
        if isinstance(feature, PolicyFeature):
            converted[str(name)] = feature
            continue
        if not isinstance(feature, dict):
            raise ValueError(
                f"`{field_name}.{name}` must be a mapping with `type` and `shape` fields."
            )
        if "type" not in feature or "shape" not in feature:
            raise ValueError(f"`{field_name}.{name}` must include `type` and `shape`.")
        shape = feature["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"`{field_name}.{name}.shape` must be a list or tuple.")
        converted[str(name)] = PolicyFeature(
            type=_feature_type_from_config(feature["type"]),
            shape=tuple(shape),
        )
    return converted


def normalize_temporal_ensemble_coeff(value: Any) -> float | None:
    """Treat non-positive and None-like values as disabled temporal ensembling."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "~"}:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(
                "`temporal_ensemble_coeff` must be a number, null, or None-like string. "
                f"Got: {value!r}"
            ) from exc

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None

    raise ValueError(
        "`temporal_ensemble_coeff` must be numeric or null-like. "
        f"Got type: {type(value).__name__}"
    )


def _validate_optional_string_field(cfg: Dict[str, Any], field_name: str) -> None:
    value = cfg.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"`{field_name}` must be a string or null. Got {type(value).__name__}.")


def _validate_tags(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise ValueError("`tags` must be null or a list of strings.")


def validate_unknown_fields(
    policy_yaml_dict: Dict[str, Any],
    allowed_fields: set[str],
    config_path: Path | None = None,
    *,
    policy_type: str = "act",
) -> None:
    unknown_fields = sorted(set(policy_yaml_dict) - allowed_fields)
    if unknown_fields:
        source = f" in {config_path}" if config_path is not None else ""
        raise ValueError(
            f"Unknown {_policy_label(policy_type)} policy config field(s){source}: {unknown_fields}. "
            f"Allowed fields are: {sorted(allowed_fields)}"
        )


def merge_legacy_policy_fields(
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None,
    allowed_fields: set[str],
    source_name: str,
    *,
    policy_type: str = "act",
) -> dict[str, Any]:
    merged = dict(policy_yaml_dict)
    legacy_policy_dict = legacy_policy_dict or {}
    legacy_overrides = {
        key: value
        for key, value in legacy_policy_dict.items()
        if key not in POLICY_CONFIG_META_KEYS and key in allowed_fields
    }
    deprecated_overrides = {
        key: value
        for key, value in legacy_overrides.items()
        if key not in POLICY_CONFIG_RUNTIME_OVERRIDE_KEYS
    }
    if deprecated_overrides:
        logging.warning(
            "[DEPRECATED] %s policy field(s) still defined in %s: %s. "
            "Please move them to scripts/policy_config/*.yaml. "
            "For backward compatibility, these values override the policy yaml for this run.",
            _policy_label(policy_type),
            source_name,
            sorted(deprecated_overrides),
        )
    if legacy_overrides:
        merged.update(legacy_overrides)
    return merged


def _act_init_device_and_restore_device(device: Any) -> tuple[Any, str | None]:
    """Allow indexed CUDA devices while keeping PreTrainedConfig's availability checks.

    This LeRobot version validates only base device strings like `cuda`, but PyTorch
    and downstream `.to(...)` calls accept `cuda:0`. We initialize ACTConfig with the
    base device, then restore the indexed string if that CUDA device is available.
    """
    if device is None:
        return None, None
    if not isinstance(device, str):
        raise ValueError(f"`device` must be a string or null. Got {type(device).__name__}.")

    text = device.strip()
    if text.startswith("cuda:"):
        return "cuda", text
    return text, None


def _restore_indexed_device_if_available(policy_cfg: Any, indexed_device: str | None) -> None:
    if indexed_device is None:
        return

    try:
        import torch

        torch_device = torch.device(indexed_device)
    except Exception as exc:
        raise ValueError(f"Invalid torch device string: {indexed_device!r}") from exc

    if torch_device.type != "cuda":
        policy_cfg.device = indexed_device
        return
    if not torch.cuda.is_available():
        logging.warning(
            "Device '%s' is not available. Keeping LeRobot-selected device '%s'.",
            indexed_device,
            policy_cfg.device,
        )
        return
    if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Configured device '{indexed_device}' is not available. "
            f"CUDA device count: {torch.cuda.device_count()}."
        )
    policy_cfg.device = indexed_device


def build_act_config(
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None = None,
    *,
    legacy_source_name: str = "config",
    runtime_overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> PreTrainedConfig:
    from lerobot.policies import ACTConfig

    allowed_fields = get_act_config_field_names()
    validate_unknown_fields(policy_yaml_dict, allowed_fields, config_path=config_path, policy_type="act")

    act_kwargs = merge_legacy_policy_fields(
        policy_yaml_dict,
        legacy_policy_dict,
        allowed_fields,
        source_name=legacy_source_name,
        policy_type="act",
    )
    if runtime_overrides:
        unknown_runtime_fields = sorted(set(runtime_overrides) - allowed_fields)
        if unknown_runtime_fields:
            raise ValueError(f"Unknown ACT runtime override field(s): {unknown_runtime_fields}")
        act_kwargs.update({key: value for key, value in runtime_overrides.items() if value is not None})

    for feature_field in ("input_features", "output_features"):
        if feature_field in act_kwargs:
            if act_kwargs[feature_field] == {}:
                act_kwargs.pop(feature_field)
            else:
                act_kwargs[feature_field] = _convert_policy_features(
                    act_kwargs[feature_field], feature_field
                )

    if "normalization_mapping" in act_kwargs:
        act_kwargs["normalization_mapping"] = convert_normalization_mapping(
            act_kwargs["normalization_mapping"]
        )

    if "temporal_ensemble_coeff" in act_kwargs:
        act_kwargs["temporal_ensemble_coeff"] = normalize_temporal_ensemble_coeff(
            act_kwargs["temporal_ensemble_coeff"]
        )

    if (
        act_kwargs.get("temporal_ensemble_coeff") is not None
        and act_kwargs.get("n_action_steps", 100) != 1
    ):
        raise ValueError(
            "temporal_ensemble_coeff is enabled, so n_action_steps must be 1 "
            "for ACT temporal ensembling."
        )

    for field_name in (
        "device",
        "repo_id",
        "license",
        "pretrained_path",
        "pretrained_backbone_weights",
    ):
        _validate_optional_string_field(act_kwargs, field_name)
    if "tags" in act_kwargs:
        _validate_tags(act_kwargs["tags"])

    init_device, indexed_device = _act_init_device_and_restore_device(act_kwargs.get("device"))
    act_kwargs["device"] = init_device
    act_config = ACTConfig(**act_kwargs)
    _restore_indexed_device_if_available(act_config, indexed_device)
    return act_config


def _list_or_tuple_to_tuple(value: Any, field_name: str, *, length: int | None = None) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"`{field_name}` must be a list or tuple. Got {type(value).__name__}.")
    converted = tuple(value)
    if length is not None and len(converted) != length:
        raise ValueError(f"`{field_name}` must contain exactly {length} values. Got {converted!r}.")
    return converted


def _normalize_diffusion_scheduler_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`noise_scheduler_type` must be a string. Got {type(value).__name__}.")
    return value.strip().upper()


def _normalize_diffusion_prediction_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`prediction_type` must be a string. Got {type(value).__name__}.")
    return value.strip().lower()


def validate_diffusion_config(policy_dict: Dict[str, Any]) -> None:
    def positive_int(field_name: str) -> None:
        value = policy_dict.get(field_name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"`{field_name}` must be > 0. Got {value!r}.")

    for field_name in (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "num_train_timesteps",
        "kernel_size",
        "n_groups",
        "diffusion_step_embed_dim",
        "spatial_softmax_num_keypoints",
    ):
        positive_int(field_name)

    if policy_dict.get("num_inference_steps") is not None:
        positive_int("num_inference_steps")

    if policy_dict.get("drop_n_last_frames") is not None and int(policy_dict["drop_n_last_frames"]) < 0:
        raise ValueError(
            f"`drop_n_last_frames` must be >= 0. Got {policy_dict['drop_n_last_frames']!r}."
        )
    if policy_dict.get("scheduler_warmup_steps") is not None and int(policy_dict["scheduler_warmup_steps"]) < 0:
        raise ValueError(
            f"`scheduler_warmup_steps` must be >= 0. Got {policy_dict['scheduler_warmup_steps']!r}."
        )

    n_obs_steps = int(policy_dict.get("n_obs_steps", 2))
    horizon = int(policy_dict.get("horizon", 16))
    n_action_steps = int(policy_dict.get("n_action_steps", 8))
    max_action_steps = horizon - n_obs_steps + 1
    if n_action_steps > max_action_steps:
        raise ValueError(
            "`n_action_steps` must be <= horizon - n_obs_steps + 1 for DiffusionPolicy "
            f"select_action caching. Got n_action_steps={n_action_steps}, horizon={horizon}, "
            f"n_obs_steps={n_obs_steps}."
        )

    crop_shape = policy_dict.get("crop_shape")
    if crop_shape is not None:
        if not isinstance(crop_shape, tuple) or len(crop_shape) != 2:
            raise ValueError("`crop_shape` must be null or a 2-item list/tuple.")
        if any(int(dim) <= 0 for dim in crop_shape):
            raise ValueError(f"`crop_shape` dimensions must be > 0. Got {crop_shape!r}.")

    down_dims = policy_dict.get("down_dims")
    if down_dims is not None:
        if not isinstance(down_dims, tuple) or len(down_dims) == 0:
            raise ValueError("`down_dims` must be a non-empty list/tuple.")
        if any(int(dim) <= 0 for dim in down_dims):
            raise ValueError(f"`down_dims` values must be > 0. Got {down_dims!r}.")

    optimizer_betas = policy_dict.get("optimizer_betas")
    if optimizer_betas is not None:
        if not isinstance(optimizer_betas, tuple) or len(optimizer_betas) != 2:
            raise ValueError("`optimizer_betas` must be a 2-item list/tuple.")
        if any(not 0 <= float(beta) < 1 for beta in optimizer_betas):
            raise ValueError(f"`optimizer_betas` must be in [0, 1). Got {optimizer_betas!r}.")

    beta_start = policy_dict.get("beta_start")
    beta_end = policy_dict.get("beta_end")
    if beta_start is not None and float(beta_start) <= 0:
        raise ValueError(f"`beta_start` must be > 0. Got {beta_start!r}.")
    if beta_end is not None and float(beta_end) <= 0:
        raise ValueError(f"`beta_end` must be > 0. Got {beta_end!r}.")
    if beta_start is not None and beta_end is not None and float(beta_start) > float(beta_end):
        raise ValueError(f"`beta_start` must be <= `beta_end`. Got {beta_start!r} > {beta_end!r}.")


def build_diffusion_config(
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None = None,
    *,
    legacy_source_name: str = "config",
    runtime_overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
    mode: str | None = None,
) -> PreTrainedConfig:
    from lerobot.policies import DiffusionConfig

    allowed_fields = get_diffusion_config_field_names()
    validate_unknown_fields(
        policy_yaml_dict,
        allowed_fields,
        config_path=config_path,
        policy_type="diffusion",
    )

    diffusion_kwargs = merge_legacy_policy_fields(
        policy_yaml_dict,
        legacy_policy_dict,
        allowed_fields,
        source_name=legacy_source_name,
        policy_type="diffusion",
    )
    if runtime_overrides:
        unknown_runtime_fields = sorted(set(runtime_overrides) - allowed_fields)
        if unknown_runtime_fields:
            raise ValueError(f"Unknown Diffusion runtime override field(s): {unknown_runtime_fields}")
        diffusion_kwargs.update({key: value for key, value in runtime_overrides.items() if value is not None})

    for feature_field in ("input_features", "output_features"):
        if feature_field in diffusion_kwargs:
            if diffusion_kwargs[feature_field] == {}:
                diffusion_kwargs.pop(feature_field)
            else:
                diffusion_kwargs[feature_field] = _convert_policy_features(
                    diffusion_kwargs[feature_field], feature_field
                )

    if "normalization_mapping" in diffusion_kwargs:
        diffusion_kwargs["normalization_mapping"] = convert_normalization_mapping(
            diffusion_kwargs["normalization_mapping"]
        )

    tuple_fields = {
        "crop_shape": 2,
        "down_dims": None,
        "optimizer_betas": 2,
    }
    for field_name, length in tuple_fields.items():
        if field_name in diffusion_kwargs and diffusion_kwargs[field_name] is not None:
            diffusion_kwargs[field_name] = _list_or_tuple_to_tuple(
                diffusion_kwargs[field_name],
                field_name,
                length=length,
            )

    if "noise_scheduler_type" in diffusion_kwargs:
        diffusion_kwargs["noise_scheduler_type"] = _normalize_diffusion_scheduler_type(
            diffusion_kwargs["noise_scheduler_type"]
        )
    if "prediction_type" in diffusion_kwargs:
        diffusion_kwargs["prediction_type"] = _normalize_diffusion_prediction_type(
            diffusion_kwargs["prediction_type"]
        )

    for field_name in (
        "device",
        "repo_id",
        "license",
        "pretrained_path",
        "pretrained_backbone_weights",
        "vision_backbone",
        "noise_scheduler_type",
        "beta_schedule",
        "prediction_type",
        "scheduler_name",
    ):
        _validate_optional_string_field(diffusion_kwargs, field_name)
    if "tags" in diffusion_kwargs:
        _validate_tags(diffusion_kwargs["tags"])

    if (
        mode is not None
        and _validate_policy_config_mode(mode) == "reason"
        and bool(diffusion_kwargs.get("crop_is_random", False))
    ):
        logging.warning(
            "Diffusion reason config should set crop_is_random=false to avoid stochastic deployment. "
            "Current config_path=%s still has crop_is_random=true.",
            config_path,
        )

    validate_diffusion_config(diffusion_kwargs)
    init_device, indexed_device = _act_init_device_and_restore_device(diffusion_kwargs.get("device"))
    diffusion_kwargs["device"] = init_device
    diffusion_config = DiffusionConfig(**diffusion_kwargs)
    _restore_indexed_device_if_available(diffusion_config, indexed_device)
    return diffusion_config


def build_policy_config(
    policy_type: str,
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None = None,
    *,
    legacy_source_name: str = "config",
    runtime_overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
    mode: str | None = None,
) -> PreTrainedConfig:
    policy_type = normalize_policy_type(policy_type)
    if policy_type == "act":
        return build_act_config(
            policy_yaml_dict,
            legacy_policy_dict=legacy_policy_dict,
            legacy_source_name=legacy_source_name,
            runtime_overrides=runtime_overrides,
            config_path=config_path,
        )
    if policy_type == "diffusion":
        return build_diffusion_config(
            policy_yaml_dict,
            legacy_policy_dict=legacy_policy_dict,
            legacy_source_name=legacy_source_name,
            runtime_overrides=runtime_overrides,
            config_path=config_path,
            mode=mode,
        )
    raise ValueError(
        f"No config for policy type: {policy_type}. "
        "Supported policy types: act | diffusion | dp | diffusion_policy"
    )


def self_test_policy_config_loader() -> None:
    from lerobot.configs.types import NormalizationMode

    def expect_error(case_name: str, cfg: Dict[str, Any], expected: str, *, policy_type: str = "act") -> None:
        try:
            if policy_type == "diffusion":
                build_diffusion_config(cfg)
            else:
                build_act_config(cfg)
        except Exception as exc:
            if expected not in str(exc):
                raise AssertionError(
                    f"{case_name}: expected error containing {expected!r}, got {exc!r}"
                ) from exc
            return
        raise AssertionError(f"{case_name}: expected an error but config loaded successfully.")

    cfg = build_act_config(
        {
            "temporal_ensemble_coeff": None,
            "n_action_steps": 100,
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        }
    )
    if cfg.temporal_ensemble_coeff is not None or cfg.n_action_steps != 100:
        raise AssertionError("temporal_ensemble_coeff: null with n_action_steps: 100 should pass.")
    if cfg.normalization_mapping["VISUAL"] is not NormalizationMode.MEAN_STD:
        raise AssertionError("normalization_mapping VISUAL was not converted to NormalizationMode.")

    cfg = build_act_config({"temporal_ensemble_coeff": 0.01, "n_action_steps": 1})
    if cfg.temporal_ensemble_coeff != 0.01 or cfg.n_action_steps != 1:
        raise AssertionError("temporal_ensemble_coeff: 0.01 with n_action_steps: 1 should pass.")

    expect_error(
        "temporal ensemble n_action_steps guard",
        {"temporal_ensemble_coeff": 0.01, "n_action_steps": 10},
        "n_action_steps must be 1",
    )
    expect_error(
        "unknown ACT field guard",
        {"temporal_ensemble_coef": 0.01},
        "Unknown ACT policy config field",
    )

    diffusion_cfg = build_diffusion_config(
        {
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MIN_MAX",
                "ACTION": "MIN_MAX",
            },
            "crop_shape": [84, 84],
            "down_dims": [512, 1024, 2048],
            "optimizer_betas": [0.95, 0.999],
            "noise_scheduler_type": "ddpm",
        }
    )
    if diffusion_cfg.normalization_mapping["STATE"] is not NormalizationMode.MIN_MAX:
        raise AssertionError("Diffusion normalization_mapping was not converted to NormalizationMode.")
    if diffusion_cfg.crop_shape != (84, 84) or diffusion_cfg.down_dims != (512, 1024, 2048):
        raise AssertionError("Diffusion list fields were not converted to tuples.")
    if diffusion_cfg.noise_scheduler_type != "DDPM":
        raise AssertionError("Diffusion noise_scheduler_type was not normalized.")

    expect_error(
        "unknown Diffusion field guard",
        {"num_inference_stepz": 10},
        "Unknown Diffusion policy config field",
        policy_type="diffusion",
    )
    expect_error(
        "Diffusion action horizon guard",
        {"horizon": 8, "n_obs_steps": 4, "n_action_steps": 8, "down_dims": [128, 256, 512]},
        "n_action_steps",
        policy_type="diffusion",
    )
    logging.info("====== [POLICY CONFIG SELF-TEST] OK ======")
