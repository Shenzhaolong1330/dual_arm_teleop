"""Persistent schema and compatibility checks for Flexiv dual-arm state data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .flexiv_force_contract import (
    GRIPPER_FORCE_SOURCE,
    GRIPPER_FORCE_UNIT,
    WRENCH_FRAME,
    WRENCH_ORDER,
    WRENCH_SOURCE,
    WRENCH_UNITS,
)

FLEXIV_STATE_SCHEMA = "flexiv_abs_rot6d_raw_force_v3"
FLEXIV_STATE_DIM = 48
FLEXIV_ACTION_DIM = 14
FLEXIV_KINEMATIC_STATE_DIM = 34
FLEXIV_FORCE_STATE_DIM = 14
FLEXIV_STATE_ROTATION_REPRESENTATION = "rotation_6d"
FLEXIV_STATE_ROTATION_REFERENCE = "absolute_rdk_world_base"
FLEXIV_ACTION_ROTATION_REPRESENTATION = "rotvec"
FLEXIV_ROTATION6D_CONVENTION = "matrix_columns_0_1"
FLEXIV_ROTATION6D_ORDER = (
    "c0x",
    "c0y",
    "c0z",
    "c1x",
    "c1y",
    "c1z",
)
FLEXIV_SCHEMA_INFO_KEY = "robot_state_schema"
FLEXIV_SCHEMA_CHECKPOINT_FILENAME = "flexiv_state_schema.json"

STATE_POSITION_FIELDS = ("x", "y", "z")
STATE_ROTATION_6D_FIELDS = tuple(
    f"{column}{axis}"
    for column in ("c0", "c1")
    for axis in ("x", "y", "z")
)
ACTION_DELTA_POSITION_FIELDS = STATE_POSITION_FIELDS
ACTION_DELTA_ROTATION_FIELDS = ("rx", "ry", "rz")
ACTION_DELTA_POSE_FIELDS = ACTION_DELTA_POSITION_FIELDS + ACTION_DELTA_ROTATION_FIELDS
STATE_FORCE_FIELDS = (
    "left_ee_ext_wrench_in_tcp_raw.fx",
    "left_ee_ext_wrench_in_tcp_raw.fy",
    "left_ee_ext_wrench_in_tcp_raw.fz",
    "left_ee_ext_wrench_in_tcp_raw.mx",
    "left_ee_ext_wrench_in_tcp_raw.my",
    "left_ee_ext_wrench_in_tcp_raw.mz",
    "left_gripper_force",
    "right_ee_ext_wrench_in_tcp_raw.fx",
    "right_ee_ext_wrench_in_tcp_raw.fy",
    "right_ee_ext_wrench_in_tcp_raw.fz",
    "right_ee_ext_wrench_in_tcp_raw.mx",
    "right_ee_ext_wrench_in_tcp_raw.my",
    "right_ee_ext_wrench_in_tcp_raw.mz",
    "right_gripper_force",
)
_DYNAMIC_SCHEMA_METADATA_KEYS = {"zero_ft_sensor_on_connect"}


class FlexivSchemaCompatibilityError(ValueError):
    """Raised when data or a checkpoint cannot satisfy the current Flexiv contract."""


def flexiv_kinematic_state_names(num_joints_per_arm: int = 7) -> list[str]:
    if int(num_joints_per_arm) != 7:
        raise ValueError(
            "Flexiv absolute rotation-6D schema requires exactly 7 joints per arm; "
            f"got {num_joints_per_arm}."
        )

    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_joint_{index}.pos" for index in range(1, 8))
        names.extend(f"{side}_ee_pose.{axis}" for axis in STATE_POSITION_FIELDS)
        names.extend(f"{side}_ee_rotation_6d.{component}" for component in STATE_ROTATION_6D_FIELDS)
        names.append(f"{side}_gripper_state_norm")
    if len(names) != FLEXIV_KINEMATIC_STATE_DIM:
        raise AssertionError(
            f"Flexiv kinematic state contract must contain {FLEXIV_KINEMATIC_STATE_DIM} fields"
        )
    return names


def flexiv_state_names(num_joints_per_arm: int = 7) -> list[str]:
    """Return complete legacy 34D kinematics followed by the 14 raw-force fields."""

    names = flexiv_kinematic_state_names(num_joints_per_arm)
    names.extend(STATE_FORCE_FIELDS)
    if len(names) != FLEXIV_STATE_DIM:
        raise AssertionError(f"Flexiv state contract must contain {FLEXIV_STATE_DIM} fields")
    return names


def flexiv_action_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_delta_ee_pose.{axis}" for axis in ACTION_DELTA_POSE_FIELDS)
    names.extend(("left_gripper_cmd", "right_gripper_cmd"))
    return names


def build_flexiv_state_schema(
    *,
    zero_ft_sensor_on_connect: bool | None = None,
) -> dict[str, Any]:
    """Return the JSON-serializable schema contract for new Flexiv recordings."""

    return {
        "state_schema": FLEXIV_STATE_SCHEMA,
        "state_dim": FLEXIV_STATE_DIM,
        "action_dim": FLEXIV_ACTION_DIM,
        "wrench_source": WRENCH_SOURCE,
        "wrench_frame": WRENCH_FRAME,
        "wrench_order": list(WRENCH_ORDER),
        "wrench_units": list(WRENCH_UNITS),
        "gripper_force_source": GRIPPER_FORCE_SOURCE,
        "gripper_force_unit": GRIPPER_FORCE_UNIT,
        "gripper_force_sign_convention": "preserve_raw_signed_value",
        "software_filter": "none",
        "zero_ft_sensor_on_connect": zero_ft_sensor_on_connect,
        "state_rotation_representation": FLEXIV_STATE_ROTATION_REPRESENTATION,
        "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
        "action_rotation_representation": FLEXIV_ACTION_ROTATION_REPRESENTATION,
        "rotation6d_convention": FLEXIV_ROTATION6D_CONVENTION,
        "rotation6d_order": list(FLEXIV_ROTATION6D_ORDER),
        "state_names": flexiv_state_names(),
        "action_names": flexiv_action_names(),
    }


def persist_flexiv_dataset_schema(
    dataset_root: str | Path,
    info: Mapping[str, Any],
    *,
    zero_ft_sensor_on_connect: bool | None = None,
) -> dict[str, Any]:
    """Add the Flexiv schema to a new dataset's standard ``meta/info.json``.

    LeRobot preserves unknown top-level info keys when it updates episode metadata,
    so no second dataset format or sidecar is needed for this contract.
    """

    if not isinstance(zero_ft_sensor_on_connect, bool):
        raise ValueError(
            "zero_ft_sensor_on_connect must be explicitly provided as a boolean when persisting "
            "a Flexiv dataset schema; automatic ZeroFTSensor is not enabled."
        )
    root = Path(dataset_root)
    info_copy = dict(info)
    schema = build_flexiv_state_schema(zero_ft_sensor_on_connect=zero_ft_sensor_on_connect)
    info_copy[FLEXIV_SCHEMA_INFO_KEY] = schema
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info_copy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return schema


def propagate_flexiv_dataset_schema(
    source_info: Mapping[str, Any],
    output_dataset_or_root: Any,
    *,
    source_features: Mapping[str, Any],
    output_features: Mapping[str, Any],
    source: str,
) -> dict[str, Any] | None:
    """Copy and validate v3 metadata through a LeRobot rewrite operation.

    Non-Flexiv datasets are ignored. Flexiv sources and outputs are validated
    before the destination metadata is written, so preprocess/split/merge/
    DAgger paths cannot silently drop the 48D contract.
    """

    if source_info.get("robot_type") != "flexiv_dual_arm":
        return None
    validate_flexiv_dataset_schema(source_info, source_features, source=f"{source} source")
    validate_flexiv_feature_schema(output_features, source=f"{source} output")

    if hasattr(output_dataset_or_root, "meta"):
        meta = output_dataset_or_root.meta
        output_root = Path(meta.root)
        output_info = dict(meta.info)
    else:
        output_root = Path(output_dataset_or_root)
        info_path = output_root / "meta" / "info.json"
        output_info = json.loads(info_path.read_text(encoding="utf-8"))

    schema = dict(source_info[FLEXIV_SCHEMA_INFO_KEY])
    output_info[FLEXIV_SCHEMA_INFO_KEY] = schema
    if hasattr(output_dataset_or_root, "meta"):
        meta.info[FLEXIV_SCHEMA_INFO_KEY] = schema
    info_path = output_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(output_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return schema


def _feature_shape(feature: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if not isinstance(feature, Mapping) or "shape" not in feature:
        return None
    try:
        return tuple(int(value) for value in feature["shape"])
    except (TypeError, ValueError):
        return None


def _feature_names(feature: Mapping[str, Any] | None) -> list[str] | None:
    if not isinstance(feature, Mapping):
        return None
    names = feature.get("names")
    return list(names) if isinstance(names, (list, tuple)) else None


def _dimension(shape: tuple[int, ...] | None) -> int | str:
    if shape is None:
        return "missing"
    if len(shape) == 1:
        return shape[0]
    return str(shape)


def _state_representation(shape: tuple[int, ...] | None, names: list[str] | None) -> str:
    if shape == (28,) or (names and any(name.endswith("ee_pose.rx") for name in names)):
        return "absolute_rotvec"
    if shape == (34,) and names and all(
        name in names for name in ("left_ee_rotation_6d.c0x", "right_ee_rotation_6d.c1z")
    ):
        return "absolute_rotation6d_v2"
    if shape == (FLEXIV_STATE_DIM,) and names and all(
        name in names for name in ("left_ee_rotation_6d.c0x", "right_ee_rotation_6d.c1z")
    ):
        return "absolute_rotation6d_raw_force_v3"
    return "unknown"


def _schema_mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    mismatches = []
    for key, expected_value in expected.items():
        if key in _DYNAMIC_SCHEMA_METADATA_KEYS:
            continue
        if actual.get(key) != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual.get(key)!r}")
    return mismatches


def validate_flexiv_feature_schema(
    features: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Validate the LeRobot feature map before recording or appending frames."""

    state_feature = features.get("observation.state")
    action_feature = features.get("action")
    state_shape = _feature_shape(state_feature)
    state_names = _feature_names(state_feature)
    action_shape = _feature_shape(action_feature)
    action_names = _feature_names(action_feature)
    expected_state_names = flexiv_state_names()
    expected_action_names = flexiv_action_names()

    state_problems = []
    if state_shape != (FLEXIV_STATE_DIM,):
        state_problems.append(
            f"state_dim={_dimension(state_shape)} representation={_state_representation(state_shape, state_names)}"
        )
    if state_names != expected_state_names:
        state_problems.append(
            f"state_names={state_names!r} expected_order={expected_state_names!r}"
        )
    if state_problems:
        raise FlexivSchemaCompatibilityError(
            f"{source} Flexiv state schema mismatch: {'; '.join(state_problems)}. "
            f"Current contract is state_dim={FLEXIV_STATE_DIM}, "
            f"representation={FLEXIV_STATE_ROTATION_REPRESENTATION}, "
            f"reference={FLEXIV_STATE_ROTATION_REFERENCE}; legacy v2/34D and v1/28D "
            "contracts cannot be resumed or appended, and no force padding is allowed."
        )

    action_problems = []
    if action_shape != (FLEXIV_ACTION_DIM,):
        action_problems.append(f"action_dim={_dimension(action_shape)}")
    if action_names != expected_action_names:
        action_problems.append(
            f"action_names={action_names!r} expected_order={expected_action_names!r}"
        )
    if action_problems:
        raise FlexivSchemaCompatibilityError(
            f"{source} Flexiv action schema mismatch: {'; '.join(action_problems)}. "
            f"Current contract requires action_dim={FLEXIV_ACTION_DIM} with unchanged "
            "3D delta-rotvec fields and ordering."
        )


def validate_flexiv_dataset_schema(
    info: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Validate both feature order and persisted metadata for an existing dataset."""

    validate_flexiv_feature_schema(features, source=source)
    persisted = info.get(FLEXIV_SCHEMA_INFO_KEY)
    if not isinstance(persisted, Mapping):
        raise FlexivSchemaCompatibilityError(
            f"{source} is missing meta/info.json[{FLEXIV_SCHEMA_INFO_KEY!r}]. "
            f"The current contract is state_schema={FLEXIV_STATE_SCHEMA!r}, "
            f"state_dim={FLEXIV_STATE_DIM}, representation={FLEXIV_STATE_ROTATION_REPRESENTATION}; "
            "an existing dataset must be migrated/exported separately and will not be modified by recording."
        )

    zero_ft = persisted.get("zero_ft_sensor_on_connect")
    if not isinstance(zero_ft, bool):
        raise FlexivSchemaCompatibilityError(
            f"{source} persisted Flexiv schema must record boolean "
            f"zero_ft_sensor_on_connect, got {zero_ft!r}."
        )

    expected = build_flexiv_state_schema()
    mismatches = _schema_mismatches(persisted, expected)
    if mismatches:
        raise FlexivSchemaCompatibilityError(
            f"{source} persisted Flexiv schema mismatch: {'; '.join(mismatches)}. "
            f"Expected state_schema={FLEXIV_STATE_SCHEMA!r}, state_dim={FLEXIV_STATE_DIM}, "
            f"representation={FLEXIV_STATE_ROTATION_REPRESENTATION}; existing data is not modified."
        )


def _load_checkpoint_schema(pretrained_dir: Path) -> Mapping[str, Any] | None:
    config_path = pretrained_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlexivSchemaCompatibilityError(
                f"Cannot read checkpoint config for Flexiv schema validation: {config_path}"
            ) from exc
        embedded = config.get(FLEXIV_SCHEMA_INFO_KEY)
        if isinstance(embedded, Mapping):
            return embedded

    candidates = (
        pretrained_dir / FLEXIV_SCHEMA_CHECKPOINT_FILENAME,
        pretrained_dir / "robot_state_schema.json",
        pretrained_dir.parent / FLEXIV_SCHEMA_CHECKPOINT_FILENAME,
        pretrained_dir.parent / "robot_state_schema.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlexivSchemaCompatibilityError(
                f"Cannot read persisted Flexiv checkpoint schema: {path}"
            ) from exc
        if isinstance(payload, Mapping) and isinstance(payload.get(FLEXIV_SCHEMA_INFO_KEY), Mapping):
            return payload[FLEXIV_SCHEMA_INFO_KEY]
        if isinstance(payload, Mapping):
            return payload
    return None


def validate_flexiv_checkpoint(
    pretrained_path: str | Path | None,
    *,
    source: str,
) -> None:
    """Fail before policy construction when a checkpoint cannot consume state 48D."""

    if pretrained_path is None:
        return
    pretrained_dir = Path(pretrained_path).expanduser()
    if not pretrained_dir.is_dir():
        return

    config_path = pretrained_dir / "config.json"
    if not config_path.is_file():
        raise FlexivSchemaCompatibilityError(
            f"{source} has no config.json for Flexiv schema validation: {config_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlexivSchemaCompatibilityError(
            f"Cannot read {source} config.json for Flexiv schema validation: {config_path}"
        ) from exc

    input_state = config.get("input_features", {}).get("observation.state", {})
    output_action = config.get("output_features", {}).get("action", {})
    state_shape = _feature_shape(input_state)
    state_names = _feature_names(input_state)
    action_shape = _feature_shape(output_action)
    action_names = _feature_names(output_action)
    state_dim = _dimension(state_shape)
    action_dim = _dimension(action_shape)
    if state_shape != (FLEXIV_STATE_DIM,) or action_shape != (FLEXIV_ACTION_DIM,):
        representation = _state_representation(state_shape, None)
        raise FlexivSchemaCompatibilityError(
            f"{source} is incompatible with the current Flexiv contract: "
            f"checkpoint state_dim={state_dim}, representation={representation}, "
            f"action_dim={action_dim}; current state_dim={FLEXIV_STATE_DIM}, "
            f"representation={FLEXIV_STATE_ROTATION_REPRESENTATION}, "
            f"action_dim={FLEXIV_ACTION_DIM}. Legacy v2/34D and v1/28D checkpoints "
            "cannot be used with the new 48D raw-force observation."
        )

    name_mismatches = []
    if state_names is not None and state_names != flexiv_state_names():
        name_mismatches.append("checkpoint observation.state names do not match the v3 order")
    if action_names is not None and action_names != flexiv_action_names():
        name_mismatches.append("checkpoint action names do not match the unchanged 14D order")
    if name_mismatches:
        raise FlexivSchemaCompatibilityError(
            f"{source} feature-name schema mismatch: {'; '.join(name_mismatches)}."
        )

    persisted = _load_checkpoint_schema(pretrained_dir)
    if persisted is None:
        raise FlexivSchemaCompatibilityError(
            f"{source} has state_dim={FLEXIV_STATE_DIM} but no persisted Flexiv schema metadata. "
            f"Cannot prove representation={FLEXIV_STATE_ROTATION_REPRESENTATION}; retrain or export "
            "the checkpoint with the current schema before using it."
        )
    mismatches = _schema_mismatches(persisted, build_flexiv_state_schema())
    if mismatches:
        raise FlexivSchemaCompatibilityError(
            f"{source} persisted Flexiv schema mismatch: {'; '.join(mismatches)}. "
            f"Expected state_schema={FLEXIV_STATE_SCHEMA!r}, state_dim={FLEXIV_STATE_DIM}, "
            f"representation={FLEXIV_STATE_ROTATION_REPRESENTATION}."
        )


def persist_flexiv_checkpoint_schema(pretrained_dir: str | Path) -> Path:
    """Persist the state contract beside a newly saved policy checkpoint."""

    path = Path(pretrained_dir) / FLEXIV_SCHEMA_CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_flexiv_state_schema(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
