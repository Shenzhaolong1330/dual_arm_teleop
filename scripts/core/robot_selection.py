"""Hardware config selection shared by recording and one-shot commands."""

from pathlib import Path

import yaml


ROBOT_DETAIL_CONFIG_FILES = {
    "dobot_dual_arm": "dobot_config.yaml",
    "franka": "franka_config.yaml",
    "franka_dual_arm": "franka_config.yaml",
    "nero_dual_arm": "nero_cofig.yaml",
    "arx_dual_arm": "arx_config.yaml",
    "flexiv_dual_arm": "flexiv_config.yaml",
}
ROBOT_CONFIG_DIR = Path("config") / "robots"
LEGACY_ROBOT_CONFIG_DIR = Path("DAS_config")


def default_record_cfg_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "record_cfg.yaml"


def _resolve_das_config_path(
    robot_type: str,
    scripts_dir: Path,
    project_root: Path,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_absolute():
            return path
        candidates = (
            project_root / path,
            scripts_dir / path,
            scripts_dir / ROBOT_CONFIG_DIR / path,
            scripts_dir / ROBOT_CONFIG_DIR / path.name,
            project_root / "scripts" / ROBOT_CONFIG_DIR / path.name,
            scripts_dir / LEGACY_ROBOT_CONFIG_DIR / path,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    config_name = ROBOT_DETAIL_CONFIG_FILES.get(robot_type)
    if config_name is None:
        raise ValueError(
            "No robot config mapping is defined for robot_type="
            f"{robot_type!r}. Add record.das_config_path or extend "
            "ROBOT_DETAIL_CONFIG_FILES."
        )
    return scripts_dir / ROBOT_CONFIG_DIR / config_name


def load_selected_robot(path: Path) -> tuple[str, dict]:
    """Resolve the selected type, detail config and inline robot overrides."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("record"), dict):
        raise ValueError(f"Config must contain a record mapping: {path}")
    record = cfg["record"]
    robot_type = record.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type:
        raise ValueError("record.robot_type must explicitly select a robot.")
    overrides = record.get("robot")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("record.robot must be a mapping.")
    scripts_dir = Path(__file__).resolve().parents[1]
    explicit = record.get("das_config_path") or record.get("robot_config_path")
    if robot_type not in ROBOT_DETAIL_CONFIG_FILES and not explicit:
        return robot_type, dict(overrides or {})
    detail_path = _resolve_das_config_path(robot_type, scripts_dir, scripts_dir.parent, explicit)
    detail = yaml.safe_load(detail_path.read_text())
    if not isinstance(detail, dict):
        raise ValueError(f"Robot config must be a mapping: {detail_path}")
    detail = detail.get("record", detail)
    if not isinstance(detail, dict) or not isinstance(detail.get("robot"), dict):
        raise ValueError(f"Robot config must contain a robot mapping: {detail_path}")
    return robot_type, {**detail["robot"], **(overrides or {})}
