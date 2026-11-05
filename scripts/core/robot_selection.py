"""Hardware config selection shared by recording and one-shot commands."""

from copy import deepcopy
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


def merge_config(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def add_selection_arguments(parser) -> None:
    parser.add_argument("--robot-type", choices=tuple(ROBOT_DETAIL_CONFIG_FILES),
                        help="Select the embodiment for this command.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--station-config", type=Path,
                       help="Station overrides; defaults to station.local.yaml beside --config.")
    group.add_argument("--no-station-config", action="store_true",
                       help="Use the shared config without this machine's station overrides.")


def load_session_config(path: Path, *, robot_type: str | None = None,
                        station_path: Path | None = None, use_station: bool = True) -> dict:
    """Keep shared source identical while resolving machine and embodiment settings."""
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Session config must be a mapping: {path}")
    station = {}
    candidate = Path(station_path).expanduser().resolve() if station_path else path.with_name("station.local.yaml")
    if use_station and (station_path is not None or candidate.is_file()):
        station = yaml.safe_load(candidate.read_text())
        if not isinstance(station, dict):
            raise ValueError(f"Station config must be a mapping: {candidate}")
    for section in ("record", "replay", "visualize"):
        overrides = station.get(section, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Station {section} must be a mapping.")
        if overrides:
            cfg[section] = merge_config(cfg.get(section, {}), overrides)
    record = cfg.get("record", {})
    if not isinstance(record, dict):
        raise ValueError(f"Config record must be a mapping: {path}")
    previous = record.get("robot_type") or cfg.get("robot_type")
    selected = robot_type or station.get("default_robot_type") or previous
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise ValueError("robot_type must be a nonempty string.")
    if selected != previous and any(key in record for key in ("robot", "teleop", "cameras", "das_config_path", "robot_config_path")):
        raise ValueError("Cannot switch robot_type with inline hardware overrides; move them to station.robots by embodiment.")
    robots = station.get("robots", {})
    if not isinstance(robots, dict):
        raise ValueError("Station robots must be a mapping.")
    hardware = robots.get(selected, {})
    if not isinstance(hardware, dict) or set(hardware) - {"robot", "teleop", "cameras"}:
        raise ValueError("Station robot entries support only robot, teleop and cameras mappings.")
    if any(not isinstance(value, dict) for value in hardware.values()):
        raise ValueError("Station hardware sections must be mappings.")
    if selected is not None:
        cfg["robot_type"] = selected
        for section in ("record", "replay"):
            if section in cfg:
                if not isinstance(cfg[section], dict):
                    raise ValueError(f"Config {section} must be a mapping: {path}")
                cfg[section]["robot_type"] = selected
        if "record" in cfg:
            cfg["record"] = merge_config(cfg["record"], hardware)
        if "replay" in cfg and "robot" in hardware:
            cfg["replay"]["robot"] = merge_config(cfg["replay"].get("robot", {}), hardware["robot"])
    return cfg


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


def load_selected_robot(path: Path, **selection) -> tuple[str, dict]:
    """Resolve the selected type, detail config and inline robot overrides."""
    cfg = load_session_config(path, **selection)
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
