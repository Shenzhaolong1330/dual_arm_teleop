"""PRIMEU dataset/checkpoint contracts; no robot or model imports."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


STATE_SLICES = {
    "left_arm": {"start": 0, "end": 7},
    "left_hand": {"start": 7, "end": 27},
    "right_arm": {"start": 27, "end": 34},
    "right_hand": {"start": 34, "end": 54},
    "head": {"start": 54, "end": 57},
    "waist": {"start": 57, "end": 60},
}
CAMERAS = ("head", "left_wrist", "right_wrist")
ACTION_GROUPS = {
    60: (("left_arm", 7), ("left_hand", 20), ("right_arm", 7), ("right_hand", 20), ("head", 3), ("waist", 3)),
    33: (("left_arm", 7), ("left_hand_synergy", 8), ("right_arm", 7), ("right_hand_synergy", 5), ("head", 3), ("waist", 3)),
    27: (("left_arm", 7), ("left_hand_synergy", 8), ("right_arm", 7), ("right_hand_synergy", 5)),
    15: (("left_arm", 7), ("left_hand_synergy", 8)),
    12: (("right_arm", 7), ("right_hand_synergy", 5)),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def expected_action_slices(action_dim):
    if action_dim not in ACTION_GROUPS:
        raise ValueError(f"Unsupported PRIMEU action dimension: {action_dim}")
    result, start = {}, 0
    for group, size in ACTION_GROUPS[action_dim]:
        result[group] = {"start": start, "end": start + size}
        start += size
    return result


def _slices(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("Missing modality mapping")
    result = {}
    for name, bounds in mapping.items():
        if not isinstance(bounds, dict) or set(("start", "end")) - bounds.keys():
            raise ValueError(f"Missing bounds for modality {name}")
        if any(type(bounds[k]) is not int for k in ("start", "end")):
            raise ValueError(f"Non-integer bounds for modality {name}")
        result[name] = {k: bounds[k] for k in ("start", "end")}
    return result


def validate_basis(path, side):
    basis = json.loads(Path(path).read_text())
    expected_names = ["index", "middle", "ring", "little"] if side == "left" else ["thumb", "index", "middle", "ring", "little"]
    latent_order = (["thumb_j1", "thumb_j2", "thumb_j3", "thumb_j4"] if side == "left" else ["thumb_synergy"]) + [f"{name}_synergy" for name in ["index", "middle", "ring", "little"]]
    if (basis.get("kind") != "chebyshev_taskspace" or basis.get("version") != 1
            or basis.get("hand") != side or basis.get("latent_order") != latent_order):
        raise ValueError(f"Invalid {side} hand basis identity/order: {path}")
    fingers = basis.get("fingers", [])
    if [finger.get("name") for finger in fingers] != expected_names:
        raise ValueError(f"Invalid {side} finger order")
    for index, finger in enumerate(fingers, start=2 if side == "left" else 1):
        coefficients = np.asarray(finger.get("coefficients"), dtype=np.float64)
        degree = finger.get("degree")
        expected_joints = [f"{side}_finger{index}_link{j}_joint" for j in range(1, 5)]
        if (type(degree) is not int or degree < 0 or coefficients.shape != (degree + 1, 4)
                or not np.isfinite(coefficients).all() or finger.get("joint_names") != expected_joints
                or finger.get("coordinate_range") != [-1.0, 1.0]):
            raise ValueError(f"Invalid {side} {finger.get('name')} curve")
        if finger.get("clip_joint_limits", False):
            lower = np.asarray(finger.get("joint_lower"), dtype=np.float64)
            upper = np.asarray(finger.get("joint_upper"), dtype=np.float64)
            if (lower.shape != (4,) or upper.shape != (4,) or not np.isfinite(lower).all()
                    or not np.isfinite(upper).all() or np.any(lower > upper)):
                raise ValueError(f"Invalid {side} finger limits")
    return basis


@dataclass(frozen=True)
class PrimeuContract:
    action_dim: int
    state_slices: dict
    action_slices: dict
    cameras: dict
    basis_files: dict
    basis_sha256: dict
    dataset_fingerprint: str
    state_dim: int = 60
    version: int = 1
    action_semantics: str = "stored_absolute_joint_and_synergy_v1"
    hold_mode: str = "initial_observation_until_reset"

    @property
    def contract_sha256(self):
        return _digest(asdict(self))

    @property
    def variant(self):
        return f"action{self.action_dim}"

    @property
    def held_groups(self):
        predicted = {name.replace("_synergy", "") for name in self.action_slices}
        return [name for name in STATE_SLICES if name not in predicted]

    def to_dict(self):
        return {**asdict(self), "contract_sha256": self.contract_sha256}


def _validate_contract(contract):
    if (contract.version != 1 or contract.state_dim != 60 or contract.state_slices != STATE_SLICES
            or contract.action_slices != expected_action_slices(contract.action_dim)
            or contract.cameras != {name: f"observation.images.{name}" for name in CAMERAS}
            or contract.action_semantics != "stored_absolute_joint_and_synergy_v1"
            or contract.hold_mode != "initial_observation_until_reset"):
        raise ValueError("Invalid PRIMEU contract layout or semantics")
    sides = {side for side in ("left", "right") if f"{side}_hand_synergy" in contract.action_slices}
    if set(contract.basis_files) != sides or set(contract.basis_sha256) != sides:
        raise ValueError("Contract basis sides do not match action layout")
    for name in contract.basis_files.values():
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Basis paths must be plain filenames")


def verify_basis_files(contract, basis_dir):
    _validate_contract(contract)
    bases = {}
    for side, name in contract.basis_files.items():
        path = Path(basis_dir) / name
        if sha256(path) != contract.basis_sha256[side]:
            raise ValueError(f"Wrong {side} hand basis SHA256: {path}")
        bases[side] = validate_basis(path, side)
    return bases


def load_dataset_contract(root):
    meta = Path(root) / "meta"
    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    features = info["features"]
    if features["observation.state"]["shape"] != [60] or len(features["action"]["shape"]) != 1:
        raise ValueError("PRIMEU requires 60D state and one-dimensional action")
    dim = features["action"]["shape"][0]
    state_slices, action_slices = _slices(modality.get("state")), _slices(modality.get("action"))
    if state_slices != STATE_SLICES or action_slices != expected_action_slices(dim):
        raise ValueError("Dataset modality differs from the supported PRIMEU field order")
    cameras = {name: modality.get("video", {}).get(name, {}).get("original_key") for name in CAMERAS}
    if cameras != {name: f"observation.images.{name}" for name in CAMERAS}:
        raise ValueError("Missing or reordered PRIMEU camera mapping")
    if any(features.get(key, {}).get("dtype") != "video" for key in cameras.values()):
        raise ValueError("Expected three PRIMEU video features")
    layout_path = meta / "action_layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}
    if dim != 60:
        if (layout.get("action_dim") != dim or layout.get("state_dim") != 60
                or _slices(layout.get("action_slices")) != action_slices
                or _slices(layout.get("state_slices")) != state_slices):
            raise ValueError("Reduced dataset requires matching action_layout.json")
        # These datasets annotate GR00T transforms here; raw Parquet remains absolute.
        raw = layout.get("raw_body_actions")
        if raw is not None and not str(raw).lower().startswith("absolute"):
            raise ValueError("Unsupported raw action semantics; expected absolute joint targets")
    basis_files = {side: layout.get(f"{side}_hand_basis") for side in ("left", "right")
                   if f"{side}_hand_synergy" in action_slices}
    if any(not isinstance(name, str) or Path(name).name != name for name in basis_files.values()):
        raise ValueError("Missing or unsafe dataset hand basis path")
    digests = {side: sha256(meta / name) for side, name in basis_files.items()}
    metadata_hashes = {name: sha256(meta / name) for name in
                       ("info.json", "modality.json", "action_layout.json", "stats.json") if (meta / name).exists()}
    contract = PrimeuContract(dim, state_slices, action_slices, cameras, basis_files, digests,
                              _digest({"metadata": metadata_hashes, "basis": digests}))
    verify_basis_files(contract, meta)
    return contract


def load_checkpoint_contract(checkpoint):
    directory = Path(checkpoint) / "primeu"
    payload = json.loads((directory / "contract.json").read_text())
    expected = payload.pop("contract_sha256")
    contract = PrimeuContract(**payload)
    if contract.contract_sha256 != expected:
        raise ValueError("Checkpoint PRIMEU contract SHA256 mismatch")
    verify_basis_files(contract, directory)
    return contract


def export_contract(dataset_root, checkpoint):
    contract = load_dataset_contract(dataset_root)
    directory = Path(checkpoint) / "primeu"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "contract.json"
    if target.exists() and load_checkpoint_contract(checkpoint).contract_sha256 != contract.contract_sha256:
        raise ValueError("Refusing to overwrite a different checkpoint contract")
    for name in contract.basis_files.values():
        shutil.copy2(Path(dataset_root) / "meta" / name, directory / name)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract.to_dict(), indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    load_checkpoint_contract(checkpoint)
    return contract
